"""
Heston stochastic-volatility model.

Purpose
-------
Implements the Heston (1993) joint dynamics for asset price and variance:

    dS_t  = mu  * S_t * dt + sqrt(v_t) * S_t * dW1_t
    dv_t  = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
    dW1 . dW2 = rho * dt

with parameters:
    kappa : mean reversion speed of variance
    theta : long-run variance
    xi    : volatility of variance ("vol of vol")
    rho   : correlation between price and variance shocks
    v0    : initial variance

Variance is propagated with the full-truncation Euler scheme, which is the
standard numerical fix that prevents the discretized variance from going
negative without distorting the long-run distribution badly.

Module connections
------------------
Upstream: ``numpy``.
Downstream:
    - ``src.calibration.calibrate_heston`` uses
      ``build_initial_parameter_guess`` + ``heston_calibration_objective``.
    - ``src.monte_carlo.forecast_realized_variance`` calls
      ``simulate_heston_paths`` when ``model_name == 'heston'``.
    - ``tests/test_gbm_heston.py`` pins shock correlation and nonneg
      variance invariants.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULT_ANNUALIZATION_FACTOR: int = 252

HESTON_PARAM_KEYS: tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0", "mu")


def build_initial_parameter_guess(
    log_returns: np.ndarray,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> dict[str, float]:
    """Build a reasonable starting point for Heston calibration from historical returns."""
    arr = np.asarray(log_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("build_initial_parameter_guess needs >= 2 returns.")

    var_annual = float(arr.var(ddof=1)) * annualization_factor
    mean_annual = float(arr.mean()) * annualization_factor
    return {
        "kappa": 2.0,
        "theta": var_annual,
        "xi": 0.30,
        "rho": -0.50,
        "v0": var_annual,
        "mu": mean_annual + 0.5 * var_annual,
    }


def validate_heston_parameters(parameters: Mapping[str, float]) -> None:
    """Raise ``ValueError`` if Heston parameters violate basic constraints."""
    missing = [k for k in ("kappa", "theta", "xi", "rho", "v0") if k not in parameters]
    if missing:
        raise ValueError(f"Heston parameters missing required keys: {missing}")
    if parameters["kappa"] <= 0:
        raise ValueError("Heston kappa must be > 0.")
    if parameters["theta"] <= 0:
        raise ValueError("Heston theta must be > 0.")
    if parameters["xi"] <= 0:
        raise ValueError("Heston xi must be > 0.")
    if parameters["v0"] < 0:
        raise ValueError("Heston v0 must be nonnegative.")
    if not -1.0 <= parameters["rho"] <= 1.0:
        raise ValueError("Heston rho must be in [-1, 1].")


def generate_correlated_shocks(
    n_paths: int, n_steps: int, rho: float, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return two arrays of shape ``(n_paths, n_steps)`` with correlation ``rho``."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1].")
    rng = np.random.default_rng(random_seed)
    z1 = rng.standard_normal(size=(n_paths, n_steps))
    z_ind = rng.standard_normal(size=(n_paths, n_steps))
    z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z_ind
    return z1, z2


def full_truncation_variance_step(
    variance: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    z2: np.ndarray,
) -> np.ndarray:
    """One full-truncation Euler step for the variance process."""
    v_plus = np.maximum(variance, 0.0)
    next_variance = (
        variance
        + kappa * (theta - v_plus) * dt
        + xi * np.sqrt(v_plus * dt) * z2
    )
    return np.maximum(next_variance, 0.0)


def heston_price_step(
    price: np.ndarray,
    variance: np.ndarray,
    mu: float,
    dt: float,
    z1: np.ndarray,
) -> np.ndarray:
    """One Euler step for the (log) price process using v_plus."""
    v_plus = np.maximum(variance, 0.0)
    log_increment = (mu - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * z1
    return price * np.exp(log_increment)


def simulate_heston_paths(
    initial_price: float,
    parameters: Mapping[str, float],
    horizon_days: int,
    n_paths: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate joint Heston price + variance paths."""
    validate_heston_parameters(parameters)
    if initial_price <= 0:
        raise ValueError("initial_price must be > 0.")
    if horizon_days < 0 or n_paths <= 0:
        raise ValueError("horizon_days >= 0 and n_paths > 0 required.")

    dt = 1.0 / annualization_factor
    kappa = float(parameters["kappa"])
    theta = float(parameters["theta"])
    xi = float(parameters["xi"])
    rho = float(parameters["rho"])
    v0 = float(parameters["v0"])
    mu = float(parameters.get("mu", 0.0))

    z1, z2 = generate_correlated_shocks(n_paths, horizon_days, rho, random_seed)

    prices = np.zeros((n_paths, horizon_days + 1))
    variances = np.zeros((n_paths, horizon_days + 1))
    prices[:, 0] = initial_price
    variances[:, 0] = v0

    for step in range(horizon_days):
        variances[:, step + 1] = full_truncation_variance_step(
            variances[:, step], kappa, theta, xi, dt, z2[:, step]
        )
        prices[:, step + 1] = heston_price_step(
            prices[:, step], variances[:, step], mu, dt, z1[:, step]
        )

    return prices, variances


def heston_calibration_objective(
    parameter_vector: np.ndarray,
    target_var_annual: float,
    target_skew_proxy: float = 0.0,
) -> float:
    """Cheap diagnostic loss: matches realized variance to ``theta``/``v0``.

    The full-fledged option-implied calibration would minimize the squared
    pricing error across a strip of liquid options. With the public data
    pack containing no options data, we substitute a moment-matching loss
    that ties Heston's long-run variance ``theta`` and initial variance
    ``v0`` to the historical annualized variance. This still exercises the
    calibration plumbing end to end.
    """
    kappa, theta, xi, rho, v0 = parameter_vector
    penalty = 0.0
    if kappa <= 0:
        penalty += 1e6
    if theta <= 0 or v0 < 0:
        penalty += 1e6
    if xi <= 0:
        penalty += 1e6
    if not -1.0 <= rho <= 1.0:
        penalty += 1e6
    target = max(target_var_annual, 1e-8)
    loss = (theta - target) ** 2 / (target ** 2) + (v0 - target) ** 2 / (target ** 2)
    # Light correlation prior: SPY equity returns and variance are typically
    # negatively correlated (leverage effect).
    loss += (rho + 0.5) ** 2
    return float(loss + penalty)
