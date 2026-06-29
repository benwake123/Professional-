"""
Geometric Brownian motion (GBM) benchmark model.

Purpose
-------
Provides the simplest possible volatility forecast: constant annualized
sigma estimated from a window of historical log returns. Used as the
benchmark in the variance-risk-premium signal so Heston-based forecasts can
be measured against it.

Module connections
------------------
Upstream:
    - ``numpy`` for random draws and array math.
Downstream:
    - ``src.calibration.calibrate_gbm`` calls ``estimate_gbm_parameters``.
    - ``src.monte_carlo.forecast_realized_variance`` dispatches to
      ``simulate_gbm_paths`` when ``model_name == 'gbm'``.
    - ``tests/test_gbm_heston.py`` pins the path-shape invariant.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ANNUALIZATION_FACTOR: int = 252


def estimate_gbm_parameters(
    log_returns: np.ndarray,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> dict[str, float]:
    """Estimate annualized drift ``mu`` and volatility ``sigma`` from log returns."""
    arr = np.asarray(log_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("estimate_gbm_parameters requires at least 2 finite returns.")
    daily_mean = float(arr.mean())
    daily_std = float(arr.std(ddof=1))
    sigma = daily_std * np.sqrt(annualization_factor)
    mu = daily_mean * annualization_factor + 0.5 * sigma * sigma
    return {"mu": float(mu), "sigma": float(sigma)}


def simulate_gbm_paths(
    initial_price: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_paths: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate ``n_paths`` GBM price paths of length ``horizon_days + 1``."""
    if initial_price <= 0:
        raise ValueError("initial_price must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if horizon_days < 0 or n_paths <= 0:
        raise ValueError("horizon_days >= 0 and n_paths > 0 required.")

    dt = 1.0 / annualization_factor
    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal(size=(n_paths, horizon_days))
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_increments = drift + diffusion

    log_paths = np.zeros((n_paths, horizon_days + 1))
    log_paths[:, 0] = np.log(initial_price)
    log_paths[:, 1:] = np.log(initial_price) + np.cumsum(log_increments, axis=1)
    return np.exp(log_paths)


def gbm_expected_variance(sigma: float) -> float:
    """Constant expected variance under GBM is simply ``sigma**2`` (already annualized)."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    return float(sigma * sigma)
