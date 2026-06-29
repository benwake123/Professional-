"""
Monte Carlo forecasting of realized variance.

Purpose
-------
Given a calibrated GBM or Heston model, simulate price paths over a forward
horizon, compute realized variance per path, and summarize the resulting
distribution into the :class:`~src.types.ModelForecast` consumed by the
signal layer. Also exposes a payoff estimator used (when option data is
available) by ``src.option_selection``.

Module connections
------------------
Upstream:
    - ``src.gbm.simulate_gbm_paths``
    - ``src.heston.simulate_heston_paths``
    - ``src.types.ModelForecast``
Downstream:
    - ``src.signals.build_volatility_signal`` calls
      :func:`forecast_realized_variance` for the model forecast.
    - ``src.option_selection`` (when options exist) calls
      :func:`estimate_expected_option_payoff` to score candidate
      structures.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping, Union

import numpy as np
import pandas as pd

from src.gbm import simulate_gbm_paths
from src.heston import simulate_heston_paths
from src.types import ModelForecast

DateLike = Union[str, date, datetime, pd.Timestamp]
DEFAULT_ANNUALIZATION_FACTOR: int = 252


def calculate_path_realized_variance(
    paths: np.ndarray, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
) -> np.ndarray:
    """Annualized realized variance per simulated path."""
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be 2-D with at least 2 time steps.")
    log_returns = np.diff(np.log(paths), axis=1)
    daily_var = np.mean(log_returns ** 2, axis=1)
    return daily_var * annualization_factor


def summarize_forecast_distribution(
    path_variances: np.ndarray,
    quantiles: tuple[float, float] = (0.10, 0.90),
) -> dict[str, float]:
    """Return mean, median, lower/upper quantiles, and a Monte Carlo SE."""
    if path_variances.size == 0:
        raise ValueError("path_variances is empty.")
    lower, upper = quantiles
    return {
        "expected_variance": float(np.mean(path_variances)),
        "median_variance": float(np.median(path_variances)),
        "lower_quantile": float(np.quantile(path_variances, lower)),
        "upper_quantile": float(np.quantile(path_variances, upper)),
        "standard_error": calculate_monte_carlo_standard_error(path_variances),
        "n_paths": int(path_variances.size),
    }


def forecast_realized_variance(
    initial_price: float,
    parameters: Mapping[str, float],
    horizon_days: int,
    n_paths: int,
    as_of_date: DateLike,
    random_seed: int | None = None,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> ModelForecast:
    """Dispatch to GBM or Heston simulator, summarize, return ``ModelForecast``."""
    model_name = str(parameters.get("model", "gbm")).lower()

    if model_name == "gbm":
        paths = simulate_gbm_paths(
            initial_price=initial_price,
            mu=float(parameters["mu"]),
            sigma=float(parameters["sigma"]),
            horizon_days=horizon_days,
            n_paths=n_paths,
            annualization_factor=annualization_factor,
            random_seed=random_seed,
        )
        kept = {k: float(parameters[k]) for k in ("mu", "sigma")}
    elif model_name == "heston":
        paths, _ = simulate_heston_paths(
            initial_price=initial_price,
            parameters=parameters,
            horizon_days=horizon_days,
            n_paths=n_paths,
            annualization_factor=annualization_factor,
            random_seed=random_seed,
        )
        kept = {k: float(parameters[k]) for k in ("kappa", "theta", "xi", "rho", "v0", "mu")}
    else:
        raise ValueError(f"Unknown model_name {model_name!r}.")

    rv = calculate_path_realized_variance(paths, annualization_factor)
    summary = summarize_forecast_distribution(rv)

    as_of = pd.Timestamp(as_of_date).date()
    return ModelForecast(
        as_of_date=as_of,
        model_name=model_name,
        expected_variance=summary["expected_variance"],
        median_variance=summary["median_variance"],
        lower_quantile=summary["lower_quantile"],
        upper_quantile=summary["upper_quantile"],
        monte_carlo_standard_error=summary["standard_error"],
        parameters=kept,
    )


def estimate_expected_option_payoff(
    terminal_prices: np.ndarray,
    legs: Iterable[Mapping[str, object]],
) -> float:
    """Average terminal payoff of a multi-leg structure across simulated paths."""
    total = np.zeros_like(terminal_prices, dtype=float)
    for leg in legs:
        strike = float(leg["strike"])
        option_type = str(leg["option_type"]).lower()
        quantity = float(leg["quantity"])
        if option_type == "call":
            payoff = np.maximum(terminal_prices - strike, 0.0)
        elif option_type == "put":
            payoff = np.maximum(strike - terminal_prices, 0.0)
        else:
            raise ValueError(f"unknown option_type {option_type!r}")
        total = total + quantity * payoff
    return float(total.mean())


def calculate_monte_carlo_standard_error(values: np.ndarray) -> float:
    """Standard error of the mean: ``sd / sqrt(n)``."""
    n = values.size
    if n < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(n))
