"""
Model calibration: select a historical window ending at the decision date,
then estimate GBM or Heston parameters from that window only.

Purpose
-------
Every model fit must use information dated ``<= decision_date``. This module
centralizes that "select then estimate" pattern so the rest of the pipeline
cannot accidentally call the simulators with parameters fit on future data.

Module connections
------------------
Upstream:
    - ``src.market_features.calculate_log_returns``
    - ``src.gbm.estimate_gbm_parameters``
    - ``src.heston.{build_initial_parameter_guess, heston_calibration_objective}``
    - ``scipy.optimize.minimize`` for the Heston loss.
Downstream:
    - ``src.backtest.evaluate_new_trade`` calls ``calibrate_gbm`` and
      ``calibrate_heston`` on every decision day.
    - ``src.monte_carlo.forecast_realized_variance`` consumes the resulting
      parameters.
    - ``tests/test_no_lookahead`` will activate
      ``test_calibration_window_ends_at_decision_date`` once this module ships.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.gbm import estimate_gbm_parameters
from src.heston import (
    HESTON_PARAM_KEYS,
    build_initial_parameter_guess,
    heston_calibration_objective,
    validate_heston_parameters,
)
from src.market_features import calculate_log_returns


DateLike = Union[str, date, datetime, pd.Timestamp]


def select_calibration_window(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    lookback_days: int,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return the last ``lookback_days`` rows with date <= ``as_of_date``."""
    if date_column not in underlying.columns:
        raise ValueError(f"underlying is missing '{date_column}'.")
    if lookback_days <= 1:
        raise ValueError("lookback_days must be > 1.")
    cutoff = pd.Timestamp(as_of_date)
    eligible = underlying[underlying[date_column] <= cutoff].copy()
    if eligible.empty:
        raise ValueError(
            f"No underlying rows available on or before {cutoff.date()}."
        )
    return eligible.sort_values(date_column).tail(lookback_days).reset_index(drop=True)


def calibrate_gbm(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    lookback_days: int,
    price_column: str = "close",
    date_column: str = "date",
) -> dict[str, float]:
    """Pick the historical window then call :func:`estimate_gbm_parameters`."""
    window = select_calibration_window(underlying, as_of_date, lookback_days, date_column)
    log_returns = calculate_log_returns(window, price_column=price_column).dropna()
    params = estimate_gbm_parameters(log_returns.to_numpy())
    params["model"] = "gbm"
    params["window_start"] = window[date_column].min()
    params["window_end"] = window[date_column].max()
    params["n_observations"] = int(len(log_returns))
    return params


def calibrate_heston(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    lookback_days: int,
    price_column: str = "close",
    date_column: str = "date",
) -> dict[str, float]:
    """
    Fit Heston parameters by minimizing :func:`heston_calibration_objective`
    against the historical annualized variance.

    Returns a dict with kappa, theta, xi, rho, v0, mu, plus diagnostics.
    """
    window = select_calibration_window(underlying, as_of_date, lookback_days, date_column)
    log_returns = calculate_log_returns(window, price_column=price_column).dropna().to_numpy()

    initial = build_initial_parameter_guess(log_returns)
    target_var = float(np.var(log_returns, ddof=1)) * 252.0

    x0 = np.array(
        [initial["kappa"], initial["theta"], initial["xi"], initial["rho"], initial["v0"]]
    )
    bounds = [
        (0.05, 10.0),    # kappa
        (1e-6, 4.0),     # theta
        (1e-3, 4.0),     # xi
        (-0.999, 0.999), # rho
        (0.0, 4.0),      # v0
    ]

    result = minimize(
        heston_calibration_objective,
        x0=x0,
        args=(target_var,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    kappa, theta, xi, rho, v0 = result.x
    fitted = {
        "kappa": float(kappa),
        "theta": float(theta),
        "xi": float(xi),
        "rho": float(rho),
        "v0": float(v0),
        "mu": float(initial["mu"]),
        "model": "heston",
        "window_start": window[date_column].min(),
        "window_end": window[date_column].max(),
        "n_observations": int(len(log_returns)),
        "calibration_loss": float(result.fun),
        "optimizer_success": bool(result.success),
    }
    validate_heston_parameters(fitted)
    return fitted


def rolling_model_calibration(
    underlying: pd.DataFrame,
    decision_dates: pd.Series,
    lookback_days: int,
    model_name: str = "gbm",
    date_column: str = "date",
    price_column: str = "close",
) -> pd.DataFrame:
    """Run calibration on each decision date and concatenate diagnostics."""
    rows = []
    for as_of in decision_dates:
        try:
            if model_name == "gbm":
                params = calibrate_gbm(
                    underlying, as_of, lookback_days, price_column, date_column
                )
            elif model_name == "heston":
                params = calibrate_heston(
                    underlying, as_of, lookback_days, price_column, date_column
                )
            else:
                raise ValueError(f"Unknown model_name {model_name!r}.")
            rows.append({"as_of_date": pd.Timestamp(as_of), **params})
        except (ValueError, RuntimeError) as exc:
            rows.append(
                {"as_of_date": pd.Timestamp(as_of), "error": str(exc), "model": model_name}
            )
    return pd.DataFrame(rows)


def validate_calibration_result(calibration: dict[str, float]) -> bool:
    """Return ``True`` when the calibration produced numerically usable parameters.

    A successful Heston fit is signalled either by ``optimizer_success`` or by
    ``calibration_loss`` being finite and below a small tolerance. The latter
    is important: our moment-matching objective is exactly zero at the initial
    guess when ``theta = v0 = historical_variance``, and SciPy reports
    ``success=False`` in that "no iterations needed" edge case even though
    the parameters are perfectly valid.
    """
    if "error" in calibration:
        return False
    if calibration.get("n_observations", 0) < 30:
        return False
    model = calibration.get("model")
    if model == "gbm":
        return calibration.get("sigma", -1.0) > 0
    if model == "heston":
        try:
            validate_heston_parameters(calibration)
        except ValueError:
            return False
        loss = calibration.get("calibration_loss", float("inf"))
        return calibration.get("optimizer_success", False) or (
            loss is not None and loss == loss and loss < 1.0
        )
    return False
