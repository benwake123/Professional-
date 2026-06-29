"""
Multi-model variance forecasting and diagnostics.

Purpose
-------
Compares five forecast methods against subsequently realized variance over
a matched horizon:

    1. Historical rolling realized variance
    2. EWMA variance
    3. GBM constant-volatility forecast
    4. Heston Monte Carlo forecast
    5. Weighted ensemble (Heston + EWMA + historical)

Module connections
------------------
Upstream:
    - ``src.calibration``, ``src.gbm``, ``src.market_features``,
      ``src.monte_carlo``.
Downstream:
    - ``src.backtest``, ``src.audit``, ``src.research``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

from src.calibration import calibrate_gbm, calibrate_heston, validate_calibration_result
from src.gbm import gbm_expected_variance
from src.market_features import calculate_ewma_variance, calculate_log_returns
from src.monte_carlo import forecast_realized_variance
from src.types import ModelForecast

DateLike = Union[str, date, datetime, pd.Timestamp]
DEFAULT_ANNUALIZATION: int = 252


def _annualized_variance_from_returns(returns: pd.Series) -> float:
    """Mean squared log return scaled to an annualized variance."""
    clean = returns.astype(float).dropna()
    if clean.size < 2:
        raise ValueError("Insufficient returns for variance estimate.")
    return float((clean ** 2).mean() * DEFAULT_ANNUALIZATION)


def historical_variance_forecast(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    lookback_days: int,
    price_column: str = "close",
    date_column: str = "date",
) -> float:
    cutoff = pd.Timestamp(as_of_date)
    window = underlying[underlying[date_column] <= cutoff].tail(lookback_days)
    returns = calculate_log_returns(window, price_column=price_column).dropna()
    if returns.size < 2:
        raise ValueError("Insufficient history for historical variance forecast.")
    return _annualized_variance_from_returns(returns)


def ewma_variance_forecast(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    lookback_days: int,
    span: int = 60,
    price_column: str = "close",
    date_column: str = "date",
) -> float:
    cutoff = pd.Timestamp(as_of_date)
    window = underlying[underlying[date_column] <= cutoff].tail(lookback_days)
    returns = calculate_log_returns(window, price_column=price_column).dropna()
    if returns.size < 2:
        raise ValueError("Insufficient history for EWMA variance forecast.")
    ewma = calculate_ewma_variance(returns, half_life_days=float(span))
    return float(ewma.iloc[-1])


def build_model_forecast(
    model_name: str,
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    spot: float,
    model_config: Mapping[str, float],
    calibration: Optional[Mapping[str, float]] = None,
) -> ModelForecast:
    """Dispatch to one named forecast model."""
    lookback = int(model_config["lookback_days"])
    horizon = int(model_config["forecast_horizon_days"])
    as_of = pd.Timestamp(as_of_date)

    if model_name == "historical":
        expected = historical_variance_forecast(underlying, as_of, lookback)
        return ModelForecast(
            as_of_date=as_of.date(),
            model_name="historical",
            expected_variance=expected,
            median_variance=expected,
            lower_quantile=expected,
            upper_quantile=expected,
            monte_carlo_standard_error=0.0,
            parameters={"lookback_days": float(lookback)},
        )

    if model_name == "ewma":
        expected = ewma_variance_forecast(underlying, as_of, lookback)
        return ModelForecast(
            as_of_date=as_of.date(),
            model_name="ewma",
            expected_variance=expected,
            median_variance=expected,
            lower_quantile=expected,
            upper_quantile=expected,
            monte_carlo_standard_error=0.0,
            parameters={"lookback_days": float(lookback), "span": 60.0},
        )

    if model_name == "gbm":
        cal = calibration or calibrate_gbm(underlying, as_of, lookback)
        expected = gbm_expected_variance(float(cal["sigma"]))
        return ModelForecast(
            as_of_date=as_of.date(),
            model_name="gbm",
            expected_variance=expected,
            median_variance=expected,
            lower_quantile=expected,
            upper_quantile=expected,
            monte_carlo_standard_error=0.0,
            parameters={"mu": float(cal["mu"]), "sigma": float(cal["sigma"])},
        )

    if model_name == "heston":
        cal = calibration or calibrate_heston(underlying, as_of, lookback)
        if not validate_calibration_result(dict(cal)):
            raise ValueError("Heston calibration failed validation.")
        return forecast_realized_variance(
            initial_price=float(spot),
            parameters=dict(cal),
            horizon_days=horizon,
            n_paths=int(model_config["simulation_paths"]),
            as_of_date=as_of,
            random_seed=int(model_config["random_seed"]),
        )

    raise ValueError(f"Unknown model_name {model_name!r}.")


def build_ensemble_forecast(
    forecasts: Mapping[str, ModelForecast],
    weights: Mapping[str, float],
) -> ModelForecast:
    """Weighted average of available component forecasts."""
    usable = {k: v for k, v in forecasts.items() if k in weights and weights[k] > 0}
    if not usable:
        raise ValueError("No usable forecasts for ensemble.")
    total_w = sum(weights[k] for k in usable)
    expected = sum(weights[k] * usable[k].expected_variance for k in usable) / total_w
    as_of = next(iter(usable.values())).as_of_date
    return ModelForecast(
        as_of_date=as_of,
        model_name="ensemble",
        expected_variance=float(expected),
        median_variance=float(expected),
        lower_quantile=float(expected),
        upper_quantile=float(expected),
        monte_carlo_standard_error=0.0,
        parameters={k: float(weights[k]) for k in usable},
    )


def select_forecast_for_decision(
    underlying: pd.DataFrame,
    as_of_date: DateLike,
    spot: float,
    model_config: Mapping[str, float],
    ensemble_weights: Mapping[str, float],
    cached_heston_calibration: Optional[Mapping[str, float]] = None,
    prefer_model: str = "ensemble",
) -> tuple[ModelForecast, dict[str, object]]:
    """Build ensemble forecast, falling back when Heston is unstable."""
    diagnostics: dict[str, object] = {"heston_used": False, "heston_failure": None}
    components: dict[str, ModelForecast] = {}

    try:
        components["historical"] = build_model_forecast(
            "historical", underlying, as_of_date, spot, model_config
        )
        components["ewma"] = build_model_forecast("ewma", underlying, as_of_date, spot, model_config)
        components["gbm"] = build_model_forecast("gbm", underlying, as_of_date, spot, model_config)
    except ValueError as exc:
        diagnostics["component_failure"] = str(exc)
        raise

    heston_cal = cached_heston_calibration
    try:
        if heston_cal is None:
            heston_cal = calibrate_heston(
                underlying, as_of_date, int(model_config["lookback_days"])
            )
        if validate_calibration_result(dict(heston_cal)):
            components["heston"] = build_model_forecast(
                "heston",
                underlying,
                as_of_date,
                spot,
                model_config,
                calibration=heston_cal,
            )
            diagnostics["heston_used"] = True
        else:
            diagnostics["heston_failure"] = "validation_failed"
    except (ValueError, RuntimeError) as exc:
        diagnostics["heston_failure"] = str(exc)

    weights = dict(ensemble_weights)
    if not diagnostics["heston_used"]:
        weights = {k: v for k, v in weights.items() if k != "heston"}
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

    if prefer_model == "ensemble":
        forecast = build_ensemble_forecast(components, weights)
    elif prefer_model in components:
        forecast = components[prefer_model]
    else:
        forecast = build_ensemble_forecast(components, weights)

    diagnostics["components"] = {k: v.expected_variance for k, v in components.items()}
    return forecast, diagnostics


def realized_variance_over_horizon(
    underlying: pd.DataFrame,
    start_date: DateLike,
    horizon_days: int,
    date_column: str = "date",
    price_column: str = "close",
) -> Optional[float]:
    """Realized variance over the next ``horizon_days`` trading rows after start."""
    start = pd.Timestamp(start_date)
    future = underlying[underlying[date_column] > start].head(horizon_days)
    if len(future) < 2:
        return None
    prices = pd.concat(
        [
            underlying[underlying[date_column] == start][[date_column, price_column]],
            future[[date_column, price_column]],
        ]
    )
    returns = calculate_log_returns(prices, price_column=price_column).dropna()
    if returns.size < 2:
        return None
    return _annualized_variance_from_returns(returns)


def evaluate_forecast_diagnostics(
    underlying: pd.DataFrame,
    forecast_log: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Attach realized variance and error metrics to the forecast log."""
    rows = []
    for _, row in forecast_log.iterrows():
        realized = realized_variance_over_horizon(
            underlying, row["as_of_date"], horizon_days
        )
        predicted = float(row["expected_variance"])
        edge = float("nan")
        if "implied_variance" in row and not pd.isna(row["implied_variance"]):
            edge = predicted - float(row["implied_variance"])
        err = float("nan") if realized is None else predicted - realized
        rows.append(
            {
                **row.to_dict(),
                "realized_variance": realized,
                "forecast_error": err,
                "abs_error": abs(err) if realized is not None else float("nan"),
                "variance_edge": edge,
                "edge_sign_correct": (
                    np.sign(edge) == np.sign((realized or 0) - float(row.get("implied_variance", np.nan)))
                    if realized is not None and not pd.isna(row.get("implied_variance"))
                    else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    valid = df.dropna(subset=["realized_variance", "expected_variance"])
    if not valid.empty:
        df.attrs["mae"] = float(valid["abs_error"].mean())
        df.attrs["rmse"] = float(np.sqrt((valid["forecast_error"] ** 2).mean()))
        df.attrs["correlation"] = float(
            valid["expected_variance"].corr(valid["realized_variance"])
        )
        ss_res = float(((valid["realized_variance"] - valid["expected_variance"]) ** 2).sum())
        ss_tot = float(((valid["realized_variance"] - valid["realized_variance"].mean()) ** 2).sum())
        df.attrs["r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        edge_valid = valid.dropna(subset=["edge_sign_correct"])
        if not edge_valid.empty:
            df.attrs["edge_sign_accuracy"] = float(edge_valid["edge_sign_correct"].mean())
    return df
