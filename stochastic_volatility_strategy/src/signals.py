"""
Variance-edge signal construction.

Purpose
-------
Builds trading direction from the forecast-vs-implied variance edge:

    variance_edge = forecast_realized_variance - implied_variance

    variance_edge > 0  -> options appear cheap  -> long volatility
    variance_edge < 0  -> options appear rich   -> short volatility

The edge is standardized with a rolling z-score. Separate long/short
thresholds can be configured independently.

Module connections
------------------
Upstream:
    - ``src.types.{ModelForecast, TradeSignal}``
    - ``src.regime.regime_adjusted_entry_threshold``
Downstream:
    - ``src.backtest.evaluate_new_trade``
    - ``tests/test_signals.py``, ``tests/test_accounting.py``
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.regime import regime_adjusted_entry_threshold
from src.types import ModelForecast, TradeSignal

DateLike = Union[str, date, datetime, pd.Timestamp]

DIRECTION_LONG = "long_vol"
DIRECTION_SHORT = "short_vol"
DIRECTION_FLAT = "flat"


def calculate_market_implied_variance(
    options_snapshot: pd.DataFrame,
    minimum_dte: int = 14,
    maximum_dte: int = 45,
    target_dte: int = 30,
    atm_moneyness_low: float = 0.98,
    atm_moneyness_high: float = 1.02,
) -> Optional[float]:
    """ATM implied variance from contracts near ``target_dte`` and spot."""
    if options_snapshot.empty:
        return None
    needed = {"implied_volatility", "quote_date", "expiration", "strike"}
    if not needed.issubset(options_snapshot.columns):
        return None
    if "spot" in options_snapshot.columns:
        spot = float(options_snapshot["spot"].iloc[0])
    else:
        spot = None

    df = options_snapshot.copy()
    df["dte"] = (df["expiration"] - df["quote_date"]).dt.days
    band = (df["dte"] >= minimum_dte) & (df["dte"] <= maximum_dte)
    df = df.loc[band].copy()
    if df.empty:
        return None

    if spot is not None and spot > 0:
        moneyness = df["strike"] / spot
        df = df[(moneyness >= atm_moneyness_low) & (moneyness <= atm_moneyness_high)]
    if df.empty:
        return None

    df["dte_distance"] = (df["dte"] - target_dte).abs()
    nearest_dte = df["dte_distance"].min()
    df = df[df["dte_distance"] == nearest_dte]
    eligible = df["implied_volatility"].dropna()
    if eligible.empty:
        return None
    avg_iv = float(eligible.mean())
    return avg_iv * avg_iv


def calculate_variance_edge(
    forecast_variance: float, implied_variance: Optional[float]
) -> Optional[float]:
    """Forecast minus implied variance (positive means options look cheap)."""
    if implied_variance is None:
        return None
    return float(forecast_variance - implied_variance)


def calculate_variance_risk_premium(
    implied_variance: Optional[float], forecast_variance: float
) -> Optional[float]:
    """Legacy VRP = implied - forecast (negative of variance_edge)."""
    edge = calculate_variance_edge(forecast_variance, implied_variance)
    if edge is None:
        return None
    return float(-edge)


def calculate_rolling_zscore(
    series: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    if window < 2:
        raise ValueError("window must be >= 2 for a z-score.")
    min_p = min_periods if min_periods is not None else window
    mean = series.rolling(window=window, min_periods=min_p).mean()
    std = series.rolling(window=window, min_periods=min_p).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def generate_raw_trade_direction(
    zscore: float,
    long_threshold: float,
    short_threshold: float,
) -> str:
    """Map variance-edge z-score to direction using separate thresholds."""
    if np.isnan(zscore):
        return DIRECTION_FLAT
    if zscore >= short_threshold:
        return DIRECTION_SHORT
    if zscore <= -long_threshold:
        return DIRECTION_LONG
    return DIRECTION_FLAT


def apply_regime_filter(direction: str, regime_label: str) -> tuple[str, bool]:
    if direction == DIRECTION_FLAT:
        return DIRECTION_FLAT, True
    vol_label, _, ts_label = regime_label.partition("__")
    if direction == DIRECTION_SHORT and ts_label == "backwardation":
        return DIRECTION_FLAT, False
    if direction == DIRECTION_LONG and ts_label == "contango" and vol_label == "low_vol":
        return DIRECTION_FLAT, False
    return direction, True


def estimate_round_trip_costs(
    structure_legs: list,
    quantity: int,
    commission_per_contract: float,
    slippage_fraction_of_spread: float,
    contract_multiplier: int = 100,
    estimated_hedge_cost: float = 0.0,
) -> dict[str, float]:
    """Estimate entry+exit spread, commission, and hedge drag."""
    spread_cost = 0.0
    for leg in structure_legs:
        bid = float(leg.bid or 0.0)
        ask = float(leg.ask or 0.0)
        if bid <= 0 or ask <= bid:
            continue
        half_spread = 0.5 * (ask - bid)
        per_leg = abs(int(leg.quantity)) * quantity * half_spread * 2.0
        spread_cost += per_leg * (1.0 + slippage_fraction_of_spread)
    commission = 2.0 * abs(quantity) * len(structure_legs) * commission_per_contract
    total = spread_cost * contract_multiplier + commission + estimated_hedge_cost
    return {
        "spread_cost": float(spread_cost * contract_multiplier),
        "commission": float(commission),
        "hedge_cost": float(estimated_hedge_cost),
        "total": float(total),
    }


def expected_edge_after_costs(
    variance_edge: float,
    implied_variance: float,
    quantity: int,
    structure_legs: list,
    commission_per_contract: float,
    slippage_fraction_of_spread: float,
    contract_multiplier: int = 100,
    estimated_hedge_cost: float = 0.0,
    structure_entry_value: float | None = None,
) -> float:
    """Variance edge scaled to dollars minus round-trip execution costs."""
    premium_notional = 0.0
    if structure_entry_value is not None and structure_entry_value != 0:
        premium_notional = abs(float(structure_entry_value)) * abs(quantity) * contract_multiplier
    else:
        for leg in structure_legs:
            if leg.bid is None or leg.ask is None:
                continue
            premium_notional += (
                0.5 * (float(leg.bid) + float(leg.ask))
                * abs(int(leg.quantity))
                * abs(quantity)
                * contract_multiplier
            )
    rel_edge = abs(float(variance_edge)) / max(float(implied_variance), 1e-8)
    gross_edge = rel_edge * premium_notional
    costs = estimate_round_trip_costs(
        structure_legs,
        quantity,
        commission_per_contract,
        slippage_fraction_of_spread,
        contract_multiplier,
        estimated_hedge_cost,
    )
    return float(gross_edge - costs["total"])


def build_volatility_signal(
    as_of_date: DateLike,
    options_snapshot: pd.DataFrame,
    forecast: ModelForecast,
    edge_history: pd.Series,
    rolling_window: int,
    long_threshold: float,
    short_threshold: float,
    regime_label: str,
    minimum_dte: int = 14,
    maximum_dte: int = 45,
    target_dte: int = 30,
    atm_moneyness_low: float = 0.98,
    atm_moneyness_high: float = 1.02,
    spot: float | None = None,
) -> TradeSignal:
    """End-to-end variance-edge signal with separate long/short thresholds."""
    options = options_snapshot.copy()
    if spot is not None and not options.empty:
        options = options.copy()
        options["spot"] = float(spot)

    implied = calculate_market_implied_variance(
        options,
        minimum_dte=minimum_dte,
        maximum_dte=maximum_dte,
        target_dte=target_dte,
        atm_moneyness_low=atm_moneyness_low,
        atm_moneyness_high=atm_moneyness_high,
    )
    edge = calculate_variance_edge(forecast.expected_variance, implied)
    vrp = calculate_variance_risk_premium(implied, forecast.expected_variance)

    if edge is None or edge_history.dropna().size < 2:
        zscore = float("nan")
    else:
        history_with_current = pd.concat(
            [edge_history.dropna(), pd.Series([edge])], ignore_index=True
        )
        zscores = calculate_rolling_zscore(history_with_current, window=rolling_window)
        zscore = float(zscores.iloc[-1]) if not zscores.empty else float("nan")

    base_threshold = 0.5 * (long_threshold + short_threshold)
    threshold = regime_adjusted_entry_threshold(base_threshold, regime_label)
    long_adj = threshold * (long_threshold / base_threshold) if base_threshold else long_threshold
    short_adj = threshold * (short_threshold / base_threshold) if base_threshold else short_threshold

    raw_direction = generate_raw_trade_direction(zscore, long_adj, short_adj)
    direction, approved = apply_regime_filter(raw_direction, regime_label)

    return TradeSignal(
        as_of_date=pd.Timestamp(as_of_date).date(),
        direction=direction,
        implied_variance=float(implied) if implied is not None else float("nan"),
        forecast_variance=float(forecast.expected_variance),
        variance_risk_premium=float(vrp) if vrp is not None else float("nan"),
        variance_edge=float(edge) if edge is not None else float("nan"),
        zscore=zscore,
        regime=regime_label,
        entry_threshold=float(short_adj if direction == DIRECTION_SHORT else long_adj),
        approved_by_regime_filter=approved,
        model_name=forecast.model_name,
    )
