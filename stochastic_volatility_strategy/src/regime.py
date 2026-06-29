"""
Volatility-regime classification.

Purpose
-------
Buckets each decision date into a vol regime + term-structure regime and
exposes regime-aware multipliers used by ``src.signals`` and
``src.position_sizing``. Thresholds are computed using an EXPANDING window
of historical data only (no future quantiles), so the classification is
strictly point-in-time.

Module connections
------------------
Upstream:
    - ``src.market_features.build_market_feature_table`` is the typical
      data source for the realized-vol and term-structure inputs.
Downstream:
    - ``src.signals.apply_regime_filter`` uses
      :func:`regime_adjusted_entry_threshold`.
    - ``src.position_sizing.apply_regime_size_multiplier`` uses
      :func:`regime_risk_multiplier`.
    - ``src.backtest.evaluate_new_trade`` calls
      :func:`classify_current_regime` on every decision day.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]

VOL_LOW = "low_vol"
VOL_NORMAL = "normal_vol"
VOL_HIGH = "high_vol"

TS_BACKWARDATION = "backwardation"
TS_FLAT = "flat"
TS_CONTANGO = "contango"


def calculate_expanding_regime_thresholds(
    realized_vol: pd.Series, lower_quantile: float = 0.25, upper_quantile: float = 0.75
) -> pd.DataFrame:
    """Expanding-window quantiles of realized vol (no future information)."""
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Need 0 < lower_quantile < upper_quantile < 1.")
    rv = realized_vol.astype(float)
    return pd.DataFrame(
        {
            "low_threshold": rv.expanding(min_periods=1).quantile(lower_quantile),
            "high_threshold": rv.expanding(min_periods=1).quantile(upper_quantile),
        }
    )


def classify_volatility_regime(
    current_vol: float, low_threshold: float, high_threshold: float
) -> str:
    """Return one of ``low_vol``, ``normal_vol``, ``high_vol``."""
    if np.isnan(current_vol):
        return VOL_NORMAL
    if current_vol <= low_threshold:
        return VOL_LOW
    if current_vol >= high_threshold:
        return VOL_HIGH
    return VOL_NORMAL


def classify_term_structure_regime(
    vix_over_vix3m: float,
    contango_band: tuple[float, float] = (0.97, 1.03),
) -> str:
    """Map the 30d/3m slope to backwardation / flat / contango."""
    if np.isnan(vix_over_vix3m):
        return TS_FLAT
    low, high = contango_band
    if vix_over_vix3m < low:
        return TS_CONTANGO
    if vix_over_vix3m > high:
        return TS_BACKWARDATION
    return TS_FLAT


def build_combined_regime_label(vol_label: str, term_label: str) -> str:
    """Concatenate volatility + term-structure into a single label."""
    return f"{vol_label}__{term_label}"


def classify_current_regime(
    feature_row: pd.Series,
    realized_vol_history: pd.Series,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> dict[str, str]:
    """Point-in-time regime classification using only historical thresholds."""
    thresholds = calculate_expanding_regime_thresholds(
        realized_vol_history, lower_quantile, upper_quantile
    )
    if thresholds.empty:
        low_threshold = high_threshold = float("nan")
    else:
        low_threshold = float(thresholds["low_threshold"].iloc[-1])
        high_threshold = float(thresholds["high_threshold"].iloc[-1])

    current_vol = float(feature_row.get("realized_volatility", float("nan")))
    term_value = float(feature_row.get("vix_over_vix3m", float("nan")))

    vol_label = classify_volatility_regime(current_vol, low_threshold, high_threshold)
    term_label = classify_term_structure_regime(term_value)
    combined = build_combined_regime_label(vol_label, term_label)
    return {
        "volatility_regime": vol_label,
        "term_structure_regime": term_label,
        "combined_regime": combined,
    }


def regime_adjusted_entry_threshold(base_threshold: float, regime_label: str) -> float:
    """Tighten or relax the z-score entry threshold based on the regime."""
    vol_label = regime_label.split("__")[0] if "__" in regime_label else regime_label
    if vol_label == VOL_HIGH:
        return base_threshold * 1.25
    if vol_label == VOL_LOW:
        return base_threshold * 0.85
    return base_threshold


def regime_risk_multiplier(regime_label: str) -> float:
    """Return a position-size multiplier suited to the current regime."""
    vol_label = regime_label.split("__")[0] if "__" in regime_label else regime_label
    if vol_label == VOL_HIGH:
        return 0.50
    if vol_label == VOL_LOW:
        return 1.10
    return 1.00
