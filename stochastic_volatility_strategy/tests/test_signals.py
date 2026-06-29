"""
Regression tests for ``src/signals.py`` variance-edge direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    DIRECTION_FLAT,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    calculate_variance_edge,
    calculate_rolling_zscore,
    generate_raw_trade_direction,
)


def test_positive_variance_edge_signals_long_vol() -> None:
    edge = calculate_variance_edge(forecast_variance=0.05, implied_variance=0.04)
    assert edge == pytest.approx(0.01)
    z = -1.5
    assert generate_raw_trade_direction(z, long_threshold=0.75, short_threshold=0.75) == DIRECTION_LONG


def test_negative_variance_edge_signals_short_vol() -> None:
    edge = calculate_variance_edge(forecast_variance=0.03, implied_variance=0.05)
    assert edge == pytest.approx(-0.02)
    z = 1.2
    assert generate_raw_trade_direction(z, long_threshold=0.75, short_threshold=0.75) == DIRECTION_SHORT


def test_neutral_zone_is_flat() -> None:
    assert generate_raw_trade_direction(0.0, 1.0, 1.0) == DIRECTION_FLAT
    assert generate_raw_trade_direction(0.5, 1.0, 1.0) == DIRECTION_FLAT
    assert generate_raw_trade_direction(-0.5, 1.0, 1.0) == DIRECTION_FLAT


def test_rolling_zscore_matches_manual_calc() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    zscores = calculate_rolling_zscore(series, window=3)
    expected = (series.iloc[-1] - series.iloc[-3:].mean()) / series.iloc[-3:].std(ddof=1)
    assert np.isclose(zscores.iloc[-1], expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
