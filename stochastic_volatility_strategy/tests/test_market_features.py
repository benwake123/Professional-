"""
Regression tests for ``src/market_features.py``.

Purpose
-------
Pins the numerical properties of the feature functions, then runs an
end-to-end test against the real ``data/raw/`` CSVs through the loader.

Unit checks (synthetic inputs):
    - Log returns sum to ``log(P_end / P_start)``.
    - Realized variance of constant returns equals zero (after burn-in).
    - Realized variance of i.i.d. ``sigma_daily * z_t`` returns
      approximately ``sigma_daily ** 2 * 252``.
    - EWMA variance shape and nonneg values.
    - Drawdown is always ``<= 0`` and matches the analytical answer on
      a hand-crafted level series.
    - VIX term structure produces the right ratios on hand-built input.
    - ``get_feature_row_as_of`` returns the latest available row and
      raises for cutoffs before the first row.

Connection / integration:
    - ``build_market_feature_table`` runs cleanly against the real
      ``MarketDataBundle``, returns one row per SPY trading day, and
      every column has the documented sign / unit characteristics.

Module connections
------------------
Upstream (this file imports from):
    - ``numpy`` / ``pandas`` / ``pytest``               : math + runner.
    - ``src.market_features``                           : unit under test.
    - ``tests.conftest`` (auto-loaded) :
        - ``underlying_frame``, ``volatility_index_frame`` (synthetic)
        - ``real_market_bundle`` (real CSVs via the loader)

Downstream:
    - ``pytest tests/test_market_features.py``
    - ``python3 tests/test_market_features.py`` (works via __main__ block).
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path shim so direct ``python3 tests/test_market_features.py`` works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.market_features import (
    build_market_feature_table,
    calculate_drawdown,
    calculate_ewma_variance,
    calculate_log_returns,
    calculate_realized_variance,
    calculate_realized_volatility,
    calculate_vix_term_structure,
    calculate_volatility_acceleration,
    get_feature_row_as_of,
)
from src.types import MarketDataBundle


def test_log_returns_telescope() -> None:
    """``log_returns.sum() == log(P_end / P_start)`` ignoring the leading NaN."""
    prices = pd.Series([100.0, 102.0, 99.5, 105.0, 110.0])
    log_returns = calculate_log_returns(prices)
    total = float(log_returns.dropna().sum())
    expected = float(np.log(prices.iloc[-1] / prices.iloc[0]))
    assert np.isclose(total, expected, atol=1e-12)


def test_realized_variance_zero_for_constant_returns() -> None:
    """Constant-return series has zero realized variance after the warm-up."""
    returns = pd.Series([np.nan] + [0.0] * 60)
    rv = calculate_realized_variance(returns, window=20)
    assert (rv.dropna() == 0.0).all()


def test_realized_variance_matches_iid_target() -> None:
    """For i.i.d. ``sigma * z_t`` returns, RV ≈ sigma^2 * 252 within sampling noise."""
    rng = np.random.default_rng(seed=12345)
    daily_sigma = 0.01
    n = 5_000
    returns = pd.Series(rng.normal(loc=0.0, scale=daily_sigma, size=n))
    rv = calculate_realized_variance(returns, window=252).dropna()
    assert np.isclose(rv.mean(), (daily_sigma**2) * 252, rtol=0.05)


def test_realized_volatility_is_sqrt_of_variance() -> None:
    rng = np.random.default_rng(seed=7)
    returns = pd.Series(rng.normal(scale=0.012, size=500))
    rv = calculate_realized_variance(returns, window=21)
    rvol = calculate_realized_volatility(returns, window=21)
    assert np.allclose(rvol.dropna(), np.sqrt(rv.dropna()), atol=1e-12)


def test_ewma_variance_nonneg_and_finite() -> None:
    rng = np.random.default_rng(seed=42)
    returns = pd.Series(rng.normal(scale=0.01, size=1_000))
    ewma = calculate_ewma_variance(returns, half_life_days=20).dropna()
    assert (ewma >= 0).all()
    assert np.isfinite(ewma).all()


def test_drawdown_matches_analytic_example() -> None:
    """Hand-built level series with a 20% drawdown."""
    levels = pd.Series([100.0, 110.0, 120.0, 96.0, 108.0, 130.0])
    drawdown = calculate_drawdown(levels)
    # Peak reaches 120; at index 3 (96): (96 - 120) / 120 == -0.20.
    assert (drawdown <= 0).all()
    assert np.isclose(drawdown.min(), -0.20, atol=1e-12)
    assert np.isclose(drawdown.iloc[-1], 0.0, atol=1e-12)


def test_vix_term_structure_ratios() -> None:
    vix_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "vix": [20.0, 15.0, 30.0],
            "vix9d": [18.0, 16.5, 33.0],
            "vix3m": [22.0, 18.0, 27.0],
        }
    )
    ts = calculate_vix_term_structure(vix_frame)
    assert np.allclose(ts["vix9d_over_vix"], [18 / 20, 16.5 / 15, 33 / 30])
    assert np.allclose(ts["vix_over_vix3m"], [20 / 22, 15 / 18, 30 / 27])


def test_volatility_acceleration_directionality() -> None:
    """If recent vol > long-run vol, acceleration must be positive."""
    rvol = pd.Series([0.10] * 30 + [0.30] * 10)
    accel = calculate_volatility_acceleration(rvol, short_window=5, long_window=20)
    assert accel.dropna().iloc[-1] > 0.0


def test_get_feature_row_as_of_returns_latest_eligible_row() -> None:
    feature_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "feature_a": [1.0, 2.0, 3.0],
        }
    )

    row = get_feature_row_as_of(feature_table, "2024-01-03")
    assert row["feature_a"] == 2.0

    row = get_feature_row_as_of(feature_table, "2024-01-05")
    assert row["feature_a"] == 3.0

    with pytest.raises(ValueError, match="No feature row available"):
        get_feature_row_as_of(feature_table, "2024-01-01")


# ---------------------------------------------------------------------------
# Connection / integration with src.data_loader output (real CSVs).
# ---------------------------------------------------------------------------


def test_build_market_feature_table_on_real_bundle(
    real_market_bundle: MarketDataBundle,
) -> None:
    """End-to-end: real loader -> build_market_feature_table -> sanity checks."""
    table = build_market_feature_table(
        real_market_bundle.underlying,
        real_market_bundle.volatility_indices,
    )

    assert len(table) == len(real_market_bundle.underlying)

    expected_columns = {
        "date",
        "log_return",
        "realized_variance",
        "realized_volatility",
        "ewma_variance",
        "volatility_acceleration",
        "drawdown",
        "vix9d_over_vix",
        "vix_over_vix3m",
    }
    assert expected_columns <= set(table.columns)

    # ``vix_over_vix3m`` is NaN because the public VIX file has no vix3m column.
    assert table["vix_over_vix3m"].isna().all()

    rv = table["realized_variance"].dropna()
    rvol = table["realized_volatility"].dropna()
    assert (rv >= 0).all()
    assert (rvol >= 0).all()

    drawdown = table["drawdown"].dropna()
    assert (drawdown <= 1e-12).all(), (
        f"drawdown must be <= 0; max observed: {drawdown.max()}"
    )

    # Annualized realized vol for SPY over this window should land in a
    # plausible historical band (5%..80% annualized).
    median_annual_vol = rvol.median()
    assert 0.05 <= median_annual_vol <= 0.80, (
        f"median realized volatility {median_annual_vol:.4f} is implausible "
        "for SPY 2014-2023."
    )

    # The 2020 COVID crash should produce a >= 25% drawdown somewhere in 2020.
    table_2020 = table[
        (table["date"] >= pd.Timestamp("2020-01-01"))
        & (table["date"] <= pd.Timestamp("2020-12-31"))
    ]
    worst_2020 = table_2020["drawdown"].min()
    assert worst_2020 <= -0.25, (
        f"Expected a >=25% drawdown during 2020, got {worst_2020:.3f}."
    )


def test_get_feature_row_as_of_on_real_table(
    real_market_bundle: MarketDataBundle,
) -> None:
    """``get_feature_row_as_of`` returns a complete row at a mid-history date."""
    table = build_market_feature_table(
        real_market_bundle.underlying,
        real_market_bundle.volatility_indices,
    )
    row = get_feature_row_as_of(table, "2020-03-31")
    assert row["date"] <= pd.Timestamp("2020-03-31")
    assert np.isfinite(row["realized_variance"])
    assert np.isfinite(row["realized_volatility"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
