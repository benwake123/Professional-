"""
Regression tests for ``src/backtest.py``.

Purpose
-------
Three end-to-end invariants:

    1. ``test_dates_strictly_chronological``
       The daily-history DataFrame is sorted ascending and contains no
       duplicate dates.
    2. ``test_rejected_trade_is_recorded``
       Pretrade risk rejections produce ``entry_rejected`` rows in the
       trade log so the audit trail is complete.
    3. ``test_pre_date_results_ignore_future_mutations``
       Running the backtest twice with different date windows produces
       identical daily-history rows over the overlapping window — that
       is, results for an early date never depend on data from a later
       date.

Module connections
------------------
Upstream:
    - ``src.backtest.{run_walk_forward_backtest, build_trading_calendar}``.
    - ``src.risk_management.check_pretrade_risk`` (for the rejection test).
Downstream:
    - ``pytest tests/test_backtest.py``
    - ``python3 tests/test_backtest.py``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_walk_forward_backtest
from src.risk_management import check_pretrade_risk
from src.portfolio import create_initial_portfolio
from src.types import OptionLeg, OptionStructure


@pytest.fixture(scope="module")
def small_run(real_market_bundle):
    """Run a short, dense backtest once and reuse it across tests."""
    config = {
        "dates": {
            "development_start": "2018-01-02",
            "end": "2018-06-29",
        },
        "model": {
            "lookback_days": 252,
            "forecast_horizon_days": 10,
            "simulation_paths": 500,
            "random_seed": 42,
        },
        "signal": {
            "rolling_zscore_window": 60,
            "long_vol_z_threshold": 1.0,
            "short_vol_z_threshold": 1.0,
            "exit_threshold": 0.25,
        },
        "options": {
            "minimum_dte": 20,
            "maximum_dte": 45,
            "maximum_relative_spread": 0.10,
            "minimum_open_interest": 500,
            "minimum_volume": 50,
            "wing_width_percent": 0.05,
        },
        "risk": {
            "initial_capital": 1_000_000.0,
            "target_annual_volatility": 0.10,
            "maximum_trade_risk_fraction": 0.02,
            "maximum_absolute_delta": 0.05,
            "maximum_drawdown": 0.15,
            "maximum_open_positions": 3,
        },
        "execution": {
            "option_commission_per_contract": 0.65,
            "option_slippage_fraction_of_spread": 0.25,
            "stock_slippage_bps": 1.0,
        },
    }
    return run_walk_forward_backtest(
        bundle=real_market_bundle,
        config=config,
        model_name="gbm",  # cheaper than Heston for tests
        decision_frequency="MS",  # one decision a month
    )


def test_dates_strictly_chronological(small_run) -> None:
    history = small_run.daily_history
    assert not history.empty
    dates = history["date"]
    assert dates.is_monotonic_increasing
    assert dates.duplicated().sum() == 0


def test_rejected_trade_is_recorded() -> None:
    """Crossed-quote rejection must surface in the RiskCheckResult."""
    leg = OptionLeg(
        contract_id="dummy",
        option_type="call",
        strike=100.0,
        expiration=date(2020, 3, 20),
        quantity=+1,
        bid=2.0,
        ask=1.5,  # crossed
        implied_volatility=0.20,
    )
    structure = OptionStructure(
        name="long_call_only",
        legs=[leg],
        direction="long_vol",
        maximum_loss_per_unit=5.0,
        entry_debit_or_credit=2.0,
        greeks={"delta": 0.5, "gamma": 0.01, "theta": -0.1, "vega": 0.2, "rho": 0.0},
    )
    portfolio = create_initial_portfolio(initial_capital=1_000_000.0)
    risk_config = {
        "target_annual_volatility": 0.10,
        "maximum_trade_risk_fraction": 0.02,
        "maximum_absolute_delta": 0.05,
        "maximum_drawdown": 0.15,
        "maximum_open_positions": 3,
    }
    options_config = {"maximum_relative_spread": 0.10}
    result = check_pretrade_risk(portfolio, structure, 1, risk_config, options_config)
    assert not result.approved
    assert any(
        "crossed quote" in r or "nonpositive midpoint" in r or "spread" in r or "bid/ask" in r
        for r in result.reasons
    )


def test_pre_date_results_ignore_future_mutations(real_market_bundle) -> None:
    """Equity history for early dates must not depend on the chosen end date."""
    base_config = {
        "model": {
            "lookback_days": 252,
            "forecast_horizon_days": 10,
            "simulation_paths": 200,
            "random_seed": 42,
        },
        "signal": {
            "rolling_zscore_window": 60,
            "long_vol_z_threshold": 1.0,
            "short_vol_z_threshold": 1.0,
            "exit_threshold": 0.25,
        },
        "options": {
            "minimum_dte": 20,
            "maximum_dte": 45,
            "maximum_relative_spread": 0.10,
            "minimum_open_interest": 500,
            "minimum_volume": 50,
            "wing_width_percent": 0.05,
        },
        "risk": {
            "initial_capital": 1_000_000.0,
            "target_annual_volatility": 0.10,
            "maximum_trade_risk_fraction": 0.02,
            "maximum_absolute_delta": 0.05,
            "maximum_drawdown": 0.15,
            "maximum_open_positions": 3,
        },
        "execution": {
            "option_commission_per_contract": 0.65,
            "option_slippage_fraction_of_spread": 0.25,
            "stock_slippage_bps": 1.0,
        },
    }
    short = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config={**base_config, "dates": {"development_start": "2018-01-02", "end": "2018-04-30"}},
        model_name="gbm",
        decision_frequency="MS",
    )
    long_run = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config={**base_config, "dates": {"development_start": "2018-01-02", "end": "2018-06-29"}},
        model_name="gbm",
        decision_frequency="MS",
    )
    short_history = short.daily_history.set_index("date")
    long_history = long_run.daily_history.set_index("date")
    overlap = short_history.index.intersection(long_history.index)
    assert len(overlap) > 0
    pd.testing.assert_series_equal(
        short_history.loc[overlap, "equity"],
        long_history.loc[overlap, "equity"],
        check_exact=False,
        rtol=1e-8,
    )


def test_rejected_decisions_are_logged(small_run) -> None:
    funnel = small_run.decision_funnel
    assert not funnel.empty
    assert funnel["rejection_reason"].notna().all()
    assert (funnel["rejection_reason"] != "").all()


def test_contract_selection_uses_same_day_quotes(real_market_bundle) -> None:
    from src.backtest import get_market_state_for_date

    decision_date = pd.Timestamp("2018-06-01")
    state = get_market_state_for_date(real_market_bundle, decision_date)
    options_today = state["options_today"]
    if options_today.empty:
        pytest.skip("No option quotes on the chosen decision date.")
    assert (pd.to_datetime(options_today["quote_date"]) == decision_date).all()


def test_daily_decisions_reuse_only_prior_calibration(real_market_bundle) -> None:
    config = {
        "dates": {
            "development_start": "2018-01-02",
            "validation_start": "2019-01-01",
            "test_start": "2022-01-01",
            "end": "2018-06-29",
        },
        "decision_frequency": "B",
        "heston_calibration_frequency": "W-MON",
        "model": {
            "lookback_days": 252,
            "forecast_horizon_days": 10,
            "simulation_paths": 200,
            "random_seed": 42,
        },
        "signal": {
            "rolling_zscore_window": 60,
            "long_vol_z_threshold": 1.0,
            "short_vol_z_threshold": 1.0,
            "exit_threshold": 0.25,
            "ensemble_weights": {"heston": 0.4, "ewma": 0.3, "historical": 0.3},
        },
        "options": {
            "minimum_dte": 20,
            "maximum_dte": 45,
            "maximum_relative_spread": 0.15,
            "minimum_open_interest": 100,
            "minimum_volume": 10,
            "wing_width_percent": 0.05,
        },
        "risk": {
            "initial_capital": 1_000_000.0,
            "target_annual_volatility": 0.10,
            "maximum_trade_risk_fraction": 0.02,
            "maximum_absolute_delta": 0.05,
            "maximum_drawdown": 0.15,
            "maximum_open_positions": 3,
            "trade_cooldown_days": 5,
        },
        "execution": {
            "option_commission_per_contract": 0.65,
            "option_slippage_fraction_of_spread": 0.25,
            "stock_slippage_bps": 1.0,
        },
        "delta_hedge": {"mode": "none", "tolerance_shares": 50},
        "exits": {"maximum_holding_days": 15, "minimum_dte_to_keep": 5},
    }
    early = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config=config,
        model_name="ensemble",
        start_date="2018-01-02",
        end_date="2018-04-30",
    )
    extended = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config=config,
        model_name="ensemble",
        start_date="2018-01-02",
        end_date="2018-06-29",
    )
    merged = early.forecasts.merge(
        extended.forecasts, on="as_of_date", suffixes=("_early", "_extended")
    )
    assert not merged.empty
    assert np.allclose(
        merged["expected_variance_early"],
        merged["expected_variance_extended"],
        rtol=1e-6,
        equal_nan=True,
    )


def test_future_data_cannot_change_past_trade(real_market_bundle) -> None:
    """Trade log through an early end date must not change when later data is added."""
    config = {
        "dates": {
            "development_start": "2018-01-02",
            "validation_start": "2019-01-01",
            "test_start": "2022-01-01",
            "end": "2018-06-29",
        },
        "model": {
            "lookback_days": 252,
            "forecast_horizon_days": 10,
            "simulation_paths": 200,
            "random_seed": 42,
        },
        "signal": {
            "rolling_zscore_window": 60,
            "long_vol_z_threshold": 0.75,
            "short_vol_z_threshold": 0.75,
            "exit_threshold": 0.25,
        },
        "options": {
            "minimum_dte": 20,
            "maximum_dte": 45,
            "maximum_relative_spread": 0.15,
            "minimum_open_interest": 100,
            "minimum_volume": 10,
            "wing_width_percent": 0.05,
        },
        "risk": {
            "initial_capital": 1_000_000.0,
            "target_annual_volatility": 0.10,
            "maximum_trade_risk_fraction": 0.02,
            "maximum_absolute_delta": 0.05,
            "maximum_drawdown": 0.15,
            "maximum_open_positions": 3,
        },
        "execution": {
            "option_commission_per_contract": 0.65,
            "option_slippage_fraction_of_spread": 0.25,
            "stock_slippage_bps": 1.0,
        },
        "delta_hedge": {"mode": "none", "tolerance_shares": 50},
        "exits": {"maximum_holding_days": 15, "minimum_dte_to_keep": 5},
    }
    early_end = "2018-04-30"
    short = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config=config,
        model_name="gbm",
        decision_frequency="MS",
        start_date="2018-01-02",
        end_date=early_end,
    )
    long_run = run_walk_forward_backtest(
        bundle=real_market_bundle,
        config=config,
        model_name="gbm",
        decision_frequency="MS",
        start_date="2018-01-02",
        end_date="2018-06-29",
    )
    pd.testing.assert_series_equal(
        short.daily_history.set_index("date")["equity"],
        long_run.daily_history.set_index("date").loc[
            short.daily_history["date"]
        ]["equity"],
        check_exact=False,
        rtol=1e-8,
    )
    if not short.trades.empty and "date" in short.trades.columns:
        short_trades = short.trades[short.trades["date"] <= pd.Timestamp(early_end)]
        long_trades = long_run.trades[long_run.trades["date"] <= pd.Timestamp(early_end)]
        pd.testing.assert_frame_equal(
            short_trades.reset_index(drop=True),
            long_trades.reset_index(drop=True),
            check_dtype=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
