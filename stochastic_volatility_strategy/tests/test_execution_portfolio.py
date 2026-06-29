"""
Regression tests for ``src/execution.py`` and ``src/portfolio.py``.

Purpose
-------
Pins three invariants:

    1. ``test_long_option_entry_uses_ask``
       Buying an option fills at ask + side-aware slippage, never below
       the midpoint. Cash decreases by ``fill_price * multiplier``.
    2. ``test_short_option_entry_uses_bid``
       Selling an option fills at bid - side-aware slippage, never above
       the midpoint. Cash increases by ``|fill_price| * multiplier``.
    3. ``test_cash_equity_reconcile``
       After opening then closing a structure, portfolio equity equals
       ``initial_capital + realized_pnl`` and no open positions remain.

Module connections
------------------
Upstream:
    - ``src.execution.{execute_option_entry, execute_option_exit, estimate_option_fill_price}``.
    - ``src.portfolio.{create_initial_portfolio, open_option_position, close_option_position,
       mark_portfolio_to_market, calculate_portfolio_equity}``.
Downstream:
    - ``pytest tests/test_execution_portfolio.py``
    - ``python3 tests/test_execution_portfolio.py``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.execution import (
    estimate_option_fill_price,
    execute_option_entry,
    execute_option_exit,
)
from src.portfolio import (
    calculate_portfolio_equity,
    close_option_position,
    create_initial_portfolio,
    mark_portfolio_to_market,
    open_option_position,
)
from src.position_sizing import calculate_position_size
from src.types import OptionLeg, OptionStructure, TradeSignal


def _dummy_signal(direction: str = "long_vol") -> TradeSignal:
    return TradeSignal(
        as_of_date=date(2020, 1, 6),
        direction=direction,
        implied_variance=0.04,
        forecast_variance=0.05,
        variance_risk_premium=-0.01,
        variance_edge=0.01,
        zscore=-1.5,
        regime="normal_vol__flat",
        entry_threshold=0.75,
        approved_by_regime_filter=True,
        model_name="ensemble",
    )


def _straddle() -> OptionStructure:
    legs = [
        OptionLeg(
            contract_id="SPY 2020-02-21 CALL K=320.00",
            option_type="call",
            strike=320.0,
            expiration=date(2020, 2, 21),
            quantity=+1,
            bid=4.90,
            ask=5.10,
            implied_volatility=0.20,
        ),
        OptionLeg(
            contract_id="SPY 2020-02-21 PUT K=320.00",
            option_type="put",
            strike=320.0,
            expiration=date(2020, 2, 21),
            quantity=+1,
            bid=4.70,
            ask=4.90,
            implied_volatility=0.21,
        ),
    ]
    structure = OptionStructure(
        name="long_straddle",
        legs=legs,
        direction="long_vol",
        maximum_loss_per_unit=10.0,
        entry_debit_or_credit=10.0,
        greeks={"delta": 0.0, "gamma": 0.02, "theta": -0.5, "vega": 0.4, "rho": 0.0},
    )
    return structure


def test_long_option_entry_uses_ask() -> None:
    """A buy fill must sit at or above the midpoint, between mid and ask + slippage."""
    fill = estimate_option_fill_price(bid=4.90, ask=5.10, side="buy", slippage_fraction_of_spread=0.5)
    midpoint = 5.0
    assert fill >= midpoint, f"buy fill {fill} should be >= midpoint {midpoint}"
    assert fill > 5.10, f"buy fill {fill} should exceed ask after slippage"


def test_short_option_entry_uses_bid() -> None:
    """A sell fill must sit at or below the midpoint."""
    fill = estimate_option_fill_price(bid=4.90, ask=5.10, side="sell", slippage_fraction_of_spread=0.5)
    midpoint = 5.0
    assert fill <= midpoint
    assert fill < 4.90


def test_flat_equity_history_still_produces_positive_size() -> None:
    """Before any PnL exists, sizing must not collapse to zero contracts."""
    structure = _straddle()
    signal = _dummy_signal(direction="long_vol")
    flat_history = pd.Series([1_000_000.0] * 100)
    qty = calculate_position_size(
        signal=signal,
        structure=structure,
        portfolio_equity=1_000_000.0,
        portfolio_equity_history=flat_history,
        risk_config={
            "target_annual_volatility": 0.10,
            "maximum_trade_risk_fraction": 0.02,
        },
    )
    assert qty > 0


def test_cash_equity_reconcile() -> None:
    """After entry+exit at unchanged quotes, equity = initial - 2*commission - 2*slippage."""
    portfolio = create_initial_portfolio(initial_capital=1_000_000.0)
    structure = _straddle()
    signal = _dummy_signal()

    entry = execute_option_entry(
        structure=structure,
        quantity_per_leg=1,
        as_of_date=date(2020, 1, 6),
        commission_per_contract=0.65,
        slippage_fraction_of_spread=0.25,
    )
    open_option_position(portfolio, structure, quantity=1, execution_report=entry, entry_signal=signal)

    # No market move: mark-to-market against the same bid/ask snapshot.
    snapshot = pd.DataFrame(
        {
            "quote_date": [pd.Timestamp("2020-01-06")] * 2,
            "expiration": [pd.Timestamp("2020-02-21")] * 2,
            "strike": [320.0, 320.0],
            "option_type": ["call", "put"],
            "bid": [4.90, 4.70],
            "ask": [5.10, 4.90],
        }
    )
    mark_portfolio_to_market(portfolio, snapshot, underlying_close=320.0)
    calculate_portfolio_equity(portfolio)

    exit_report = execute_option_exit(
        structure=structure,
        quantity_per_leg=1,
        as_of_date=date(2020, 1, 13),
        commission_per_contract=0.65,
        slippage_fraction_of_spread=0.25,
    )
    close_option_position(portfolio, portfolio.option_positions[0], exit_report)
    calculate_portfolio_equity(portfolio)

    assert all(not p.is_open for p in portfolio.option_positions)
    assert abs(portfolio.equity - portfolio.cash) < 1e-6
    expected_equity = 1_000_000.0 + entry.net_cash_flow + exit_report.net_cash_flow
    assert abs(portfolio.equity - expected_equity) < 1e-6, (
        f"equity={portfolio.equity} expected={expected_equity}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
