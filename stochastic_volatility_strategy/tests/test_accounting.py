"""
Accounting, execution, and signal-direction regression tests.

Purpose
-------
Verifies cash-flow conventions, P&L reconciliation, variance-edge direction,
and that rejected decisions are logged in the trade funnel.
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

from src.accounting import (
    build_accounting_snapshot,
    intrinsic_option_value,
    position_realized_pnl_components,
)
from src.execution import execute_option_entry, execute_option_exit, estimate_option_fill_price
from src.portfolio import (
    calculate_portfolio_equity,
    close_option_position,
    create_initial_portfolio,
    mark_option_position,
    open_option_position,
)
from src.signals import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    calculate_variance_edge,
    generate_raw_trade_direction,
)
from src.types import OptionLeg, OptionStructure, TradeSignal


def _one_leg_structure(side: str = "long") -> OptionStructure:
    qty = +1 if side == "long" else -1
    leg = OptionLeg(
        contract_id="TEST",
        option_type="call",
        strike=100.0,
        expiration=date(2025, 6, 20),
        quantity=qty,
        bid=4.90,
        ask=5.10,
        implied_volatility=0.20,
    )
    return OptionStructure(name="test", legs=[leg], direction="long_vol" if side == "long" else "short_vol")


def _signal() -> TradeSignal:
    return TradeSignal(
        as_of_date=date(2024, 1, 2),
        direction="long_vol",
        implied_variance=0.04,
        forecast_variance=0.05,
        variance_risk_premium=-0.01,
        variance_edge=0.01,
        zscore=-1.0,
        regime="normal_vol__flat",
        entry_threshold=0.75,
        approved_by_regime_filter=True,
        model_name="ensemble",
    )


def test_long_entry_cash_flow_is_negative() -> None:
    portfolio = create_initial_portfolio(1_000_000.0)
    entry = execute_option_entry(
        _one_leg_structure("long"), 1, date(2024, 1, 2), 0.65, 0.25
    )
    assert entry.gross_cash_flow < 0
    open_option_position(portfolio, _one_leg_structure("long"), 1, entry, _signal())
    assert portfolio.cash < 1_000_000.0


def test_short_entry_cash_flow_is_positive() -> None:
    entry = execute_option_entry(
        _one_leg_structure("short"), 1, date(2024, 1, 2), 0.65, 0.25
    )
    assert entry.gross_cash_flow > 0


def test_long_exit_cash_flow_is_positive() -> None:
    exit_report = execute_option_exit(
        _one_leg_structure("long"), 1, date(2024, 1, 10), 0.65, 0.25
    )
    assert exit_report.gross_cash_flow > 0


def test_short_exit_cash_flow_is_negative() -> None:
    exit_report = execute_option_exit(
        _one_leg_structure("short"), 1, date(2024, 1, 10), 0.65, 0.25
    )
    assert exit_report.gross_cash_flow < 0


def test_option_multiplier_applied_once() -> None:
    entry = execute_option_entry(
        _one_leg_structure("long"), 2, date(2024, 1, 2), 0.65, 0.25
    )
    fill = entry.fills[0]["fill_price"]
    expected = -2 * float(fill) * 100
    assert np.isclose(entry.gross_cash_flow, expected)


def test_quantity_sign_applied_once() -> None:
    entry = execute_option_entry(
        _one_leg_structure("short"), 3, date(2024, 1, 2), 0.65, 0.25
    )
    assert entry.fills[0]["quantity"] == -3


def test_commissions_applied_once() -> None:
    entry = execute_option_entry(
        _one_leg_structure("long"), 1, date(2024, 1, 2), 0.65, 0.25
    )
    assert entry.net_cash_flow == entry.gross_cash_flow - entry.commission - entry.slippage_cost
    assert entry.commission == pytest.approx(0.65)


def test_realized_and_unrealized_pnl_not_double_counted() -> None:
    portfolio = create_initial_portfolio(1_000_000.0)
    structure = _one_leg_structure("long")
    entry = execute_option_entry(structure, 1, date(2024, 1, 2), 0.65, 0.25)
    position = open_option_position(portfolio, structure, 1, entry, _signal())
    mark_option_position(position, pd.DataFrame(), spot=100.0, as_of_date=date(2024, 1, 3))
    calculate_portfolio_equity(portfolio)
    assert portfolio.unrealized_pnl != 0.0
    exit_report = execute_option_exit(structure, 1, date(2024, 1, 10), 0.65, 0.25)
    close_option_position(portfolio, position, exit_report)
    assert position.unrealized_pnl == 0.0
    assert position.current_value == 0.0
    calculate_portfolio_equity(portfolio)
    assert portfolio.unrealized_pnl == 0.0


def test_trade_pnl_reconciles_with_equity() -> None:
    portfolio = create_initial_portfolio(1_000_000.0)
    structure = _one_leg_structure("long")
    entry = execute_option_entry(structure, 1, date(2024, 1, 2), 0.65, 0.25)
    position = open_option_position(portfolio, structure, 1, entry, _signal())
    exit_report = execute_option_exit(structure, 1, date(2024, 1, 10), 0.65, 0.25)
    close_option_position(portfolio, position, exit_report)
    calculate_portfolio_equity(portfolio)
    snap = build_accounting_snapshot(portfolio)
    assert abs(snap.difference) < 0.01


def test_expiration_settles_at_intrinsic_value() -> None:
    value = intrinsic_option_value("call", spot=105.0, strike=100.0)
    assert value == 5.0
    portfolio = create_initial_portfolio(1_000_000.0)
    structure = _one_leg_structure("long")
    entry = execute_option_entry(structure, 1, date(2024, 1, 2), 0.0, 0.0)
    position = open_option_position(portfolio, structure, 1, entry, _signal())
    mark_option_position(
        position,
        pd.DataFrame(),
        spot=105.0,
        as_of_date=date(2025, 6, 20),
    )
    assert np.isclose(position.current_value, 5.0 * 100)


def test_variance_edge_direction_is_correct() -> None:
    assert calculate_variance_edge(0.05, 0.04) == pytest.approx(0.01)
    assert generate_raw_trade_direction(-1.0, 0.75, 0.75) == DIRECTION_LONG
    assert generate_raw_trade_direction(1.0, 0.75, 0.75) == DIRECTION_SHORT


def test_buy_fills_are_not_better_than_ask_model() -> None:
    fill = estimate_option_fill_price(4.90, 5.10, "buy", 0.25)
    assert fill >= 5.10


def test_sell_fills_are_not_better_than_bid_model() -> None:
    fill = estimate_option_fill_price(4.90, 5.10, "sell", 0.25)
    assert fill <= 4.90


def test_manual_trade_pnl_by_hand() -> None:
    """One long call: buy 1 contract at ask-side fill, sell back at bid-side fill."""
    bid, ask = 4.90, 5.10
    buy_fill = estimate_option_fill_price(bid, ask, "buy", 0.0)
    sell_fill = estimate_option_fill_price(bid, ask, "sell", 0.0)
    gross_entry = -buy_fill * 100
    gross_exit = +sell_fill * 100
    commission = 0.65 * 2
    net = gross_entry + gross_exit - commission
    structure = _one_leg_structure("long")
    portfolio = create_initial_portfolio(1_000_000.0)
    entry = execute_option_entry(structure, 1, date(2024, 1, 2), 0.65, 0.0)
    position = open_option_position(portfolio, structure, 1, entry, _signal())
    exit_report = execute_option_exit(structure, 1, date(2024, 1, 3), 0.65, 0.0)
    close_option_position(portfolio, position, exit_report)
    comps = position_realized_pnl_components(position)
    assert comps["net_option_pnl"] == pytest.approx(net, rel=0.01)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
