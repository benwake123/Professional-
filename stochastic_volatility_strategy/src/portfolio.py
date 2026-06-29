"""
Portfolio state container + helpers.

Purpose
-------
Implements the mutable :class:`~src.types.PortfolioState` that the
backtester carries forward day by day. Cash flows, commissions, slippage,
and gross P&L components are tracked explicitly so
:mod:`src.accounting` can reconcile ending equity to within one cent.

Module connections
------------------
Upstream:
    - ``src.accounting.{apply_execution_to_portfolio_ledger, intrinsic_option_value}``.
    - ``src.types.{ExecutionReport, OptionPosition, OptionStructure,
      PortfolioState, TradeSignal}``.
Downstream:
    - ``src.backtest`` orchestrates these primitives every day.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd

from src.accounting import apply_execution_to_portfolio_ledger, intrinsic_option_value
from src.types import (
    ExecutionReport,
    OptionPosition,
    OptionStructure,
    PortfolioState,
    TradeSignal,
)

DateLike = Union[str, date, datetime, pd.Timestamp]
CONTRACT_MULTIPLIER: int = 100


def create_initial_portfolio(initial_capital: float) -> PortfolioState:
    return PortfolioState(
        initial_capital=float(initial_capital),
        cash=float(initial_capital),
        stock_shares=0,
        stock_reference_price=0.0,
        option_positions=[],
        equity=float(initial_capital),
        equity_peak=float(initial_capital),
        drawdown=0.0,
    )


def _next_position_id(portfolio: PortfolioState) -> str:
    return f"P{len(portfolio.option_positions) + 1:04d}"


def open_option_position(
    portfolio: PortfolioState,
    structure: OptionStructure,
    quantity: int,
    execution_report: ExecutionReport,
    entry_signal: TradeSignal,
) -> OptionPosition:
    """Apply entry execution and append a new open position."""
    apply_execution_to_portfolio_ledger(portfolio, execution_report, instrument="option")

    position = OptionPosition(
        position_id=_next_position_id(portfolio),
        structure=structure,
        quantity=int(quantity),
        entry_date=execution_report.execution_date,
        entry_execution=execution_report,
        entry_signal=entry_signal,
        current_value=0.0,
    )
    portfolio.option_positions.append(position)
    portfolio.last_entry_date = execution_report.execution_date
    return position


def close_option_position(
    portfolio: PortfolioState,
    position: OptionPosition,
    execution_report: ExecutionReport,
) -> None:
    """Apply exit execution and finalize realized P&L on the position."""
    apply_execution_to_portfolio_ledger(portfolio, execution_report, instrument="option")

    entry_cash = position.entry_execution.net_cash_flow
    exit_cash = execution_report.net_cash_flow
    position.realized_pnl = float(entry_cash + exit_cash)
    position.unrealized_pnl = 0.0
    position.current_value = 0.0
    position.is_open = False
    position.exit_date = execution_report.execution_date
    position.exit_execution = execution_report
    portfolio.realized_pnl += position.realized_pnl


def update_stock_hedge(
    portfolio: PortfolioState, execution_report: ExecutionReport, reference_price: float
) -> None:
    """Apply hedge cash flow and update share count."""
    apply_execution_to_portfolio_ledger(portfolio, execution_report, instrument="hedge")
    if execution_report.fills:
        portfolio.stock_shares += int(execution_report.fills[0].get("shares", 0))
        portfolio.stock_reference_price = float(reference_price)


def _leg_market_value(
    leg,
    options_snapshot: pd.DataFrame,
    spot: float,
    as_of_date: pd.Timestamp,
) -> float:
    """Mark one leg to quote mid or intrinsic settlement when expired."""
    expiry = pd.Timestamp(leg.expiration)
    if as_of_date >= expiry:
        return float(leg.quantity) * intrinsic_option_value(
            leg.option_type, spot, leg.strike
        )

    if not options_snapshot.empty:
        match = options_snapshot[
            (options_snapshot["expiration"] == expiry)
            & (options_snapshot["strike"] == leg.strike)
            & (options_snapshot["option_type"].str.lower() == leg.option_type.lower())
        ]
        if not match.empty:
            latest = match.iloc[-1]
            midpoint = 0.5 * (float(latest["bid"]) + float(latest["ask"]))
            return float(leg.quantity) * midpoint

    if leg.bid is not None and leg.ask is not None and leg.bid > 0 and leg.ask >= leg.bid:
        midpoint = 0.5 * (float(leg.bid) + float(leg.ask))
        return float(leg.quantity) * midpoint

    return float(leg.quantity) * intrinsic_option_value(leg.option_type, spot, leg.strike)


def mark_option_position(
    position: OptionPosition,
    options_snapshot: pd.DataFrame,
    spot: float,
    as_of_date: DateLike,
) -> float:
    """Mark one open position; expired legs settle at intrinsic value."""
    if not position.is_open:
        return position.current_value

    as_of = pd.Timestamp(as_of_date)
    structure_value = sum(
        _leg_market_value(leg, options_snapshot, spot, as_of)
        for leg in position.structure.legs
    )
    position.current_value = float(
        position.quantity * structure_value * CONTRACT_MULTIPLIER
    )
    position.unrealized_pnl = float(
        position.entry_execution.net_cash_flow + position.current_value
    )
    return position.current_value


def mark_portfolio_to_market(
    portfolio: PortfolioState,
    options_snapshot: pd.DataFrame,
    underlying_close: float,
    as_of_date: DateLike | None = None,
) -> None:
    """Mark every open option position and refresh stock reference price."""
    portfolio.stock_reference_price = float(underlying_close)
    portfolio.unrealized_pnl = 0.0
    mark_date = as_of_date if as_of_date is not None else pd.Timestamp.today()
    for position in portfolio.option_positions:
        if position.is_open:
            mark_option_position(
                position, options_snapshot, underlying_close, mark_date
            )
            portfolio.unrealized_pnl += position.unrealized_pnl


def calculate_portfolio_equity(portfolio: PortfolioState) -> float:
    """Cash + stock value + open option mark-to-market."""
    option_value = sum(
        p.current_value for p in portfolio.option_positions if p.is_open
    )
    stock_value = float(portfolio.stock_shares) * float(portfolio.stock_reference_price)
    portfolio.unrealized_pnl = float(
        sum(p.unrealized_pnl for p in portfolio.option_positions if p.is_open)
    )
    portfolio.equity = float(portfolio.cash + stock_value + option_value)
    return portfolio.equity


def update_equity_peak_and_drawdown(portfolio: PortfolioState) -> None:
    if portfolio.equity > portfolio.equity_peak:
        portfolio.equity_peak = portfolio.equity
    if portfolio.equity_peak > 0:
        portfolio.drawdown = portfolio.equity / portfolio.equity_peak - 1.0
    else:
        portfolio.drawdown = 0.0


def record_portfolio_snapshot(
    portfolio: PortfolioState, as_of_date: DateLike
) -> dict[str, float | int | str]:
    """Return a dict suitable for appending to the daily history DataFrame."""
    open_positions = [p for p in portfolio.option_positions if p.is_open]
    return {
        "date": pd.Timestamp(as_of_date),
        "equity": portfolio.equity,
        "cash": portfolio.cash,
        "stock_shares": portfolio.stock_shares,
        "stock_value": float(portfolio.stock_shares) * float(portfolio.stock_reference_price),
        "option_value": sum(p.current_value for p in open_positions),
        "open_positions": len(open_positions),
        "unrealized_pnl": portfolio.unrealized_pnl,
        "realized_pnl": portfolio.realized_pnl,
        "hedge_pnl": portfolio.cumulative_hedge_gross_pnl,
        "cumulative_commissions": portfolio.cumulative_commissions,
        "cumulative_slippage": portfolio.cumulative_slippage,
        "drawdown": portfolio.drawdown,
        "equity_peak": portfolio.equity_peak,
    }
