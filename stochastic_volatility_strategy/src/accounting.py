"""
Portfolio accounting and reconciliation.

Purpose
-------
Tracks explicit P&L components and verifies:

    ending_equity
    ≈ initial_equity
      + realized_option_pnl_gross
      + unrealized_option_pnl
      + hedge_pnl_gross
      - commissions
      - slippage

where gross option P&L uses modeled fill prices and open positions contribute
through ``unrealized_option_pnl = entry_net + current_market_value``.

Module connections
------------------
Upstream: ``src.types.{PortfolioState, OptionPosition, ExecutionReport}``.
Downstream: ``src.audit``, ``src.run_pipeline``, ``tests/test_accounting.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.types import ExecutionReport, OptionPosition, PortfolioState

CONTRACT_MULTIPLIER: int = 100


@dataclass
class AccountingSnapshot:
    initial_equity: float
    ending_equity: float
    realized_option_pnl: float
    unrealized_option_pnl: float
    hedge_pnl: float
    commissions: float
    slippage: float
    reconciliation_rhs: float
    difference: float
    open_option_market_value: float
    stock_market_value: float
    cash: float
    net_pnl: float


def intrinsic_option_value(option_type: str, spot: float, strike: float) -> float:
    if option_type.lower() == "call":
        return max(float(spot) - float(strike), 0.0)
    return max(float(strike) - float(spot), 0.0)


def apply_execution_to_portfolio_ledger(
    portfolio: PortfolioState, report: ExecutionReport, instrument: str
) -> None:
    portfolio.cash += report.net_cash_flow
    portfolio.cumulative_commissions += report.commission
    portfolio.cumulative_slippage += report.slippage_cost
    if instrument == "option":
        portfolio.cumulative_option_gross_pnl += report.gross_cash_flow
        portfolio.cumulative_option_net_pnl += report.net_cash_flow
    elif instrument == "hedge":
        portfolio.cumulative_hedge_gross_pnl += report.gross_cash_flow
        portfolio.cumulative_hedge_net_pnl += report.net_cash_flow


def closed_option_gross_pnl(portfolio: PortfolioState) -> float:
    total = 0.0
    for position in portfolio.option_positions:
        if position.is_open or position.exit_execution is None:
            continue
        total += float(
            position.entry_execution.gross_cash_flow
            + position.exit_execution.gross_cash_flow
        )
    return total


def build_accounting_snapshot(portfolio: PortfolioState) -> AccountingSnapshot:
    open_option_value = float(
        sum(p.current_value for p in portfolio.option_positions if p.is_open)
    )
    stock_value = float(portfolio.stock_shares) * float(portfolio.stock_reference_price)
    unrealized = float(portfolio.unrealized_pnl)
    realized_gross = closed_option_gross_pnl(portfolio)
    hedge_gross = float(portfolio.cumulative_hedge_gross_pnl)

    rhs = (
        portfolio.initial_capital
        + realized_gross
        + unrealized
        + hedge_gross
        - portfolio.cumulative_commissions
        - portfolio.cumulative_slippage
    )
    difference = float(portfolio.equity - rhs)

    return AccountingSnapshot(
        initial_equity=float(portfolio.initial_capital),
        ending_equity=float(portfolio.equity),
        realized_option_pnl=realized_gross,
        unrealized_option_pnl=unrealized,
        hedge_pnl=hedge_gross,
        commissions=float(portfolio.cumulative_commissions),
        slippage=float(portfolio.cumulative_slippage),
        reconciliation_rhs=float(rhs),
        difference=difference,
        open_option_market_value=open_option_value,
        stock_market_value=stock_value,
        cash=float(portfolio.cash),
        net_pnl=float(portfolio.equity - portfolio.initial_capital),
    )


def accounting_snapshot_to_frame(snapshot: AccountingSnapshot) -> pd.DataFrame:
    return pd.DataFrame([snapshot.__dict__])


def position_realized_pnl_components(position: OptionPosition) -> dict[str, float]:
    if position.exit_execution is None:
        raise ValueError("Position is not closed.")
    entry = position.entry_execution
    exit_report = position.exit_execution
    return {
        "gross_option_pnl": float(entry.gross_cash_flow + exit_report.gross_cash_flow),
        "commission": float(entry.commission + exit_report.commission),
        "slippage": float(entry.slippage_cost + exit_report.slippage_cost),
        "net_option_pnl": float(entry.net_cash_flow + exit_report.net_cash_flow),
    }
