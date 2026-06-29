"""
Delta-hedging utilities.

Purpose
-------
Computes the net delta of every open option position + stock holdings, and
asks "should we rehedge today?" based on a tolerance threshold expressed in
shares. When rehedging is required, returns the number of shares to trade
so the resulting portfolio delta lands inside the tolerance band.

Module connections
------------------
Upstream:
    - ``src.types.{PortfolioState, OptionPosition}``.
Downstream:
    - ``src.backtest`` invokes :func:`rebalance_delta_hedge` after marking
      open positions to market each day.
"""

from __future__ import annotations

from datetime import date
from typing import Union

import pandas as pd

from src.execution import execute_stock_hedge
from src.portfolio import update_stock_hedge
from src.types import ExecutionReport, OptionPosition, PortfolioState

CONTRACT_MULTIPLIER: int = 100

DateLike = Union[str, date, pd.Timestamp]


def calculate_position_delta(position: OptionPosition) -> float:
    """Signed share-equivalent delta of one option position."""
    return float(
        position.quantity * position.structure.greeks.get("delta", 0.0) * CONTRACT_MULTIPLIER
    )


def calculate_portfolio_net_delta(portfolio: PortfolioState) -> float:
    """Stock + sum of option-position deltas, in shares."""
    option_delta = sum(
        calculate_position_delta(p) for p in portfolio.option_positions if p.is_open
    )
    return float(option_delta + portfolio.stock_shares)


def should_rehedge(net_delta_shares: float, tolerance_shares: float) -> bool:
    """``True`` when |net delta| > tolerance."""
    return abs(net_delta_shares) > tolerance_shares


def calculate_required_hedge_shares(net_delta_shares: float) -> int:
    """Number of shares to trade to neutralize delta (sign-flipped)."""
    return int(round(-net_delta_shares))


def apply_delta_hedge(
    portfolio: PortfolioState,
    reference_price: float,
    as_of_date: DateLike,
    execution_config: dict[str, float],
    hedge_config: dict[str, float],
) -> ExecutionReport | None:
    """Apply delta hedge according to configured mode: none, daily, threshold."""
    mode = str(hedge_config.get("mode", "threshold")).lower()
    if mode == "none":
        return None
    tolerance = float(hedge_config.get("tolerance_shares", 50.0))
    if mode == "daily":
        tolerance = 0.0
    return rebalance_delta_hedge(
        portfolio=portfolio,
        reference_price=reference_price,
        tolerance_shares=tolerance,
        slippage_bps=float(execution_config["stock_slippage_bps"]),
        as_of_date=as_of_date,
    )


def rebalance_delta_hedge(
    portfolio: PortfolioState,
    reference_price: float,
    tolerance_shares: float,
    slippage_bps: float,
    as_of_date: DateLike,
    per_share_commission: float = 0.0,
) -> ExecutionReport | None:
    """Execute a hedge trade when delta drifts outside the tolerance band."""
    net_delta = calculate_portfolio_net_delta(portfolio)
    if not should_rehedge(net_delta, tolerance_shares):
        return None
    shares = calculate_required_hedge_shares(net_delta)
    if shares == 0:
        return None
    report = execute_stock_hedge(
        shares=shares,
        reference_price=reference_price,
        as_of_date=as_of_date,
        slippage_bps=slippage_bps,
        per_share_commission=per_share_commission,
    )
    update_stock_hedge(portfolio, report, reference_price)
    return report


def close_delta_hedge(
    portfolio: PortfolioState,
    reference_price: float,
    slippage_bps: float,
    as_of_date: DateLike,
    per_share_commission: float = 0.0,
) -> ExecutionReport | None:
    """Liquidate the entire stock hedge (used when all option positions close)."""
    if portfolio.stock_shares == 0:
        return None
    shares = -int(portfolio.stock_shares)
    report = execute_stock_hedge(
        shares=shares,
        reference_price=reference_price,
        as_of_date=as_of_date,
        slippage_bps=slippage_bps,
        per_share_commission=per_share_commission,
    )
    update_stock_hedge(portfolio, report, reference_price)
    return report
