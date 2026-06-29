"""
Configurable position exit rules.

Purpose
-------
Centralizes every reason a position may be closed before natural expiry:
signal convergence, profit target, max loss, max holding period, minimum
DTE, forecast reversal, and portfolio risk circuit breakers.

Module connections
------------------
Upstream: ``src.types.{OptionPosition, TradeSignal, PortfolioState}``.
Downstream: ``src.backtest.manage_open_positions``.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Union

import pandas as pd

from src.types import OptionPosition, PortfolioState, TradeSignal

DateLike = Union[str, date, pd.Timestamp]


def holding_days(entry_date: date, as_of_date: DateLike) -> int:
    return int((pd.Timestamp(as_of_date) - pd.Timestamp(entry_date)).days)


def minimum_dte(position: OptionPosition, as_of_date: DateLike) -> int:
    expiry = min(pd.Timestamp(leg.expiration) for leg in position.structure.legs)
    return int((expiry - pd.Timestamp(as_of_date)).days)


def should_exit_position(
    position: OptionPosition,
    as_of_date: DateLike,
    current_signal: Optional[TradeSignal],
    portfolio: PortfolioState,
    exit_config: dict[str, float],
    structure_max_loss_dollars: Optional[float] = None,
) -> tuple[bool, Optional[str]]:
    """Return ``(True, reason)`` when any exit rule fires."""
    days_held = holding_days(position.entry_date, as_of_date)
    dte = minimum_dte(position, as_of_date)

    if dte <= int(exit_config.get("minimum_dte_to_keep", 5)):
        return True, "minimum_dte"

    max_hold = int(exit_config.get("maximum_holding_days", 20))
    if days_held >= max_hold:
        return True, "maximum_holding_period"

    if structure_max_loss_dollars and structure_max_loss_dollars > 0:
        stop_frac = float(exit_config.get("maximum_loss_fraction", 0.50))
        if position.unrealized_pnl <= -stop_frac * structure_max_loss_dollars:
            return True, "maximum_loss"

    profit_target = float(exit_config.get("profit_target_fraction", 0.0))
    if (
        profit_target > 0
        and structure_max_loss_dollars
        and structure_max_loss_dollars > 0
        and position.unrealized_pnl >= profit_target * structure_max_loss_dollars
    ):
        return True, "profit_target"

    if current_signal is not None:
        z = current_signal.zscore
        exit_threshold = float(exit_config.get("signal_convergence_threshold", 0.25))
        if not pd.isna(z) and abs(z) <= exit_threshold:
            return True, "signal_convergence"

        if position.structure.direction == "long_vol" and current_signal.direction == "short_vol":
            return True, "forecast_reversal"
        if position.structure.direction == "short_vol" and current_signal.direction == "long_vol":
            return True, "forecast_reversal"

    max_dd = float(exit_config.get("portfolio_max_drawdown_exit", 0.15))
    if portfolio.drawdown <= -max_dd:
        return True, "portfolio_risk"

    return False, None
