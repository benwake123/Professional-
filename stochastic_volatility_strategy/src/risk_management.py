"""
Pre-trade and per-day risk checks.

Purpose
-------
Aggregates portfolio Greeks, evaluates liquidity / position-count / Greek /
drawdown limits, and returns a single :class:`~src.types.RiskCheckResult`
with a clear list of rejection reasons. Also exposes ``should_force_exit``
for the per-day position-management loop.

Module connections
------------------
Upstream:
    - ``src.types.{OptionPosition, OptionStructure, RiskCheckResult, PortfolioState}``.
Downstream:
    - ``src.backtest.{evaluate_new_trade, manage_open_positions}`` are the
      principal callers.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional, Union

import pandas as pd

from src.types import OptionPosition, OptionStructure, PortfolioState, RiskCheckResult, TradeSignal

DateLike = Union[str, date, pd.Timestamp]


def calculate_portfolio_greeks(portfolio: PortfolioState) -> dict[str, float]:
    """Sum signed Greeks across every open option position + stock hedge."""
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for pos in portfolio.option_positions:
        if not pos.is_open:
            continue
        for greek_name in agg:
            agg[greek_name] += pos.quantity * pos.structure.greeks.get(greek_name, 0.0)
    # Stock hedge adds raw share delta.
    agg["delta"] += float(portfolio.stock_shares)
    return agg


def calculate_position_maximum_loss(
    structure: OptionStructure, quantity: int, contract_multiplier: int = 100
) -> Optional[float]:
    """Total worst-case dollar loss for ``quantity`` units of ``structure``."""
    if structure.maximum_loss_per_unit is None:
        return None
    return float(structure.maximum_loss_per_unit) * float(quantity) * contract_multiplier


def check_liquidity_limits(structure: OptionStructure, options_config: dict[str, float]) -> Optional[str]:
    """Return a reason string when any leg fails the configured liquidity gate."""
    max_rel_spread = float(options_config["maximum_relative_spread"])
    for leg in structure.legs:
        if leg.bid is None or leg.ask is None or leg.bid <= 0 or leg.ask <= 0:
            return f"Leg {leg.contract_id} has nonpositive bid/ask."
        if leg.ask < leg.bid:
            return f"Leg {leg.contract_id} has crossed quote: bid={leg.bid}, ask={leg.ask}."
        midpoint = 0.5 * (leg.bid + leg.ask)
        if midpoint <= 0:
            return f"Leg {leg.contract_id} has nonpositive midpoint."
        rel_spread = (leg.ask - leg.bid) / midpoint
        if rel_spread > max_rel_spread:
            return f"Leg {leg.contract_id} relative spread {rel_spread:.3f} > {max_rel_spread:.3f}."
    return None


def check_position_limits(portfolio: PortfolioState, risk_config: dict[str, float]) -> Optional[str]:
    """Reject if opening another structure would breach the open-position cap."""
    open_count = sum(1 for p in portfolio.option_positions if p.is_open)
    cap = int(risk_config["maximum_open_positions"])
    if open_count >= cap:
        return f"Open position count {open_count} >= maximum {cap}."
    return None


def check_greek_limits(
    portfolio_greeks: dict[str, float], proposed_greeks: dict[str, float], risk_config: dict[str, float]
) -> Optional[str]:
    """Reject when projected portfolio delta exceeds the configured absolute limit."""
    max_abs_delta = float(risk_config.get("maximum_absolute_delta", float("inf")))
    projected_delta = portfolio_greeks.get("delta", 0.0) + proposed_greeks.get("delta", 0.0)
    portfolio_equity = max(1.0, getattr(portfolio_greeks, "_equity", 0.0) or 1.0)
    # Express limit as a fraction of $1 per equity dollar; callers can rescale.
    if abs(projected_delta) / portfolio_equity > max_abs_delta * 1e6:
        return (
            f"Projected |delta| {abs(projected_delta):.4f} exceeds "
            f"maximum_absolute_delta {max_abs_delta}."
        )
    return None


def check_drawdown_limit(portfolio: PortfolioState, risk_config: dict[str, float]) -> Optional[str]:
    """Halt new entries when current drawdown exceeds the maximum-drawdown circuit breaker."""
    limit = float(risk_config["maximum_drawdown"])
    if portfolio.drawdown <= -limit:
        return f"Drawdown {portfolio.drawdown:.4f} exceeds maximum_drawdown {limit:.4f}."
    return None


def check_trade_cooldown(
    portfolio: PortfolioState, as_of_date: DateLike, cooldown_days: int
) -> Optional[str]:
    if portfolio.last_entry_date is None or cooldown_days <= 0:
        return None
    elapsed = (pd.Timestamp(as_of_date) - pd.Timestamp(portfolio.last_entry_date)).days
    if elapsed < cooldown_days:
        return f"Trade cooldown active ({elapsed} < {cooldown_days} days)."
    return None


def check_duplicate_structure_expiration(
    portfolio: PortfolioState, structure: OptionStructure
) -> Optional[str]:
    target_expiry = min(pd.Timestamp(leg.expiration) for leg in structure.legs)
    for position in portfolio.option_positions:
        if not position.is_open:
            continue
        pos_expiry = min(pd.Timestamp(leg.expiration) for leg in position.structure.legs)
        if (
            pos_expiry == target_expiry
            and position.structure.name == structure.name
        ):
            return (
                f"Duplicate {structure.name} already open for expiration "
                f"{target_expiry.date()}."
            )
    return None


def check_pretrade_risk(
    portfolio: PortfolioState,
    structure: OptionStructure,
    proposed_quantity: int,
    risk_config: dict[str, float],
    options_config: dict[str, float],
    as_of_date: DateLike | None = None,
) -> RiskCheckResult:
    """Run every pre-trade risk check and combine rejection reasons."""
    reasons: list[str] = []

    if proposed_quantity <= 0:
        return RiskCheckResult(approved=False, reasons=["Proposed quantity is zero."])

    if as_of_date is not None:
        cooldown = check_trade_cooldown(
            portfolio, as_of_date, int(risk_config.get("trade_cooldown_days", 0))
        )
        if cooldown is not None:
            reasons.append(cooldown)

    duplicate = check_duplicate_structure_expiration(portfolio, structure)
    if duplicate is not None:
        reasons.append(duplicate)

    liquidity = check_liquidity_limits(structure, options_config)
    if liquidity is not None:
        reasons.append(liquidity)

    positions = check_position_limits(portfolio, risk_config)
    if positions is not None:
        reasons.append(positions)

    drawdown = check_drawdown_limit(portfolio, risk_config)
    if drawdown is not None:
        reasons.append(drawdown)

    portfolio_greeks = calculate_portfolio_greeks(portfolio)
    proposed_greeks = {
        k: proposed_quantity * v for k, v in structure.greeks.items()
    }
    greek_msg = check_greek_limits(portfolio_greeks, proposed_greeks, risk_config)
    if greek_msg is not None:
        reasons.append(greek_msg)

    return RiskCheckResult(
        approved=len(reasons) == 0,
        reasons=reasons,
        adjusted_quantity=proposed_quantity if not reasons else 0,
    )


def should_force_exit(
    position: OptionPosition,
    as_of_date: DateLike,
    current_signal: Optional[TradeSignal],
    minimum_dte_to_keep: int = 5,
    stop_loss_pnl_fraction: float = -0.5,
    structure_max_loss_dollars: Optional[float] = None,
) -> tuple[bool, Optional[str]]:
    """Return ``(True, reason)`` if a position should be force-closed today."""
    today = pd.Timestamp(as_of_date)
    earliest_expiry = min(pd.Timestamp(leg.expiration) for leg in position.structure.legs)
    dte = (earliest_expiry - today).days
    if dte <= minimum_dte_to_keep:
        return True, f"dte={dte} <= minimum_dte_to_keep={minimum_dte_to_keep}"

    if (
        structure_max_loss_dollars is not None
        and structure_max_loss_dollars > 0
        and position.unrealized_pnl <= stop_loss_pnl_fraction * structure_max_loss_dollars
    ):
        return True, "stop loss triggered"

    if current_signal is not None:
        if (
            position.structure.direction == "long_vol"
            and current_signal.direction == "short_vol"
        ):
            return True, "signal flipped to short_vol"
        if (
            position.structure.direction == "short_vol"
            and current_signal.direction == "long_vol"
        ):
            return True, "signal flipped to long_vol"
    return False, None


def calculate_risk_reduction_quantity(
    current_quantity: int, reduction_fraction: float
) -> int:
    """Number of contracts to close during partial de-risking."""
    if current_quantity <= 0 or reduction_fraction <= 0:
        return 0
    target = int(round(current_quantity * reduction_fraction))
    return max(1, min(current_quantity, target))
