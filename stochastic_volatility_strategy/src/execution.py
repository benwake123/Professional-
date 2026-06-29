"""
Modeled execution: bid/ask-aware fills, commissions, and slippage.

Purpose
-------
Produces :class:`~src.types.ExecutionReport` objects that describe the cash
flows and costs of opening or closing a multi-leg option structure and of
buying/selling SPY shares for delta hedging. Side-aware: buys are filled at
ask (plus a fraction of the spread) and sells at bid (minus a fraction).

Module connections
------------------
Upstream:
    - ``src.types.{ExecutionReport, OptionStructure, OptionLeg}``.
Downstream:
    - ``src.backtest`` calls :func:`execute_option_entry`,
      :func:`execute_option_exit`, and :func:`execute_stock_hedge`.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

from src.types import ExecutionReport, OptionLeg, OptionStructure

DateLike = Union[str, date, pd.Timestamp]


def calculate_mid_price(bid: float, ask: float) -> float:
    return 0.5 * (float(bid) + float(ask))


def estimate_option_fill_price(
    bid: float, ask: float, side: str, slippage_fraction_of_spread: float
) -> float:
    """Side-aware fill price at bid/ask plus optional spread slippage."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'.")
    half_spread = 0.5 * (float(ask) - float(bid))
    extra = float(slippage_fraction_of_spread) * half_spread
    if side == "buy":
        return float(ask) + extra
    return float(bid) - extra


def estimate_stock_fill_price(reference_price: float, side: str, slippage_bps: float) -> float:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'.")
    direction = 1.0 if side == "buy" else -1.0
    return float(reference_price) * (1.0 + direction * slippage_bps / 10_000.0)


def calculate_option_commission(
    leg_count: int, contracts_per_leg: int, commission_per_contract: float
) -> float:
    return float(abs(contracts_per_leg) * leg_count * commission_per_contract)


def calculate_stock_commission(shares: int, per_share_commission: float = 0.0) -> float:
    return float(abs(shares) * per_share_commission)


def validate_executable_quotes(structure: OptionStructure) -> None:
    """Reject stale, missing, crossed, or non-positive option quotes."""
    for leg in structure.legs:
        bid = leg.bid
        ask = leg.ask
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError(
                f"Leg {leg.contract_id} has unusable quote: bid={bid}, ask={ask}."
            )


def _build_leg_fill(
    leg: OptionLeg,
    quantity: int,
    side: str,
    slippage_fraction_of_spread: float,
    fill_at_midpoint: bool = False,
) -> dict[str, object]:
    midpoint = calculate_mid_price(leg.bid, leg.ask)
    if fill_at_midpoint:
        fill_price = midpoint
        slippage_per_contract = 0.0
    else:
        fill_price = estimate_option_fill_price(
            bid=leg.bid, ask=leg.ask, side=side, slippage_fraction_of_spread=slippage_fraction_of_spread
        )
        half_spread = 0.5 * (float(leg.ask) - float(leg.bid))
        slippage_per_contract = float(slippage_fraction_of_spread) * half_spread
    return {
        "contract_id": leg.contract_id,
        "option_type": leg.option_type,
        "strike": leg.strike,
        "expiration": leg.expiration,
        "quantity": int(quantity),
        "fill_price": float(fill_price),
        "midpoint": float(midpoint),
        "slippage_per_contract": float(slippage_per_contract),
    }


def execute_option_entry(
    structure: OptionStructure,
    quantity_per_leg: int,
    as_of_date: DateLike,
    commission_per_contract: float,
    slippage_fraction_of_spread: float,
    contract_multiplier: int = 100,
    fill_at_midpoint: bool = False,
) -> ExecutionReport:
    """Generate fills + costs for opening ``quantity_per_leg`` of each leg."""
    validate_executable_quotes(structure)
    fills: list[dict[str, object]] = []
    gross_cash = 0.0
    slippage_total = 0.0

    for leg in structure.legs:
        leg_qty = int(quantity_per_leg) * int(leg.quantity)
        side = "buy" if leg_qty > 0 else "sell"
        fill = _build_leg_fill(
            leg, leg_qty, side, slippage_fraction_of_spread, fill_at_midpoint=fill_at_midpoint
        )
        fills.append(fill)
        gross_cash -= leg_qty * fill["fill_price"] * contract_multiplier
        slippage_total += abs(leg_qty) * fill["slippage_per_contract"] * contract_multiplier

    commission = calculate_option_commission(
        leg_count=len(structure.legs),
        contracts_per_leg=quantity_per_leg,
        commission_per_contract=commission_per_contract,
    )
    net_cash = gross_cash - commission - slippage_total

    return ExecutionReport(
        execution_date=pd.Timestamp(as_of_date).date(),
        action="entry",
        fills=fills,
        gross_cash_flow=float(gross_cash),
        commission=float(commission),
        slippage_cost=float(slippage_total),
        net_cash_flow=float(net_cash),
    )


def execute_option_exit(
    structure: OptionStructure,
    quantity_per_leg: int,
    as_of_date: DateLike,
    commission_per_contract: float,
    slippage_fraction_of_spread: float,
    contract_multiplier: int = 100,
    fill_at_midpoint: bool = False,
) -> ExecutionReport:
    """Generate fills + costs for closing ``quantity_per_leg`` of each leg."""
    validate_executable_quotes(structure)
    fills: list[dict[str, object]] = []
    gross_cash = 0.0
    slippage_total = 0.0

    for leg in structure.legs:
        leg_qty = -int(quantity_per_leg) * int(leg.quantity)  # exit reverses signs
        side = "buy" if leg_qty > 0 else "sell"
        fill = _build_leg_fill(
            leg, leg_qty, side, slippage_fraction_of_spread, fill_at_midpoint=fill_at_midpoint
        )
        fills.append(fill)
        gross_cash -= leg_qty * fill["fill_price"] * contract_multiplier
        slippage_total += abs(leg_qty) * fill["slippage_per_contract"] * contract_multiplier

    commission = calculate_option_commission(
        leg_count=len(structure.legs),
        contracts_per_leg=quantity_per_leg,
        commission_per_contract=commission_per_contract,
    )
    net_cash = gross_cash - commission - slippage_total

    return ExecutionReport(
        execution_date=pd.Timestamp(as_of_date).date(),
        action="exit",
        fills=fills,
        gross_cash_flow=float(gross_cash),
        commission=float(commission),
        slippage_cost=float(slippage_total),
        net_cash_flow=float(net_cash),
    )


def execute_stock_hedge(
    shares: int,
    reference_price: float,
    as_of_date: DateLike,
    slippage_bps: float,
    per_share_commission: float = 0.0,
) -> ExecutionReport:
    """Buy or sell SPY shares to neutralize delta."""
    if shares == 0:
        return ExecutionReport(
            execution_date=pd.Timestamp(as_of_date).date(),
            action="hedge",
            fills=[],
            gross_cash_flow=0.0,
            commission=0.0,
            slippage_cost=0.0,
            net_cash_flow=0.0,
        )
    side = "buy" if shares > 0 else "sell"
    fill_price = estimate_stock_fill_price(reference_price, side, slippage_bps)
    slippage_per_share = (fill_price - reference_price) if side == "buy" else (reference_price - fill_price)
    gross = -float(shares) * float(fill_price)
    commission = calculate_stock_commission(shares, per_share_commission)
    return ExecutionReport(
        execution_date=pd.Timestamp(as_of_date).date(),
        action="hedge",
        fills=[
            {
                "instrument": "SPY",
                "side": side,
                "shares": int(shares),
                "reference_price": float(reference_price),
                "fill_price": float(fill_price),
                "slippage_per_share": float(slippage_per_share),
            }
        ],
        gross_cash_flow=float(gross),
        commission=float(commission),
        slippage_cost=float(abs(shares) * slippage_per_share),
        net_cash_flow=float(gross - commission - abs(shares) * slippage_per_share),
    )
