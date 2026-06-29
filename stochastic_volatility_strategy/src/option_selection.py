"""
Option contract selection.

Purpose
-------
Given a point-in-time options snapshot and a target direction, picks the
matched contracts that make up the trade structure (long straddle for
``long_vol``, iron butterfly for ``short_vol``), applies liquidity filters,
and assembles an :class:`~src.types.OptionStructure`. Returns ``None`` if
the snapshot is empty or no contracts pass the filters (the public data
pack ships with no options, so that branch is exercised by the pipeline).

Module connections
------------------
Upstream:
    - ``src.black_scholes.calculate_structure_greeks`` aggregates Greeks
      across the legs we choose.
    - ``src.types.{OptionLeg, OptionStructure}``.
Downstream:
    - ``src.backtest.evaluate_new_trade`` calls
      :func:`select_trade_structure` whenever the signal layer produces a
      directional decision.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.black_scholes import calculate_structure_greeks as _bs_structure_greeks
from src.types import OptionLeg, OptionStructure


DateLike = Union[str, date, datetime, pd.Timestamp]


def calculate_days_to_expiration(
    quote_date: pd.Series, expiration: pd.Series
) -> pd.Series:
    """Calendar days between two date columns."""
    return (expiration - quote_date).dt.days


def calculate_relative_bid_ask_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """``(ask - bid) / midpoint``. NaN where midpoint is zero."""
    midpoint = (bid + ask) / 2.0
    spread = ask - bid
    rel = spread / midpoint.replace(0.0, np.nan)
    return rel


def filter_liquid_options(
    options: pd.DataFrame,
    minimum_dte: int,
    maximum_dte: int,
    maximum_relative_spread: float,
    minimum_open_interest: int,
    minimum_volume: int,
) -> pd.DataFrame:
    """Apply DTE, spread, volume, open-interest, and quote-validity filters."""
    if options.empty:
        return options.copy()

    df = options.copy()
    df["dte"] = calculate_days_to_expiration(df["quote_date"], df["expiration"])
    df["relative_spread"] = calculate_relative_bid_ask_spread(df["bid"], df["ask"])

    keep = (
        (df["dte"] >= minimum_dte)
        & (df["dte"] <= maximum_dte)
        & (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["relative_spread"] <= maximum_relative_spread)
        & (df["open_interest"] >= minimum_open_interest)
        & (df["volume"] >= minimum_volume)
        & (df["implied_volatility"] > 0)
    )
    return df.loc[keep].reset_index(drop=True)


def select_target_expiration(
    options: pd.DataFrame, target_dte: int
) -> Optional[pd.Timestamp]:
    """Return the eligible expiration nearest to ``target_dte``."""
    if options.empty:
        return None
    distances = (options["dte"] - target_dte).abs()
    idx = distances.idxmin()
    return pd.Timestamp(options.loc[idx, "expiration"])


def find_atm_call_and_put(
    options: pd.DataFrame,
    expiration: pd.Timestamp,
    spot: float,
    atm_moneyness_low: float = 0.98,
    atm_moneyness_high: float = 1.02,
) -> Optional[tuple[pd.Series, pd.Series]]:
    """Return ATM call/put on ``expiration`` within the moneyness band."""
    if options.empty or spot <= 0:
        return None
    same_exp = options[options["expiration"] == expiration]
    calls = same_exp[same_exp["option_type"] == "call"]
    puts = same_exp[same_exp["option_type"] == "put"]
    if calls.empty or puts.empty:
        return None
    calls = calls[
        (calls["strike"] / spot >= atm_moneyness_low)
        & (calls["strike"] / spot <= atm_moneyness_high)
    ]
    puts = puts[
        (puts["strike"] / spot >= atm_moneyness_low)
        & (puts["strike"] / spot <= atm_moneyness_high)
    ]
    if calls.empty or puts.empty:
        calls = same_exp[same_exp["option_type"] == "call"]
        puts = same_exp[same_exp["option_type"] == "put"]
    call_row = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    put_row = puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[0]]
    return call_row, put_row


def find_wing_options(
    options: pd.DataFrame,
    expiration: pd.Timestamp,
    spot: float,
    wing_width_percent: float,
) -> Optional[tuple[pd.Series, pd.Series]]:
    """Find OTM call and OTM put used as wings for an iron butterfly."""
    if options.empty:
        return None
    same_exp = options[options["expiration"] == expiration]
    upper_target = spot * (1.0 + wing_width_percent)
    lower_target = spot * (1.0 - wing_width_percent)

    calls = same_exp[(same_exp["option_type"] == "call") & (same_exp["strike"] >= upper_target)]
    puts = same_exp[(same_exp["option_type"] == "put") & (same_exp["strike"] <= lower_target)]
    if calls.empty or puts.empty:
        return None
    wing_call = calls.iloc[(calls["strike"] - upper_target).abs().argsort().iloc[0]]
    wing_put = puts.iloc[(puts["strike"] - lower_target).abs().argsort().iloc[0]]
    return wing_call, wing_put


def _row_to_leg(row: pd.Series, quantity: int) -> OptionLeg:
    contract_id = (
        f"SPY {pd.Timestamp(row['expiration']).date()} "
        f"{row['option_type'].upper()} K={float(row['strike']):.2f}"
    )
    return OptionLeg(
        contract_id=contract_id,
        option_type=str(row["option_type"]).lower(),
        strike=float(row["strike"]),
        expiration=pd.Timestamp(row["expiration"]).date(),
        quantity=int(quantity),
        bid=float(row.get("bid", float("nan"))),
        ask=float(row.get("ask", float("nan"))),
        implied_volatility=float(row.get("implied_volatility", float("nan"))),
    )


def build_long_straddle(
    call_row: pd.Series, put_row: pd.Series, quantity: int = 1
) -> OptionStructure:
    """Long ATM call + long ATM put."""
    return OptionStructure(
        name="long_straddle",
        legs=[_row_to_leg(call_row, +quantity), _row_to_leg(put_row, +quantity)],
        direction="long_vol",
    )


def build_iron_butterfly(
    atm_call: pd.Series,
    atm_put: pd.Series,
    wing_call: pd.Series,
    wing_put: pd.Series,
    quantity: int = 1,
) -> OptionStructure:
    """Short ATM call + short ATM put + long wing call + long wing put."""
    legs = [
        _row_to_leg(atm_call, -quantity),
        _row_to_leg(atm_put, -quantity),
        _row_to_leg(wing_call, +quantity),
        _row_to_leg(wing_put, +quantity),
    ]
    return OptionStructure(name="iron_butterfly", legs=legs, direction="short_vol")


def calculate_structure_entry_value(structure: OptionStructure) -> float:
    """Quoted net debit (positive) or credit (negative) before execution costs."""
    total = 0.0
    for leg in structure.legs:
        if leg.bid is None or leg.ask is None:
            continue
        midpoint = 0.5 * (leg.bid + leg.ask)
        total += float(leg.quantity) * midpoint
    return float(total)


def calculate_structure_greeks(
    structure: OptionStructure,
    spot: float,
    as_of_date: DateLike,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Forward to :func:`src.black_scholes.calculate_structure_greeks`."""
    return _bs_structure_greeks(
        legs=structure.legs,
        spot=spot,
        as_of_date=as_of_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def select_trade_structure(
    options_snapshot: pd.DataFrame,
    spot: float,
    direction: str,
    options_config: dict[str, float],
    as_of_date: DateLike,
    risk_free_rate: float,
) -> Optional[OptionStructure]:
    """Full filtering + contract-selection workflow for one direction."""
    if direction not in ("long_vol", "short_vol"):
        return None
    if options_snapshot.empty:
        return None

    liquid = filter_liquid_options(
        options_snapshot,
        minimum_dte=int(options_config["minimum_dte"]),
        maximum_dte=int(options_config["maximum_dte"]),
        maximum_relative_spread=float(options_config["maximum_relative_spread"]),
        minimum_open_interest=int(options_config["minimum_open_interest"]),
        minimum_volume=int(options_config["minimum_volume"]),
    )
    if liquid.empty:
        return None

    target_dte = int(options_config.get("target_dte", 0.5 * (options_config["minimum_dte"] + options_config["maximum_dte"])))
    atm_low = float(options_config.get("atm_moneyness_low", 0.98))
    atm_high = float(options_config.get("atm_moneyness_high", 1.02))
    expiration = select_target_expiration(liquid, target_dte)
    if expiration is None:
        return None

    atm = find_atm_call_and_put(liquid, expiration, spot, atm_low, atm_high)
    if atm is None:
        return None
    atm_call, atm_put = atm

    if direction == "long_vol":
        structure = build_long_straddle(atm_call, atm_put)
        max_loss = calculate_structure_entry_value(structure)
        structure.maximum_loss_per_unit = max_loss if max_loss > 0 else None
    else:
        wings = find_wing_options(
            liquid, expiration, spot, float(options_config["wing_width_percent"])
        )
        if wings is None:
            return None
        wing_call, wing_put = wings
        structure = build_iron_butterfly(atm_call, atm_put, wing_call, wing_put)
        wing_distance = float(options_config["wing_width_percent"]) * spot
        credit = -calculate_structure_entry_value(structure)
        # Per-spread max loss = wing width minus net credit received. When the
        # quoted credit exceeds the wing width (mispriced chain), fall back to
        # the wing width itself so risk caps still bind.
        per_unit_max_loss = wing_distance - credit
        if per_unit_max_loss <= 0:
            per_unit_max_loss = wing_distance
        structure.maximum_loss_per_unit = float(per_unit_max_loss)

    structure.entry_debit_or_credit = calculate_structure_entry_value(structure)
    structure.greeks = calculate_structure_greeks(
        structure, spot=spot, as_of_date=as_of_date, risk_free_rate=risk_free_rate
    )
    return structure
