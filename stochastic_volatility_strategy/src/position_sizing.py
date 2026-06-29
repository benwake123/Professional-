"""
Position sizing: volatility-targeted + risk-capped.

Purpose
-------
Determines how many contracts of a proposed structure to open. Composes
five inputs in a strict order:

    signal-strength multiplier
    volatility-target size
    optional fractional-Kelly cap
    regime-based size multiplier
    per-trade risk cap (max-loss limit)
    rounded to whole contracts

Module connections
------------------
Upstream:
    - ``src.regime.regime_risk_multiplier`` for the regime-aware multiplier.
    - ``src.types.{OptionStructure, TradeSignal}`` for typed inputs.
Downstream:
    - ``src.backtest.evaluate_new_trade`` is the primary caller.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.regime import regime_risk_multiplier
from src.types import OptionStructure, TradeSignal


def estimate_strategy_pnl_volatility(
    portfolio_equity_history: pd.Series,
    window: int = 60,
    annualization_factor: int = 252,
    default_volatility: float = 0.10,
) -> float:
    """Annualized stdev of recent strategy returns.

    Falls back to ``default_volatility`` when history is too short or when
    equity has not moved yet (all returns zero), which is the normal state
    at the start of a backtest before the first trade is opened.
    """
    if portfolio_equity_history is None or portfolio_equity_history.size < 2:
        return default_volatility
    returns = portfolio_equity_history.pct_change().dropna().tail(window)
    if returns.size < 2:
        return default_volatility
    vol = float(returns.std(ddof=1) * np.sqrt(annualization_factor))
    if vol <= 0 or np.isnan(vol):
        return default_volatility
    return vol


def calculate_signal_strength_multiplier(
    zscore: float, entry_threshold: float, cap: float = 2.0
) -> float:
    """Map z-score magnitude into a [0, cap] multiplier."""
    if np.isnan(zscore) or np.isnan(entry_threshold) or entry_threshold <= 0:
        return 0.0
    raw = (abs(zscore) - entry_threshold) / entry_threshold
    if raw <= 0:
        return 0.0
    return float(min(cap, 1.0 + raw))


def calculate_volatility_target_size(
    portfolio_equity: float,
    structure: OptionStructure,
    target_annual_volatility: float,
    strategy_pnl_volatility: float,
    contract_multiplier: int = 100,
) -> float:
    """Exposure-implied number of contracts to hit a target portfolio vol."""
    if strategy_pnl_volatility <= 0:
        return 0.0
    structure_vega = abs(structure.greeks.get("vega", 0.0)) * contract_multiplier
    if structure_vega <= 0:
        # No vega -> fall back to delta-based exposure.
        structure_delta = abs(structure.greeks.get("delta", 0.0)) * contract_multiplier
        if structure_delta <= 0:
            return 0.0
        return (portfolio_equity * target_annual_volatility) / (
            structure_delta * strategy_pnl_volatility
        )
    return (portfolio_equity * target_annual_volatility) / (
        structure_vega * strategy_pnl_volatility
    )


def calculate_fractional_kelly_size(
    expected_edge: float,
    structure_max_loss: Optional[float],
    portfolio_equity: float,
    kelly_fraction: float = 0.25,
    contract_multiplier: int = 100,
) -> Optional[float]:
    """Cap implied by a fractional-Kelly bet sizing."""
    if structure_max_loss is None or structure_max_loss <= 0:
        return None
    if expected_edge <= 0:
        return 0.0
    fraction = kelly_fraction * min(1.0, expected_edge)
    return fraction * portfolio_equity / (structure_max_loss * contract_multiplier)


def apply_regime_size_multiplier(quantity: float, regime_label: str) -> float:
    """Multiply by the regime-derived multiplier (e.g. ``0.5`` in high-vol regimes)."""
    return float(quantity * regime_risk_multiplier(regime_label))


def apply_trade_risk_cap(
    quantity: float,
    structure_max_loss: Optional[float],
    portfolio_equity: float,
    max_trade_risk_fraction: float,
    contract_multiplier: int = 100,
) -> float:
    """Cap the quantity so worst-case loss stays inside the configured fraction."""
    if structure_max_loss is None or structure_max_loss <= 0:
        return quantity
    max_dollar_loss = portfolio_equity * max_trade_risk_fraction
    cap = max_dollar_loss / (structure_max_loss * contract_multiplier)
    return float(min(quantity, cap))


def round_to_contract_quantity(quantity: float) -> int:
    """Round to whole contracts and clamp to nonnegative."""
    if quantity <= 0 or np.isnan(quantity):
        return 0
    return int(math.floor(quantity))


def calculate_position_size(
    signal: TradeSignal,
    structure: OptionStructure,
    portfolio_equity: float,
    portfolio_equity_history: Optional[pd.Series],
    risk_config: dict[str, float],
    contract_multiplier: int = 100,
) -> int:
    """Vol-target sizing capped by per-trade maximum-loss budget."""
    if signal.direction == "flat":
        return 0

    base_risk_budget = portfolio_equity * float(risk_config["maximum_trade_risk_fraction"])
    max_loss_per_unit = structure.maximum_loss_per_unit
    if max_loss_per_unit is None or max_loss_per_unit <= 0:
        risk_cap_qty = 1
    else:
        risk_cap_qty = base_risk_budget / (max_loss_per_unit * contract_multiplier)

    strategy_vol = estimate_strategy_pnl_volatility(portfolio_equity_history)
    strength = calculate_signal_strength_multiplier(
        signal.zscore,
        signal.entry_threshold if signal.entry_threshold > 0 else 1.0,
    )
    vol_target_size = calculate_volatility_target_size(
        portfolio_equity=portfolio_equity,
        structure=structure,
        target_annual_volatility=float(risk_config["target_annual_volatility"]),
        strategy_pnl_volatility=strategy_vol,
        contract_multiplier=contract_multiplier,
    )

    sized = min(vol_target_size * strength, risk_cap_qty)
    sized = apply_regime_size_multiplier(sized, signal.regime)

    max_vega = float(risk_config.get("maximum_portfolio_vega", float("inf")))
    structure_vega = abs(structure.greeks.get("vega", 0.0)) * contract_multiplier
    if structure_vega > 0 and np.isfinite(max_vega):
        sized = min(sized, max_vega / structure_vega)

    return round_to_contract_quantity(sized)
