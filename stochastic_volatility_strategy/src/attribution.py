"""
PnL attribution.

Purpose
-------
Decomposes total PnL across (a) option structures, (b) the delta hedge,
(c) transaction-cost drag, and (d) the regime in effect at trade entry.
Outputs a dict of small DataFrames keyed by attribution slice.

Module connections
------------------
Upstream:
    - ``src.types.BacktestResults`` is the typical input.
Downstream:
    - ``src.run_pipeline.export_results`` writes each frame to disk.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def calculate_option_leg_pnl(
    entry_fill: float, exit_fill: float, quantity: int, contract_multiplier: int = 100
) -> float:
    """Per-leg PnL: (exit - entry) * signed_quantity * multiplier."""
    return float((exit_fill - entry_fill) * quantity * contract_multiplier)


def calculate_option_structure_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized PnL per closed structure."""
    if trades.empty:
        return pd.DataFrame(columns=["position_id", "realized_pnl"])
    exits = trades[trades["action"] == "exit"].copy()
    if exits.empty:
        return pd.DataFrame(columns=["position_id", "realized_pnl"])
    return (
        exits.groupby("position_id")["realized_pnl"]
        .sum()
        .reset_index()
        .sort_values("realized_pnl", ascending=False)
    )


def calculate_hedge_pnl(trades: pd.DataFrame) -> float:
    """Net cash flow from every stock hedge action."""
    if trades.empty or "action" not in trades.columns:
        return 0.0
    hedges = trades[trades["action"] == "hedge"]
    if hedges.empty:
        return 0.0
    return float(hedges["net_cash_flow"].sum())


def calculate_transaction_cost_drag(daily_history: pd.DataFrame) -> dict[str, float]:
    """Total commission + slippage cost dollars."""
    if daily_history.empty:
        return {"commission": 0.0, "slippage": 0.0, "total": 0.0}
    commission = float(daily_history["cumulative_commissions"].iloc[-1])
    slippage = float(daily_history["cumulative_slippage"].iloc[-1])
    return {"commission": commission, "slippage": slippage, "total": commission + slippage}


def calculate_greek_pnl_approximation(
    delta: float, gamma: float, vega: float, theta: float, dS: float, dSigma: float, dT: float
) -> dict[str, float]:
    """Greek-decomposition PnL: delta + gamma + vega + theta."""
    return {
        "delta_pnl": float(delta * dS),
        "gamma_pnl": float(0.5 * gamma * dS * dS),
        "vega_pnl": float(vega * dSigma),
        "theta_pnl": float(theta * dT),
    }


def aggregate_pnl_by_regime(
    decisions: pd.DataFrame, trades: pd.DataFrame
) -> pd.DataFrame:
    """Group realized PnL by the regime in effect on the entry decision date."""
    if trades.empty or decisions.empty:
        return pd.DataFrame(columns=["regime", "trade_count", "total_pnl", "average_pnl"])

    exits = trades[trades["action"] == "exit"].copy()
    if exits.empty:
        return pd.DataFrame(columns=["regime", "trade_count", "total_pnl", "average_pnl"])

    # No entry_date here, so map via position_id -> entry date via the trade log.
    entries = trades[trades["action"] == "entry"][["date", "position_id"]].copy()
    entries = entries.rename(columns={"date": "entry_date"})
    exits = exits.merge(entries, on="position_id", how="left")
    exits = exits.merge(
        decisions[["as_of_date", "regime"]],
        left_on="entry_date",
        right_on="as_of_date",
        how="left",
    )
    summary = (
        exits.groupby("regime", dropna=False)["realized_pnl"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "trade_count",
                "sum": "total_pnl",
                "mean": "average_pnl",
            }
        )
    )
    return summary


def build_pnl_attribution(
    daily_history: pd.DataFrame,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Bundle every attribution slice into a single dict."""
    return {
        "by_structure": calculate_option_structure_pnl(trades),
        "by_regime": aggregate_pnl_by_regime(decisions, trades),
        "transaction_cost_drag": pd.DataFrame(
            [calculate_transaction_cost_drag(daily_history)]
        ),
        "hedge_pnl_total": pd.DataFrame(
            [{"hedge_net_cash_flow": calculate_hedge_pnl(trades)}]
        ),
    }
