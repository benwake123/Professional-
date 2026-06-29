"""
Backtest audit artifacts: decision funnel, trade audit, reconciliation.

Purpose
-------
Produces the CSV audit files requested in Part 1 of the research brief and
summarizes where decision dates drop out of the pipeline.

Module connections
------------------
Upstream: ``src.accounting``, ``src.types.BacktestResults``.
Downstream: ``src.run_pipeline.export_results``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.accounting import build_accounting_snapshot, position_realized_pnl_components
from src.types import BacktestResults, OptionPosition, PortfolioState


FUNNEL_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "has_required_market_data",
    "has_valid_option_chain",
    "has_sufficient_model_history",
    "model_calibration_succeeded",
    "forecast_created",
    "signal_exceeded_threshold",
    "eligible_expiration_found",
    "atm_contracts_found",
    "liquidity_filter_passed",
    "position_size_positive",
    "risk_check_passed",
    "trade_executed",
    "rejection_reason",
)


@dataclass
class DecisionFunnelRow:
    decision_date: pd.Timestamp
    has_required_market_data: bool = False
    has_valid_option_chain: bool = False
    has_sufficient_model_history: bool = False
    model_calibration_succeeded: bool = False
    forecast_created: bool = False
    signal_exceeded_threshold: bool = False
    eligible_expiration_found: bool = False
    atm_contracts_found: bool = False
    liquidity_filter_passed: bool = False
    position_size_positive: bool = False
    risk_check_passed: bool = False
    trade_executed: bool = False
    rejection_reason: str = "not_evaluated"

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_date": self.decision_date,
            "has_required_market_data": self.has_required_market_data,
            "has_valid_option_chain": self.has_valid_option_chain,
            "has_sufficient_model_history": self.has_sufficient_model_history,
            "model_calibration_succeeded": self.model_calibration_succeeded,
            "forecast_created": self.forecast_created,
            "signal_exceeded_threshold": self.signal_exceeded_threshold,
            "eligible_expiration_found": self.eligible_expiration_found,
            "atm_contracts_found": self.atm_contracts_found,
            "liquidity_filter_passed": self.liquidity_filter_passed,
            "position_size_positive": self.position_size_positive,
            "risk_check_passed": self.risk_check_passed,
            "trade_executed": self.trade_executed,
            "rejection_reason": self.rejection_reason,
        }


def summarize_decision_funnel(funnel: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage surviving each stage."""
    if funnel.empty:
        return pd.DataFrame()
    total = len(funnel)
    stages = [c for c in FUNNEL_COLUMNS if c not in ("decision_date", "rejection_reason")]
    rows = []
    for stage in stages:
        count = int(funnel[stage].sum())
        rows.append(
            {
                "stage": stage,
                "count": count,
                "pct_of_decisions": 100.0 * count / total,
            }
        )
    return pd.DataFrame(rows)


def summarize_rejections(funnel: pd.DataFrame) -> pd.DataFrame:
    if funnel.empty:
        return pd.DataFrame()
    summary = (
        funnel.groupby("rejection_reason", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["pct_of_decisions"] = 100.0 * summary["count"] / len(funnel)
    return summary


def build_trade_audit_rows(
    portfolio: PortfolioState,
    underlying: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """One row per closed trade with signal, execution, and P&L fields."""
    rows: list[dict[str, object]] = []
    for position in portfolio.option_positions:
        if position.is_open or position.exit_execution is None:
            continue
        signal = position.entry_signal
        comps = position_realized_pnl_components(position)
        entry_date = pd.Timestamp(position.entry_date)
        exit_date = pd.Timestamp(position.exit_date)
        spot_entry = _spot_on(underlying, entry_date)
        spot_exit = _spot_on(underlying, exit_date)
        atm_strike = position.structure.legs[0].strike
        expiry = min(pd.Timestamp(leg.expiration) for leg in position.structure.legs)
        entry_iv = float(
            np.nanmean([leg.implied_volatility for leg in position.structure.legs])
        )
        subsequent_realized = _realized_after(underlying, entry_date, horizon_days)
        rows.append(
            {
                "decision_date": entry_date,
                "model_name": signal.model_name,
                "forecast_realized_variance": signal.forecast_variance,
                "implied_variance": signal.implied_variance,
                "variance_edge": signal.variance_edge,
                "variance_edge_zscore": signal.zscore,
                "trade_direction": signal.direction,
                "structure_name": position.structure.name,
                "spot_at_entry": spot_entry,
                "strike": atm_strike,
                "expiration": expiry.date(),
                "days_to_expiration": int((expiry - entry_date).days),
                "entry_option_iv": entry_iv,
                "exit_option_iv": entry_iv,
                "subsequent_realized_variance": subsequent_realized,
                "underlying_return": (
                    (spot_exit / spot_entry - 1.0) if spot_entry and spot_exit else float("nan")
                ),
                "gross_option_pnl": comps["gross_option_pnl"],
                "hedge_pnl": float("nan"),
                "commission": comps["commission"],
                "slippage": comps["slippage"],
                "net_pnl": comps["net_option_pnl"],
                "holding_days": int((exit_date - entry_date).days),
                "exit_reason": position.exit_execution.action,
            }
        )
    return pd.DataFrame(rows)


def _spot_on(underlying: pd.DataFrame, as_of: pd.Timestamp) -> Optional[float]:
    row = underlying[underlying["date"] == as_of]
    if row.empty:
        return None
    return float(row["close"].iloc[-1])


def _realized_after(
    underlying: pd.DataFrame, start: pd.Timestamp, horizon_days: int
) -> Optional[float]:
    from src.forecast_engine import realized_variance_over_horizon

    return realized_variance_over_horizon(underlying, start, horizon_days)


def attach_audit_artifacts(
    results: BacktestResults,
    portfolio: PortfolioState,
    underlying: pd.DataFrame,
    horizon_days: int,
) -> BacktestResults:
    snapshot = build_accounting_snapshot(portfolio)
    results.accounting_reconciliation = pd.DataFrame([snapshot.__dict__])
    if hasattr(results, "decision_funnel") and isinstance(results.decision_funnel, pd.DataFrame):
        results.rejection_summary = summarize_rejections(results.decision_funnel)
    results.trade_audit = build_trade_audit_rows(portfolio, underlying, horizon_days)
    if not results.forecasts.empty:
        from src.forecast_engine import evaluate_forecast_diagnostics

        merged_forecasts = _merge_implied_into_forecasts(
            results.forecasts, results.trade_decisions
        )
        results.forecast_diagnostics = evaluate_forecast_diagnostics(
            underlying, merged_forecasts, horizon_days
        )
    return results


def _merge_implied_into_forecasts(
    forecasts: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    if forecasts.empty or decisions.empty:
        return forecasts
    implied_cols = [c for c in ("implied_variance", "variance_edge") if c in decisions.columns]
    if not implied_cols or "as_of_date" not in decisions.columns:
        return forecasts
    merged = forecasts.merge(
        decisions[["as_of_date", *implied_cols]].drop_duplicates("as_of_date"),
        on="as_of_date",
        how="left",
    )
    return merged
