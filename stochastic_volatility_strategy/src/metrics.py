"""
Performance metrics.

Purpose
-------
Translates the backtest's daily-history DataFrame into the standard set of
risk/return statistics: annualized return, annualized volatility, Sharpe,
Sortino, max drawdown, Calmar, VaR, ES. Also computes per-trade stats.

Module connections
------------------
Upstream:
    - ``src.types.BacktestResults`` is the consumer of
      :func:`build_performance_summary`.
Downstream:
    - ``src.run_pipeline.export_results`` writes the summary.
    - ``src.attribution`` reuses :func:`calculate_daily_returns`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_ANNUALIZATION_FACTOR: int = 252


def calculate_daily_returns(equity_series: pd.Series) -> pd.Series:
    return equity_series.pct_change().fillna(0.0)


def annualized_return(daily_returns: pd.Series, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR) -> float:
    if daily_returns.size == 0:
        return 0.0
    return float((1.0 + daily_returns).prod() ** (annualization_factor / max(1, daily_returns.size)) - 1.0)


def annualized_volatility(
    daily_returns: pd.Series, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
) -> float:
    if daily_returns.size < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(annualization_factor))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    if daily_returns.size < 2:
        return 0.0
    excess = daily_returns - risk_free_rate_annual / annualization_factor
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(annualization_factor))


def sortino_ratio(
    daily_returns: pd.Series, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
) -> float:
    if daily_returns.size < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    if downside.size == 0:
        return float("inf") if daily_returns.mean() > 0 else 0.0
    sd = downside.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(daily_returns.mean() / sd * np.sqrt(annualization_factor))


def maximum_drawdown(equity_series: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative number."""
    if equity_series.size == 0:
        return 0.0
    running_max = equity_series.cummax()
    drawdowns = equity_series / running_max - 1.0
    return float(drawdowns.min())


def calmar_ratio(
    daily_returns: pd.Series,
    equity_series: pd.Series,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    mdd = abs(maximum_drawdown(equity_series))
    if mdd == 0:
        return 0.0
    return annualized_return(daily_returns, annualization_factor) / mdd


def historical_value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Negative-tail quantile of historical daily returns."""
    if daily_returns.size == 0:
        return 0.0
    return float(np.quantile(daily_returns, 1.0 - confidence))


def expected_shortfall(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Mean of returns at or below the VaR threshold."""
    if daily_returns.size == 0:
        return 0.0
    var = historical_value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def calculate_trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Win rate and P&L from closed option exits; hedge P&L reported separately."""
    if trades.empty or "action" not in trades.columns:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "average_pnl": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "option_pnl_total": 0.0,
            "hedge_pnl_total": 0.0,
            "net_pnl_total": 0.0,
        }
    exits = trades[trades["action"] == "exit"].copy()
    if "realized_pnl" not in exits.columns:
        exits["realized_pnl"] = np.nan
    exits = exits.dropna(subset=["realized_pnl"])
    hedges = trades[trades["action"] == "hedge"]
    wins = exits[exits["realized_pnl"] > 0]["realized_pnl"]
    losses = exits[exits["realized_pnl"] < 0]["realized_pnl"]
    option_pnl = float(exits["realized_pnl"].sum()) if not exits.empty else 0.0
    hedge_pnl = float(hedges["gross_cash_flow"].sum()) if not hedges.empty else 0.0
    return {
        "trade_count": int(len(exits)),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": float(len(wins) / max(1, len(exits))),
        "average_pnl": float(exits["realized_pnl"].mean()) if not exits.empty else 0.0,
        "average_winner": float(wins.mean()) if len(wins) > 0 else 0.0,
        "average_loser": float(losses.mean()) if len(losses) > 0 else 0.0,
        "option_pnl_total": option_pnl,
        "hedge_pnl_total": hedge_pnl,
        "net_pnl_total": option_pnl + hedge_pnl,
    }


def build_performance_summary(
    daily_history: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, float]:
    """Compose the full summary dictionary written to ``results/``."""
    if daily_history.empty:
        return {}
    equity = daily_history["equity"].astype(float).reset_index(drop=True)
    returns = calculate_daily_returns(equity)
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])

    summary = {
        "initial_equity": initial,
        "final_equity": final,
        "total_return": float(final / initial - 1.0),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate_annual),
        "sortino_ratio": sortino_ratio(returns),
        "maximum_drawdown": maximum_drawdown(equity),
        "calmar_ratio": calmar_ratio(returns, equity),
        "value_at_risk_95": historical_value_at_risk(returns, 0.95),
        "expected_shortfall_95": expected_shortfall(returns, 0.95),
        "trading_days": int(len(daily_history)),
    }
    summary.update(calculate_trade_statistics(trades))
    return summary
