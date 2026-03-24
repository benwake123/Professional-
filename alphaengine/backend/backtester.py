"""Vectorized backtester: long/short from signal ranks, transaction costs, metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    signal_type: str
    lookback_days: int = 21
    rebalance_freq: str = "daily"  # daily, weekly, monthly
    tc_bps: float = 5.0
    slippage_bps: float = 2.0
    top_quintile_long: bool = True
    bottom_quintile_short: bool = True


def _to_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = pd.Series(1, index=dates).sort_index()
    if freq == "daily":
        return s.index
    if freq == "weekly":
        return s.resample("W").last().dropna().index
    if freq == "monthly":
        return s.resample("ME").last().dropna().index
    return s.index


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
    benchmark_col: Optional[str] = "SPY",
) -> dict:
    """Run vectorized backtest. df: columns [date, ticker, close, signal, fwd_1d]. Returns metrics + series."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"])

    if "fwd_1d" not in df.columns:
        df["fwd_1d"] = df.groupby("ticker")["close"].pct_change().shift(-1)

    df = df.dropna(subset=["signal", "fwd_1d"])

    rb_dates = _to_rebalance_dates(df["date"].unique(), config.rebalance_freq)
    rb_set = set(rb_dates)

    def get_weights(g: pd.DataFrame) -> pd.Series:
        r = g["signal"].rank(pct=True)
        w = pd.Series(0.0, index=g.index)
        if config.top_quintile_long:
            w[r >= 0.8] = 1.0
        if config.bottom_quintile_short:
            w[r <= 0.2] = -1.0
        s = w.abs().sum()
        if s > 0:
            w = w / s
        return w

    weights_list = []
    prev_w = None
    for dt in df["date"].unique():
        g = df[df["date"] == dt]
        if dt in rb_set or prev_w is None:
            w = get_weights(g)
            prev_w = w
        else:
            w = prev_w.reindex(g.index).fillna(0)
        weights_list.append(w)

    df["w"] = pd.concat(weights_list)

    daily = df.groupby("date").apply(
        lambda x: (x["w"] * x["fwd_1d"]).sum()
    ).rename("gross_ret")

    w_pivot = df.pivot(index="date", columns="ticker", values="w").fillna(0)
    turnover = w_pivot.diff().abs().sum(axis=1).fillna(0)
    tc = turnover * (config.tc_bps + config.slippage_bps) / 10000.0
    daily = daily.to_frame()
    daily["turnover"] = turnover
    daily["tc"] = tc
    daily["net_ret"] = daily["gross_ret"] - daily["tc"]
    daily["log_ret"] = np.log1p(daily["net_ret"])
    daily["equity"] = (1 + daily["net_ret"]).cumprod()
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["peak"] - 1

    rf = 0.0
    ann_ret = (1 + daily["net_ret"]).prod() ** (252 / max(1, len(daily))) - 1
    ann_vol = daily["net_ret"].std() * np.sqrt(252) if len(daily) > 1 else 0.0
    sharpe = (ann_ret - rf) / (ann_vol + 1e-12) if ann_vol > 0 else 0.0
    down_ret = daily["net_ret"][daily["net_ret"] < 0]
    sortino = (ann_ret - rf) / (down_ret.std() * np.sqrt(252) + 1e-12) if len(down_ret) > 0 else 0.0
    mdd = daily["drawdown"].min()
    calmar = ann_ret / (abs(mdd) + 1e-12) if mdd != 0 else 0.0

    dd_series = daily["drawdown"]
    in_dd = dd_series < 0
    dd_duration = (in_dd.groupby((~in_dd).cumsum()).cumsum().abs()).max() if in_dd.any() else 0

    wins = (daily["net_ret"] > 0).sum()
    losses = (daily["net_ret"] < 0).sum()
    win_rate = wins / max(1, wins + losses)
    gross_profit = daily["net_ret"][daily["net_ret"] > 0].sum()
    gross_loss = abs(daily["net_ret"][daily["net_ret"] < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)

    bench_ret = None
    if benchmark_col and benchmark_col in df["ticker"].values:
        bench = df[df["ticker"] == benchmark_col].set_index("date")["fwd_1d"]
        daily = daily.join(bench.rename("bench_ret"), how="left")
        bench_ret = daily["bench_ret"].fillna(0)
        cov = daily["net_ret"].cov(bench_ret)
        bench_var = bench_ret.var()
        beta = cov / (bench_var + 1e-12) if bench_var > 0 else 0.0
        bench_ann = (1 + bench_ret).prod() ** (252 / max(1, len(daily))) - 1
        alpha = ann_ret - (rf + beta * (bench_ann - rf))
    else:
        beta = 0.0
        alpha = ann_ret

    return {
        "equity_curve": daily["equity"].tolist(),
        "dates": daily.index.astype(str).tolist(),
        "drawdown": daily["drawdown"].tolist(),
        "log_returns": daily["log_ret"].tolist(),
        "metrics": {
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(mdd),
            "max_drawdown_duration_days": int(dd_duration),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "beta": float(beta),
            "alpha": float(alpha),
            "avg_turnover": float(turnover.mean()),
            "total_return": float(daily["equity"].iloc[-1] - 1) if len(daily) > 0 else 0.0,
        },
    }
