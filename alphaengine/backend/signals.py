"""Alpha signal generation and evaluation (IC, ICIR, cross-sectional ranking)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

SignalType = Literal["momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m", "mean_reversion", "volume_trend", "vwap_deviation"]

MOMENTUM_DAYS = {"momentum_1m": 21, "momentum_3m": 63, "momentum_6m": 126, "momentum_12m": 252}
MEAN_REVERSION_WINDOW = 20
VOLUME_TREND_WINDOW = 20
VWAP_WINDOW = 20


def _forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fwd_{h}d"] = out.groupby("ticker")["close"].pct_change(h).shift(-h)
    return out


def _cross_sectional_rank(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.get_level_values(0) if isinstance(s.index, pd.MultiIndex) else s.index).rank(pct=True)


def compute_signal(df: pd.DataFrame, signal_type: SignalType) -> pd.Series:
    """Compute signal series aligned with df (date, ticker)."""
    df = df.set_index(["date", "ticker"]) if "date" in df.columns and "ticker" in df.columns else df
    ret = df.groupby(level="ticker")["close"].pct_change()

    if signal_type.startswith("momentum_"):
        w = MOMENTUM_DAYS.get(signal_type, 21)
        sig = df.groupby(level="ticker")["close"].pct_change(w)
    elif signal_type == "mean_reversion":
        roll_mean = df.groupby(level="ticker")["close"].transform(lambda x: x.rolling(MEAN_REVERSION_WINDOW).mean())
        roll_std = df.groupby(level="ticker")["close"].transform(lambda x: x.rolling(MEAN_REVERSION_WINDOW).std())
        sig = -(df["close"] - roll_mean) / (roll_std + 1e-12)
    elif signal_type == "volume_trend":
        obv = (df.groupby(level="ticker")["volume"] * np.sign(ret)).groupby(level="ticker").cumsum()
        sig = obv.groupby(level="ticker").pct_change(VOLUME_TREND_WINDOW)
    elif signal_type == "vwap_deviation":
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical * df["volume"]).groupby(level="ticker").transform(
            lambda x: x.rolling(VWAP_WINDOW).sum() / (df["volume"].groupby(level="ticker").transform(
                lambda v: v.rolling(VWAP_WINDOW).sum()) + 1e-12))
        sig = (df["close"] - vwap) / (vwap + 1e-12)
    else:
        w = 21
        sig = df.groupby(level="ticker")["close"].pct_change(w)

    return sig


def run_signal_analysis(
    df: pd.DataFrame,
    signal_type: SignalType,
    horizons: list[int] | None = None,
) -> dict:
    """Generate signal, rank cross-sectionally, compute IC and ICIR by horizon."""
    horizons = horizons or [1, 5, 10, 21]
    df = df.copy()
    df = _forward_returns(df, horizons)

    sig = compute_signal(df, signal_type)
    df["signal"] = sig.values if hasattr(sig, "values") else sig

    df = df.dropna(subset=["signal"])
    results: dict = {"ic_by_horizon": {}, "icir_by_horizon": {}, "rolling_ic": {}, "quintile_returns": {}}

    for h in horizons:
        col = f"fwd_{h}d"
        if col not in df.columns:
            continue
        sub = df[["signal", col]].dropna()
        if len(sub) < 10:
            continue
        ic, _ = stats.spearmanr(sub["signal"], sub[col])
        results["ic_by_horizon"][h] = float(ic) if not np.isnan(ic) else 0.0

        ics = []
        for dt, g in sub.groupby(level="date" if "date" in sub.index.names else sub.index):
            if len(g) >= 5:
                r, _ = stats.spearmanr(g["signal"], g[col])
                ics.append(r if not np.isnan(r) else 0.0)
        ics_arr = np.array(ics)
        results["icir_by_horizon"][h] = float(np.mean(ics_arr) / (np.std(ics_arr) + 1e-12)) if len(ics_arr) > 1 else 0.0

    df_wide = df.reset_index()
    if "date" in df_wide.columns:
        df_wide["signal_rank"] = df_wide.groupby("date")["signal"].rank(pct=True)
        for h in horizons:
            col = f"fwd_{h}d"
            if col in df_wide.columns:
                q = pd.qcut(df_wide["signal_rank"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
                qret = df_wide.groupby(q)[col].mean()
                results["quintile_returns"][h] = qret.to_dict()

    return results
