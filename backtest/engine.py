import numpy as np
import pandas as pd


def signal_from_preds(df: pd.DataFrame, long_q: float = 0.9, short_q: float = 0.1) -> pd.DataFrame:
    out = df.copy()
    long_thr = out.groupby("date")["pred"].transform(lambda x: x.quantile(long_q))
    short_thr = out.groupby("date")["pred"].transform(lambda x: x.quantile(short_q))
    out["w"] = 0.0
    out.loc[out["pred"] >= long_thr, "w"] = 1.0
    out.loc[out["pred"] <= short_thr, "w"] = -1.0
    abs_sum = out.groupby("date")["w"].transform(lambda x: x.abs().sum())
    out["w"] = out["w"] / abs_sum.replace(0.0, np.nan)
    out["w"] = out["w"].fillna(0.0)
    return out


def apply_risk_controls(df: pd.DataFrame, max_weight: float = 0.02) -> pd.DataFrame:
    out = df.copy()
    out["w"] = out["w"].clip(-max_weight, max_weight)
    abs_sum = out.groupby("date")["w"].transform(lambda x: x.abs().sum())
    out["w"] = out["w"] / abs_sum.replace(0.0, np.nan)
    out["w"] = out["w"].fillna(0.0)
    return out


def run_backtest(df: pd.DataFrame, tc_bps: float = 5.0, target_annual_vol: float = 0.10) -> pd.DataFrame:
    out = df.copy().sort_values(["date", "ticker"])
    daily = out.groupby("date").apply(lambda g: (g["w"] * g["target"]).sum()).rename("gross_ret").to_frame()

    w = out.pivot(index="date", columns="ticker", values="w").fillna(0.0)
    daily["turnover"] = w.diff().abs().sum(axis=1).fillna(0.0)
    daily["tc"] = daily["turnover"] * (tc_bps / 10000.0)
    daily["net_ret"] = daily["gross_ret"] - daily["tc"]

    vol20 = daily["net_ret"].rolling(20).std() * np.sqrt(252)
    scale = (target_annual_vol / (vol20 + 1e-8)).clip(0, 3.0).fillna(1.0)
    daily["net_ret_vt"] = daily["net_ret"] * scale
    return daily.reset_index()
