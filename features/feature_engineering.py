import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    out["ret_1d"] = out.groupby("ticker")["close"].pct_change()
    out["target"] = out.groupby("ticker")["close"].pct_change().shift(-1)

    for w in [3, 5, 10, 20]:
        out[f"mom_{w}"] = out.groupby("ticker")["close"].pct_change(w)
        out[f"vol_{w}"] = (
            out.groupby("ticker")["ret_1d"].rolling(w).std().reset_index(level=0, drop=True)
        )
        out[f"ma_ratio_{w}"] = (
            out["close"]
            / out.groupby("ticker")["close"].rolling(w).mean().reset_index(level=0, drop=True)
            - 1
        )

    out["dollar_vol"] = out["close"] * out["volume"]
    out["range"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)
    out["turnover_proxy"] = (
        out["volume"]
        / (out.groupby("ticker")["volume"].rolling(20).mean().reset_index(level=0, drop=True) + 1e-12)
    )

    factor_cols = [
        c
        for c in out.columns
        if c
        not in ["date", "ticker", "open", "high", "low", "close", "volume", "target"]
    ]

    for c in factor_cols:
        out[c] = out.groupby("date")[c].rank(pct=True) - 0.5

    return out
