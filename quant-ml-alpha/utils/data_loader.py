import pandas as pd
import yfinance as yf
from pathlib import Path


def download_ohlcv(tickers, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    frames = []
    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue
        df = data[ticker].copy().reset_index()
        df["ticker"] = ticker
        df.columns = [c.lower() for c in df.columns]
        frames.append(df.rename(columns={"date": "date"}))

    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
