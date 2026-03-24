"""Data fetching with yfinance and TTL caching."""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS", "MS", "BAC",
    "XOM", "CVX", "JNJ", "UNH", "PFE", "WMT", "HD", "COST", "BA", "CAT",
    "SPY", "QQQ",
]

_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_CACHE_TTL_SEC = 60


def _cache_key(symbols: tuple[str, ...], start: str, end: str) -> str:
    return f"{','.join(sorted(symbols))}|{start}|{end}"


def fetch_ohlcv(
    symbols: Optional[list[str]] = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for symbols. Returns long-format DataFrame with columns: Date, ticker, Open, High, Low, Close, Volume."""
    syms = symbols or UNIVERSE
    syms = [s for s in syms if s in UNIVERSE or s in ["SPY", "QQQ"]]
    if not syms:
        syms = UNIVERSE
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    key = _cache_key(tuple(syms), start, end)
    if use_cache and key in _CACHE:
        df, ts = _CACHE[key]
        if time.time() - ts < _CACHE_TTL_SEC:
            return df.copy()

    data = yf.download(
        tickers=syms,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if len(syms) == 1:
        data = data.copy()
        data["ticker"] = syms[0]
        data = data.reset_index()
        data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
        if "date" not in data.columns:
            data = data.rename(columns={data.columns[0]: "date"})
    else:
        frames = []
        for s in syms:
            try:
                level = data.columns.names
                if s in data.columns.get_level_values(0):
                    df = data[s].copy().reset_index()
                    df["ticker"] = s
                    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
                    if "date" not in df.columns:
                        df = df.rename(columns={df.columns[0]: "date"})
                    frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])
        data = pd.concat(frames, ignore_index=True)

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    if use_cache:
        _CACHE[key] = (data.copy(), time.time())
    return data


def fetch_stock_analysis(ticker: str) -> dict:
    """Fetch chart + quote data for a single ticker. Used by frontend stock analysis."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    hist = t.history(period="1y", auto_adjust=True)
    if hist.empty or len(hist) < 2:
        raise ValueError(f"No price data for {ticker}")
    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else "Datetime" if "Datetime" in hist.columns else hist.columns[0]
    hist["date"] = pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d")
    chart_data = [
        {
            "date": row["date"],
            "timestamp": int(pd.Timestamp(row[date_col]).timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        }
        for _, row in hist.iterrows()
    ]
    info = t.info
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or chart_data[-1]["close"]
    current = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or chart_data[-1]["close"]
    def _float_or_none(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    quote = {
        "previousClose": float(prev_close),
        "marketCap": float(info.get("marketCap", 0) or 0),
        "fiftyTwoWeekHigh": float(info.get("fiftyTwoWeekHigh", 0) or 0),
        "fiftyTwoWeekLow": float(info.get("fiftyTwoWeekLow", 0) or 0),
        "volume": int(info.get("volume", 0) or 0),
        "averageVolume": int(info.get("averageVolume", 0) or 0),
        "beta": _float_or_none(info.get("beta")),
        "trailingPE": _float_or_none(info.get("trailingPE")),
    }
    return {
        "ticker": ticker.upper(),
        "chartData": chart_data,
        "quote": quote,
        "currentPrice": float(current),
    }
