"""
Data loading for the stochastic-volatility options strategy.

Purpose
-------
Each loader reads one CSV from disk, checks that the columns documented in
``data/README.md`` are present, normalizes types (dates ->
``datetime64[ns]``, numeric fields -> ``float``), sorts by the relevant date
column, and deduplicates obvious key collisions. The four single-dataset
loaders are composed by :func:`load_all_market_data` into a
:class:`~src.types.MarketDataBundle`. :func:`slice_data_as_of` returns a
point-in-time view of either a single dataframe or an entire bundle so the
backtest can guarantee no future information leaks into a decision.

The deeper "is this dataset internally consistent?" checks
(no crossed quotes, OHLC ordering, sane rate ranges, etc.) live in
:mod:`src.data_validation`, which is run immediately after these loaders.

Module connections
------------------
Upstream (this module imports from):
    - ``pandas``                  : core dataframe type and CSV reader.
    - ``src.types.MarketDataBundle`` : container returned by
                                       :func:`load_all_market_data` and
                                       accepted by :func:`slice_data_as_of`.

Downstream (this module is imported / called by):
    - ``src.run_pipeline.main``       : calls :func:`load_all_market_data`
                                        using paths produced by
                                        :func:`src.config.resolve_project_paths`.
    - ``src.data_validation`` (next)  : its ``validate_all_data`` consumes
                                        the bundle this module produces;
                                        its ``assert_no_future_information``
                                        is the read-side counterpart of
                                        :func:`slice_data_as_of`.
    - ``src.backtest``                : uses :func:`slice_data_as_of` on
                                        every trading day to build a
                                        point-in-time market snapshot.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Union

import pandas as pd

from src.types import MarketDataBundle


UNDERLYING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

OPTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "quote_date",
    "expiration",
    "option_type",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
)
OPTION_OPTIONAL_COLUMNS: tuple[str, ...] = ("delta", "gamma", "theta", "vega")

VOLATILITY_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "vix", "vix9d")
VOLATILITY_INDEX_OPTIONAL_COLUMNS: tuple[str, ...] = ("vix3m", "vvix")

RISK_FREE_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "annual_rate")


PathLike = Union[str, Path]
DateLike = Union[str, date, datetime, pd.Timestamp]


def load_underlying_prices(path: PathLike) -> pd.DataFrame:
    """Load and normalize daily SPY OHLCV data from ``path``."""
    frame = _read_csv(path, parse_date_columns=["date"])
    _require_columns(frame, UNDERLYING_REQUIRED_COLUMNS, "underlying prices")

    numeric_columns = ("open", "high", "low", "close", "adjusted_close", "volume")
    frame = _coerce_numeric(frame, numeric_columns)

    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return frame


def load_option_chain(path: PathLike) -> pd.DataFrame:
    """Load contract-level historical option quotes from ``path``."""
    frame = _read_csv(path, parse_date_columns=["quote_date", "expiration"])
    _require_columns(frame, OPTION_REQUIRED_COLUMNS, "option chain")

    numeric_columns = (
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "implied_volatility",
    )
    optional_numeric = tuple(c for c in OPTION_OPTIONAL_COLUMNS if c in frame.columns)
    frame = _coerce_numeric(frame, numeric_columns + optional_numeric)

    frame["option_type"] = (
        frame["option_type"].astype("string").str.strip().str.lower()
    )

    frame = (
        frame.dropna(subset=["quote_date", "expiration", "strike", "option_type"])
        .drop_duplicates(
            subset=["quote_date", "expiration", "option_type", "strike"], keep="last"
        )
        .sort_values(["quote_date", "expiration", "option_type", "strike"])
        .reset_index(drop=True)
    )
    return frame


def load_volatility_indices(path: PathLike) -> pd.DataFrame:
    """Load VIX-family time series from ``path``."""
    frame = _read_csv(path, parse_date_columns=["date"])
    _require_columns(frame, VOLATILITY_INDEX_REQUIRED_COLUMNS, "volatility indices")

    numeric_columns = ["vix", "vix9d"] + [
        c for c in VOLATILITY_INDEX_OPTIONAL_COLUMNS if c in frame.columns
    ]
    frame = _coerce_numeric(frame, numeric_columns)

    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return frame


def load_risk_free_rates(path: PathLike) -> pd.DataFrame:
    """Load annualized short-rate observations from ``path``."""
    frame = _read_csv(path, parse_date_columns=["date"])
    _require_columns(frame, RISK_FREE_REQUIRED_COLUMNS, "risk-free rates")

    frame = _coerce_numeric(frame, ["annual_rate"])

    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return frame


def load_all_market_data(paths: Mapping[str, PathLike]) -> MarketDataBundle:
    """
    Call all four single-dataset loaders and return a :class:`MarketDataBundle`.

    Parameters
    ----------
    paths:
        Mapping with the same keys produced by
        :func:`src.config.resolve_project_paths`:

        ``underlying_csv``, ``options_csv``, ``vix_csv``, ``risk_free_csv``.
    """
    required = ("underlying_csv", "options_csv", "vix_csv", "risk_free_csv")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Missing required path entries: {missing}")

    return MarketDataBundle(
        underlying=load_underlying_prices(paths["underlying_csv"]),
        options=load_option_chain(paths["options_csv"]),
        volatility_indices=load_volatility_indices(paths["vix_csv"]),
        risk_free_rates=load_risk_free_rates(paths["risk_free_csv"]),
    )


def slice_data_as_of(
    data: Union[pd.DataFrame, MarketDataBundle],
    as_of_date: DateLike,
) -> Union[pd.DataFrame, MarketDataBundle]:
    """
    Return only the information that was available on or before ``as_of_date``.

    For an individual dataframe, the slicing column is auto-detected:

    - ``"quote_date"`` when present (option chain),
    - otherwise ``"date"``.

    For a :class:`MarketDataBundle`, every contained dataframe is sliced and a
    new bundle is returned. The input objects are never mutated.
    """
    cutoff = _coerce_timestamp(as_of_date)

    if isinstance(data, MarketDataBundle):
        return MarketDataBundle(
            underlying=_slice_frame(data.underlying, cutoff, "date"),
            options=_slice_frame(data.options, cutoff, "quote_date"),
            volatility_indices=_slice_frame(data.volatility_indices, cutoff, "date"),
            risk_free_rates=_slice_frame(data.risk_free_rates, cutoff, "date"),
        )

    if isinstance(data, pd.DataFrame):
        date_column = "quote_date" if "quote_date" in data.columns else "date"
        return _slice_frame(data, cutoff, date_column)

    raise TypeError(
        "slice_data_as_of expects a pandas DataFrame or MarketDataBundle, "
        f"got {type(data).__name__}."
    )


def _read_csv(path: PathLike, parse_date_columns: list[str]) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path, parse_dates=parse_date_columns)


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], dataset_name: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} dataset is missing required column(s): {missing}"
        )


def _coerce_numeric(frame: pd.DataFrame, columns) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _slice_frame(
    frame: pd.DataFrame, cutoff: pd.Timestamp, date_column: str
) -> pd.DataFrame:
    if date_column not in frame.columns:
        raise ValueError(
            f"Cannot slice as-of: required column '{date_column}' is missing."
        )
    mask = frame[date_column] <= cutoff
    return frame.loc[mask].reset_index(drop=True)


def _coerce_timestamp(value: DateLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"as_of_date is not a valid date: {value!r}")
    return timestamp
