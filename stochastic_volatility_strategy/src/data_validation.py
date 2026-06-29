"""
Data validation for the stochastic-volatility options strategy.

Purpose
-------
Enforces dataset-level integrity invariants AFTER the raw CSVs have been
parsed by :mod:`src.data_loader` but BEFORE any feature, signal, calibration,
or backtest code consumes them. Two responsibilities:

1. Per-dataset structural checks
   - required columns present,
   - date columns sorted and unique,
   - underlying prices positive and OHLC-consistent,
   - option quotes uncrossed, strikes positive, expirations >= quote_date,
   - VIX-family values nonnegative,
   - risk-free rates expressed as decimals inside a sane range.

2. A point-in-time look-ahead guard, :func:`assert_no_future_information`,
   that makes sure no dataframe contains information dated after a given
   decision date.

Every validation failure raises :class:`ValueError` so callers can wrap calls
in a uniform ``try/except ValueError`` block. Type misuse raises
:class:`TypeError`.

Module connections
------------------
Upstream (this module imports from):
    - ``pandas`` : dataframe operations.
    - ``src.types.MarketDataBundle`` : container produced by
                                       :func:`src.data_loader.load_all_market_data`
                                       and accepted by
                                       :func:`validate_all_data`.

Downstream (this module is imported / called by):
    - ``src.run_pipeline.main`` : runs :func:`validate_all_data` immediately
                                  after :func:`src.data_loader.load_all_market_data`
                                  (see ``docs/CALL_GRAPH.md``).
    - ``src.backtest``          : calls :func:`assert_no_future_information`
                                  every time a market state is built for a
                                  specific decision date, paired with
                                  :func:`src.data_loader.slice_data_as_of`.

Column contracts here are kept aligned with the ones declared in
:mod:`src.data_loader` and ``data/README.md``; if those change, update both
sides together.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional, Union

import pandas as pd

from src.types import MarketDataBundle


DateLike = Union[str, date, datetime, pd.Timestamp]


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

VOLATILITY_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "vix", "vix9d")
VOLATILITY_INDEX_OPTIONAL_NONNEGATIVE: tuple[str, ...] = ("vix3m", "vvix")

RISK_FREE_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "annual_rate")

# Annualized short rates have historically lived between slightly negative
# (European policy rates) and ~20% (early-1980s US). This range guards
# against unit errors (basis points stored as integers vs. decimals).
RISK_FREE_MIN_RATE: float = -0.05
RISK_FREE_MAX_RATE: float = 0.50

VALID_OPTION_TYPES: frozenset[str] = frozenset({"call", "put"})


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """Raise ``ValueError`` if any required column is missing from ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{dataset_name} must be a pandas DataFrame.")
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required column(s): {missing}"
        )


def validate_sorted_unique_dates(
    frame: pd.DataFrame,
    date_column: str = "date",
    dataset_name: str = "dataset",
) -> None:
    """Raise ``ValueError`` if ``date_column`` is unsorted, NaN, or duplicated."""
    validate_required_columns(frame, [date_column], dataset_name)
    series = frame[date_column]

    if series.isna().any():
        raise ValueError(f"{dataset_name}.{date_column} contains NaN values.")
    if not series.is_monotonic_increasing:
        raise ValueError(
            f"{dataset_name}.{date_column} is not sorted in ascending order."
        )
    if not series.is_unique:
        duplicates = series[series.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"{dataset_name}.{date_column} contains duplicate values: {duplicates}"
        )


def validate_underlying_data(frame: pd.DataFrame) -> None:
    """Validate the SPY OHLCV dataframe."""
    validate_required_columns(frame, UNDERLYING_REQUIRED_COLUMNS, "underlying")
    validate_sorted_unique_dates(frame, "date", "underlying")

    price_columns = ("open", "high", "low", "close", "adjusted_close")
    for column in price_columns:
        if (frame[column] <= 0).any():
            raise ValueError(f"underlying.{column} must be strictly positive.")

    intraday_high_floor = frame[["open", "close", "low"]].max(axis=1)
    if (frame["high"] < intraday_high_floor).any():
        raise ValueError("underlying.high must be >= max(open, close, low).")
    intraday_low_ceiling = frame[["open", "close", "high"]].min(axis=1)
    if (frame["low"] > intraday_low_ceiling).any():
        raise ValueError("underlying.low must be <= min(open, close, high).")

    if (frame["volume"] < 0).any():
        raise ValueError("underlying.volume must be nonnegative.")


def validate_option_data(frame: pd.DataFrame) -> None:
    """Validate the historical option-chain dataframe."""
    validate_required_columns(frame, OPTION_REQUIRED_COLUMNS, "options")

    invalid_type_mask = ~frame["option_type"].isin(VALID_OPTION_TYPES)
    if invalid_type_mask.any():
        bad_types = (
            frame.loc[invalid_type_mask, "option_type"].unique().tolist()[:5]
        )
        raise ValueError(
            f"options.option_type must be one of {sorted(VALID_OPTION_TYPES)}; "
            f"got: {bad_types}"
        )

    if (frame["strike"] <= 0).any():
        raise ValueError("options.strike must be strictly positive.")

    if (frame["expiration"] < frame["quote_date"]).any():
        raise ValueError(
            "options.expiration must be on or after options.quote_date "
            "(expired contracts must be dropped at load time)."
        )

    for column in ("bid", "ask"):
        if (frame[column] < 0).any():
            raise ValueError(f"options.{column} must be nonnegative.")
    if (frame["ask"] < frame["bid"]).any():
        raise ValueError(
            "options.ask must be >= options.bid (crossed quotes are invalid)."
        )

    iv = frame["implied_volatility"]
    if (iv.notna() & (iv < 0)).any():
        raise ValueError(
            "options.implied_volatility must be nonnegative when present."
        )

    for column in ("volume", "open_interest"):
        if (frame[column] < 0).any():
            raise ValueError(f"options.{column} must be nonnegative.")


def validate_volatility_index_data(frame: pd.DataFrame) -> None:
    """Validate the VIX-family dataframe."""
    validate_required_columns(
        frame, VOLATILITY_INDEX_REQUIRED_COLUMNS, "volatility_indices"
    )
    validate_sorted_unique_dates(frame, "date", "volatility_indices")

    for column in ("vix", "vix9d"):
        if (frame[column] < 0).any():
            raise ValueError(f"volatility_indices.{column} must be nonnegative.")

    for column in VOLATILITY_INDEX_OPTIONAL_NONNEGATIVE:
        if column in frame.columns and (frame[column].dropna() < 0).any():
            raise ValueError(
                f"volatility_indices.{column} must be nonnegative when present."
            )


def validate_risk_free_data(frame: pd.DataFrame) -> None:
    """Validate the risk-free rate dataframe."""
    validate_required_columns(frame, RISK_FREE_REQUIRED_COLUMNS, "risk_free_rates")
    validate_sorted_unique_dates(frame, "date", "risk_free_rates")

    if frame["annual_rate"].isna().any():
        raise ValueError("risk_free_rates.annual_rate contains NaN values.")

    out_of_range = (
        (frame["annual_rate"] < RISK_FREE_MIN_RATE)
        | (frame["annual_rate"] > RISK_FREE_MAX_RATE)
    )
    if out_of_range.any():
        raise ValueError(
            "risk_free_rates.annual_rate must be expressed as a decimal in "
            f"[{RISK_FREE_MIN_RATE}, {RISK_FREE_MAX_RATE}] "
            "(possible basis-points vs decimal unit error)."
        )


def validate_all_data(bundle: MarketDataBundle) -> None:
    """Run every dataset-specific validator against a ``MarketDataBundle``."""
    if not isinstance(bundle, MarketDataBundle):
        raise TypeError(
            "validate_all_data expects a MarketDataBundle, got "
            f"{type(bundle).__name__}."
        )
    validate_underlying_data(bundle.underlying)
    validate_option_data(bundle.options)
    validate_volatility_index_data(bundle.volatility_indices)
    validate_risk_free_data(bundle.risk_free_rates)


def assert_no_future_information(
    data: Union[pd.DataFrame, MarketDataBundle],
    as_of_date: DateLike,
    date_column: Optional[str] = None,
) -> None:
    """
    Raise ``ValueError`` if any row's information date exceeds ``as_of_date``.

    For a single :class:`pandas.DataFrame` the column is either
    ``date_column`` (when supplied) or auto-detected: ``quote_date`` if
    present, else ``date``.

    For a :class:`MarketDataBundle` every contained dataframe is checked
    using the same convention as :func:`src.data_loader.slice_data_as_of`.
    """
    cutoff = pd.Timestamp(as_of_date)
    if pd.isna(cutoff):
        raise ValueError(f"as_of_date is not a valid date: {as_of_date!r}")

    if isinstance(data, MarketDataBundle):
        _check_max_date(data.underlying, "underlying.date", "date", cutoff)
        _check_max_date(data.options, "options.quote_date", "quote_date", cutoff)
        _check_max_date(
            data.volatility_indices, "volatility_indices.date", "date", cutoff
        )
        _check_max_date(
            data.risk_free_rates, "risk_free_rates.date", "date", cutoff
        )
        return

    if isinstance(data, pd.DataFrame):
        column = date_column or (
            "quote_date" if "quote_date" in data.columns else "date"
        )
        _check_max_date(data, column, column, cutoff)
        return

    raise TypeError(
        "assert_no_future_information expects a DataFrame or MarketDataBundle, "
        f"got {type(data).__name__}."
    )


def _check_max_date(
    frame: pd.DataFrame, label: str, date_column: str, cutoff: pd.Timestamp
) -> None:
    if date_column not in frame.columns:
        raise ValueError(f"{label}: required column '{date_column}' is missing.")
    if frame.empty:
        return
    max_observed = frame[date_column].max()
    if pd.notna(max_observed) and max_observed > cutoff:
        raise ValueError(
            f"Look-ahead detected: {label} max={max_observed!r} > "
            f"as_of_date={cutoff.date()!r}"
        )
