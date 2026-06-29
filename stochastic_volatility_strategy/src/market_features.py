"""
Market features for the stochastic-volatility options strategy.

Purpose
-------
Computes the time-series feature inputs the rest of the pipeline relies on:
log returns, rolling realized variance/volatility, EWMA variance, drawdown
from running peak, VIX term-structure slopes, short/long realized-vol
acceleration, plus a composed ``build_market_feature_table`` that merges
everything onto a single ``date`` calendar.

Every function is strictly point-in-time. ``pandas.rolling`` and ``ewm`` are
left-aligned by default, so a value at row ``t`` only ever uses data from
rows ``<= t``. As long as callers pass historical frames (or use
:func:`src.data_loader.slice_data_as_of` first), no future information can
leak in.

Module connections
------------------
Upstream (this module imports from):
    - ``pandas``, ``numpy``                 : math + DataFrame ops.

Downstream (this module is consumed by):
    - ``src.calibration``  : uses ``calculate_log_returns`` /
                             ``calculate_realized_variance`` for GBM and
                             Heston parameter estimation.
    - ``src.regime``       : uses realized vol + VIX term structure for
                             regime classification.
    - ``src.signals``      : pulls a point-in-time row via
                             :func:`get_feature_row_as_of`.
    - ``src.run_pipeline`` : builds the master feature table once at startup
                             via :func:`build_market_feature_table`.

Conventions
-----------
Realized variance is reported as an *annualized* number, computed as
``mean(r_t^2) * 252`` where ``r_t`` is the daily log return. Realized
volatility is the square root of that. This matches the convention used
when comparing against implied variance / VIX.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd


DEFAULT_ANNUALIZATION_FACTOR: int = 252
DateLike = Union[str, date, datetime, pd.Timestamp]


def calculate_log_returns(
    prices: Union[pd.DataFrame, pd.Series],
    price_column: str = "close",
) -> pd.Series:
    """
    Close-to-close log returns.

    Accepts either a :class:`pandas.DataFrame` (column selected by
    ``price_column``) or a :class:`pandas.Series`. The output Series keeps
    the input's index; the first row is ``NaN`` (no prior price).
    """
    if isinstance(prices, pd.DataFrame):
        if price_column not in prices.columns:
            raise ValueError(f"prices is missing column '{price_column}'.")
        series = prices[price_column].astype(float)
    elif isinstance(prices, pd.Series):
        series = prices.astype(float)
    else:
        raise TypeError(
            f"prices must be a DataFrame or Series, got {type(prices).__name__}."
        )

    if (series <= 0).any():
        raise ValueError("price series contains nonpositive values.")
    return np.log(series).diff().rename("log_return")


def calculate_realized_variance(
    log_returns: pd.Series,
    window: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Rolling annualized realized variance: ``mean(r^2) * annualization_factor``."""
    _require_positive_int(window, "window")
    squared = log_returns.astype(float) ** 2
    rv = squared.rolling(window=window, min_periods=window).mean() * annualization_factor
    return rv.rename(f"realized_variance_{window}")


def calculate_realized_volatility(
    log_returns: pd.Series,
    window: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Square root of :func:`calculate_realized_variance`."""
    rv = calculate_realized_variance(log_returns, window, annualization_factor)
    return np.sqrt(rv).rename(f"realized_volatility_{window}")


def calculate_ewma_variance(
    log_returns: pd.Series,
    half_life_days: float,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Annualized EWMA variance with the given half-life."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0.")
    squared = log_returns.astype(float) ** 2
    ewma_daily = squared.ewm(halflife=half_life_days, adjust=False).mean()
    return (ewma_daily * annualization_factor).rename(
        f"ewma_variance_hl{half_life_days}"
    )


def calculate_drawdown(levels: pd.Series) -> pd.Series:
    """
    Drawdown from the running peak: ``(level - cummax) / cummax``.

    Output is always ``<= 0``. Use ``-drawdown.min()`` to recover the
    largest peak-to-trough loss as a positive number.
    """
    series = levels.astype(float)
    peaks = series.cummax()
    return ((series - peaks) / peaks).rename("drawdown")


def calculate_vix_term_structure(
    vix_frame: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Return a DataFrame of VIX-family slope features.

    Columns: ``date_column``, ``vix9d_over_vix``, ``vix_over_vix3m``.

    - ``vix9d_over_vix > 1`` : short-term stress dominates the 30-day
      forward expectation (commonly observed near event days).
    - ``vix_over_vix3m > 1`` : backwardation across the 30d/3m curve
      (stress / regime change). ``< 1`` is contango (calm).

    If ``vix3m`` is missing from the input, ``vix_over_vix3m`` is filled
    with ``NaN`` so downstream code can keep a stable column layout.
    """
    required = {date_column, "vix", "vix9d"}
    missing = required - set(vix_frame.columns)
    if missing:
        raise ValueError(f"vix_frame is missing column(s): {sorted(missing)}")

    out = pd.DataFrame({date_column: vix_frame[date_column].values})

    vix = vix_frame["vix"].astype(float).replace(0.0, np.nan)
    vix9d = vix_frame["vix9d"].astype(float)
    out["vix9d_over_vix"] = (vix9d / vix).values

    if "vix3m" in vix_frame.columns:
        vix3m = vix_frame["vix3m"].astype(float).replace(0.0, np.nan)
        out["vix_over_vix3m"] = (vix / vix3m).values
    else:
        out["vix_over_vix3m"] = np.nan

    return out


def calculate_volatility_acceleration(
    realized_volatility: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Short-window mean realized vol minus long-window mean realized vol.

    Positive values mean recent realized vol exceeds the longer-run mean
    (acceleration); negative values mean deceleration.
    """
    _require_positive_int(short_window, "short_window")
    _require_positive_int(long_window, "long_window")
    if short_window >= long_window:
        raise ValueError("short_window must be < long_window.")

    series = realized_volatility.astype(float)
    short_mean = series.rolling(window=short_window, min_periods=short_window).mean()
    long_mean = series.rolling(window=long_window, min_periods=long_window).mean()
    return (short_mean - long_mean).rename("volatility_acceleration")


def build_market_feature_table(
    underlying: pd.DataFrame,
    vix_frame: pd.DataFrame,
    realized_window: int = 21,
    ewma_half_life: float = 10.0,
    short_window: int = 5,
    long_window: int = 21,
    date_column: str = "date",
    price_column: str = "close",
) -> pd.DataFrame:
    """
    Compose every feature function into one date-keyed DataFrame.

    Columns: ``date``, ``log_return``, ``realized_variance``,
    ``realized_volatility``, ``ewma_variance``,
    ``volatility_acceleration``, ``drawdown``,
    ``vix9d_over_vix``, ``vix_over_vix3m``.
    """
    if date_column not in underlying.columns:
        raise ValueError(f"underlying is missing '{date_column}'.")
    if price_column not in underlying.columns:
        raise ValueError(f"underlying is missing '{price_column}'.")

    sorted_under = (
        underlying.sort_values(date_column)
        .reset_index(drop=True)
        .copy()
    )
    log_returns = calculate_log_returns(sorted_under, price_column=price_column)
    rv = calculate_realized_variance(log_returns, realized_window)
    rvol = calculate_realized_volatility(log_returns, realized_window)
    ewma_var = calculate_ewma_variance(log_returns, ewma_half_life)
    accel = calculate_volatility_acceleration(rvol, short_window, long_window)
    drawdown = calculate_drawdown(sorted_under[price_column])

    table = pd.DataFrame(
        {
            date_column: sorted_under[date_column].values,
            "log_return": log_returns.values,
            "realized_variance": rv.values,
            "realized_volatility": rvol.values,
            "ewma_variance": ewma_var.values,
            "volatility_acceleration": accel.values,
            "drawdown": drawdown.values,
        }
    )

    term_structure = calculate_vix_term_structure(vix_frame, date_column=date_column)
    table = table.merge(term_structure, on=date_column, how="left")
    return table


def get_feature_row_as_of(
    feature_table: pd.DataFrame,
    as_of_date: DateLike,
    date_column: str = "date",
) -> pd.Series:
    """
    Return the latest row whose ``date_column`` is on or before ``as_of_date``.

    Raises :class:`ValueError` if no row qualifies (e.g. ``as_of_date`` is
    earlier than every row in ``feature_table``).
    """
    if date_column not in feature_table.columns:
        raise ValueError(f"feature_table is missing '{date_column}'.")
    cutoff = pd.Timestamp(as_of_date)
    if pd.isna(cutoff):
        raise ValueError(f"as_of_date is not a valid date: {as_of_date!r}")
    eligible = feature_table[feature_table[date_column] <= cutoff]
    if eligible.empty:
        raise ValueError(
            f"No feature row available on or before {cutoff.date()} "
            f"(earliest row is {feature_table[date_column].min()})."
        )
    return eligible.iloc[-1]


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
