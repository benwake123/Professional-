"""Load and validate the daily Apple earnings-volatility CSV.

The full pipeline is fed by a single file: ``aapl_earnings_volatility_data.csv``.
This module is deliberately strict.  It parses dates, sorts rows in
chronological order, rejects duplicate trading days, and verifies that every
price column is positive.  Loud errors are preferable to silent corruption,
especially because every downstream feature is computed from rolling windows
on these prices.

Beginner orientation
--------------------
* ``date``                   - trading day (YYYY-MM-DD).
* ``aapl_*``                 - Apple split-adjusted daily OHLC and volume.
* ``vxapl_close``            - Cboe Apple VIX Index close (annualized %).
* ``vix_close``, ``vix9d_*`` - reference S&P 500 vol indices.
* ``earnings_flag``          - 1 on an Apple earnings-announcement day else 0.
* ``eps_*``                  - reported / estimated EPS (descriptive only).

The file is the only sanctioned input: ``run_analysis`` never reaches the
internet, never scrapes a website, and never calls a vendor API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "aapl_open",
    "aapl_high",
    "aapl_low",
    "aapl_close",
    "aapl_volume",
    "vxapl_open",
    "vxapl_high",
    "vxapl_low",
    "vxapl_close",
    "vix_open",
    "vix_high",
    "vix_low",
    "vix_close",
    "vix9d_open",
    "vix9d_high",
    "vix9d_low",
    "vix9d_close",
    "earnings_flag",
    "eps_estimate",
    "eps_actual",
    "eps_surprise_pct",
)

# Columns that must be strictly positive on every row.  AAPL OHLC values feed
# the daily log-return calculation; the VIX/VIX9D closes are needed for the
# market-term-slope feature on every event date.
STRICT_POSITIVE_COLUMNS: tuple[str, ...] = (
    "aapl_open",
    "aapl_high",
    "aapl_low",
    "aapl_close",
    "vix_close",
    "vix9d_close",
)

# Columns where occasional missing values exist in the prepared CSV.  Any
# present value must still be positive; ``NaN`` is allowed because the
# underlying historical VXAPL data has a handful of gaps that never overlap
# an actual earnings-event window.
ALLOW_NAN_POSITIVE_COLUMNS: tuple[str, ...] = (
    "vxapl_open",
    "vxapl_high",
    "vxapl_low",
    "vxapl_close",
)


class DataValidationError(ValueError):
    """Raised when the daily file cannot be safely consumed downstream."""


def _check_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Raise ``DataValidationError`` if any required column is missing."""

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Daily file is missing required columns: {missing!r}. "
            "The schema is defined in data/DATA_DICTIONARY.md."
        )


def load_daily_panel(csv_path: str | Path) -> pd.DataFrame:
    """Read the daily Apple / VXAPL / VIX panel and validate it.

    Parameters
    ----------
    csv_path
        Path to the prepared CSV file.

    Returns
    -------
    pandas.DataFrame
        Strict, chronologically sorted daily panel with ``date`` parsed as
        ``pandas.Timestamp`` and ``earnings_flag`` cast to ``int``.

    Notes
    -----
    The function rejects duplicate ``date`` values and any non-positive price
    in :data:`POSITIVE_PRICE_COLUMNS`.  Both conditions would silently bias
    rolling realized-variance calculations and must be repaired upstream.
    """

    path = Path(csv_path)
    if not path.exists():
        raise DataValidationError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    _check_required_columns(df, REQUIRED_COLUMNS)

    # Parse the trading day as a true Timestamp and reject anything that does
    # not parse.  A bad date here would propagate through every rolling window.
    parsed = pd.to_datetime(df["date"], errors="coerce")
    bad_mask = parsed.isna()
    if bad_mask.any():
        bad_rows = df.loc[bad_mask, "date"].tolist()
        raise DataValidationError(
            f"Could not parse the following date values: {bad_rows!r}"
        )
    df = df.assign(date=parsed)

    # Chronological order is a precondition for every feature: pre-event
    # windows look backwards, post-event windows look forwards.
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Duplicate trading days would inflate variance estimates and break the
    # uniqueness assumption used by the event walk-forward loop.
    duplicates = df["date"].duplicated()
    if duplicates.any():
        dupes = df.loc[duplicates, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise DataValidationError(
            f"Duplicate trading dates detected: {dupes!r}"
        )

    # Strict positivity: NaN and non-positive values are both fatal because
    # any one of these columns is required by every downstream computation.
    for col in STRICT_POSITIVE_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            raise DataValidationError(
                f"Column {col!r} contains non-finite values; all rows must be "
                "populated for this column."
            )
        if np.any(values <= 0):
            raise DataValidationError(
                f"Column {col!r} contains non-positive values. "
                "All prices must be strictly positive."
            )

    # Soft positivity: ``NaN`` values are tolerated (these specific gaps do
    # not overlap with the event windows we evaluate) but any populated value
    # must still be positive.  Sanity-check rows still tied to event dates.
    earnings_mask = df["earnings_flag"].astype(int) == 1
    for col in ALLOW_NAN_POSITIVE_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if np.any(values[finite] <= 0):
            raise DataValidationError(
                f"Column {col!r} contains a non-positive value. "
                "All present prices must be strictly positive."
            )
        # Earnings rows must always have a populated VXAPL close, otherwise
        # the implied-variance feature cannot be formed for that event.
        if col == "vxapl_close":
            event_values = values[earnings_mask.to_numpy()]
            if np.any(~np.isfinite(event_values)):
                raise DataValidationError(
                    "vxapl_close is missing on at least one earnings event; "
                    "the implied-variance feature cannot be constructed."
                )

    df["earnings_flag"] = df["earnings_flag"].astype(int)
    if not set(df["earnings_flag"].unique()).issubset({0, 1}):
        raise DataValidationError("earnings_flag must take only the values 0 or 1.")

    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append the daily close-to-close log return ``log_return``.

    The log return is the canonical input for realized-variance estimators.
    The first row receives ``NaN`` because no prior close exists.

    Parameters
    ----------
    df
        Daily panel produced by :func:`load_daily_panel`.

    Returns
    -------
    pandas.DataFrame
        Same panel with an extra ``log_return`` column.
    """

    if "aapl_close" not in df.columns:
        raise DataValidationError("Cannot compute log returns without aapl_close.")

    closes = df["aapl_close"].to_numpy(dtype=float)
    log_return = np.empty_like(closes)
    log_return[0] = np.nan
    # log(P_t / P_{t-1}); numpy operates element-wise on the slices.
    log_return[1:] = np.log(closes[1:] / closes[:-1])
    return df.assign(log_return=log_return)
