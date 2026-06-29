"""
Regression tests for ``src/data_validation.py``.

Purpose
-------
Pins three of the most important integrity invariants the project depends
on. If any of these break, downstream calibration, signal, and backtest
code can silently consume corrupt data, so these tests act as
guard-rails:

    1. ``test_rejects_duplicate_underlying_dates``
       The underlying-price calendar must be unique - duplicate dates
       would corrupt every rolling/realized statistic.

    2. ``test_rejects_crossed_option_quotes``
       Option ``ask < bid`` is non-physical and would invert every signed
       cost calculation downstream.

    3. ``test_rejects_expired_contracts``
       ``expiration < quote_date`` would let the optimizer "trade"
       contracts that no longer exist.

Module connections
------------------
Upstream (this file imports from):
    - ``pytest``                              : runner.
    - ``pandas``                              : building bad-data frames.
    - ``src.data_validation``                 : the unit under test
                                                (functions whose contracts
                                                we're pinning).
    - ``tests.conftest`` (auto-loaded)        : ``underlying_frame``,
                                                ``option_frame`` fixtures.

Downstream (this file is run by):
    - ``pytest tests/test_data_validation.py``
    - Anything that wraps pytest (CI, ``pytest`` from the repo root, etc.).

These tests are intentionally narrow: ``implementation_tests/test_data_validation.py``
covers the wider per-validator surface and the loader-validator handshake.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python3 tests/test_data_validation.py`` (no pytest wrapper) by
# putting the project root on sys.path before importing from ``src``.
# pytest already does this via conftest.py; this shim is for direct runs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data_validation import (
    validate_option_data,
    validate_sorted_unique_dates,
    validate_underlying_data,
)


def test_rejects_duplicate_underlying_dates(underlying_frame: pd.DataFrame) -> None:
    """Two rows with the same ``date`` must be rejected as a key collision."""
    duplicated = pd.concat(
        [underlying_frame, underlying_frame.iloc[[1]]], ignore_index=True
    ).sort_values("date", kind="mergesort").reset_index(drop=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_underlying_data(duplicated)

    with pytest.raises(ValueError, match="duplicate"):
        validate_sorted_unique_dates(duplicated, "date", "underlying")


def test_rejects_crossed_option_quotes(option_frame: pd.DataFrame) -> None:
    """An option row where ``ask < bid`` must be rejected (crossed quote)."""
    crossed = option_frame.copy()
    crossed.loc[0, "ask"] = crossed.loc[0, "bid"] - 0.10

    with pytest.raises(ValueError, match=r"ask must be >= .*\bbid\b"):
        validate_option_data(crossed)


def test_rejects_expired_contracts(option_frame: pd.DataFrame) -> None:
    """Contracts whose ``expiration < quote_date`` must be rejected as expired."""
    expired = option_frame.copy()
    expired.loc[0, "expiration"] = expired.loc[0, "quote_date"] - pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="expiration must be on or after"):
        validate_option_data(expired)


if __name__ == "__main__":
    # Allows ``python3 tests/test_data_validation.py`` to run the file under
    # pytest (which provides the fixtures from tests/conftest.py).
    raise SystemExit(pytest.main([__file__, "-v"]))
