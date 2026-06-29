"""
Regression tests that exercise the data layer against the real CSVs
shipped under ``data/raw/``.

Purpose
-------
The synthetic fixtures in ``tests/conftest.py`` keep the unit tests fast
and deterministic, but they don't prove anything about the actual data the
research uses. This file does. It loads the four CSVs declared in
``config.json`` through the real ``src.data_loader`` functions, checks the
loader output against the ground-truth row counts and date ranges in
``data/raw/verification.json``, then runs ``src.data_validation`` end to
end on the resulting bundle (including the no-lookahead guard).

Why this matters: if any of these tests fail, either (a) the real data
violates the contract documented in ``data/README.md`` / ``data/raw/
DATA_README.md``, or (b) one of our modules has drifted from that
contract. Either case must be fixed before signal/backtest code can rely
on the data.

Module connections
------------------
Upstream (this file imports from / depends on):
    - ``pytest``                                  : runner.
    - ``pandas``                                  : Timestamp arithmetic.
    - ``src.data_loader.{load_*, load_all_market_data, slice_data_as_of}``
                                                  : functions under test.
    - ``src.data_validation.{validate_all_data, assert_no_future_information}``
                                                  : functions under test.
    - ``tests.conftest`` (auto-loaded fixtures):
        - ``real_paths``                          : absolute paths from
                                                    ``config.json``.
        - ``real_market_bundle``                  : output of
                                                    ``load_all_market_data``.
        - ``real_verification``                   : parsed
                                                    ``data/raw/verification.json``.

Downstream (this file is run by):
    - ``pytest tests/test_real_data.py``
    - ``pytest`` from the repo root (collected automatically).

Special handling for ``spy_options.csv``: the public data pack ships only
the header (DATA_README explicitly says options data must be sourced
separately because fabricating it would invalidate the strategy). Tests
treat an empty options frame as expected and skip option-specific
assertions when the file has zero rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python3 tests/test_real_data.py`` (no pytest wrapper) by putting
# the project root on sys.path before importing from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data_loader import (
    load_option_chain,
    load_risk_free_rates,
    load_underlying_prices,
    load_volatility_indices,
    slice_data_as_of,
)
from src.data_validation import assert_no_future_information, validate_all_data
from src.types import MarketDataBundle


def test_real_underlying_matches_verification(
    real_paths: dict[str, Path], real_verification: dict[str, object]
) -> None:
    """SPY prices: row count and first/last date must match verification.json."""
    frame = load_underlying_prices(real_paths["underlying_csv"])

    assert len(frame) == real_verification["spy_prices_rows"]
    assert frame["date"].min() == pd.Timestamp(real_verification["spy_first_date"])
    assert frame["date"].max() == pd.Timestamp(real_verification["spy_last_date"])
    assert (frame["close"] > 0).all()


def test_real_vix_matches_verification(
    real_paths: dict[str, Path], real_verification: dict[str, object]
) -> None:
    """VIX: row count and first/last date must match verification.json."""
    frame = load_volatility_indices(real_paths["vix_csv"])

    assert len(frame) == real_verification["vix_rows"]
    assert frame["date"].min() == pd.Timestamp(real_verification["vix_first_date"])
    assert frame["date"].max() == pd.Timestamp(real_verification["vix_last_date"])
    assert (frame["vix"] >= 0).all() and (frame["vix9d"] >= 0).all()


def test_real_rates_matches_verification(
    real_paths: dict[str, Path], real_verification: dict[str, object]
) -> None:
    """Risk-free rates: row count, dates, and decimal-unit invariant."""
    frame = load_risk_free_rates(real_paths["risk_free_csv"])

    assert len(frame) == real_verification["risk_free_rows"]
    assert frame["date"].min() == pd.Timestamp(real_verification["risk_free_first_date"])
    assert frame["date"].max() == pd.Timestamp(real_verification["risk_free_last_date"])
    assert (frame["annual_rate"].abs() <= 0.50).all(), (
        "annual_rate looks like basis points; should be decimal in [-0.05, 0.50]."
    )


def test_real_options_loads_with_expected_schema(real_paths: dict[str, Path]) -> None:
    """Options CSV must load cleanly; when rows exist, basic sanity checks apply."""
    frame = load_option_chain(real_paths["options_csv"])
    assert isinstance(frame, pd.DataFrame)
    expected = {
        "quote_date",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "implied_volatility",
    }
    assert expected.issubset(frame.columns)

    if frame.empty:
        pytest.skip(
            "spy_options.csv is header-only; run data/synthesize_options.py "
            "to generate a backtestable chain."
        )

    assert (frame["ask"] > frame["bid"]).all()
    assert (frame["implied_volatility"] > 0).all()
    assert frame["option_type"].str.lower().isin({"call", "put"}).all()


def test_real_bundle_passes_validation(real_market_bundle: MarketDataBundle) -> None:
    """``validate_all_data`` must accept the bundle assembled from real CSVs."""
    validate_all_data(real_market_bundle)


@pytest.mark.parametrize(
    "as_of_date",
    ["2015-06-30", "2018-12-31", "2020-03-31", "2022-01-04"],
)
def test_real_bundle_slice_then_no_lookahead(
    real_market_bundle: MarketDataBundle, as_of_date: str
) -> None:
    """``slice_data_as_of`` output must satisfy the no-lookahead guard."""
    sliced = slice_data_as_of(real_market_bundle, as_of_date)
    assert_no_future_information(sliced, as_of_date)

    cutoff = pd.Timestamp(as_of_date)
    assert (sliced.underlying["date"] <= cutoff).all()
    assert (sliced.volatility_indices["date"] <= cutoff).all()
    assert (sliced.risk_free_rates["date"] <= cutoff).all()


def test_real_unsliced_bundle_fails_no_lookahead_guard(
    real_market_bundle: MarketDataBundle,
) -> None:
    """Picking a cutoff strictly before the dataset's max date must trip the guard."""
    early_cutoff = "2014-01-02"
    with pytest.raises(ValueError, match="Look-ahead detected"):
        assert_no_future_information(real_market_bundle, early_cutoff)


def test_real_calendar_consistency(real_market_bundle: MarketDataBundle) -> None:
    """SPY and VIX should share the same trading calendar (both use NYSE).

    Rates use a different convention (Fed business days), so we only sanity
    check that the rate date range sits inside the price date range and that
    most rate dates land on NYSE trading days.
    """
    underlying_dates = set(real_market_bundle.underlying["date"])
    vix_dates = set(real_market_bundle.volatility_indices["date"])
    assert underlying_dates == vix_dates, (
        "SPY price and VIX calendars diverged; "
        f"prices_only={len(underlying_dates - vix_dates)} extra rows, "
        f"vix_only={len(vix_dates - underlying_dates)} extra rows."
    )

    rate_dates = set(real_market_bundle.risk_free_rates["date"])
    rate_min = min(rate_dates)
    rate_max = max(rate_dates)
    price_min = min(underlying_dates)
    price_max = max(underlying_dates)
    assert price_min <= rate_min and rate_max <= price_max, (
        f"Rate date range [{rate_min.date()}, {rate_max.date()}] should sit "
        f"inside SPY date range [{price_min.date()}, {price_max.date()}]."
    )

    overlap_count = len(rate_dates & underlying_dates)
    assert overlap_count >= int(0.95 * len(rate_dates)), (
        f"Only {overlap_count}/{len(rate_dates)} rate dates align with "
        "NYSE trading days; calendars look badly mismatched."
    )


if __name__ == "__main__":
    # Allows ``python3 tests/test_real_data.py`` to run the file under
    # pytest (which provides the fixtures from tests/conftest.py).
    raise SystemExit(pytest.main([__file__, "-v"]))
