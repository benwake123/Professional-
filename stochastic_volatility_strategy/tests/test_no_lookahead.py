"""
Regression tests that pin the project's "no future information" invariant.

Purpose
-------
This file enforces the most important rule in the codebase: every decision
the strategy makes on date ``D`` must be based on information dated ``<= D``.
Look-ahead leakage is the single easiest way to fabricate a "great" backtest,
so we test the guard at every layer as those layers come online.

The three original skeleton tests (calibration window, regime thresholds,
forecast invariance) require modules that are not yet implemented; they are
preserved here as named tests and SKIPPED with module-specific reasons so
each will become active automatically as soon as its target module ships.

The remaining tests are runnable today against the data layer:

    - ``test_slice_then_validate_round_trip`` proves the loader and validator
      agree on the cutoff convention for every dataset in a bundle.
    - ``test_unsliced_bundle_fails_no_lookahead_guard`` proves the guard
      actually catches a bundle that contains future rows.
    - ``test_option_chain_slices_on_quote_date`` proves the option-chain
      uses ``quote_date`` (not ``expiration``) for slicing, which is the
      single most common look-ahead bug to introduce later.

Module connections
------------------
Upstream (this file imports from):
    - ``pytest``                              : runner / parametrization.
    - ``pandas``                              : timestamp arithmetic.
    - ``src.data_loader.slice_data_as_of``    : produces the "as-of"
                                                view of a bundle.
    - ``src.data_validation.assert_no_future_information`` : the look-ahead
                                                guard being pinned.
    - ``tests.conftest`` (auto-loaded)        : ``market_bundle``,
                                                ``option_frame`` fixtures.

Downstream (this file is run by):
    - ``pytest tests/test_no_lookahead.py``
    - ``pytest`` from the repo root (collected automatically).

Pending tests (will be activated as modules ship):
    - ``test_calibration_window_ends_at_decision_date``
        depends on ``src.calibration``.
    - ``test_regime_thresholds_ignore_future_rows``
        depends on ``src.regime``.
    - ``test_forecast_unchanged_when_future_data_changes``
        depends on ``src.monte_carlo`` and ``src.calibration``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Allow ``python3 tests/test_no_lookahead.py`` (no pytest wrapper) by
# putting the project root on sys.path before importing from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data_loader import slice_data_as_of
from src.data_validation import assert_no_future_information
from src.types import MarketDataBundle


def _module_is_implemented(module_name: str) -> bool:
    """
    Return ``True`` only if no function defined in ``module_name`` still
    raises :class:`NotImplementedError` when called.

    The stub convention used in ``src/`` is ``def foo(*args, **kwargs):
    raise NotImplementedError(...)``. Calling a stub with no arguments
    surfaces that error; any other behavior counts as "the function has
    a real body". Attributes that aren't functions defined locally in
    ``module_name`` (re-imports such as ``typing.Any``) are ignored.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    for attribute_name in dir(module):
        if attribute_name.startswith("_"):
            continue
        attribute = getattr(module, attribute_name)
        if not callable(attribute):
            continue
        if getattr(attribute, "__module__", None) != module_name:
            continue
        try:
            attribute()
        except NotImplementedError:
            return False
        except Exception:
            continue
    return True


# ---------------------------------------------------------------------------
# Active no-lookahead tests (work against the data layer today)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", ["2024-01-02", "2024-01-03", "2024-01-04"])
def test_slice_then_validate_round_trip(
    market_bundle: MarketDataBundle, cutoff: str
) -> None:
    """``slice_data_as_of`` output must satisfy ``assert_no_future_information``."""
    sliced = slice_data_as_of(market_bundle, cutoff)
    assert_no_future_information(sliced, cutoff)


def test_unsliced_bundle_fails_no_lookahead_guard(
    market_bundle: MarketDataBundle,
) -> None:
    """A bundle with rows after ``as_of_date`` must be rejected by the guard."""
    with pytest.raises(ValueError, match="Look-ahead detected"):
        assert_no_future_information(market_bundle, "2024-01-03")


def test_option_chain_slices_on_quote_date(option_frame: pd.DataFrame) -> None:
    """
    The option-chain must slice on ``quote_date``, not ``expiration``.

    All synthetic rows have ``expiration = 2024-02-16``; the only correct
    behavior at ``cutoff = 2024-01-03`` is to keep rows whose
    ``quote_date <= 2024-01-03``. If the loader ever regresses to slicing
    on ``expiration`` instead, every row would survive and this assertion
    would fail.
    """
    cutoff = pd.Timestamp("2024-01-03")
    sliced = slice_data_as_of(option_frame, cutoff)

    assert isinstance(sliced, pd.DataFrame)
    assert (sliced["quote_date"] <= cutoff).all()
    expected_row_count = int((option_frame["quote_date"] <= cutoff).sum())
    assert len(sliced) == expected_row_count, (
        f"slice_data_as_of returned {len(sliced)} rows; expected "
        f"{expected_row_count}. Did the loader regress to slicing on expiration?"
    )
    assert_no_future_information(sliced, cutoff)


# ---------------------------------------------------------------------------
# Pending no-lookahead tests for modules that are not yet implemented.
# Each will become active the moment its target module ships - just delete
# the `pytest.skip(...)` call and write the body.
# ---------------------------------------------------------------------------


def test_calibration_window_ends_at_decision_date(real_market_bundle) -> None:
    """The calibration window must end on, never past, the decision date."""
    if not _module_is_implemented("src.calibration"):
        pytest.skip("src.calibration is not yet implemented.")
    from src.calibration import calibrate_gbm

    decision_date = pd.Timestamp("2018-06-01")
    result = calibrate_gbm(
        real_market_bundle.underlying,
        as_of_date=decision_date,
        lookback_days=252,
    )
    assert pd.Timestamp(result["window_end"]) <= decision_date


def test_regime_thresholds_ignore_future_rows() -> None:
    """Regime thresholds at date ``D`` must use only data with date <= ``D``."""
    if not _module_is_implemented("src.regime"):
        pytest.skip("src.regime is not yet implemented.")
    from src.regime import calculate_expanding_regime_thresholds

    realized_vol = pd.Series(
        [0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.40, 0.55, 0.80]
    )
    full = calculate_expanding_regime_thresholds(realized_vol)
    truncated = calculate_expanding_regime_thresholds(realized_vol.iloc[:5])
    pd.testing.assert_frame_equal(
        full.iloc[:5].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_exact=False,
    )


def test_forecast_unchanged_when_future_data_changes(real_market_bundle) -> None:
    """Mutating data dated > D must not change a forecast produced on D."""
    if not _module_is_implemented(
        "src.monte_carlo"
    ) or not _module_is_implemented("src.calibration"):
        pytest.skip("src.monte_carlo and/or src.calibration are not yet implemented.")

    from src.calibration import calibrate_gbm
    from src.monte_carlo import forecast_realized_variance

    underlying = real_market_bundle.underlying.copy()
    decision_date = pd.Timestamp("2018-06-01")
    spot = float(underlying.loc[underlying["date"] <= decision_date, "close"].iloc[-1])
    params = calibrate_gbm(underlying, decision_date, lookback_days=252)
    forecast_before = forecast_realized_variance(
        initial_price=spot,
        parameters=params,
        horizon_days=10,
        n_paths=500,
        as_of_date=decision_date,
        random_seed=42,
    )

    # Corrupt rows strictly AFTER the decision date with absurd values.
    corrupted = underlying.copy()
    mask = corrupted["date"] > decision_date
    corrupted.loc[mask, "close"] = 1.0  # nonsensical price

    params_after = calibrate_gbm(corrupted, decision_date, lookback_days=252)
    forecast_after = forecast_realized_variance(
        initial_price=spot,
        parameters=params_after,
        horizon_days=10,
        n_paths=500,
        as_of_date=decision_date,
        random_seed=42,
    )
    assert forecast_before.expected_variance == forecast_after.expected_variance
    assert forecast_before.median_variance == forecast_after.median_variance


if __name__ == "__main__":
    # Allows ``python3 tests/test_no_lookahead.py`` to run the file under
    # pytest (which provides the fixtures from tests/conftest.py).
    raise SystemExit(pytest.main([__file__, "-v"]))
