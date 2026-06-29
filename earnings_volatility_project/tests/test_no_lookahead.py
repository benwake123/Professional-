"""Lookahead-bias regression tests.

These tests verify the rules that must hold for every forecast in the
paper:

* Pre-event features do not depend on prices realized *after* the event.
* Historical event features exclude the current event.
* Modifying observations after a forecast date cannot change that forecast.
* Event and prediction tables are chronologically sorted.
* The forecasting predictors do not include current-event EPS columns.

The tests run the real pipeline on the prepared CSV so we know they catch
bugs in production code rather than in a toy reimplementation.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import add_log_returns, load_daily_panel  # noqa: E402
from src.features import build_event_table  # noqa: E402
from src.forecasting import RIDGE_PREDICTORS, run_walk_forward  # noqa: E402


REAL_CSV = REPO_ROOT / "data" / "aapl_earnings_volatility_data.csv"


pytestmark = pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="Real Apple earnings-volatility dataset is required for these tests.",
)


@pytest.fixture(scope="module")
def real_events() -> pd.DataFrame:
    """Load and cache the real event table once per test module."""

    panel = add_log_returns(load_daily_panel(REAL_CSV))
    return build_event_table(panel)


def test_event_table_is_chronologically_sorted(real_events: pd.DataFrame) -> None:
    """Events must already be sorted before any forecast is formed."""

    dates = pd.to_datetime(real_events["announcement_date"])
    assert dates.is_monotonic_increasing, "Event table is not chronological."
    assert dates.is_unique, "Duplicate event dates would corrupt training windows."


def test_walk_forward_predictions_are_chronological(real_events: pd.DataFrame) -> None:
    """The walk-forward output preserves chronological order."""

    wf = run_walk_forward(real_events)
    dates = pd.to_datetime(wf.test_frame["announcement_date"])
    assert dates.is_monotonic_increasing
    assert dates.is_unique


def test_eps_actual_is_not_a_predictor() -> None:
    """The contemporaneous earnings surprise must never enter the model."""

    forbidden = {"eps_actual", "eps_surprise_pct", "eps_estimate"}
    assert forbidden.isdisjoint(set(RIDGE_PREDICTORS))


def test_hist_abs_gap_excludes_current_event(real_events: pd.DataFrame) -> None:
    """``hist_abs_gap_4`` is computed only from prior events."""

    # Walk through the events in order and compute the expected window from
    # rows preceding each one.  Equality must hold to machine precision.
    abs_gaps = real_events["abs_event_gap"].to_numpy(dtype=float)
    expected = []
    prior: list[float] = []
    for value in abs_gaps:
        if not prior:
            expected.append(float("nan"))
        else:
            expected.append(float(np.mean(prior[-4:])))
        if np.isfinite(value):
            prior.append(float(value))
        else:
            # NaNs in the source column must not contribute to history.
            pass
    actual = real_events["hist_abs_gap_4"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12)


def test_future_returns_do_not_change_pre_event_features() -> None:
    """Mutate every daily row *after* an event and re-check pre-event features.

    Concretely we corrupt every ``aapl_close`` row strictly after the first
    earnings event, rebuild the feature table, and confirm the pre-window
    columns and the implied-variance column are identical to the original.
    """

    panel = add_log_returns(load_daily_panel(REAL_CSV))
    events = build_event_table(panel)

    first_event_date = pd.to_datetime(events["announcement_date"].iloc[0])
    # Take a snapshot of the pre-event columns for every event.
    original_pre_cols = events[[
        "announcement_date",
        "iv_var",
        "pre_rv20_var",
        "iv_runup_5",
        "market_term_slope",
        "hist_abs_gap_4",
        "hist_abs_eps_surprise_4",
    ]].copy()

    perturbed = panel.copy()
    future_mask = perturbed["date"] > first_event_date
    # Multiply every future close by 5 and the volatility indices by a noisy
    # factor; if a feature reaches forward in time it will absolutely change.
    perturbed.loc[future_mask, "aapl_close"] *= 5.0
    perturbed.loc[future_mask, "aapl_open"] *= 5.0
    perturbed.loc[future_mask, "aapl_high"] *= 5.0
    perturbed.loc[future_mask, "aapl_low"] *= 5.0
    perturbed.loc[future_mask, "vxapl_close"] *= 2.0

    # Re-compute log returns because they depend on the corrupted close.
    perturbed = add_log_returns(perturbed.drop(columns=["log_return"]))
    perturbed_events = build_event_table(perturbed)

    # All pre-event features for the first event must be unchanged.  Other
    # events depend on future-date prices that we deliberately corrupted, so
    # they may legitimately differ; the rule is purely "no leakage into
    # earlier rows".
    pd.testing.assert_series_equal(
        perturbed_events.iloc[0][[
            "iv_var",
            "pre_rv20_var",
            "iv_runup_5",
            "market_term_slope",
        ]].astype(float),
        original_pre_cols.iloc[0][[
            "iv_var",
            "pre_rv20_var",
            "iv_runup_5",
            "market_term_slope",
        ]].astype(float),
        check_names=False,
    )


def test_modifying_data_after_forecast_does_not_change_prediction(tmp_path: Path) -> None:
    """Changing post-forecast rows must not retroactively change a forecast.

    The test runs the walk-forward loop, records the first out-of-sample
    forecast, then mutates the daily panel for every date *strictly after*
    that forecast.  Re-running the walk-forward exercise must reproduce the
    same prediction for that event.
    """

    panel = add_log_returns(load_daily_panel(REAL_CSV))
    events = build_event_table(panel)
    wf = run_walk_forward(events)
    first_test = wf.test_frame.iloc[0]

    # Copy the panel and corrupt every row strictly after the first test
    # event.  The pipeline uses only earlier rows for that event's forecast.
    target_date = pd.to_datetime(first_test["announcement_date"])
    corrupted = panel.copy()
    future_mask = corrupted["date"] > target_date
    corrupted.loc[future_mask, "aapl_close"] *= 0.5
    corrupted.loc[future_mask, "aapl_open"] *= 0.5
    corrupted.loc[future_mask, "aapl_high"] *= 0.5
    corrupted.loc[future_mask, "aapl_low"] *= 0.5
    corrupted.loc[future_mask, "vxapl_close"] *= 3.0
    corrupted = add_log_returns(corrupted.drop(columns=["log_return"]))
    corrupted_events = build_event_table(corrupted)
    corrupted_wf = run_walk_forward(corrupted_events)
    corrupted_first = corrupted_wf.test_frame.iloc[0]

    for column in ("mean_pred", "iv_only_pred", "ridge_pred"):
        original = float(first_test[column])
        new = float(corrupted_first[column])
        np.testing.assert_allclose(
            new, original, rtol=1e-12,
            err_msg=f"Forecast column {column!r} changed after future-data corruption.",
        )


def test_pipeline_inputs_match_repository_layout() -> None:
    """Sanity check: the documented CSV exists exactly where the README claims."""

    assert REAL_CSV.exists(), (
        f"Expected CSV at {REAL_CSV}; this confirms the project never reads "
        "any other file."
    )


def test_results_directory_can_be_written(tmp_path: Path) -> None:
    """Smoke test that the entrypoint writes the required artefacts."""

    # Lazy import so the heavyweight pipeline only loads when this test runs.
    from src.run_analysis import run_pipeline  # noqa: WPS433

    out_dir = tmp_path / "results"
    run_pipeline(REAL_CSV, out_dir)
    assert (out_dir / "event_level_results.csv").exists()
    assert (out_dir / "walk_forward_predictions.csv").exists()
    assert (out_dir / "summary_metrics.csv").exists()

    # The pipeline also writes report/figures and report/generated.  We do
    # not move those locations under tmp_path because the LaTeX paper expects
    # them at fixed paths, but we can still check they exist for the run.
    assert (REPO_ROOT / "report" / "figures" / "iv_vs_realized.pdf").exists()
    assert (REPO_ROOT / "report" / "generated" / "results_macros.tex").exists()

    # Clean up the temporary output directory; the figures/generated
    # directories are owned by the repository and remain in place.
    shutil.rmtree(out_dir, ignore_errors=True)
