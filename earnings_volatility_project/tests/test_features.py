"""Unit tests for :mod:`src.features`.

These tests build a small synthetic daily panel where every feature can be
computed by hand.  Because the underlying formulas only depend on simple
windowed sums, the expected values are derived inline so a reader can audit
both the production code and the tests in the same file.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Make the package importable when running ``pytest tests`` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import add_log_returns, load_daily_panel  # noqa: E402
from src.features import (  # noqa: E402
    TRADING_DAYS_PER_YEAR,
    build_event_table,
    feature_complete_mask,
)
from src.forecasting import RIDGE_PREDICTORS  # noqa: E402


# ---------------------------------------------------------------------------
# A small handcrafted panel where every event-level number can be checked.
# ---------------------------------------------------------------------------


def _toy_panel() -> pd.DataFrame:
    """Build a 30-row daily panel with two earnings events.

    Prices follow a controlled geometric series so the squared log-return
    sums collapse to a known closed form.  The two earnings flags are placed
    far enough into the sample that pre/post windows are always defined.
    """

    # 60 daily rows give us plenty of room for both pre-event and post-event
    # 20-day windows around two synthetic earnings dates.
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    aapl_close = 100.0 * np.exp(0.01 * np.arange(n))  # log return = 0.01 each day
    aapl_open = aapl_close * 1.005
    vxapl = np.linspace(20.0, 30.0, n)
    vix = np.linspace(15.0, 18.0, n)
    vix9d = np.linspace(14.0, 17.0, n)

    earnings_flag = np.zeros(n, dtype=int)
    earnings_flag[25] = 1  # first event: 25 prior returns, >= 20 forward ones
    earnings_flag[35] = 1  # second event: still leaves 24 forward returns

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "aapl_open": aapl_open,
            "aapl_high": aapl_close * 1.01,
            "aapl_low": aapl_close * 0.99,
            "aapl_close": aapl_close,
            "aapl_volume": np.full(n, 1_000_000),
            "vxapl_open": vxapl,
            "vxapl_high": vxapl,
            "vxapl_low": vxapl,
            "vxapl_close": vxapl,
            "vix_open": vix,
            "vix_high": vix,
            "vix_low": vix,
            "vix_close": vix,
            "vix9d_open": vix9d,
            "vix9d_high": vix9d,
            "vix9d_low": vix9d,
            "vix9d_close": vix9d,
            "earnings_flag": earnings_flag,
            "fiscal_quarter_end": [None] * n,
            "eps_estimate": [None] * n,
            "eps_actual": [None] * n,
            "eps_surprise_pct": [None] * n,
            "source_ids": ["TEST"] * n,
        }
    )
    return df


@pytest.fixture
def toy_events(tmp_path: Path) -> pd.DataFrame:
    """Run the real loader against the toy CSV so the test reflects production."""

    csv_path = tmp_path / "toy.csv"
    _toy_panel().to_csv(csv_path, index=False)
    panel = add_log_returns(load_daily_panel(csv_path))
    return build_event_table(panel)


def test_iv_var_is_squared_vxapl_over_one_hundred(toy_events: pd.DataFrame) -> None:
    """``iv_var`` must equal ``(VXAPL / 100) ** 2`` row by row."""

    expected = (toy_events["vxapl_close"] / 100.0) ** 2
    np.testing.assert_allclose(toy_events["iv_var"], expected, rtol=1e-12)


def test_pre_rv20_uses_only_prior_returns(toy_events: pd.DataFrame) -> None:
    """The synthetic returns are constant ``0.01`` per day, so
    ``pre_rv20_var = (252/20) * 20 * 0.01 ** 2``."""

    expected = (TRADING_DAYS_PER_YEAR / 20.0) * 20.0 * (0.01 ** 2)
    np.testing.assert_allclose(toy_events["pre_rv20_var"], expected, rtol=1e-12)


def test_post_rv_windows_match_closed_form(toy_events: pd.DataFrame) -> None:
    """20-day and 5-day forward RVs collapse to closed forms."""

    expected_post20 = (TRADING_DAYS_PER_YEAR / 20.0) * 20.0 * (0.01 ** 2)
    expected_post5 = (TRADING_DAYS_PER_YEAR / 5.0) * 5.0 * (0.01 ** 2)
    np.testing.assert_allclose(toy_events["post_rv20_var"], expected_post20, rtol=1e-12)
    np.testing.assert_allclose(toy_events["post_rv5_var"], expected_post5, rtol=1e-12)


def test_iv_runup_5_matches_vxapl_ratio(toy_events: pd.DataFrame) -> None:
    """``iv_runup_5`` reproduces ``VXAPL_t / VXAPL_{t-5} - 1``."""

    # Reload the panel so we have access to the daily VXAPL series.
    panel = _toy_panel()
    event_positions = np.flatnonzero(panel["earnings_flag"].to_numpy() == 1)
    expected = []
    for t in event_positions:
        v_t = float(panel.loc[t, "vxapl_close"])
        v_lag = float(panel.loc[t - 5, "vxapl_close"])
        expected.append(v_t / v_lag - 1.0)
    np.testing.assert_allclose(toy_events["iv_runup_5"], expected, rtol=1e-12)


def test_event_gap_uses_next_day_open(toy_events: pd.DataFrame) -> None:
    """The event gap is ``log(aapl_open_{t+1} / aapl_close_t)``."""

    panel = _toy_panel()
    event_positions = np.flatnonzero(panel["earnings_flag"].to_numpy() == 1)
    expected = []
    for t in event_positions:
        next_open = float(panel.loc[t + 1, "aapl_open"])
        close_t = float(panel.loc[t, "aapl_close"])
        expected.append(math.log(next_open / close_t))
    np.testing.assert_allclose(toy_events["event_gap"], expected, rtol=1e-12)


def test_hist_abs_gap_4_excludes_current_event(toy_events: pd.DataFrame) -> None:
    """``hist_abs_gap_4`` of the first event is NaN; the second is the first's |gap|."""

    assert math.isnan(float(toy_events.loc[0, "hist_abs_gap_4"]))
    expected_second = float(np.abs(toy_events.loc[0, "event_gap"]))
    np.testing.assert_allclose(toy_events.loc[1, "hist_abs_gap_4"], expected_second,
                               rtol=1e-12)


def test_eps_surprise_is_not_in_predictors() -> None:
    """The forecasting model must not list current-event EPS columns."""

    forbidden = {"eps_actual", "eps_surprise_pct", "eps_estimate"}
    assert forbidden.isdisjoint(set(RIDGE_PREDICTORS))


def test_feature_complete_mask_matches_nonan(toy_events: pd.DataFrame) -> None:
    """``feature_complete_mask`` agrees with ``DataFrame.notna().all`` on the predictors."""

    mask = feature_complete_mask(toy_events, RIDGE_PREDICTORS)
    expected = toy_events[list(RIDGE_PREDICTORS)].notna().all(axis=1)
    pd.testing.assert_series_equal(mask, expected, check_names=False)


def test_real_panel_reproduces_validation_targets() -> None:
    """Cross-check the published reference numbers against the real CSV."""

    real_csv = REPO_ROOT / "data" / "aapl_earnings_volatility_data.csv"
    if not real_csv.exists():  # pragma: no cover - dataset only present in repo
        pytest.skip("Real dataset is not available in this environment.")
    panel = add_log_returns(load_daily_panel(real_csv))
    events = build_event_table(panel)
    assert int(events["iv_var"].notna().sum()) == 40
    assert events["iv_var"].notna().sum() == 40
    np.testing.assert_allclose(events["vxapl_implied_vol"].mean(), 33.8278, atol=1e-3)
    np.testing.assert_allclose(events["post_rv20_vol"].mean(), 27.8762, atol=1e-3)
    np.testing.assert_allclose(events["vrp20"].dropna().mean(), 0.03277, atol=1e-4)
    pct_positive = float((events["vrp20"].dropna() > 0).mean())
    np.testing.assert_allclose(pct_positive, 0.70, atol=1e-3)
