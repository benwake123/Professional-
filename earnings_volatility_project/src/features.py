"""Construct event-level features from the daily panel.

For every Apple earnings-announcement date ``t`` the pipeline reduces the
daily panel to a single row of features and outcomes.

Variable definitions (all variances are annualized with the trading-day
convention of 252 days per year):

* ``iv_var``           = ``(vxapl_close / 100) ** 2``
* ``pre_rv20_var``     = ``(252 / 20) * sum_{i=t-19..t} r_i ** 2``
* ``post_rv20_var``    = ``(252 / 20) * sum_{i=t+1..t+20} r_i ** 2``
* ``post_rv5_var``     = ``(252 / 5)  * sum_{i=t+1..t+5}  r_i ** 2``
* ``vrp20``            = ``iv_var - post_rv20_var``
* ``vrp5``             = ``iv_var - post_rv5_var``
* ``iv_runup_5``       = ``vxapl_close_t / vxapl_close_{t-5} - 1``
* ``market_term_slope``= ``(vix_close - vix9d_close) / 100``
* ``event_gap``        = ``log(aapl_open_{t+1} / aapl_close_t)``
* ``hist_abs_gap_4``   = rolling mean of ``|event_gap|`` over the previous
  four earnings events, **not** including the current one.
* ``hist_abs_eps_surprise_4`` = analogous rolling mean of
  ``|eps_surprise_pct|`` from previous events only.

The two ``hist_*_4`` features are the only ones that condition on earlier
*earnings events*: every other window operates on daily rows.  Each event
feature is documented with the timing convention used downstream by the
tests in ``tests/test_no_lookahead.py``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Trading-day annualization constant used in every realized-variance formula.
TRADING_DAYS_PER_YEAR: int = 252


def _annualized_realized_variance(squared_returns: np.ndarray, window: int) -> float:
    """Return ``(252 / window) * sum(squared_returns)``.

    The helper centralizes the annualization convention so all callers stay
    consistent.  ``window`` is the number of *returns* in the sum; a 20-day
    realized-variance estimate uses ``window == 20``.
    """

    if squared_returns.size != window:
        raise ValueError(
            f"Expected exactly {window} squared returns, received "
            f"{squared_returns.size}."
        )
    return float(TRADING_DAYS_PER_YEAR / window) * float(np.sum(squared_returns))


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    """Return ``log(numerator / denominator)`` with positivity checks."""

    if numerator <= 0 or denominator <= 0:
        raise ValueError(
            "Cannot take a log of a non-positive price ratio. "
            f"Received numerator={numerator}, denominator={denominator}."
        )
    return float(np.log(numerator / denominator))


def _ratio_change(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator - 1`` with positivity checks."""

    if denominator <= 0:
        raise ValueError(
            "Cannot compute a relative change with non-positive denominator. "
            f"Received denominator={denominator}."
        )
    return float(numerator / denominator - 1.0)


def _rolling_history_mean(values: Sequence[float], window: int) -> list[float]:
    """Mean of the previous ``window`` non-null entries.

    Mimics ``pd.Series.shift(1).rolling(window, min_periods=1).mean()`` but
    is written in pure Python for transparency.  The current observation is
    explicitly excluded so the resulting feature cannot leak information from
    the event being predicted.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    history: list[float] = []
    out: list[float] = []
    for value in values:
        if not history:
            out.append(float("nan"))
        else:
            tail = history[-window:]
            out.append(float(np.mean(tail)))
        # Append the current value *after* recording the output so the
        # current event never contributes to its own history.
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            history.append(float(value))
        else:
            # Skip NaNs so the window only averages valid prior events.
            pass
    return out


def build_event_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per earnings event with features and outcomes.

    Parameters
    ----------
    panel
        Daily panel produced by
        :func:`src.data_loader.load_daily_panel` and augmented with
        ``log_return`` by :func:`src.data_loader.add_log_returns`.

    Returns
    -------
    pandas.DataFrame
        Event-level table sorted chronologically by ``announcement_date``.
        Columns include ``iv_var``, ``pre_rv20_var``, ``post_rv20_var``,
        ``post_rv5_var``, ``vrp20``, ``vrp5``, ``iv_runup_5``,
        ``market_term_slope``, ``event_gap``, ``abs_event_gap``,
        ``hist_abs_gap_4``, ``hist_abs_eps_surprise_4``, the descriptive
        ``eps_actual``/``eps_surprise_pct`` columns, and feature-completeness
        flags ``has_pre_rv20``, ``has_iv_runup_5`` and ``has_hist_abs_gap_4``.

    Notes
    -----
    Events without 20 prior returns (typically the first event in the
    sample) still appear in the table, but their pre-window columns are
    ``NaN`` and they are flagged ``has_pre_rv20=False``.  Walk-forward
    training only consumes feature-complete rows.
    """

    required = {"date", "aapl_open", "aapl_close", "vxapl_close",
                "vix_close", "vix9d_close", "earnings_flag", "log_return",
                "eps_actual", "eps_estimate", "eps_surprise_pct"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"Daily panel is missing required columns: {sorted(missing)!r}."
        )
    if not panel["date"].is_monotonic_increasing:
        raise ValueError("Daily panel must be sorted by date before building events.")

    # Use numpy for fixed-size windowing because integer position math is
    # easier to audit than label-based pandas slicing.
    dates = panel["date"].to_numpy()
    aapl_close = panel["aapl_close"].to_numpy(dtype=float)
    aapl_open = panel["aapl_open"].to_numpy(dtype=float)
    vxapl = panel["vxapl_close"].to_numpy(dtype=float)
    vix = panel["vix_close"].to_numpy(dtype=float)
    vix9d = panel["vix9d_close"].to_numpy(dtype=float)
    log_returns = panel["log_return"].to_numpy(dtype=float)
    earn_flag = panel["earnings_flag"].to_numpy(dtype=int)

    eps_estimate = panel["eps_estimate"].to_numpy(dtype=float)
    eps_actual = panel["eps_actual"].to_numpy(dtype=float)
    eps_surprise_pct = panel["eps_surprise_pct"].to_numpy(dtype=float)
    fiscal_quarter_end = (
        panel["fiscal_quarter_end"].to_numpy()
        if "fiscal_quarter_end" in panel.columns
        else np.full(len(panel), None, dtype=object)
    )

    event_positions = np.flatnonzero(earn_flag == 1)

    rows: list[dict] = []
    for t in event_positions:
        # Implied variance from VXAPL.  VXAPL is quoted in percentage points
        # so we divide by 100 before squaring.
        iv_var = float((vxapl[t] / 100.0) ** 2)

        # Pre-event 20-day realized variance.  We need 20 returns ending at t,
        # which corresponds to positions [t-19, ..., t] (inclusive).
        pre_start = t - 19
        if pre_start >= 1 and not np.any(np.isnan(log_returns[pre_start:t + 1])):
            pre_rv20_var = _annualized_realized_variance(
                log_returns[pre_start:t + 1] ** 2, window=20
            )
            has_pre_rv20 = True
        else:
            pre_rv20_var = float("nan")
            has_pre_rv20 = False

        # Post-event realized variances.  These look strictly forward and are
        # therefore outcomes, never predictors.
        post20_end = t + 20  # inclusive index of the last forward return
        if post20_end < len(log_returns) and not np.any(np.isnan(log_returns[t + 1:post20_end + 1])):
            post_rv20_var = _annualized_realized_variance(
                log_returns[t + 1:post20_end + 1] ** 2, window=20
            )
        else:
            post_rv20_var = float("nan")

        post5_end = t + 5
        if post5_end < len(log_returns) and not np.any(np.isnan(log_returns[t + 1:post5_end + 1])):
            post_rv5_var = _annualized_realized_variance(
                log_returns[t + 1:post5_end + 1] ** 2, window=5
            )
        else:
            post_rv5_var = float("nan")

        # VXAPL run-up over the previous five trading days.  ``has_iv_runup_5``
        # is only ``True`` when both the lookback and the current VXAPL close
        # are populated; the prepared CSV has a few NaN VXAPL days on
        # non-event rows and we must propagate those gaps honestly.
        if t - 5 >= 0 and np.isfinite(vxapl[t]) and np.isfinite(vxapl[t - 5]):
            iv_runup_5 = _ratio_change(vxapl[t], vxapl[t - 5])
            has_iv_runup_5 = True
        else:
            iv_runup_5 = float("nan")
            has_iv_runup_5 = False

        market_term_slope = float((vix[t] - vix9d[t]) / 100.0)

        # Event gap requires the next trading day's open.  It is a descriptive
        # outcome (known only after the announcement is digested overnight).
        if t + 1 < len(panel):
            event_gap = _safe_log_ratio(aapl_open[t + 1], aapl_close[t])
        else:
            event_gap = float("nan")

        rows.append(
            {
                "announcement_date": pd.Timestamp(dates[t]),
                "fiscal_quarter_end": fiscal_quarter_end[t],
                "eps_estimate": eps_estimate[t],
                "eps_actual": eps_actual[t],
                "eps_surprise_pct": eps_surprise_pct[t],
                "vxapl_close": float(vxapl[t]),
                "aapl_close": float(aapl_close[t]),
                "iv_var": iv_var,
                "pre_rv20_var": pre_rv20_var,
                "post_rv20_var": post_rv20_var,
                "post_rv5_var": post_rv5_var,
                "vrp20": iv_var - post_rv20_var if not np.isnan(post_rv20_var) else float("nan"),
                "vrp5": iv_var - post_rv5_var if not np.isnan(post_rv5_var) else float("nan"),
                "iv_runup_5": iv_runup_5,
                "market_term_slope": market_term_slope,
                "event_gap": event_gap,
                "abs_event_gap": float(np.abs(event_gap)) if not np.isnan(event_gap) else float("nan"),
                "has_pre_rv20": has_pre_rv20,
                "has_iv_runup_5": has_iv_runup_5,
            }
        )

    events = pd.DataFrame(rows).sort_values("announcement_date").reset_index(drop=True)

    # Rolling historical earnings-event features.  These use only earlier
    # earnings events.  ``_rolling_history_mean`` excludes the current row by
    # construction.
    events["hist_abs_gap_4"] = _rolling_history_mean(
        events["abs_event_gap"].tolist(), window=4
    )
    abs_surprise = events["eps_surprise_pct"].abs().tolist()
    events["hist_abs_eps_surprise_4"] = _rolling_history_mean(abs_surprise, window=4)
    events["has_hist_abs_gap_4"] = events["hist_abs_gap_4"].notna()

    # Convenience volatility columns expressed in annualized percentage
    # points so summary tables print intuitive numbers.
    events["vxapl_implied_vol"] = np.sqrt(events["iv_var"]) * 100.0
    events["post_rv20_vol"] = np.sqrt(events["post_rv20_var"]) * 100.0
    events["pre_rv20_vol"] = np.sqrt(events["pre_rv20_var"]) * 100.0

    return events


def feature_complete_mask(events: pd.DataFrame, predictors: Iterable[str]) -> pd.Series:
    """Boolean mask of events whose forecasting predictors are all present.

    The walk-forward loop iterates over feature-complete events only so the
    multivariate model receives the same training rows at every step.
    """

    cols = list(predictors)
    missing = [c for c in cols if c not in events.columns]
    if missing:
        raise KeyError(f"Predictor columns missing from event table: {missing!r}")
    return events[cols].notna().all(axis=1)
