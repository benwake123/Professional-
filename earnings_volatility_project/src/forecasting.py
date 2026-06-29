"""Walk-forward forecasting of realized variance around earnings events.

Three forecasting models are compared with strictly expanding training
windows:

1. **Expanding historical mean** of ``post_rv20_var``.
2. **IV-only OLS** regressing ``post_rv20_var`` on ``iv_var``.
3. **Ridge regression** using the predictors
   ``["iv_var", "pre_rv20_var", "iv_runup_5", "hist_abs_gap_4"]``.

For every model the rule is identical: to forecast event ``i`` we may use
events ``0..i-1`` and nothing else.  The first 16 feature-complete events
form the initial training window; the rest are out-of-sample test rows.

A separate ``test_no_lookahead`` integration test verifies the strict
time-ordering at the file level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .regression import fit_ols, fit_ridge, predict_ridge


# Predictors used by the regularized model and the descriptive multivariate
# regression.  They are stored once here so other modules import the same
# list and stay in sync.
RIDGE_PREDICTORS: tuple[str, ...] = (
    "iv_var",
    "pre_rv20_var",
    "iv_runup_5",
    "hist_abs_gap_4",
)

RIDGE_PENALTY: float = 10.0
INITIAL_TRAIN_EVENTS: int = 16


@dataclass(frozen=True)
class WalkForwardResults:
    """Container for the three sequences of predictions and their dates.

    Attributes
    ----------
    test_frame
        Event-level dataframe restricted to the 23 (or however many)
        feature-complete test events, with predictions appended.
    n_train
        Size of the initial training window.
    """

    test_frame: pd.DataFrame
    n_train: int = INITIAL_TRAIN_EVENTS
    predictors: tuple[str, ...] = field(default=RIDGE_PREDICTORS)
    ridge_lambda: float = RIDGE_PENALTY


def _expanding_mean_forecast(train_y: np.ndarray) -> float:
    """Naive benchmark: predict the next event with the training mean."""

    return float(np.mean(train_y))


def _iv_only_forecast(train_y: np.ndarray, train_iv: np.ndarray, test_iv: float) -> float:
    """Expanding-window OLS using only ``iv_var``."""

    result = fit_ols(train_y, train_iv)
    intercept = result.coefficients[0]
    slope = result.coefficients[1]
    return float(intercept + slope * test_iv)


def _ridge_forecast(
    train_y: np.ndarray,
    train_X: np.ndarray,
    test_X: np.ndarray,
    *,
    lam: float,
) -> float:
    """Closed-form ridge with standardization on training moments only."""

    model = fit_ridge(train_y, train_X, lam=lam)
    prediction = predict_ridge(model, test_X.reshape(1, -1))
    return float(prediction[0])


def run_walk_forward(
    events: pd.DataFrame,
    *,
    predictors: Sequence[str] = RIDGE_PREDICTORS,
    initial_train: int = INITIAL_TRAIN_EVENTS,
    ridge_lambda: float = RIDGE_PENALTY,
) -> WalkForwardResults:
    """Generate the three forecast sequences in chronological order.

    Parameters
    ----------
    events
        Output of :func:`src.features.build_event_table`.  Must already be
        sorted by ``announcement_date``.
    predictors
        Ordered tuple of feature columns used by the ridge model.
    initial_train
        Number of feature-complete events held back as the training window
        before any forecast is produced.
    ridge_lambda
        Penalty applied to the standardized ridge coefficients.

    Returns
    -------
    WalkForwardResults
        Predictions, actuals and contextual columns for the test sample.

    Raises
    ------
    ValueError
        If the events are not chronological, the predictor list is empty,
        or the sample is too small to leave anything to forecast.
    """

    if not events["announcement_date"].is_monotonic_increasing:
        raise ValueError("Event table must be sorted chronologically before forecasting.")
    if not predictors:
        raise ValueError("At least one predictor is required for the ridge model.")
    if initial_train <= 0:
        raise ValueError("initial_train must be positive.")

    needed_columns = {"announcement_date", "iv_var", "post_rv20_var", *predictors}
    missing = needed_columns - set(events.columns)
    if missing:
        raise KeyError(f"Event table is missing required columns: {sorted(missing)!r}.")

    # Feature-complete events are the only rows used by the ridge regression.
    # Using the same set across all three models keeps the comparison fair.
    feature_cols = list(predictors)
    complete_mask = (
        events[feature_cols].notna().all(axis=1) & events["post_rv20_var"].notna()
    )
    eligible = events.loc[complete_mask].reset_index(drop=True)
    if len(eligible) <= initial_train:
        raise ValueError(
            f"Need more than {initial_train} feature-complete events to run a "
            f"walk-forward forecast; received {len(eligible)}."
        )

    y_all = eligible["post_rv20_var"].to_numpy(dtype=float)
    iv_all = eligible["iv_var"].to_numpy(dtype=float)
    X_all = eligible[feature_cols].to_numpy(dtype=float)

    records: list[dict] = []
    for i in range(initial_train, len(eligible)):
        train_y = y_all[:i]
        train_iv = iv_all[:i]
        train_X = X_all[:i]
        test_iv = float(iv_all[i])
        test_X = X_all[i]
        actual_rv = float(y_all[i])

        mean_pred = _expanding_mean_forecast(train_y)
        iv_pred = _iv_only_forecast(train_y, train_iv, test_iv)
        ridge_pred = _ridge_forecast(train_y, train_X, test_X, lam=ridge_lambda)

        row = {
            "announcement_date": eligible["announcement_date"].iloc[i],
            "iv_var": test_iv,
            "actual_rv": actual_rv,
            "train_size": int(i),
            "mean_pred": mean_pred,
            "iv_only_pred": iv_pred,
            "ridge_pred": ridge_pred,
            "mean_spread_pred": test_iv - mean_pred,
            "iv_only_spread_pred": test_iv - iv_pred,
            "ridge_spread_pred": test_iv - ridge_pred,
            "actual_spread": test_iv - actual_rv,
        }
        # Persist the predictor values so the CSV is fully audit-friendly.
        for col, val in zip(feature_cols, test_X):
            row[f"feature_{col}"] = float(val)
        records.append(row)

    test_frame = pd.DataFrame(records)
    if not test_frame["announcement_date"].is_monotonic_increasing:
        # Defensive: this can only fail if eligible was somehow unsorted.
        raise RuntimeError("Walk-forward output is not chronological.")
    return WalkForwardResults(
        test_frame=test_frame,
        n_train=initial_train,
        predictors=tuple(predictors),
        ridge_lambda=ridge_lambda,
    )
