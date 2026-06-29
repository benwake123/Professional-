"""Statistical helpers: bootstrap, error metrics, and sign accuracy.

All routines are pure NumPy so the project does not depend on
``statsmodels`` or ``scikit-learn``.  Each function is small enough to read
in one screen.
"""

from __future__ import annotations

import numpy as np


def event_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap confidence interval for a mean.

    Parameters
    ----------
    values
        1-D array of event-level observations (for example, ``vrp20``).
    n_resamples
        Number of bootstrap samples.  The reference design uses 20,000.
    confidence
        Two-sided coverage probability, in ``(0, 1)``.
    seed
        Seed for ``numpy.random.default_rng``.  Fixed by default so the
        published intervals are reproducible.

    Returns
    -------
    (lower, upper, draws)
        Lower and upper percentile cutoffs and the full vector of bootstrap
        mean draws.  The latter is useful for diagnostic plots.
    """

    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    # ``rng.integers`` draws with replacement; this is the classical
    # non-parametric bootstrap for the mean.  We pre-allocate the draw
    # matrix so the call stays vectorized.
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    draws = values[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lower, upper, draws


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return ``mean(|actual - predicted|)``."""

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must share the same shape.")
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return ``sqrt(mean((actual - predicted) ** 2))``."""

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must share the same shape.")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def out_of_sample_r_squared(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray
) -> float:
    """``1 - sum((y - yhat)^2) / sum((y - ybar_benchmark)^2)``.

    The benchmark is the per-step expanding-mean prediction.  A positive
    value means the candidate forecast beats the historical-mean benchmark
    out of sample, exactly as in Goyal-Welch style regression evaluations.
    """

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    benchmark = np.asarray(benchmark, dtype=float).ravel()
    if not (actual.shape == predicted.shape == benchmark.shape):
        raise ValueError("All inputs must share the same shape.")
    ss_model = float(np.sum((actual - predicted) ** 2))
    ss_bench = float(np.sum((actual - benchmark) ** 2))
    if ss_bench <= 0:
        return float("nan")
    return 1.0 - ss_model / ss_bench


def forecast_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation between actual and predicted vectors."""

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must share the same shape.")
    if actual.size < 2:
        return float("nan")
    if np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def variance_spread_sign_accuracy(
    iv_var: np.ndarray,
    predicted_rv: np.ndarray,
    actual_rv: np.ndarray,
) -> float:
    """Share of events where the predicted and actual VRP signs agree.

    The variance-risk-premium proxy is ``IV - RV``.  We compare the sign of
    the *predicted* spread with the sign of the *realized* spread.  Both
    zeros count as agreement to keep the metric well defined at boundary
    cases.
    """

    iv_var = np.asarray(iv_var, dtype=float).ravel()
    predicted_rv = np.asarray(predicted_rv, dtype=float).ravel()
    actual_rv = np.asarray(actual_rv, dtype=float).ravel()
    if not (iv_var.shape == predicted_rv.shape == actual_rv.shape):
        raise ValueError("All inputs must share the same shape.")
    if iv_var.size == 0:
        return float("nan")
    predicted_spread = iv_var - predicted_rv
    actual_spread = iv_var - actual_rv
    return float(np.mean(np.sign(predicted_spread) == np.sign(actual_spread)))
