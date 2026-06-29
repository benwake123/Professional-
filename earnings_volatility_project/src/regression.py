"""Linear-regression utilities written from scratch in NumPy.

The project deliberately avoids ``statsmodels`` and ``scikit-learn``.  Every
estimator is implemented here so a beginner can follow the math line by
line.  The two estimators we need are:

* Ordinary least squares with an intercept solved via ``numpy.linalg.lstsq``.
* Closed-form ridge regression with an *unpenalized* intercept used for
  walk-forward forecasts.

We also implement HC3 heteroskedasticity-robust standard errors because the
sample is small and residuals can scale with implied variance.  HC3 inflates
high-leverage residuals more than HC0/HC1, which is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSResult:
    """Bundle of statistics returned by :func:`fit_ols`.

    Attributes
    ----------
    coefficients
        Intercept first, then one entry per column of ``X``.
    fitted_values
        ``X @ coefficients`` for the training sample.
    residuals
        ``y - fitted_values``.
    r_squared
        Coefficient of determination computed against the training mean.
    hc3_std_errors
        Heteroskedasticity-consistent standard errors, same order as
        ``coefficients``.  ``np.nan`` if the sample is too small.
    t_statistics
        ``coefficients / hc3_std_errors``.
    n_obs
        Number of training observations.
    k_features
        Number of right-hand-side predictors (excluding the intercept).
    """

    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    hc3_std_errors: np.ndarray
    t_statistics: np.ndarray
    n_obs: int
    k_features: int


def _design_matrix(features: np.ndarray) -> np.ndarray:
    """Return the ``[1, x_1, ..., x_k]`` design matrix.

    Accepts both ``(n,)`` and ``(n, k)`` shapes for convenience.
    """

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            f"features must be 1-D or 2-D, got shape {features.shape!r}."
        )
    n = features.shape[0]
    intercept = np.ones((n, 1))
    return np.concatenate([intercept, features], axis=1)


def fit_ols(y: np.ndarray, features: np.ndarray) -> OLSResult:
    """Solve ``y = alpha + features @ beta + error`` by least squares.

    Parameters
    ----------
    y
        ``(n,)`` array of targets.
    features
        ``(n,)`` or ``(n, k)`` array of predictors.  The function adds the
        intercept column itself.

    Returns
    -------
    OLSResult
        Coefficients, residuals, R-squared, and HC3 standard errors.

    Notes
    -----
    * Uses ``numpy.linalg.lstsq`` so the function is numerically stable even
      when the predictor matrix is mildly ill-conditioned.
    * HC3 standard errors follow MacKinnon and White (1985):
      :math:`\\widehat{V} = (X^\\top X)^{-1}
      \\left[\\sum_i x_i x_i^\\top \\left(e_i / (1 - h_i)\\right)^2\\right]
      (X^\\top X)^{-1}`,
      where :math:`h_i` are the diagonal entries of the hat matrix.
    """

    y = np.asarray(y, dtype=float).ravel()
    X = _design_matrix(np.asarray(features, dtype=float))
    n, p = X.shape  # p = k + 1 including the intercept

    # rcond=None tells numpy to use the machine-precision default cutoff for
    # small singular values, which is the recommended modern behaviour.
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    residuals = y - fitted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    hc3_se = _hc3_standard_errors(X, residuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(hc3_se > 0, coef / hc3_se, np.nan)

    return OLSResult(
        coefficients=coef,
        fitted_values=fitted,
        residuals=residuals,
        r_squared=r_squared,
        hc3_std_errors=hc3_se,
        t_statistics=t_stats,
        n_obs=n,
        k_features=p - 1,
    )


def _hc3_standard_errors(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Return HC3 (MacKinnon-White 1985) standard errors.

    Parameters
    ----------
    X
        Design matrix already including the intercept column.
    residuals
        OLS residuals from ``y - X @ coef``.

    Returns
    -------
    numpy.ndarray
        Standard errors in the same order as the columns of ``X``.  Returns
        a NaN-filled vector when the design matrix is singular or when a row
        has leverage one.
    """

    n, p = X.shape
    if n <= p:
        return np.full(p, np.nan)

    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)

    # Hat-matrix leverages h_i = x_i' (X'X)^{-1} x_i computed in a
    # vectorized form: row-wise dot product of X with X @ (X'X)^{-1}.
    leverage = np.sum(X * (X @ xtx_inv), axis=1)
    if np.any(leverage >= 1.0 - 1e-12):
        return np.full(p, np.nan)

    scaled = residuals / (1.0 - leverage)
    # HC3 "meat" matrix:  sum_i x_i x_i' (e_i / (1 - h_i))^2.
    meat = (X * (scaled ** 2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv
    variances = np.diag(cov)
    variances = np.where(variances > 0, variances, np.nan)
    return np.sqrt(variances)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeModel:
    """Parameters and fitting-time statistics for a ridge model.

    Attributes
    ----------
    intercept
        Unpenalized intercept on the original scale of the target.
    coefficients
        Slope estimates on the original scale of the predictors.
    feature_means, feature_stds
        Training-window standardization moments.  Stored so the same
        transform can be applied to the test point.
    safe_columns
        Boolean mask of columns whose training standard deviation was
        strictly positive.  Zero-variance columns are coerced to a zero
        slope, never divided by zero.
    """

    intercept: float
    coefficients: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    safe_columns: np.ndarray


def fit_ridge(
    y: np.ndarray, features: np.ndarray, *, lam: float
) -> RidgeModel:
    """Closed-form ridge regression with an unpenalized intercept.

    Parameters
    ----------
    y
        Training-sample targets, shape ``(n,)``.
    features
        Training-sample predictors, shape ``(n, k)``.
    lam
        Ridge penalty applied to the standardized coefficients.  Must be
        non-negative.

    Returns
    -------
    RidgeModel
        Fitted parameters expressed on the original feature scale so the
        caller can multiply by a new row directly.

    Notes
    -----
    Standardization uses *training* moments only.  Any column whose training
    standard deviation is zero is given a zero coefficient: the feature is
    constant and therefore cannot help the model.
    """

    if lam < 0:
        raise ValueError("Ridge penalty lam must be non-negative.")
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D for fit_ridge, got {X.shape!r}.")
    n, k = X.shape
    if y.shape[0] != n:
        raise ValueError("y and features have inconsistent first-dimension length.")

    feature_means = X.mean(axis=0)
    # ``ddof=0`` matches the convention used by the typical machine-learning
    # standardizer.  Either convention works as long as it is applied
    # consistently to train and test rows.
    feature_stds = X.std(axis=0, ddof=0)
    safe = feature_stds > 0

    Z = np.zeros_like(X)
    Z[:, safe] = (X[:, safe] - feature_means[safe]) / feature_stds[safe]

    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    # Closed-form ridge on the centered, standardized design.  The intercept
    # is fitted unpenalized and equals the training mean of ``y`` because the
    # standardized predictors are mean-zero by construction.
    gram = Z.T @ Z + lam * np.eye(k)
    # ``np.linalg.solve`` is numerically more stable than ``inv`` + multiply.
    beta_standardized = np.linalg.solve(gram, Z.T @ y_centered)

    # Convert back to original-scale coefficients so the caller can use the
    # familiar ``intercept + features @ coefficients`` form.
    coef_original = np.zeros(k)
    coef_original[safe] = beta_standardized[safe] / feature_stds[safe]
    intercept = y_mean - float(np.dot(coef_original[safe], feature_means[safe]))

    return RidgeModel(
        intercept=intercept,
        coefficients=coef_original,
        feature_means=feature_means,
        feature_stds=feature_stds,
        safe_columns=safe,
    )


def predict_ridge(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    """Apply a fitted :class:`RidgeModel` to one or more new rows."""

    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != model.coefficients.shape[0]:
        raise ValueError(
            "Number of features in input does not match fitted model: "
            f"got {X.shape[1]}, expected {model.coefficients.shape[0]}."
        )
    return model.intercept + X @ model.coefficients
