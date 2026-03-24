"""Risk module: VaR, CVaR, stress tests, factor exposure, Kelly."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

STRESS_PERIODS = {
    "covid": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-01", "2022-10-12"),
    "gfc": ("2008-09-01", "2009-03-09"),
}


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (percentile). Negative = loss."""
    return float(returns.quantile(1 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) VaR."""
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_paths: int = 10000,
    seed: Optional[int] = 42,
) -> float:
    """Monte Carlo VaR (bootstrap returns, take percentile)."""
    rng = np.random.default_rng(seed)
    sim = rng.choice(returns.dropna().values, size=n_paths)
    return float(np.percentile(sim, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall."""
    var = returns.quantile(1 - confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def monte_carlo_fan(
    returns: pd.Series,
    n_paths: int = 10000,
    horizon: int = 21,
    percentiles: list[int] = [5, 25, 50, 75, 95],
    seed: Optional[int] = 42,
) -> dict:
    """Monte Carlo fan: simulate horizon-day paths, return percentile bands."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = 1.0
    for t in range(1, horizon + 1):
        draws = rng.choice(r, size=n_paths)
        paths[:, t] = paths[:, t - 1] * (1 + draws)
    bands = {p: np.percentile(paths, p, axis=0).tolist() for p in percentiles}
    return {"bands": bands, "horizon": horizon}


def stress_pnl(
    portfolio_returns: pd.Series,
    period_name: str,
) -> float:
    """Cumulative return during stress period."""
    start, end = STRESS_PERIODS.get(period_name, (None, None))
    if not start or not end:
        return 0.0
    mask = (portfolio_returns.index >= pd.Timestamp(start)) & (portfolio_returns.index <= pd.Timestamp(end))
    subset = portfolio_returns[mask]
    return float((1 + subset).prod() - 1) if len(subset) > 0 else 0.0


def factor_exposure(
    returns: pd.Series,
    market: pd.Series,
    size_proxy: Optional[pd.Series] = None,
    momentum_proxy: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Regress returns on market, size, momentum proxies. Return factor loadings."""
    common = returns.align(market, join="inner")[0].dropna()
    mkt = market.reindex(common.index).ffill().bfill()
    X = pd.DataFrame({"MKT": mkt})
    if size_proxy is not None:
        sz = size_proxy.reindex(common.index).ffill().bfill()
        X["SMB"] = sz
    if momentum_proxy is not None:
        mom = momentum_proxy.reindex(common.index).ffill().bfill()
        X["HML"] = mom
    X = X.dropna(how="any")
    y = common.reindex(X.index).dropna()
    idx = X.index.intersection(y.index)
    if len(idx) < 10:
        return {"market_beta": 0.0, "size": 0.0, "momentum": 0.0}
    from numpy.linalg import lstsq
    Xm = np.column_stack([np.ones(len(idx)), X.loc[idx].values])
    coef, _, _, _ = lstsq(Xm, y.loc[idx].values, rcond=None)
    out = {"market_beta": float(coef[1]) if len(coef) > 1 else 0.0}
    if len(coef) > 2:
        out["size"] = float(coef[2])
    if len(coef) > 3:
        out["momentum"] = float(coef[3])
    return out


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Full Kelly: f = p - (1-p)/b where b = avg_win/avg_loss."""
    p = win_rate
    b = win_loss_ratio
    f = p - (1 - p) / b if b > 0 else 0.0
    return max(0.0, min(1.0, f))


def fractional_kelly(full_kelly: float, fraction: float = 0.5) -> float:
    return full_kelly * fraction
