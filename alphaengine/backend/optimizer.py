"""Mean-variance optimization: efficient frontier, min vol, max Sharpe, risk parity."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OptMode = Literal["min_vol", "max_sharpe", "risk_parity"]


def _annualize_rets(r: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    return (1 + r.mean(axis=0)) ** periods_per_year - 1


def _annualize_cov(cov: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    return cov * periods_per_year


def optimize(
    returns: pd.DataFrame,
    mode: OptMode = "max_sharpe",
    max_weight: float = 0.10,
    long_only: bool = True,
) -> dict:
    """Optimize portfolio. returns: columns = assets, index = dates."""
    rets = returns.dropna(how="all").fillna(0)
    if rets.empty or len(rets.columns) < 2:
        return {"weights": {}, "metrics": {}, "efficient_frontier": []}

    mu = rets.mean().values
    cov = rets.cov().values
    n = len(mu)
    rf = 0.0

    def port_return(w: np.ndarray) -> float:
        return float(w @ mu)

    def port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    def neg_sharpe(w: np.ndarray) -> float:
        r = port_return(w)
        v = port_vol(w)
        return -(r - rf) / (v + 1e-12)

    def risk_parity_obj(w: np.ndarray) -> float:
        mc = cov @ w
        rc = w * mc
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    bounds = [(0.0, max_weight) if long_only else (-max_weight, max_weight) for _ in range(n)]
    sum_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    if mode == "min_vol":
        res = minimize(lambda w: port_vol(w) ** 2, x0, bounds=bounds, constraints=[sum_one], method="SLSQP")
    elif mode == "max_sharpe":
        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=[sum_one], method="SLSQP")
    elif mode == "risk_parity":
        res = minimize(risk_parity_obj, x0, bounds=bounds, constraints=[sum_one], method="SLSQP")
    else:
        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=[sum_one], method="SLSQP")

    w_opt = res.x
    w_opt = np.clip(w_opt, 0 if long_only else -max_weight, max_weight)
    w_opt = w_opt / w_opt.sum()

    ann_ret = (1 + port_return(w_opt)) ** 252 - 1
    ann_vol = port_vol(w_opt) * np.sqrt(252)
    sharpe = (ann_ret - rf) / (ann_vol + 1e-12)

    weights = {returns.columns[i]: float(w_opt[i]) for i in range(n) if abs(w_opt[i]) > 1e-6}

    ef = []
    for target_ret in np.linspace(mu.min(), mu.max(), 20):
        tr = float(target_ret)
        try:
            r = minimize(
                lambda w: w @ cov @ w,
                x0,
                bounds=bounds,
                constraints=[
                    sum_one,
                    {"type": "eq", "fun": (lambda t: lambda w: w @ mu - t)(tr)},
                ],
                method="SLSQP",
            )
            if r.success:
                vol = np.sqrt(r.fun) * np.sqrt(252)
                ret_ann = (1 + target_ret) ** 252 - 1
                ef.append({"return": float(ret_ann), "volatility": float(vol)})
        except Exception:
            continue

    return {
        "weights": weights,
        "metrics": {
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
        },
        "efficient_frontier": ef,
    }
