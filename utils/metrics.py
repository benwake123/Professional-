import numpy as np
import pandas as pd


def sharpe(r: pd.Series) -> float:
    return float(np.sqrt(252) * r.mean() / (r.std() + 1e-12))


def sortino(r: pd.Series) -> float:
    dn = r[r < 0].std()
    return float(np.sqrt(252) * r.mean() / (dn + 1e-12))


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def annual_return(r: pd.Series) -> float:
    eq = float((1 + r).prod())
    years = len(r) / 252
    if years <= 0:
        return 0.0
    return float(eq ** (1 / years) - 1)
