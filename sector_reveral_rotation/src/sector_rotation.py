"""Sector Rotation Research Backtest.
Uses local CSV if supplied; otherwise downloads adjusted data from Yahoo Finance.
Educational research only; not investment advice.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SECTORS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
ALL = ['SPY'] + SECTORS

def load_prices(csv_path: str | None, start: str, end: str) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        raw = pd.read_csv(csv_path, parse_dates=['date'])
        prices = raw.pivot(index='date', columns='symbol', values='close').sort_index()
        return prices.reindex(columns=ALL)

    import yfinance as yf

    # threads=False avoids occasional local yfinance SQLite/cache locking errors.
    data = yf.download(
        ALL,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how='all').ffill()
    if close.empty or 'SPY' not in close.columns:
        raise RuntimeError('Yahoo download returned no usable data. Use --csv with the included sample.')

    # Reindex instead of close[ALL] so the script still runs if Yahoo temporarily
    # fails on one ticker or if an ETF did not exist for the full backtest window.
    return close.reindex(columns=ALL)

def backtest(prices: pd.DataFrame, lookback=20, top_n=3, holding=20, cost_bps=10):
    sector_cols = [c for c in SECTORS if c in prices.columns]
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    trailing = prices[sector_cols].pct_change(lookback)
    weights = pd.DataFrame(0.0, index=prices.index, columns=sector_cols)
    selections=[]
    for i in range(lookback, len(prices), holding):
        ranked = trailing.iloc[i].dropna().nsmallest(top_n)  # weakest = mean reversion
        selected = ranked.index.tolist()
        if not selected:
            continue
        weights.loc[prices.index[i:min(i+holding,len(prices))], selected] = 1.0/len(selected)
        selections.append({'signal_date':prices.index[i], 'selected':', '.join(selected)})
    # Signal at close t; position becomes active on t+1, eliminating same-bar look-ahead.
    weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    strategy = (weights * returns[sector_cols]).sum(axis=1) - (cost_bps/10000.0)*turnover
    benchmark = returns['SPY']
    return strategy.rename('Strategy'), benchmark.rename('SPY'), weights, pd.DataFrame(selections)

def stats(r: pd.Series) -> dict:
    r=r.dropna(); equity=(1+r).cumprod(); years=len(r)/252
    cagr=equity.iloc[-1]**(1/years)-1 if years>0 else np.nan
    vol=r.std()*np.sqrt(252)
    sharpe=(r.mean()/r.std())*np.sqrt(252) if r.std()>0 else np.nan
    dd=equity/equity.cummax()-1
    return {'Total Return':equity.iloc[-1]-1,'CAGR':cagr,'Annualized Volatility':vol,
            'Sharpe (rf=0)':sharpe,'Max Drawdown':dd.min(),'Positive Days':(r>0).mean()}

def save_outputs(prices, strategy, benchmark, weights, selections, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    comparison=pd.DataFrame({'Strategy':strategy,'SPY':benchmark})
    equity=(1+comparison).cumprod()
    metrics=pd.DataFrame({'Strategy':stats(strategy),'SPY':stats(benchmark)})
    metrics.to_csv(out/'performance_metrics.csv')
    comparison.to_csv(out/'daily_returns.csv')
    weights.to_csv(out/'portfolio_weights.csv')
    selections.to_csv(out/'rebalance_selections.csv',index=False)
    equity.to_csv(out/'equity_curve.csv')
    plt.figure(figsize=(10,6)); equity.plot(ax=plt.gca()); plt.title('Growth of $1: Sector Mean Reversion vs SPY'); plt.ylabel('Portfolio Value'); plt.tight_layout(); plt.savefig(out/'equity_curve.png',dpi=180); plt.close()
    dd=equity/equity.cummax()-1
    plt.figure(figsize=(10,5)); dd.plot(ax=plt.gca()); plt.title('Drawdowns'); plt.ylabel('Drawdown'); plt.tight_layout(); plt.savefig(out/'drawdowns.png',dpi=180); plt.close()
    pretty_metrics = metrics.apply(lambda col: col.map(lambda x: f'{x:.4f}' if pd.notna(x) else 'nan'))
    print(pretty_metrics)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/index_prices.csv')
    p.add_argument('--start',default='2005-01-01'); p.add_argument('--end',default=None)
    p.add_argument('--lookback',type=int,default=20); p.add_argument('--top-n',type=int,default=3)
    p.add_argument('--holding',type=int,default=20); p.add_argument('--cost-bps',type=float,default=10)
    p.add_argument('--outdir',default='outputs')
    a=p.parse_args(); prices=load_prices(a.csv,a.start,a.end)
    s,b,w,sel=backtest(prices,a.lookback,a.top_n,a.holding,a.cost_bps)
    save_outputs(prices,s,b,w,sel,a.outdir)
if __name__=='__main__': main()