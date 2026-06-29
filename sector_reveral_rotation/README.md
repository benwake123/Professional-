# Sector Reversal Rotation Backtest

A reproducible quantitative-finance project that ranks S&P 500 sector ETFs by their prior 20-trading-day return and invests equally in the three weakest sectors for the next 20 trading days. The model is a cross-sectional mean-reversion strategy with explicit transaction costs and one-day signal execution lag.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/sector_rotation.py --csv data/index_prices.csv
```
To use a longer live Yahoo Finance history, omit the local sample:
```bash
python src/sector_rotation.py --csv "" --start 2005-01-01
```
To see the outputs used for the window tested in this project run:
'''bash
python src/sector_rotation.py --csv "" --start 2019-01-01 --end 2020-01-01 --lookback 20 --top-n 2 --holding 20 --mode reversal --outdir outputs/best_window
'''
## Outputs
`outputs/` contains metrics, daily returns, portfolio weights, rebalance selections, equity curves, and drawdown charts.

## Important research caveat
The default parameters were selected after examining the included 2019 sample. Therefore the displayed result is in-sample and subject to selection/lookback bias. It is suitable as a research demonstration, not evidence of future profitability or a live trading recommendation.
