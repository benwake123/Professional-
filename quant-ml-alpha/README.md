# quant-ml-alpha

Quantitative ML alpha pipeline: daily equity features, walk-forward training, long/short backtest.

**Run from this directory** (`quant-ml-alpha/quant-ml-alpha` in the repo, or `quant-ml-alpha/` after `cd` from repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

Config: `config/config.yaml`. Optional per-stock charts: `.\.venv\Scripts\python utils\stock_summary_charts.py`
