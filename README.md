# Professional-

This repository holds multiple projects. Each lives in its own top-level folder.

## Projects

| Folder | Description |
|--------|-------------|
| [`quant-ml-alpha/`](./quant-ml-alpha/) | ML-based equity alpha pipeline (features, walk-forward training, backtest). |

Add new projects as sibling folders next to `quant-ml-alpha/`.

## Clone & run (quant-ml-alpha)

```powershell
cd quant-ml-alpha
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

See [`quant-ml-alpha/README.md`](./quant-ml-alpha/README.md) for details.
