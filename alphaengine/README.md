# AlphaEngine

A full-stack quantitative trading research platform showcasing signal research, backtesting, risk management, and portfolio construction — the four pillars quantitative trading firms look for.

---

## Project Overview

AlphaEngine is a professional-grade research platform built to demonstrate quant finance fundamentals. It includes:

- **Signal Research Engine** — Momentum, mean-reversion, volume, and VWAP signals with IC/ICIR analysis
- **Backtesting Engine** — Vectorized long/short backtester with transaction costs and full metrics
- **Risk Management** — VaR (historical, parametric, Monte Carlo), CVaR, stress tests, Kelly criterion
- **Portfolio Construction** — Mean-variance optimization, efficient frontier, risk parity
- **Paper Trading Simulator** — Virtual $1M account with order execution and P&L tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)                 │
│  Dashboard | Signal Lab | Backtest Studio | Risk | Portfolio |   │
│                         Paper Trading                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼───────────────────────────────────┐
│                     Backend (Python FastAPI)                      │
│  /api/signals  /api/backtest  /api/risk  /api/optimize           │
│  /api/prices   /api/order     /api/positions                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Data Layer (yfinance)  │  SQLite (SQLAlchemy)  │  Paper Engine  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

| Term | Definition |
|------|------------|
| **IC** | Information Coefficient — correlation between signal and forward returns. Higher = better predictive power. |
| **ICIR** | IC / std(IC) — risk-adjusted signal quality. Measures consistency of signal performance. |
| **Sharpe** | (μ − rf) / σ × √252 — annualized excess return per unit of risk. |
| **CVaR** | Conditional VaR (Expected Shortfall) — average loss in the worst α% of outcomes. |
| **Kelly** | Optimal position size for geometric growth: f* = (p·b − q) / b where p=win prob, q=1−p, b=win/loss ratio. |
| **Risk Parity** | Portfolio where each asset contributes equally to total risk. |

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd alphaengine/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Frontend

```bash
cd alphaengine/frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Stock Universe

AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, GS, MS, BAC, XOM, CVX, JNJ, UNH, PFE, WMT, HD, COST, BA, CAT, SPY, QQQ

---

## Screenshots

<!-- Add screenshots here -->

---

## Future Improvements

- **ML Signal Generation** — Neural networks or gradient boosting for alpha prediction
- **Live Broker Integration** — Interactive Brokers API for live paper/live trading
- **Regime Detection** — Hidden Markov Models for market regime identification
