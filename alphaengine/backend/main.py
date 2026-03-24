"""FastAPI app: signals, backtest, risk, optimizer, paper trading."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backtester import BacktestConfig, run_backtest
from data import UNIVERSE, fetch_ohlcv, fetch_stock_analysis
from models import BacktestRun, SessionLocal, init_db
from optimizer import optimize as run_optimize
from paper_trading import get_engine
from risk import (
    STRESS_PERIODS,
    cvar,
    fractional_kelly,
    historical_var,
    kelly_fraction,
    monte_carlo_fan,
    monte_carlo_var,
    parametric_var,
    stress_pnl,
)
from signals import run_signal_analysis
from signals import compute_signal

app = FastAPI(title="AlphaEngine API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()


# --- Request/Response models ---

class BacktestRequest(BaseModel):
    signal_type: str = "momentum_3m"
    lookback_days: int = 21
    rebalance_freq: str = "daily"
    tc_bps: float = 5.0
    slippage_bps: float = 2.0
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None


class OptimizeRequest(BaseModel):
    mode: str = "max_sharpe"
    max_weight: float = 0.10
    long_only: bool = True
    start_date: str = "2022-01-01"
    end_date: Optional[str] = None


class OrderRequest(BaseModel):
    ticker: str
    side: str
    shares: float
    limit_price: Optional[float] = None


# --- Endpoints ---

@app.get("/api/signals/{signal_type}")
async def get_signal(signal_type: str, start: str = "2020-01-01", end: Optional[str] = None) -> dict[str, Any]:
    try:
        df = fetch_ohlcv(start=start, end=end or "")
        if df.empty:
            raise HTTPException(400, "No data")
        valid = ["momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m", "mean_reversion", "volume_trend", "vwap_deviation"]
        if signal_type not in valid:
            signal_type = "momentum_3m"
        result = run_signal_analysis(df, signal_type)
        return {"signal_type": signal_type, **result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/backtest")
async def run_backtest_endpoint(req: BacktestRequest) -> dict[str, Any]:
    try:
        df = fetch_ohlcv(start=req.start_date, end=req.end_date or "")
        if df.empty:
            raise HTTPException(400, "No data")
        df["ret"] = df.groupby("ticker")["close"].pct_change()
        df["fwd_1d"] = df.groupby("ticker")["ret"].shift(-1)
        df_idx = df.set_index(["date", "ticker"])
        sig = compute_signal(df_idx, req.signal_type)
        df_idx["signal"] = sig
        df = df_idx.reset_index().dropna(subset=["signal", "fwd_1d"])
        config = BacktestConfig(
            signal_type=req.signal_type,
            lookback_days=req.lookback_days,
            rebalance_freq=req.rebalance_freq,
            tc_bps=req.tc_bps,
            slippage_bps=req.slippage_bps,
        )
        out = run_backtest(df, config)
        db = SessionLocal()
        try:
            run = BacktestRun(
                config_json=json.dumps(req.model_dump()),
                metrics_json=json.dumps(out["metrics"]),
                equity_curve_json=json.dumps({"dates": out["dates"], "equity": out["equity_curve"]}),
            )
            db.add(run)
            db.commit()
            out["run_id"] = run.id
        finally:
            db.close()
        return out
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/risk/{portfolio_id}")
async def get_risk(portfolio_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run = db.query(BacktestRun).filter(BacktestRun.id == portfolio_id).first()
        if not run:
            raise HTTPException(404, "Backtest not found")
        import pandas as pd
        eq = json.loads(run.equity_curve_json or "{}")
        dates = eq.get("dates", [])
        equity = eq.get("equity", [])
        if len(dates) < 2:
            raise HTTPException(400, "Insufficient data")
        rets = pd.Series(equity).pct_change().dropna()
        var_h95 = historical_var(rets, 0.95)
        var_p95 = parametric_var(rets, 0.95)
        var_mc95 = monte_carlo_var(rets, 0.95)
        cvar95 = cvar(rets, 0.95)
        cvar99 = cvar(rets, 0.99)
        fan = monte_carlo_fan(rets)
        stress = {k: stress_pnl(rets, k) for k in STRESS_PERIODS}
        return {
            "var": {"historical_95": var_h95, "parametric_95": var_p95, "monte_carlo_95": var_mc95},
            "cvar": {"95": cvar95, "99": cvar99},
            "monte_carlo_fan": fan,
            "stress_pnl": stress,
        }
    finally:
        db.close()


@app.post("/api/optimize")
async def optimize_portfolio(req: OptimizeRequest) -> dict[str, Any]:
    try:
        df = fetch_ohlcv(start=req.start_date, end=req.end_date or "")
        if df.empty:
            raise HTTPException(400, "No data")
        wide = df.pivot(index="date", columns="ticker", values="close").pct_change().dropna(how="all")
        wide = wide.dropna(axis=1, how="all")
        if wide.empty or len(wide.columns) < 2:
            raise HTTPException(400, "Insufficient return data")
        mode = req.mode if req.mode in ("min_vol", "max_sharpe", "risk_parity") else "max_sharpe"
        return run_optimize(wide, mode=mode, max_weight=req.max_weight, long_only=req.long_only)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/prices")
async def get_prices() -> dict[str, Any]:
    df = fetch_ohlcv(use_cache=True)
    if df.empty:
        raise HTTPException(500, "No price data")
    latest = df.groupby("ticker").last().reset_index()
    prices = {r["ticker"]: float(r["close"]) for _, r in latest.iterrows()}
    return {"prices": prices, "universe": UNIVERSE}


@app.post("/api/order")
async def submit_order(req: OrderRequest) -> dict[str, Any]:
    eng = get_engine()
    prices = (await get_prices())["prices"]
    eng.update_prices(prices)
    order = eng.submit_order(req.ticker, req.side, req.shares, req.limit_price)
    return {"order_id": len(eng.orders), "status": order.status, "filled_shares": order.filled_shares}


@app.get("/api/positions")
async def get_positions() -> dict[str, Any]:
    eng = get_engine()
    prices = (await get_prices())["prices"]
    eng.update_prices(prices)
    return eng.get_summary()


@app.get("/api/backtest/runs")
async def list_backtest_runs() -> list[dict]:
    db = SessionLocal()
    try:
        runs = db.query(BacktestRun).order_by(BacktestRun.id.desc()).limit(50).all()
        return [
            {"id": r.id, "config": r.config_json, "metrics": r.metrics_json, "created_at": str(r.created_at)}
            for r in runs
        ]
    finally:
        db.close()


@app.get("/api/backtest/{run_id}")
async def get_backtest_run(run_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
        if not run:
            raise HTTPException(404, "Backtest not found")
        eq = json.loads(run.equity_curve_json or "{}")
        return {
            "id": run.id,
            "config": run.config_json,
            "metrics": run.metrics_json,
            "equity_curve": eq.get("equity", []),
            "dates": eq.get("dates", []),
        }
    finally:
        db.close()


@app.get("/api/stock/{ticker}")
async def get_stock_analysis(ticker: str) -> dict[str, Any]:
    """Proxy for frontend stock analysis. Fetches from yfinance (no CORS)."""
    try:
        return fetch_stock_analysis(ticker.strip().upper())
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
