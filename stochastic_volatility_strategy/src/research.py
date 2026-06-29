"""
Walk-forward parameter search on development and validation periods.

Purpose
-------
Grid-searches signal thresholds, holding periods, and liquidity settings
using only 2014-2018 (development) and 2019-2021 (validation). The 2022-2023
test window remains untouched until a final candidate is selected.

Module connections
------------------
Upstream: ``src.backtest.run_walk_forward_backtest``, ``src.metrics``.
Downstream: ``src.run_pipeline`` writes ``results/parameter_search.csv``.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Mapping

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.metrics import build_performance_summary
from src.types import MarketDataBundle


def _period_metrics(
    bundle: MarketDataBundle,
    base_config: dict,
    params: dict,
    start: str,
    end: str,
) -> dict[str, float]:
    config = deepcopy(base_config)
    config["signal"]["long_vol_z_threshold"] = params["long_threshold"]
    config["signal"]["short_vol_z_threshold"] = params["short_threshold"]
    config["exits"]["maximum_holding_days"] = params["max_holding_days"]
    config["options"]["maximum_relative_spread"] = params["max_relative_spread"]
    config["options"]["minimum_open_interest"] = params["min_open_interest"]
    config["options"]["minimum_volume"] = params["min_volume"]
    config["delta_hedge"]["mode"] = params["hedge_mode"]

    results = run_walk_forward_backtest(
        bundle=bundle,
        config=config,
        model_name="ensemble",
        start_date=start,
        end_date=end,
    )
    summary = build_performance_summary(results.daily_history, results.trades)
    trades = results.trades
    exits = trades[trades["action"] == "exit"] if not trades.empty else trades
    option_pnl = float(exits["realized_pnl"].sum()) if "realized_pnl" in exits.columns else 0.0
    hedges = trades[trades["action"] == "hedge"] if not trades.empty else trades
    hedge_pnl = float(hedges["gross_cash_flow"].sum()) if not hedges.empty else 0.0
    return {
        **summary,
        "trade_count": int(len(exits)),
        "option_pnl": option_pnl,
        "hedge_pnl": hedge_pnl,
        "net_pnl": float(summary.get("final_equity", 0) - summary.get("initial_equity", 0)),
    }


def run_parameter_search(
    bundle: MarketDataBundle,
    base_config: dict,
    full_grid: bool = False,
) -> pd.DataFrame:
    """Evaluate candidate grids on development and validation windows."""
    dates = base_config["dates"]
    if full_grid:
        long_thresholds = [0.50, 0.75, 1.00, 1.25]
        short_thresholds = [0.50, 0.75, 1.00, 1.25]
        holding_periods = [5, 10, 15, 20]
        spread_levels = [0.05, 0.10, 0.15]
        oi_levels = [0, 100, 500]
        volume_levels = [0, 10, 50]
        hedge_modes = ["none", "daily", "threshold"]
    else:
        long_thresholds = [0.75, 1.00]
        short_thresholds = [0.75, 1.00]
        holding_periods = [10, 15]
        spread_levels = [0.10, 0.15]
        oi_levels = [100, 500]
        volume_levels = [10, 50]
        hedge_modes = ["none", "threshold"]

    rows: list[dict[str, object]] = []
    for combo in itertools.product(
        long_thresholds,
        short_thresholds,
        holding_periods,
        spread_levels,
        oi_levels,
        volume_levels,
        hedge_modes,
    ):
        params = {
            "long_threshold": combo[0],
            "short_threshold": combo[1],
            "max_holding_days": combo[2],
            "max_relative_spread": combo[3],
            "min_open_interest": combo[4],
            "min_volume": combo[5],
            "hedge_mode": combo[6],
        }
        dev = _period_metrics(
            bundle,
            base_config,
            params,
            dates["development_start"],
            "2018-12-31",
        )
        val = _period_metrics(
            bundle,
            base_config,
            params,
            dates["validation_start"],
            "2021-12-31",
        )
        rows.append({**params, **{f"dev_{k}": v for k, v in dev.items()}, **{f"val_{k}": v for k, v in val.items()}})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["selection_score"] = (
        df["val_trade_count"].clip(lower=0)
        * (df["val_net_pnl"] > 0).astype(float)
        * (df["val_sharpe_ratio"] > 0).astype(float)
        * (df["val_trade_count"] >= 25).astype(float)
    )
    return df.sort_values("selection_score", ascending=False)


def run_liquidity_sensitivity(
    bundle: MarketDataBundle,
    base_config: dict,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Liquidity grid sensitivity on a fixed window (validation by default)."""
    spread_levels = [0.05, 0.10, 0.15]
    oi_levels = [0, 100, 500]
    volume_levels = [0, 10, 50]
    rows: list[dict[str, object]] = []
    for spread, oi, volume in itertools.product(spread_levels, oi_levels, volume_levels):
        params = {
            "long_threshold": base_config["signal"]["long_vol_z_threshold"],
            "short_threshold": base_config["signal"]["short_vol_z_threshold"],
            "max_holding_days": base_config["exits"]["maximum_holding_days"],
            "max_relative_spread": spread,
            "min_open_interest": oi,
            "min_volume": volume,
            "hedge_mode": base_config["delta_hedge"]["mode"],
        }
        metrics = _period_metrics(bundle, base_config, params, start, end)
        rows.append(
            {
                "maximum_relative_spread": spread,
                "minimum_open_interest": oi,
                "minimum_volume": volume,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_signal_direction_diagnostics(
    bundle: MarketDataBundle,
    config: dict,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Four diagnostic backtests for signal / cost attribution."""
    scenarios = [
        ("long_only", {"direction_filter": "long_vol"}),
        ("short_only", {"direction_filter": "short_vol"}),
        ("both_directions", {}),
        ("both_zero_cost_mid", {"zero_cost_mode": True, "midpoint_fill_mode": True}),
    ]
    rows: list[dict[str, object]] = []
    for label, kwargs in scenarios:
        results = run_walk_forward_backtest(
            bundle=bundle,
            config=config,
            model_name=str(config.get("forecast_model", "ensemble")),
            start_date=start,
            end_date=end,
            **kwargs,
        )
        summary = build_performance_summary(results.daily_history, results.trades)
        trades = results.trades
        exits = trades[trades["action"] == "exit"] if not trades.empty else trades
        hedges = trades[trades["action"] == "hedge"] if not trades.empty else trades
        rows.append(
            {
                "scenario": label,
                "trade_count": int(len(exits)),
                "wins": int((exits["realized_pnl"] > 0).sum()) if "realized_pnl" in exits.columns else 0,
                "losses": int((exits["realized_pnl"] < 0).sum()) if "realized_pnl" in exits.columns else 0,
                "option_pnl": float(exits["realized_pnl"].sum()) if "realized_pnl" in exits.columns else 0.0,
                "hedge_pnl": float(hedges["gross_cash_flow"].sum()) if not hedges.empty else 0.0,
                "net_pnl": float(summary.get("final_equity", 0) - summary.get("initial_equity", 0)),
                "sharpe_ratio": float(summary.get("sharpe_ratio", 0.0)),
                "maximum_drawdown": float(summary.get("maximum_drawdown", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def select_best_parameters(search_results: pd.DataFrame) -> dict[str, object]:
    """Pick a stable candidate from validation metrics, not max return alone."""
    if search_results.empty:
        return {}
    eligible = search_results[
        (search_results["val_trade_count"] >= 25)
        & (search_results["val_net_pnl"] > 0)
    ]
    if eligible.empty:
        eligible = search_results.sort_values("val_trade_count", ascending=False).head(10)
    best = eligible.iloc[0]
    return {
        "long_vol_z_threshold": float(best["long_threshold"]),
        "short_vol_z_threshold": float(best["short_threshold"]),
        "maximum_holding_days": int(best["max_holding_days"]),
        "maximum_relative_spread": float(best["max_relative_spread"]),
        "minimum_open_interest": int(best["min_open_interest"]),
        "minimum_volume": int(best["min_volume"]),
        "delta_hedge_mode": str(best["hedge_mode"]),
    }
