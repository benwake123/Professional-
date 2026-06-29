"""
End-to-end pipeline entry point with audit exports and parameter search.

Module connections
------------------
Upstream: config, data, backtest, audit, research, metrics modules.
Downstream: ``python -m src.run_pipeline``.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.attribution import build_pnl_attribution
from src.audit import attach_audit_artifacts, summarize_decision_funnel, summarize_rejections
from src.backtest import run_walk_forward_backtest
from src.config import load_config, resolve_project_paths
from src.data_loader import load_all_market_data
from src.data_validation import validate_all_data
from src.metrics import build_performance_summary
from src.plots import generate_all_figures
from src.research import (
    run_liquidity_sensitivity,
    run_parameter_search,
    run_signal_direction_diagnostics,
    select_best_parameters,
)


def parse_command_line_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SV options strategy pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--model", type=str, default="ensemble")
    parser.add_argument("--decision-frequency", type=str, default=None)
    parser.add_argument("--full-parameter-search", action="store_true")
    parser.add_argument("--skip-parameter-search", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _make_progress_callback(quiet: bool):
    if quiet:
        return None

    def progress(pct: int, as_of_date: pd.Timestamp) -> None:
        print(f"[backtest] {pct:3d}%  {pd.Timestamp(as_of_date).date()}", flush=True)

    return progress


def _apply_selected_parameters(config: dict, selected: dict[str, object]) -> dict:
    if not selected:
        return config
    updated = deepcopy(config)
    updated["signal"]["long_vol_z_threshold"] = selected["long_vol_z_threshold"]
    updated["signal"]["short_vol_z_threshold"] = selected["short_vol_z_threshold"]
    updated["exits"]["maximum_holding_days"] = selected["maximum_holding_days"]
    updated["options"]["maximum_relative_spread"] = selected["maximum_relative_spread"]
    updated["options"]["minimum_open_interest"] = selected["minimum_open_interest"]
    updated["options"]["minimum_volume"] = selected["minimum_volume"]
    updated["delta_hedge"]["mode"] = selected["delta_hedge_mode"]
    return updated


def export_results(results, output_dir: Path, figures_dir: Path, performance_summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.daily_history.to_csv(output_dir / "daily_history.csv", index=False)
    results.trades.to_csv(output_dir / "trades.csv", index=False)
    results.trade_decisions.to_csv(output_dir / "trade_decisions.csv", index=False)
    results.forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    if hasattr(results, "calibrations") and not results.calibrations.empty:
        results.calibrations.to_csv(output_dir / "calibrations.csv", index=False)
    if hasattr(results, "decision_funnel") and not results.decision_funnel.empty:
        results.decision_funnel.to_csv(output_dir / "decision_funnel.csv", index=False)
        summarize_decision_funnel(results.decision_funnel).to_csv(
            output_dir / "decision_funnel_summary.csv", index=False
        )
    if hasattr(results, "trade_audit") and not results.trade_audit.empty:
        results.trade_audit.to_csv(output_dir / "trade_audit.csv", index=False)
    if hasattr(results, "accounting_reconciliation") and not results.accounting_reconciliation.empty:
        results.accounting_reconciliation.to_csv(
            output_dir / "accounting_reconciliation.csv", index=False
        )
    if hasattr(results, "forecast_diagnostics") and not results.forecast_diagnostics.empty:
        results.forecast_diagnostics.to_csv(output_dir / "forecast_diagnostics.csv", index=False)
    if hasattr(results, "rejection_summary") and not results.rejection_summary.empty:
        results.rejection_summary.to_csv(output_dir / "rejection_summary.csv", index=False)

    with open(output_dir / "performance_summary.json", "w") as fh:
        json.dump({k: _jsonable(v) for k, v in performance_summary.items()}, fh, indent=2)

    attribution_dir = output_dir / "attribution"
    attribution_dir.mkdir(exist_ok=True)
    for name, df in results.pnl_attribution.items():
        df.to_csv(attribution_dir / f"{name}.csv", index=False)

    generate_all_figures(
        daily_history=results.daily_history,
        decisions=results.trade_decisions,
        forecasts=results.forecasts,
        calibrations=getattr(results, "calibrations", pd.DataFrame()),
        attribution=results.pnl_attribution,
        figures_dir=figures_dir,
    )


def _jsonable(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def print_final_report(results, summary: dict, accounting_diff: float) -> None:
    funnel = results.decision_funnel
    trades = results.trades
    exits = trades[trades["action"] == "exit"] if not trades.empty else trades
    entries = trades[trades["action"] == "entry"] if not trades.empty else trades
    hedges = trades[trades["action"] == "hedge"] if not trades.empty else trades

    print("\n[pipeline] final report", flush=True)
    print(f"  total decision dates          {len(funnel)}", flush=True)
    print(f"  valid model forecasts         {int(funnel['forecast_created'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  non-flat signals              {int(funnel['signal_exceeded_threshold'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  contract-selection successes  {int(funnel['atm_contracts_found'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  liquidity approvals           {int(funnel['liquidity_filter_passed'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  risk approvals                {int(funnel['risk_check_passed'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  executed trades               {int(funnel['trade_executed'].sum()) if not funnel.empty else 0}", flush=True)
    print(f"  long-volatility trades        {int((entries.get('direction') == 'long_vol').sum()) if 'direction' in entries.columns else 0}", flush=True)
    print(f"  short-volatility trades       {int((entries.get('direction') == 'short_vol').sum()) if 'direction' in entries.columns else 0}", flush=True)
    wins = int((exits["realized_pnl"] > 0).sum()) if "realized_pnl" in exits.columns else 0
    losses = int((exits["realized_pnl"] < 0).sum()) if "realized_pnl" in exits.columns else 0
    print(f"  wins                          {wins}", flush=True)
    print(f"  losses                        {losses}", flush=True)
    gross_option = float(exits["gross_option_pnl"].sum()) if "gross_option_pnl" in exits.columns else float("nan")
    hedge_pnl = float(hedges["gross_cash_flow"].sum()) if not hedges.empty else 0.0
    costs = float(summary.get("final_equity", 0) - summary.get("initial_equity", 0) - gross_option - hedge_pnl)
    print(f"  gross option P&L              {gross_option:.2f}", flush=True)
    print(f"  transaction costs (net)       {costs:.2f}", flush=True)
    print(f"  hedge P&L                     {hedge_pnl:.2f}", flush=True)
    print(f"  net P&L                       {summary.get('final_equity', 0) - summary.get('initial_equity', 0):.2f}", flush=True)
    print(f"  annualized return             {summary.get('annualized_return', 0.0)}", flush=True)
    print(f"  annualized volatility         {summary.get('annualized_volatility', 0.0)}", flush=True)
    print(f"  Sharpe ratio                  {summary.get('sharpe_ratio', 0.0)}", flush=True)
    print(f"  maximum drawdown              {summary.get('maximum_drawdown', 0.0)}", flush=True)
    print(f"  accounting difference         {accounting_diff:.6f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_command_line_arguments(argv)
    config = load_config(args.config)
    project_root = args.config.resolve().parent
    paths = resolve_project_paths(config, project_root=project_root)

    print(f"[pipeline] start: {datetime.now(timezone.utc).isoformat()}Z", flush=True)
    bundle = load_all_market_data(paths)
    validate_all_data(bundle)
    print("[pipeline] data validation passed.", flush=True)

    results_dir = Path(paths.get("results_dir", project_root / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    audit_start = config["dates"]["development_start"]
    audit_end = "2021-12-31"
    print(
        f"[pipeline] signal-direction diagnostics {audit_start} -> {audit_end}",
        flush=True,
    )
    direction_diag = run_signal_direction_diagnostics(
        bundle, config, audit_start, audit_end
    )
    direction_diag.to_csv(results_dir / "signal_direction_diagnostics.csv", index=False)

    print(
        f"[pipeline] liquidity sensitivity on validation {config['dates']['validation_start']} -> {audit_end}",
        flush=True,
    )
    liquidity_table = run_liquidity_sensitivity(
        bundle,
        config,
        config["dates"]["validation_start"],
        audit_end,
    )
    liquidity_table.to_csv(results_dir / "liquidity_sensitivity.csv", index=False)

    if not args.skip_parameter_search:
        print("[pipeline] running parameter search (development + validation)...", flush=True)
        search = run_parameter_search(bundle, config, full_grid=args.full_parameter_search)
        search.to_csv(results_dir / "parameter_search.csv", index=False)
        selected = select_best_parameters(search)
        config = _apply_selected_parameters(config, selected)
        print(f"[pipeline] selected parameters: {selected}", flush=True)

    progress = _make_progress_callback(args.quiet)
    test_start = config["dates"]["test_start"]
    test_end = config["dates"]["end"]
    print(f"[pipeline] running untouched test-period backtest {test_start} -> {test_end}", flush=True)

    results = run_walk_forward_backtest(
        bundle=bundle,
        config=config,
        model_name=args.model,
        decision_frequency=args.decision_frequency,
        progress_callback=progress,
        start_date=test_start,
        end_date=test_end,
    )
    portfolio = getattr(results, "_portfolio", None)
    if portfolio is not None:
        attach_audit_artifacts(
            results,
            portfolio,
            bundle.underlying,
            int(config["model"]["forecast_horizon_days"]),
        )
        results.rejection_summary = summarize_rejections(results.decision_funnel)

    summary = build_performance_summary(results.daily_history, results.trades)
    results.performance_summary = summary
    results.pnl_attribution = build_pnl_attribution(
        results.daily_history, results.trades, results.trade_decisions
    )

    figures_dir = project_root / "report" / "figures"
    export_results(results, Path(paths.get("results_dir", project_root / "results")), figures_dir, summary)

    accounting_diff = 0.0
    if not results.accounting_reconciliation.empty:
        accounting_diff = float(results.accounting_reconciliation.iloc[0]["difference"])
    print_final_report(results, summary, accounting_diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
