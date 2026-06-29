"""
Reporting plots.

Purpose
-------
Generates the figures referenced in ``report/figures/``. All plots use the
non-interactive ``matplotlib`` ``Agg`` backend so :func:`generate_all_figures`
can run inside CI / headless containers.

Module connections
------------------
Upstream:
    - ``src.types.BacktestResults``.
Downstream:
    - ``src.run_pipeline.export_results`` calls
      :func:`generate_all_figures` after building the performance summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _save_or_close(fig: plt.Figure, output_path: Path | None) -> Path | None:
    if output_path is None:
        plt.close(fig)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_equity_curve(daily_history: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not daily_history.empty:
        ax.plot(daily_history["date"], daily_history["equity"], color="#1f77b4")
    ax.set_title("Strategy equity")
    ax.set_xlabel("date")
    ax.set_ylabel("equity ($)")
    ax.grid(True, alpha=0.3)
    return _save_or_close(fig, output_path)


def plot_drawdown(daily_history: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 3))
    if not daily_history.empty:
        ax.fill_between(daily_history["date"], daily_history["drawdown"], 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    return _save_or_close(fig, output_path)


def plot_signal_history(decisions: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not decisions.empty and "zscore" in decisions.columns:
        ax.plot(decisions["as_of_date"], decisions["zscore"], color="#2ca02c", marker=".", linestyle="-")
        if "entry_threshold" in decisions.columns:
            ax.plot(decisions["as_of_date"], decisions["entry_threshold"], color="black", linestyle="--", label="+threshold")
            ax.plot(decisions["as_of_date"], -decisions["entry_threshold"], color="black", linestyle="--", label="-threshold")
        ax.legend(loc="upper left")
    ax.set_title("VRP z-score over time")
    ax.set_xlabel("date")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.3)
    return _save_or_close(fig, output_path)


def plot_forecast_vs_realized(
    forecasts: pd.DataFrame, daily_history: pd.DataFrame, output_path: Path | None = None
) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not forecasts.empty:
        ax.plot(forecasts["as_of_date"], forecasts["expected_variance"], color="#9467bd", label="forecast")
    ax.set_title("Model forecast variance")
    ax.set_xlabel("date")
    ax.set_ylabel("variance (annual)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return _save_or_close(fig, output_path)


def plot_regime_performance(by_regime: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not by_regime.empty:
        ax.bar(by_regime["regime"], by_regime["total_pnl"], color="#ff7f0e")
        ax.set_xticks(range(len(by_regime)))
        ax.set_xticklabels(by_regime["regime"], rotation=30, ha="right")
    ax.set_title("Realized PnL by regime")
    ax.set_ylabel("total PnL ($)")
    ax.grid(True, alpha=0.3, axis="y")
    return _save_or_close(fig, output_path)


def plot_pnl_attribution(attribution: Mapping[str, pd.DataFrame], output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cost_row = attribution.get("transaction_cost_drag")
    if cost_row is not None and not cost_row.empty:
        ax.bar(["commission", "slippage"], [cost_row.iloc[0]["commission"], cost_row.iloc[0]["slippage"]], color="#8c564b")
    ax.set_title("Transaction cost drag")
    ax.set_ylabel("$ cost")
    ax.grid(True, alpha=0.3, axis="y")
    return _save_or_close(fig, output_path)


def plot_parameter_stability(calibrations: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not calibrations.empty and "as_of_date" in calibrations.columns:
        for column in ("sigma", "theta", "v0"):
            if column in calibrations.columns:
                ax.plot(calibrations["as_of_date"], calibrations[column], label=column)
        ax.legend(loc="upper right")
    ax.set_title("Calibration parameter time series")
    ax.set_xlabel("date")
    ax.set_ylabel("parameter value")
    ax.grid(True, alpha=0.3)
    return _save_or_close(fig, output_path)


def generate_all_figures(
    daily_history: pd.DataFrame,
    decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    calibrations: pd.DataFrame,
    attribution: Mapping[str, pd.DataFrame],
    figures_dir: Path,
) -> list[Path]:
    """Write every standard figure into ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.append(plot_equity_curve(daily_history, figures_dir / "equity_curve.png"))
    outputs.append(plot_drawdown(daily_history, figures_dir / "drawdown.png"))
    outputs.append(plot_signal_history(decisions, figures_dir / "signal_history.png"))
    outputs.append(
        plot_forecast_vs_realized(forecasts, daily_history, figures_dir / "forecast_variance.png")
    )
    outputs.append(plot_regime_performance(attribution.get("by_regime", pd.DataFrame()), figures_dir / "regime_pnl.png"))
    outputs.append(plot_pnl_attribution(attribution, figures_dir / "transaction_costs.png"))
    outputs.append(plot_parameter_stability(calibrations, figures_dir / "calibration_stability.png"))
    return [p for p in outputs if p is not None]
