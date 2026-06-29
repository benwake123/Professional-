"""Command-line entry point that runs the full earnings-volatility pipeline.

Usage::

    python -m src.run_analysis \
        --data data/aapl_earnings_volatility_data.csv \
        --output results

The script intentionally has no implicit defaults that reach outside the
repository.  Every input is supplied on the command line and the project
reads exactly one CSV.

The pipeline executes the following steps:

1. Load the daily panel and compute log returns.
2. Build the event-level feature/outcome table.
3. Estimate summary statistics, bootstrap confidence intervals, and the
   two descriptive regressions.
4. Run the walk-forward forecast comparison.
5. Write CSV result tables, LaTeX tables and macros, and PDF figures so
   ``report/research_paper.tex`` compiles without manual editing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import add_log_returns, load_daily_panel
from .features import build_event_table
from .forecasting import (
    INITIAL_TRAIN_EVENTS,
    RIDGE_PENALTY,
    RIDGE_PREDICTORS,
    run_walk_forward,
)
from .metrics import (
    event_bootstrap_mean_ci,
    forecast_correlation,
    mean_absolute_error,
    out_of_sample_r_squared,
    root_mean_squared_error,
    variance_spread_sign_accuracy,
)
from .plots import (
    plot_iv_rv_scatter,
    plot_iv_vs_realized,
    plot_vrp_distribution,
    plot_walk_forward_forecasts,
)
from .regression import fit_ols


# Module-level repository root so the LaTeX helpers always write to
# ``report/generated`` regardless of where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_FIGURES = REPO_ROOT / "report" / "figures"
REPORT_GENERATED = REPO_ROOT / "report" / "generated"


# ---------------------------------------------------------------------------
# LaTeX writers.  These are intentionally small and concrete so the column
# names in the paper are easy to audit.
# ---------------------------------------------------------------------------


def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format ``value`` as ``"33.83\\%"``.  Missing values become ``"--"``."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value * 100:.{decimals}f}\\%"


def _fmt_num(value: float, decimals: int = 4) -> str:
    """Format ``value`` as ``"0.0328"``.  Missing values become ``"--"``."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to ``path`` after ensuring the directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_summary_table(events: pd.DataFrame, dest: Path) -> None:
    """LaTeX table of mean / median statistics used in Section 4."""

    def stats(col: str) -> tuple[float, float]:
        series = events[col].dropna()
        return float(series.mean()), float(series.median())

    mean_vxapl_vol, med_vxapl_vol = stats("vxapl_implied_vol")
    mean_pre_rv_vol, med_pre_rv_vol = stats("pre_rv20_vol")
    mean_post_rv_vol, med_post_rv_vol = stats("post_rv20_vol")
    mean_iv, med_iv = stats("iv_var")
    mean_post_rv, med_post_rv = stats("post_rv20_var")
    mean_vrp, med_vrp = stats("vrp20")
    abs_gap = events["abs_event_gap"].dropna()
    mean_gap = float(abs_gap.mean()) if not abs_gap.empty else float("nan")
    med_gap = float(abs_gap.median()) if not abs_gap.empty else float("nan")

    rows = [
        ("VXAPL implied volatility",
         _fmt_pct(mean_vxapl_vol / 100.0), _fmt_pct(med_vxapl_vol / 100.0)),
        ("Pre-announcement 20-day realized volatility",
         _fmt_pct(mean_pre_rv_vol / 100.0), _fmt_pct(med_pre_rv_vol / 100.0)),
        ("Post-announcement 20-day realized volatility",
         _fmt_pct(mean_post_rv_vol / 100.0), _fmt_pct(med_post_rv_vol / 100.0)),
        ("Implied variance", _fmt_num(mean_iv, 4), _fmt_num(med_iv, 4)),
        ("Post-announcement realized variance", _fmt_num(mean_post_rv, 4), _fmt_num(med_post_rv, 4)),
        ("Variance spread $IV-RV$", _fmt_num(mean_vrp, 4), _fmt_num(med_vrp, 4)),
        ("Absolute overnight earnings gap", _fmt_pct(mean_gap), _fmt_pct(med_gap)),
    ]

    body = "\n".join(f"{name} & {mean} & {median} \\\\" for name, mean, median in rows)
    n_events = int(events["iv_var"].notna().sum())
    text = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Summary statistics across Apple earnings announcements}\n"
        "\\label{tab:summary}\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Statistic & Mean & Median \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{minipage}{0.92\\linewidth}\n"
        "\\footnotesize \\textit{Notes:} The sample contains "
        f"{n_events} quarterly announcements. Volatility is annualized. "
        "Implied variance equals $(VXAPL/100)^2$. Realized variance is the "
        "annualized sum of squared close-to-close log returns over the next 20 "
        "trading days.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    _write_text(dest, text)


def _write_regression_table(
    events: pd.DataFrame, iv_result, full_result, dest: Path
) -> None:
    """LaTeX table for the IV-only and public-feature OLS regressions."""

    iv_coef, iv_int_coef = float(iv_result.coefficients[1]), float(iv_result.coefficients[0])
    iv_se_slope = float(iv_result.hc3_std_errors[1])
    iv_se_int = float(iv_result.hc3_std_errors[0])

    full_coefs = full_result.coefficients
    full_ses = full_result.hc3_std_errors

    text = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Explaining post-announcement realized variance}\n"
        "\\label{tab:regression}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        " & (1) IV only & (2) Public-feature model \\\\\n"
        "\\midrule\n"
        f"Intercept & {iv_int_coef:.4f} & {full_coefs[0]:.4f} \\\\\n"
        f" & ({iv_se_int:.4f}) & ({full_ses[0]:.4f}) \\\\\n"
        f"Implied variance & {iv_coef:.4f} & {full_coefs[1]:.4f} \\\\\n"
        f" & ({iv_se_slope:.4f}) & ({full_ses[1]:.4f}) \\\\\n"
        f"Pre-event realized variance &  & {full_coefs[2]:.4f} \\\\\n"
        f" &  & ({full_ses[2]:.4f}) \\\\\n"
        f"Five-day VXAPL run-up &  & {full_coefs[3]:.4f} \\\\\n"
        f" &  & ({full_ses[3]:.4f}) \\\\\n"
        f"Prior-four-event mean absolute gap &  & {full_coefs[4]:.4f} \\\\\n"
        f" &  & ({full_ses[4]:.4f}) \\\\\n"
        "\\midrule\n"
        f"Observations & {iv_result.n_obs} & {full_result.n_obs} \\\\\n"
        f"$R^2$ & {iv_result.r_squared:.3f} & {full_result.r_squared:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{minipage}{0.92\\linewidth}\n"
        "\\footnotesize \\textit{Notes:} The dependent variable is annualized "
        "realized variance over the 20 trading days following the announcement. "
        "Parentheses contain HC3 heteroskedasticity-consistent standard errors. "
        "The public-feature model excludes the contemporaneous earnings surprise "
        "because that value is not known when the forecast is formed.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    _write_text(dest, text)


def _write_oos_table(oos_metrics: pd.DataFrame, dest: Path) -> None:
    """LaTeX table summarising walk-forward performance per model."""

    rows = []
    for _, row in oos_metrics.iterrows():
        rows.append(
            f"{row['display_name']} & {row['mae']:.4f} & {row['rmse']:.4f} & "
            f"{row['oos_r_squared']:.3f} & {row['sign_accuracy'] * 100:.1f}\\% \\\\"
        )
    body = "\n".join(rows)
    text = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Walk-forward out-of-sample forecast performance}\n"
        "\\label{tab:oos}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Model & MAE & RMSE & $R^2_{OOS}$ & VRP sign accuracy \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{minipage}{0.92\\linewidth}\n"
        "\\footnotesize \\textit{Notes:} Every forecast uses only earlier "
        "events. $R^2_{OOS}=1-\\sum(y-\\hat y)^2/\\sum(y-\\bar y_{train})^2$. "
        "The regularized model uses a fixed ridge penalty of "
        f"{RIDGE_PENALTY:g} after training-sample standardization.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    _write_text(dest, text)


def _write_horizon_table(events: pd.DataFrame, dest: Path) -> None:
    """Compare the 5-day mismatched horizon with the 20-day matched horizon."""

    vrp5 = events["vrp5"].dropna().to_numpy(dtype=float)
    vrp20 = events["vrp20"].dropna().to_numpy(dtype=float)
    mean_5 = float(np.mean(vrp5)) if vrp5.size else float("nan")
    mean_20 = float(np.mean(vrp20)) if vrp20.size else float("nan")
    pos_5 = float(np.mean(vrp5 > 0) * 100.0) if vrp5.size else float("nan")
    pos_20 = float(np.mean(vrp20 > 0) * 100.0) if vrp20.size else float("nan")

    text = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Horizon alignment and the measured variance spread}\n"
        "\\label{tab:horizon}\n"
        "\\small\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "RV horizon & Mean spread & Positive frequency & Interpretation \\\\\n"
        "\\midrule\n"
        f"5 trading days & {mean_5:.4f} & {pos_5:.1f}\\% & Event shock dominates \\\\\n"
        f"20 trading days & {mean_20:.4f} & {pos_20:.1f}\\% & Approximate horizon match \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{minipage}{0.92\\linewidth}\n"
        "\\footnotesize \\textit{Notes:} Both rows use 30-calendar-day VXAPL "
        "implied variance. The five-day comparison is deliberately horizon-"
        "mismatched and therefore should not be interpreted as a variance "
        "risk premium. It is reported to show how the concentrated earnings "
        "jump can reverse the sign of the spread.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    _write_text(dest, text)


def _write_results_macros(
    events: pd.DataFrame,
    iv_result,
    oos_metrics: pd.DataFrame,
    bootstrap_low: float,
    bootstrap_high: float,
    test_frame: pd.DataFrame,
    dest: Path,
) -> None:
    """Single ``\\newcommand`` file consumed by ``research_paper.tex``."""

    vrp20 = events["vrp20"].dropna().to_numpy(dtype=float)
    vrp5 = events["vrp5"].dropna().to_numpy(dtype=float)
    sample_dates = pd.to_datetime(events["announcement_date"].dropna()).sort_values()
    sample_start = sample_dates.iloc[0].strftime("%B %Y") if not sample_dates.empty else ""
    sample_end = sample_dates.iloc[-1].strftime("%B %Y") if not sample_dates.empty else ""

    test_dates = pd.to_datetime(test_frame["announcement_date"]).sort_values()
    test_start = test_dates.iloc[0].strftime("%B %Y") if not test_dates.empty else ""
    test_end = test_dates.iloc[-1].strftime("%B %Y") if not test_dates.empty else ""

    by_model = oos_metrics.set_index("model")
    mean_rmse = float(by_model.loc["mean", "rmse"])
    iv_rmse = float(by_model.loc["iv_only", "rmse"])
    iv_oos = float(by_model.loc["iv_only", "oos_r_squared"])
    ridge_rmse = float(by_model.loc["ridge", "rmse"])
    ridge_oos = float(by_model.loc["ridge", "oos_r_squared"])

    text = (
        f"\\newcommand{{\\SampleN}}{{{int(events['iv_var'].notna().sum())}}}\n"
        f"\\newcommand{{\\TestN}}{{{int(len(test_frame))}}}\n"
        f"\\newcommand{{\\SampleStart}}{{{sample_start}}}\n"
        f"\\newcommand{{\\SampleEnd}}{{{sample_end}}}\n"
        f"\\newcommand{{\\TestStart}}{{{test_start}}}\n"
        f"\\newcommand{{\\TestEnd}}{{{test_end}}}\n"
        f"\\newcommand{{\\MeanVXAPL}}{{{events['vxapl_implied_vol'].mean():.2f}\\%}}\n"
        f"\\newcommand{{\\MeanRV}}{{{events['post_rv20_vol'].mean():.2f}\\%}}\n"
        f"\\newcommand{{\\MeanVRP}}{{{vrp20.mean():.4f}}}\n"
        f"\\newcommand{{\\VRPCILow}}{{{bootstrap_low:.4f}}}\n"
        f"\\newcommand{{\\VRPCIHigh}}{{{bootstrap_high:.4f}}}\n"
        f"\\newcommand{{\\VRPPositiveRate}}{{{(vrp20 > 0).mean() * 100:.1f}\\%}}\n"
        f"\\newcommand{{\\IVSlope}}{{{iv_result.coefficients[1]:.3f}}}\n"
        f"\\newcommand{{\\IVTstat}}{{{iv_result.t_statistics[1]:.2f}}}\n"
        f"\\newcommand{{\\IVRsquared}}{{{iv_result.r_squared:.3f}}}\n"
        f"\\newcommand{{\\IVOOSRsquared}}{{{iv_oos:.3f}}}\n"
        f"\\newcommand{{\\IVRMSE}}{{{iv_rmse:.4f}}}\n"
        f"\\newcommand{{\\MeanRMSE}}{{{mean_rmse:.4f}}}\n"
        f"\\newcommand{{\\RidgeRMSE}}{{{ridge_rmse:.4f}}}\n"
        f"\\newcommand{{\\RidgeOOSRsquared}}{{{ridge_oos:.3f}}}\n"
        f"\\newcommand{{\\FiveDayVRP}}{{{vrp5.mean():.4f}}}\n"
    )
    _write_text(dest, text)


# ---------------------------------------------------------------------------
# Helper that turns walk-forward predictions into a tidy metrics dataframe.
# ---------------------------------------------------------------------------


def _summarize_walk_forward(test_frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-model MAE/RMSE/R^2/sign-accuracy as a long-form table."""

    actual = test_frame["actual_rv"].to_numpy(dtype=float)
    iv_var = test_frame["iv_var"].to_numpy(dtype=float)
    benchmark = test_frame["mean_pred"].to_numpy(dtype=float)

    rows = []
    for model_key, display_name, column in (
        ("mean", "Expanding historical mean", "mean_pred"),
        ("iv_only", "Implied-variance OLS", "iv_only_pred"),
        ("ridge", "Regularized public-feature model", "ridge_pred"),
    ):
        predictions = test_frame[column].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_key,
                "display_name": display_name,
                "mae": mean_absolute_error(actual, predictions),
                "rmse": root_mean_squared_error(actual, predictions),
                "oos_r_squared": out_of_sample_r_squared(actual, predictions, benchmark),
                "forecast_correlation": forecast_correlation(actual, predictions),
                "sign_accuracy": variance_spread_sign_accuracy(iv_var, predictions, actual),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(data_path: Path, output_dir: Path) -> dict[str, object]:
    """Execute every step and write all required artefacts.

    Returns a dictionary of in-memory objects so the function is also usable
    from notebooks and integration tests.
    """

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = add_log_returns(load_daily_panel(data_path))
    events = build_event_table(panel)

    # ------------------------------------------------------------------
    # Descriptive statistics and bootstrap interval for vrp20.
    # ------------------------------------------------------------------
    vrp20 = events["vrp20"].dropna().to_numpy(dtype=float)
    ci_low, ci_high, _ = event_bootstrap_mean_ci(vrp20, n_resamples=20_000, seed=42)

    # ------------------------------------------------------------------
    # Full-sample regressions.
    # ------------------------------------------------------------------
    iv_only = events.dropna(subset=["iv_var", "post_rv20_var"])
    iv_result = fit_ols(
        iv_only["post_rv20_var"].to_numpy(dtype=float),
        iv_only["iv_var"].to_numpy(dtype=float),
    )

    multi_cols = list(RIDGE_PREDICTORS)
    multivariate = events.dropna(subset=["post_rv20_var", *multi_cols])
    multi_result = fit_ols(
        multivariate["post_rv20_var"].to_numpy(dtype=float),
        multivariate[multi_cols].to_numpy(dtype=float),
    )

    # ------------------------------------------------------------------
    # Walk-forward forecasts.
    # ------------------------------------------------------------------
    wf = run_walk_forward(events)
    oos_metrics = _summarize_walk_forward(wf.test_frame)

    # ------------------------------------------------------------------
    # CSV outputs in the requested ``results/`` directory.
    # ------------------------------------------------------------------
    event_csv = output_dir / "event_level_results.csv"
    wf_csv = output_dir / "walk_forward_predictions.csv"
    summary_csv = output_dir / "summary_metrics.csv"

    events.to_csv(event_csv, index=False)
    wf.test_frame.to_csv(wf_csv, index=False)

    summary_rows = [
        {"metric": "n_events", "value": int(events["iv_var"].notna().sum())},
        {"metric": "n_test_events", "value": int(len(wf.test_frame))},
        {"metric": "n_initial_train", "value": int(wf.n_train)},
        {"metric": "mean_vxapl_pct", "value": float(events["vxapl_implied_vol"].mean())},
        {"metric": "mean_post_rv20_vol_pct", "value": float(events["post_rv20_vol"].mean())},
        {"metric": "mean_vrp20", "value": float(np.mean(vrp20))},
        {"metric": "positive_vrp20_share", "value": float(np.mean(vrp20 > 0))},
        {"metric": "vrp20_ci_low_95", "value": ci_low},
        {"metric": "vrp20_ci_high_95", "value": ci_high},
        {"metric": "iv_only_slope", "value": float(iv_result.coefficients[1])},
        {"metric": "iv_only_hc3_t", "value": float(iv_result.t_statistics[1])},
        {"metric": "iv_only_r_squared", "value": float(iv_result.r_squared)},
        {"metric": "multivariate_r_squared", "value": float(multi_result.r_squared)},
        {"metric": "ridge_lambda", "value": float(wf.ridge_lambda)},
    ]
    for _, row in oos_metrics.iterrows():
        prefix = row["model"]
        summary_rows.extend([
            {"metric": f"{prefix}_mae", "value": float(row["mae"])},
            {"metric": f"{prefix}_rmse", "value": float(row["rmse"])},
            {"metric": f"{prefix}_oos_r_squared", "value": float(row["oos_r_squared"])},
            {"metric": f"{prefix}_forecast_correlation", "value": float(row["forecast_correlation"])},
            {"metric": f"{prefix}_sign_accuracy", "value": float(row["sign_accuracy"])},
        ])
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot_iv_vs_realized(events, REPORT_FIGURES / "iv_vs_realized.pdf")
    plot_iv_rv_scatter(events, REPORT_FIGURES / "iv_rv_scatter.pdf")
    plot_vrp_distribution(events, REPORT_FIGURES / "vrp_distribution.pdf")
    plot_walk_forward_forecasts(wf.test_frame, REPORT_FIGURES / "walk_forward_forecasts.pdf")

    # ------------------------------------------------------------------
    # LaTeX assets so ``research_paper.tex`` compiles without manual fixes.
    # ------------------------------------------------------------------
    REPORT_GENERATED.mkdir(parents=True, exist_ok=True)
    _write_summary_table(events, REPORT_GENERATED / "summary_table.tex")
    _write_regression_table(events, iv_result, multi_result,
                            REPORT_GENERATED / "regression_table.tex")
    _write_oos_table(oos_metrics, REPORT_GENERATED / "oos_table.tex")
    _write_horizon_table(events, REPORT_GENERATED / "horizon_table.tex")
    _write_results_macros(events, iv_result, oos_metrics, ci_low, ci_high,
                          wf.test_frame, REPORT_GENERATED / "results_macros.tex")

    return {
        "events": events,
        "iv_only_ols": iv_result,
        "multivariate_ols": multi_result,
        "walk_forward": wf,
        "oos_metrics": oos_metrics,
        "bootstrap_ci": (ci_low, ci_high),
    }


def _print_console_summary(results: dict[str, object]) -> None:
    """Echo the validation targets so the user can eyeball the run."""

    events = results["events"]
    iv_result = results["iv_only_ols"]
    oos = results["oos_metrics"].set_index("model")
    ci_low, ci_high = results["bootstrap_ci"]
    n_events = int(events["iv_var"].notna().sum())
    n_test = int(len(results["walk_forward"].test_frame))
    vrp = events["vrp20"].dropna().to_numpy()

    print("\n=== Apple earnings-volatility pipeline summary ===")
    print(f"Earnings events:                  {n_events}")
    print(f"Walk-forward test events:         {n_test}")
    print(f"Mean VXAPL:                       {events['vxapl_implied_vol'].mean():.4f}%")
    print(f"Mean 20-day realized volatility:  {events['post_rv20_vol'].mean():.4f}%")
    print(f"Mean vrp20:                       {vrp.mean():.5f}")
    print(f"Positive vrp20 frequency:         {(vrp > 0).mean() * 100:.1f}%")
    print(f"Bootstrap 95% interval:           [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"IV-only full-sample slope:        {iv_result.coefficients[1]:.4f}")
    print(f"HC3 t-statistic:                  {iv_result.t_statistics[1]:.3f}")
    print(f"Full-sample R-squared:            {iv_result.r_squared:.4f}")
    print(f"IV-only RMSE:                     {oos.loc['iv_only', 'rmse']:.5f}")
    print(f"IV-only OOS R-squared:            {oos.loc['iv_only', 'oos_r_squared']:.4f}")
    print(f"IV-only spread-sign accuracy:     {oos.loc['iv_only', 'sign_accuracy'] * 100:.1f}%")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argument parser shared by ``__main__`` and the test harness."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Apple earnings-volatility research pipeline using only "
            "the prepared CSV input."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "aapl_earnings_volatility_data.csv",
        help="Path to the prepared daily CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory in which to write CSV result tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = run_pipeline(args.data, args.output)
    _print_console_summary(results)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
