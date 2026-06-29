"""Figure generation for the research paper.

Each function takes the prepared frames produced by ``run_analysis`` and
writes one PDF (and one PNG twin for quick previewing) into
``report/figures``.  ``matplotlib`` runs in the ``Agg`` backend so the
analysis is fully scriptable on a headless CI runner.

The figures follow the structure described in the paper:

* ``iv_vs_realized.pdf``     time series of VXAPL implied vol and the
  matched 20-day realized vol at every event.
* ``iv_rv_scatter.pdf``      scatter plot of implied vs realized variance
  with a 45-degree reference line.
* ``vrp_distribution.pdf``   histogram of the matched-horizon variance
  spread together with the sample mean.
* ``walk_forward_forecasts.pdf``  realized variance against the three
  out-of-sample forecast paths.

The plotting functions only depend on ``matplotlib``; no seaborn, no
plotly, no extra packages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for CLI and CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_directory(path: Path) -> None:
    """Create the parent directory of ``path`` if it does not exist yet."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, pdf_path: Path) -> None:
    """Save ``fig`` as PDF and a matching PNG for fast preview."""

    _ensure_directory(pdf_path)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(png_path, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_iv_vs_realized(events: pd.DataFrame, pdf_path: Path) -> None:
    """Time-series view of implied and realized volatility at each event."""

    fig, ax = plt.subplots(figsize=(10, 5))
    dates = pd.to_datetime(events["announcement_date"])
    ax.plot(
        dates,
        events["vxapl_implied_vol"],
        marker="o",
        linestyle="-",
        color="#1f4e79",
        label="VXAPL implied vol (%)",
    )
    ax.plot(
        dates,
        events["post_rv20_vol"],
        marker="s",
        linestyle="--",
        color="#c0504d",
        label="Post-event 20-day realized vol (%)",
    )
    ax.set_title("Implied vs realized volatility at Apple earnings events")
    ax.set_xlabel("Announcement date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, pdf_path)


def plot_iv_rv_scatter(events: pd.DataFrame, pdf_path: Path) -> None:
    """Scatter of implied variance against subsequent realized variance."""

    iv = events["iv_var"].to_numpy(dtype=float)
    rv = events["post_rv20_var"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(iv, rv, color="#1f4e79", alpha=0.85, edgecolor="white", s=60)

    # 45-degree reference line over a small padded range.
    if iv.size > 0:
        lo = float(min(iv.min(), rv.min()))
        hi = float(max(iv.max(), rv.max()))
        span = (hi - lo) * 0.05
        ax.plot([lo - span, hi + span], [lo - span, hi + span], "--", color="grey",
                label="45-degree line")
    ax.set_xlabel("Implied variance (VXAPL/100)^2")
    ax.set_ylabel("Realized variance (next 20 days, annualized)")
    ax.set_title("Implied variance vs post-event realized variance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    _save(fig, pdf_path)


def plot_vrp_distribution(events: pd.DataFrame, pdf_path: Path) -> None:
    """Histogram of the matched-horizon variance spread."""

    vrp = events["vrp20"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(vrp, bins=12, color="#1f4e79", alpha=0.85, edgecolor="white")
    mean_value = float(np.mean(vrp)) if vrp.size else float("nan")
    ax.axvline(mean_value, color="#c0504d", linestyle="--",
               label=f"Mean = {mean_value:.4f}")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Implied variance minus realized variance")
    ax.set_ylabel("Number of earnings events")
    ax.set_title("Distribution of the matched-horizon variance spread")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, pdf_path)


def plot_walk_forward_forecasts(test_frame: pd.DataFrame, pdf_path: Path) -> None:
    """Side-by-side actual vs three model forecasts."""

    dates = pd.to_datetime(test_frame["announcement_date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, test_frame["actual_rv"], color="black", marker="o",
            linestyle="-", linewidth=2, label="Actual realized variance")
    ax.plot(dates, test_frame["mean_pred"], color="#999999", marker="x",
            linestyle="--", label="Expanding historical mean")
    ax.plot(dates, test_frame["iv_only_pred"], color="#1f4e79", marker="s",
            linestyle="--", label="IV-only OLS")
    ax.plot(dates, test_frame["ridge_pred"], color="#c0504d", marker="^",
            linestyle="--", label="Ridge public-feature model")
    ax.set_title("Walk-forward forecasts of realized variance")
    ax.set_xlabel("Announcement date")
    ax.set_ylabel("Realized variance (next 20 days, annualized)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, pdf_path)
