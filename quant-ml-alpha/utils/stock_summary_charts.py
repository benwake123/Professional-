from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    return float(a.corr(b))


def generate_stock_summary_charts(
    predictions_csv: str = "data/processed/predictions.csv",
    output_dir: str = "outputs/stocks",
) -> list[str]:
    pred_path = Path(predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    required_cols = {"date", "ticker", "target", "pred"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"Missing required columns in predictions CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    written_files: list[str] = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy().sort_values("date")

        # Signal strategy for a single stock: long if pred > 0, short if pred < 0.
        g["position"] = np.sign(g["pred"])
        g["strategy_ret"] = g["position"] * g["target"]
        g["cum_stock"] = (1 + g["target"]).cumprod()
        g["cum_strategy"] = (1 + g["strategy_ret"]).cumprod()
        g["rolling_corr_20d"] = g["pred"].rolling(20).corr(g["target"])

        rmse = float(np.sqrt(np.mean((g["pred"] - g["target"]) ** 2)))
        corr = _safe_corr(g["pred"], g["target"])
        hit_rate = float((np.sign(g["pred"]) == np.sign(g["target"])).mean())
        avg_target = float(g["target"].mean())
        avg_pred = float(g["pred"].mean())

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{ticker} Model Summary", fontsize=14, fontweight="bold")

        # 1) Cumulative returns
        axes[0, 0].plot(g["date"], g["cum_stock"], label="Buy & Hold", linewidth=1.6)
        axes[0, 0].plot(g["date"], g["cum_strategy"], label="Sign(pred) Strategy", linewidth=1.6)
        axes[0, 0].set_title("Cumulative Performance")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Growth of $1")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()

        # 2) Prediction vs realized scatter
        axes[0, 1].scatter(g["pred"], g["target"], alpha=0.5, s=14)
        axes[0, 1].axhline(0.0, linestyle="--", linewidth=1)
        axes[0, 1].axvline(0.0, linestyle="--", linewidth=1)
        axes[0, 1].set_title("Prediction vs Realized Return")
        axes[0, 1].set_xlabel("Predicted Return")
        axes[0, 1].set_ylabel("Realized Return")
        axes[0, 1].grid(alpha=0.25)

        # 3) Rolling correlation
        axes[1, 0].plot(g["date"], g["rolling_corr_20d"], linewidth=1.6)
        axes[1, 0].axhline(0.0, linestyle="--", linewidth=1)
        axes[1, 0].set_title("Rolling Correlation (20D)")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Corr(pred, target)")
        axes[1, 0].grid(alpha=0.3)

        # 4) Metrics panel
        axes[1, 1].axis("off")
        text = (
            f"Observations: {len(g)}\n"
            f"RMSE: {rmse:.6f}\n"
            f"Corr(pred,target): {corr:.4f}\n"
            f"Hit rate (sign): {hit_rate:.2%}\n"
            f"Avg predicted return: {avg_pred:.6f}\n"
            f"Avg realized return: {avg_target:.6f}\n"
            f"Final buy&hold multiple: {g['cum_stock'].iloc[-1]:.3f}\n"
            f"Final strategy multiple: {g['cum_strategy'].iloc[-1]:.3f}"
        )
        axes[1, 1].text(0.02, 0.98, text, va="top", ha="left", fontsize=10)
        axes[1, 1].set_title("Summary Metrics")

        fig.tight_layout()
        output_file = out_dir / f"{ticker}_summary.png"
        fig.savefig(output_file, dpi=160)
        plt.close(fig)
        written_files.append(str(output_file))

    return written_files


if __name__ == "__main__":
    files = generate_stock_summary_charts()
    print("Generated stock charts:")
    for f in files:
        print(f"- {f}")
