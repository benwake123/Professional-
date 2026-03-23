from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_performance_plots(bt: pd.DataFrame, output_dir: str = "outputs") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = bt.copy().sort_values("date")
    data["date"] = pd.to_datetime(data["date"])
    data["equity_curve"] = (1 + data["net_ret_vt"]).cumprod()
    data["drawdown"] = data["equity_curve"] / data["equity_curve"].cummax() - 1
    data["rolling_sharpe_20d"] = (
        data["net_ret_vt"].rolling(20).mean()
        / (data["net_ret_vt"].rolling(20).std() + 1e-12)
    ) * (252 ** 0.5)

    # Cumulative return / equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(data["date"], data["equity_curve"], linewidth=1.5)
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    # Drawdown
    plt.figure(figsize=(10, 4))
    plt.plot(data["date"], data["drawdown"], linewidth=1.2)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close()

    # Rolling Sharpe
    plt.figure(figsize=(10, 4))
    plt.plot(data["date"], data["rolling_sharpe_20d"], linewidth=1.2)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.title("Rolling Sharpe (20D)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_sharpe_20d.png", dpi=150)
    plt.close()

