from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score

from backtest.engine import apply_risk_controls, run_backtest, signal_from_preds
from features.feature_engineering import build_features
from models.train import get_model
from utils.data_loader import download_ohlcv, save_parquet
from utils.metrics import annual_return, max_drawdown, sharpe, sortino
from utils.plotting import save_performance_plots
from utils.preprocessing import fit_transform_scaler, split_walk_forward


def main():
    cfg_path = Path("config/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    raw = download_ohlcv(cfg["tickers"], cfg["start_date"], cfg["end_date"])
    if raw.empty:
        raise RuntimeError("No market data downloaded. Check internet/tickers/date range.")
    save_parquet(raw, "data/raw/ohlcv.parquet")
    raw.to_csv("data/raw/ohlcv.csv", index=False)

    df = build_features(raw)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "ticker", "open", "high", "low", "close", "volume", "target"]
    ]
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    windows = split_walk_forward(
        df["date"].unique(),
        train_years=cfg["train_years"],
        val_months=cfg["val_months"],
        test_months=cfg["test_months"],
    )
    if not windows:
        raise RuntimeError("No walk-forward windows created. Increase date range or reduce split sizes.")

    all_test = []
    for tr_dates, _va_dates, te_dates in windows:
        tr = df[df["date"].isin(tr_dates)]
        te = df[df["date"].isin(te_dates)]
        if tr.empty or te.empty:
            continue

        x_tr, x_te = fit_transform_scaler(tr, te, feature_cols)
        y_tr = tr["target"].values

        preds = []
        for model_name in cfg["models"]:
            model = get_model(model_name, seed=cfg["seed"])
            model.fit(x_tr, y_tr)
            preds.append(model.predict(x_te))

        te_out = te[["date", "ticker", "target"]].copy()
        te_out["pred"] = pd.DataFrame(preds).mean(axis=0).values
        all_test.append(te_out)

    pred_df = pd.concat(all_test, ignore_index=True).sort_values(["date", "ticker"])
    save_parquet(pred_df, "data/processed/predictions.parquet")
    pred_df.to_csv("data/processed/predictions.csv", index=False)

    rmse = mean_squared_error(pred_df["target"], pred_df["pred"]) ** 0.5
    r2 = r2_score(pred_df["target"], pred_df["pred"])

    signal_df = signal_from_preds(
        pred_df,
        long_q=cfg["long_quantile"],
        short_q=cfg["short_quantile"],
    )
    signal_df = apply_risk_controls(signal_df, max_weight=cfg["max_weight"])
    bt = run_backtest(
        signal_df,
        tc_bps=cfg["transaction_cost_bps"],
        target_annual_vol=cfg["target_annual_vol"],
    )
    save_parquet(bt, "data/processed/backtest_daily.parquet")
    bt.to_csv("data/processed/backtest_daily.csv", index=False)
    save_performance_plots(bt, output_dir="outputs")

    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Sharpe: {sharpe(bt['net_ret_vt']):.3f}")
    print(f"Sortino: {sortino(bt['net_ret_vt']):.3f}")
    print(f"Max Drawdown: {max_drawdown(bt['net_ret_vt']):.3%}")
    print(f"Annual Return: {annual_return(bt['net_ret_vt']):.3%}")
    print("Saved:")
    print("- data/raw/ohlcv.parquet")
    print("- data/raw/ohlcv.csv")
    print("- data/processed/predictions.parquet")
    print("- data/processed/predictions.csv")
    print("- data/processed/backtest_daily.parquet")
    print("- data/processed/backtest_daily.csv")
    print("- outputs/equity_curve.png")
    print("- outputs/drawdown.png")
    print("- outputs/rolling_sharpe_20d.png")


if __name__ == "__main__":
    main()
