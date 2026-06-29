# Regime-Aware Stochastic-Volatility Options Strategy

This repository is an **implementation skeleton**. It contains the complete project layout, function names, responsibilities, and call relationships, but no trading-model implementation.

## Research question

Can GBM and Heston forecasts of future SPY realized variance identify dislocations relative to option-implied variance, and can those dislocations be traded with delta-hedged option structures after liquidity filters, transaction costs, and regime-aware risk controls?

## Implementation order

1. Configuration and data loading
2. Data validation and look-ahead tests
3. Market features
4. Black--Scholes and Greeks
5. GBM benchmark
6. Heston model and calibration
7. Monte Carlo forecasting
8. Regime classification and signal generation
9. Option selection
10. Position sizing, risk, execution, and hedging
11. Portfolio accounting
12. Walk-forward backtest
13. Attribution, metrics, and plots
14. End-to-end pipeline and tests

Read `docs/FUNCTION_MAP.md` and `docs/CALL_GRAPH.md` before writing code.

Intended command after implementation:

```bash
python -m src.run_pipeline --config config.json
```

Do not report performance until the untouched test period has been run with realistic bid/ask fills, commissions, and hedge costs.
