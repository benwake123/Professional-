# Changelog

## Research audit and engine overhaul (2026-06-24)

### Accounting and P&L reconciliation

- Added `src/accounting.py` with an explicit reconciliation identity:
  `ending_equity ≈ initial_equity + realized_option_pnl + unrealized_option_pnl + hedge_pnl - commissions - slippage`.
- Fixed the root cause of the reported trade-P&L vs equity mismatch: trade statistics were option-only while delta-hedge P&L flowed through portfolio cash/equity separately.
- Execution fills now use ask/bid plus optional spread slippage; slippage and commissions are subtracted once in `net_cash_flow`.
- Open positions mark to mid (or intrinsic at expiry); closing removes mark-to-market value from equity.
- `src/metrics.py` reports option P&L, hedge P&L, and net P&L separately.

### Audit outputs (Part 1)

- Added `src/audit.py` with decision funnel tracking, trade audit rows, rejection summaries, and accounting snapshots.
- Pipeline exports:
  - `results/decision_funnel.csv`
  - `results/trade_audit.csv`
  - `results/accounting_reconciliation.csv`
  - `results/forecast_diagnostics.csv`
  - `results/rejection_summary.csv`
  - `results/signal_direction_diagnostics.csv`
  - `results/liquidity_sensitivity.csv`
  - `results/parameter_search.csv`

### Forecasting framework (Part 2)

- Added `src/forecast_engine.py` with historical, EWMA, GBM, Heston, and ensemble forecasts.
- Ensemble weights are configurable; Heston is excluded automatically when calibration fails validation.
- Forecast diagnostics compare predicted variance to subsequently realized variance over a matched horizon.

### Trading frequency and contract rules (Part 3)

- Daily business-day decisions (`decision_frequency: "B"`) with weekly Heston recalibration (`heston_calibration_frequency: "W-MON"`).
- Expanded contract band: DTE 14–45, target 30, ATM moneyness 0.98–1.02.
- Added trade cooldown, duplicate-structure guard, and `maximum_open_positions`.

### Signals, exits, hedging, sizing (Parts 4–6)

- Signals use `variance_edge = forecast - implied`; separate long/short z thresholds.
- Long vol → ATM straddle; short vol → iron butterfly/condor (no uncovered short straddle).
- Edge-after-costs gate now scales expected gross edge relative to structure premium notional (fixes a unit bug that blocked all entries).
- Delta hedge modes: `none`, `daily`, `threshold`.
- Volatility-targeted, max-loss-capped position sizing in `src/position_sizing.py`.

### Research workflow (Part 7)

- Fixed splits: development 2014–2018, validation 2019–2021, final test 2022–2023.
- `src/research.py` runs walk-forward parameter search on dev/validation only; final test period stays untouched until parameters are fixed.

### Tests (Part 8)

- Added `tests/test_accounting.py` for cash-flow conventions, reconciliation, variance-edge direction, and hand-calculated trade P&L.
- Updated execution, signal, and backtest tests for new APIs and audit behavior.

### Configuration

- `config.json` updated with daily decisions, ensemble defaults, separate thresholds, exit rules, delta-hedge settings, and expanded option filters.
