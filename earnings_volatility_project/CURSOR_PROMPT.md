# Cursor implementation prompt

Build a complete, reproducible Python research repository for the project **“Can Earnings Volatility Be Timed? Forecasting Apple’s Implied–Realized Variance Spread with Public Data.”**

## Non-negotiable constraints

- Read data only from `data/aapl_earnings_volatility_data.csv`. Do not call APIs, scrape websites, or download data.
- Use only `pandas`, `numpy`, and `matplotlib`, plus Python’s standard library. Do not use `yfinance`, `scipy`, `statsmodels`, `scikit-learn`, or `cvxpy`.
- Write clear beginner-readable code with type hints, docstrings, and comments explaining the financial reasoning.
- Never use random train/test splits. Preserve chronological order and prevent look-ahead bias.
- Treat all variance-strategy outputs as **non-tradable variance-payoff proxies**, not actual option returns, because VXAPL is an index and the CSV does not contain option contracts or bid–ask quotes.

## Repository structure

```text
earnings-volatility-research/
├── README.md
├── requirements.txt
├── research_log.md
├── data/
│   ├── aapl_earnings_volatility_data.csv
│   └── DATA_DICTIONARY.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── regression.py
│   ├── forecasting.py
│   ├── metrics.py
│   ├── plots.py
│   └── run_analysis.py
├── tests/
│   ├── test_features.py
│   └── test_no_lookahead.py
├── results/
└── report/
    ├── research_paper.tex
    ├── figures/
    └── generated/
```

## Event construction and formulas

1. Parse `date`, sort ascending, reject duplicate dates, and verify positive prices.
2. Calculate daily log returns from `aapl_close`:
   `r_t = log(P_t / P_{t-1})`.
3. Keep rows where `earnings_flag == 1` as events.
4. At each event date `t`, calculate:
   - `iv_var = (vxapl_close / 100)^2`
   - `pre_rv20_var = (252/20) * sum(r^2)` over the 20 returns ending at `t`
   - `post_rv20_var = (252/20) * sum(r^2)` over the next 20 returns, `t+1` through `t+20`
   - `post_rv5_var = (252/5) * sum(r^2)` over `t+1` through `t+5`
   - `vrp20 = iv_var - post_rv20_var`
   - `vrp5 = iv_var - post_rv5_var`
   - `iv_runup_5 = vxapl_close_t / vxapl_close_{t-5} - 1`
   - `market_term_slope = (vix_close - vix9d_close) / 100`
   - `event_gap = log(aapl_open_{t+1} / aapl_close_t)`
5. Create `hist_abs_gap_4` as the mean absolute event gap from the **previous four earnings events only**. Use an explicit one-event shift before rolling.
6. Create `hist_abs_eps_surprise_4` from the previous four events only. Keep the current event’s `eps_actual` and `eps_surprise_pct` for descriptive output, but never use them as predictors.

## Statistical analysis

Implement the following from NumPy rather than external modeling packages:

- Event summary statistics.
- A 20,000-resample event bootstrap, seed 42, for the mean `vrp20` confidence interval.
- OLS with an intercept using `numpy.linalg.lstsq`.
- HC3 heteroskedasticity-robust standard errors using the hat-matrix leverage values.
- Full-sample regression: `post_rv20_var ~ 1 + iv_var`.
- An additional descriptive multivariate regression using `iv_var`, `pre_rv20_var`, `iv_runup_5`, and `hist_abs_gap_4`.

## Strict walk-forward forecasts

Use the first 16 feature-complete events as the initial training window. For every later event, fit using prior events only and then forecast that one event.

Compare:

1. **Expanding mean:** training-sample mean of `post_rv20_var`.
2. **IV-only OLS:** expanding OLS using only `iv_var`.
3. **Regularized public-feature model:** ridge regression using `iv_var`, `pre_rv20_var`, `iv_runup_5`, and `hist_abs_gap_4`.

For ridge:

- Standardize each feature using training-window means and standard deviations only.
- Include an unpenalized intercept.
- Use the closed-form NumPy solution.
- Fix `lambda = 10`; do not tune it on the test sample.
- Handle zero-variance columns safely.

Report MAE, RMSE, forecast correlation, out-of-sample R-squared relative to the expanding-mean forecasts, and variance-spread sign accuracy. Define sign accuracy by comparing `sign(iv_var - predicted_rv)` with `sign(iv_var - actual_rv)`.

## Required outputs

Running

```bash
python -m src.run_analysis --data data/aapl_earnings_volatility_data.csv --output results
```

must create:

- `results/event_level_results.csv`
- `results/walk_forward_predictions.csv`
- `results/summary_metrics.csv`
- `report/figures/iv_vs_realized.pdf`
- `report/figures/iv_rv_scatter.pdf`
- `report/figures/vrp_distribution.pdf`
- `report/figures/walk_forward_forecasts.pdf`
- `report/generated/summary_table.tex`
- `report/generated/regression_table.tex`
- `report/generated/oos_table.tex`
- `report/generated/horizon_table.tex`
- `report/generated/results_macros.tex`

Copy the supplied `report/research_paper.tex` into the repository and make the generated filenames compatible with its `\input{}` and figure commands.

## Validation targets

Results should be close to these reference values, allowing small floating-point differences:

- 40 earnings events
- mean VXAPL: 33.8278%
- mean 20-day realized volatility: 27.8762%
- mean `vrp20`: 0.03277
- positive `vrp20`: 70.0%
- bootstrap 95% interval: approximately [0.01499, 0.05000]
- IV-only full-sample slope: 0.5944
- IV-only HC3 t-statistic: 2.695
- full-sample R-squared: 0.2596
- 23 walk-forward test events
- IV-only test RMSE: approximately 0.06662
- IV-only OOS R-squared: approximately 0.1150
- IV-only spread-sign accuracy: approximately 73.9%

## Tests and documentation

Add tests that prove:

- future prices are not included in pre-event features;
- rolling historical event features exclude the current event;
- changing data after a forecast date cannot change that date’s training features or prediction;
- events and forecasts remain chronologically sorted.

The README must explain setup, formulas, output files, limitations, and the distinction between a variance forecast and executable option P&L. The research log must list the fixed hypotheses, model choices, and validation targets before presenting results. Favor transparent functions over clever abstractions, and raise informative errors when required columns or sufficient history are missing.
