# Apple earnings-volatility research project

Companion code for the paper **"Can Earnings Volatility Be Timed?
Forecasting Apple's Implied-Realized Variance Spread with Public Data"**.

The project quantifies how informative Apple's option-implied volatility
(VXAPL) is about realized variance around scheduled quarterly earnings,
and whether a small set of public conditioning variables improves an
implied-variance-only forecast in a strictly walk-forward exercise.

## Research question

> Can Apple option-implied volatility forecast realized variance around
> quarterly earnings, and can a small set of public variables improve
> that forecast out of sample?

## What is and what is not

* The pipeline reads one CSV: ``data/aapl_earnings_volatility_data.csv``.
* It never connects to the internet, never scrapes vendor sites, and uses
  only ``numpy``, ``pandas``, ``matplotlib`` and the Python standard
  library.  ``scikit-learn``, ``statsmodels``, ``yfinance``, ``scipy``
  and ``cvxpy`` are deliberately absent.
* The empirical output is a **variance forecast**, evaluated through
  forecast error and a non-tradable variance-spread proxy.  It is not an
  option-trading backtest: VXAPL is an index, the CSV has no individual
  option quotes, no bid-ask spreads, no expirations, no Greeks, no open
  interest, and no liquidity filters.  Any quantity that mentions ``IV -
  RV`` is labelled a variance-payoff proxy in the code and in the paper.

## Repository layout

```text
earnings-volatility-research/
├── README.md
├── requirements.txt
├── research_log.md
├── data/
│   ├── aapl_earnings_volatility_data.csv     # prepared daily input
│   ├── data_sources.csv                      # provenance
│   ├── reference_event_results.csv           # validation reference numbers
│   └── DATA_DICTIONARY.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # parse + validate the daily CSV
│   ├── features.py        # build the event-level feature table
│   ├── regression.py      # OLS + HC3 SEs + ridge in pure NumPy
│   ├── forecasting.py     # expanding-window walk-forward forecasts
│   ├── metrics.py         # bootstrap, MAE, RMSE, R^2_OOS, sign accuracy
│   ├── plots.py           # matplotlib figure writers
│   └── run_analysis.py    # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   └── test_no_lookahead.py
├── results/               # auto-generated CSV outputs
└── report/
    ├── research_paper.tex
    ├── references.bib
    ├── figures/
    └── generated/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python -m src.run_analysis \
    --data data/aapl_earnings_volatility_data.csv \
    --output results
```

This single command writes:

* CSV result tables to ``results/``
* PDF/PNG figures to ``report/figures/``
* LaTeX tables and ``\newcommand`` macros to ``report/generated/``

so that ``pdflatex report/research_paper.tex`` compiles without manual
intervention.

## Run the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The suite verifies, among other things, that

* pre-event features never look forward in time,
* historical-event features exclude the current event,
* modifying data after a forecast date leaves the prior forecasts
  identical, and
* the event and prediction tables are chronologically sorted.

## Formulas implemented in the code

At every earnings date ``t``:

```text
log_return_t      = log(aapl_close_t / aapl_close_{t-1})

iv_var            = (vxapl_close_t / 100) ** 2

pre_rv20_var      = (252 / 20)
                    * sum_{i = t-19..t} log_return_i ** 2

post_rv20_var     = (252 / 20)
                    * sum_{i = t+1..t+20} log_return_i ** 2

post_rv5_var      = (252 / 5)
                    * sum_{i = t+1..t+5}  log_return_i ** 2

vrp20             = iv_var - post_rv20_var
vrp5              = iv_var - post_rv5_var

iv_runup_5        = vxapl_close_t / vxapl_close_{t-5} - 1

market_term_slope = (vix_close_t - vix9d_close_t) / 100

event_gap_t       = log(aapl_open_{t+1} / aapl_close_t)
```

The two historical-event features average earlier observations only:

```text
hist_abs_gap_4_t         = mean( |event_gap| over the previous 4 events )
hist_abs_eps_surprise_4  = mean( |eps_surprise_pct| over the previous 4 events )
```

Walk-forward forecasts compare three models at every test event:

1. **Expanding historical mean** of ``post_rv20_var``.
2. **IV-only OLS**: ``post_rv20_var ~ 1 + iv_var``.
3. **Ridge** on the standardized predictors
   ``[iv_var, pre_rv20_var, iv_runup_5, hist_abs_gap_4]`` with a fixed
   penalty ``lambda = 10`` and an unpenalized intercept.

The first 16 feature-complete events are the initial training window;
the next 23 events are scored.  Sign accuracy uses

```text
sign(iv_var - predicted_rv) == sign(iv_var - actual_rv).
```

## Output files

### ``results/``

| File | Description |
|---|---|
| ``event_level_results.csv`` | One row per Apple earnings event with every feature and outcome described above. |
| ``walk_forward_predictions.csv`` | Test events with each model's prediction and the realized variance. |
| ``summary_metrics.csv`` | Long-form table of scalar metrics (mean VRP, bootstrap interval, OLS slope, MAE/RMSE/R^2 etc.). |

### ``report/figures/``

* ``iv_vs_realized.pdf`` - VXAPL implied vol vs post-event realized vol
* ``iv_rv_scatter.pdf`` - implied variance against realized variance
* ``vrp_distribution.pdf`` - histogram of the matched-horizon variance spread
* ``walk_forward_forecasts.pdf`` - actual RV against each model's predictions

### ``report/generated/``

* ``summary_table.tex`` - descriptive statistics
* ``regression_table.tex`` - IV-only and public-feature OLS
* ``oos_table.tex`` - walk-forward MAE / RMSE / R^2_OOS / sign accuracy
* ``horizon_table.tex`` - 5-day vs 20-day horizon-mismatch comparison
* ``results_macros.tex`` - ``\newcommand`` definitions consumed by ``research_paper.tex``

## Reference validation numbers

Running the pipeline against the provided CSV should reproduce, up to
floating-point differences:

| Quantity | Value |
|---|---|
| Earnings events | 40 |
| Mean VXAPL | 33.83% |
| Mean post-event 20-day realized vol | 27.88% |
| Mean ``vrp20`` | 0.0328 |
| Positive ``vrp20`` share | 70.0% |
| Bootstrap 95% CI for mean ``vrp20`` | [0.0150, 0.0500] |
| IV-only OLS slope | 0.5944 |
| HC3 t-statistic | 2.70 |
| Full-sample R^2 | 0.260 |
| Walk-forward test events | 23 |
| IV-only RMSE | 0.0666 |
| IV-only OOS R^2 | 0.115 |
| IV-only spread-sign accuracy | 73.9% |

## Why these are variance forecasts, not option-strategy returns

The dataset and the methodology deliberately stop at variance.  A
defensible option-return study requires:

* contract-level bid and ask quotes,
* exact expirations bracketing the event,
* strikes (especially the OTM put skew),
* open interest and volume to filter stale or illiquid contracts,
* delta-hedging frictions, financing costs and margin treatment.

None of these are in the CSV.  The Cboe Apple VIX Index summarizes
30-day implied volatility but is itself not directly tradable as an
Apple variance swap.  Quantities such as ``IV - RV`` therefore measure a
**variance-payoff proxy** that signals whether option-implied risk
exceeded realized risk, not an executable P&L.  We label every such
quantity as a proxy in the code, the tables and the paper.

## Known limitations

* Apple is one firm: 40 events spread across one decade is a small
  sample.  The published intervals reflect that.
* The ridge penalty ``lambda = 10`` is pre-specified in
  ``research_log.md`` and **not** tuned on the test sample.  A future
  extension could use chronological nested cross-validation.
* The walk-forward exercise drops the very first event (no
  ``hist_abs_gap_4`` history available); 39 of the 40 events are
  feature-complete and 23 are held out.
* Real option market microstructure (skew, term structure, liquidity,
  open interest) is not in scope.  The paper enumerates the institutional
  data that would be required to extend the study.

## Errors raised by the pipeline

The loader raises ``DataValidationError`` for any of:

* missing required columns (``date``, ``aapl_*``, ``vxapl_close`` etc.),
* unparseable date strings,
* duplicate trading dates,
* non-positive prices in any strict-positive column,
* a missing VXAPL close on an actual earnings event.

Feature construction raises informative ``ValueError``s when a window
asks for negative-index data or when ``log`` would receive a
non-positive argument.

## License and citation

This is an educational research project.  Cite the prepared CSV's
underlying sources (Cboe historical indices, public Apple OHLC mirror,
and AlphaQuery earnings history) before reusing the dataset.
