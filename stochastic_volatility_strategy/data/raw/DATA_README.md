# SPY Stochastic-Volatility Project Data Pack

## Included files

- `spy_prices.csv`: 2,516 real daily SPY OHLCV observations.
- `vix_data.csv`: 2,516 real official Cboe VIX/VIX9D observations.
- `risk_free_rates.csv`: 2,501 real official Treasury-rate observations.
- `spy_options.csv`: schema only; it contains no option observations.
- `data_sources.csv`: provenance and limitations.
- `config_public_data.json`: configuration aligned to this pack.

## Common research window

The supplied public data are filtered to January 1, 2014 through
December 31, 2023.

## Critical options-data limitation

A valid version of this project requires historical, point-in-time SPY option
chains that include expired contracts. The options file in this pack is only
a header template because fabricating bid, ask, implied-volatility, volume, or
open-interest values would make the strategy results invalid.

Obtain the real options data through a source such as:

- OptionMetrics through a university WRDS subscription
- Cboe DataShop
- ThetaData
- ORATS
- Polygon historical options

Export or transform the data into the exact columns already present in
`spy_options.csv`.

## SPY adjusted-close warning

The public SPY source supplies OHLCV but does not supply an adjusted-close
series. To retain the expected project schema, `adjusted_close` currently
duplicates `close`. Use `close` for preliminary code development. Before
publishing final results, replace this column with a verified split- and
distribution-adjusted series or explicitly model distributions.

## Risk-free-rate units

`annual_rate` is stored as a decimal:

- `0.0525` means 5.25%
- `0.0010` means 0.10%

## Installation location

Copy these four primary files into:

```text
data/raw/
├── spy_prices.csv
├── spy_options.csv
├── vix_data.csv
└── risk_free_rates.csv
```

The data loader can then use the paths in `config_public_data.json`.
