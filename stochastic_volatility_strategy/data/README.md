# Required CSV Files

## spy_prices.csv
`date, open, high, low, close, adjusted_close, volume`

## spy_options.csv
`quote_date, expiration, option_type, strike, bid, ask, last, volume, open_interest, implied_volatility`

Optional: `delta, gamma, theta, vega`

## vix_data.csv
`date, vix, vix9d` and optionally `vix3m, vvix`

## risk_free_rates.csv
`date, annual_rate`

All rows must use information available by the stated timestamp. Never backfill future values.
