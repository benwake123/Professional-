# Data dictionary

The analysis reads only `aapl_earnings_volatility_data.csv`. Each row is a U.S. trading day.

| Column | Meaning |
|---|---|
| `date` | Trading date in `YYYY-MM-DD` format. |
| `aapl_open`, `aapl_high`, `aapl_low`, `aapl_close` | Split-adjusted Apple daily prices. |
| `aapl_volume` | Apple daily share volume. |
| `vxapl_open`, `vxapl_high`, `vxapl_low`, `vxapl_close` | Cboe Apple VIX Index daily OHLC. VXAPL is quoted in annualized volatility percentage points. |
| `vix_open`, `vix_high`, `vix_low`, `vix_close` | Cboe VIX daily OHLC. |
| `vix9d_open`, `vix9d_high`, `vix9d_low`, `vix9d_close` | Cboe 9-Day Volatility Index daily OHLC. |
| `earnings_flag` | `1` on an Apple earnings-announcement date, otherwise `0`. |
| `fiscal_quarter_end` | Fiscal quarter associated with the announcement. Populated only on event rows. |
| `eps_estimate` | Consensus EPS estimate for the event. Descriptive only. |
| `eps_actual` | Reported EPS. Not available before the release and must not be used as an ex ante feature. |
| `eps_surprise_pct` | `(actual - estimate) / abs(estimate)`. Current-event value is descriptive only. |
| `source_ids` | Semicolon-separated keys linking the row to `data_sources.csv`. |

## Timing convention

Apple generally announces after the regular market close. The project treats features measured at the listed event-date close as observable before the release. The next trading day's close-to-close return and open gap are outcomes, not predictors.

## Limitations

This is a prepared public-data research file, not a point-in-time institutional options database. It contains a Cboe implied-volatility index, not individual option quotes, strikes, expirations, bid–ask spreads, volume, open interest, Greeks, or delisting records. It therefore supports volatility forecasting and a variance-spread proxy, but not a defensible reconstruction of actual straddle or variance-swap returns.
