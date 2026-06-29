# Function Map

Every function below exists as a stub in `src/`.

## `src/config.py`

- **`load_config(...)`** — Read JSON, call validate_config, and return the configuration.
- **`validate_config(...)`** — Check required sections, date ordering, and parameter ranges.
- **`resolve_project_paths(...)`** — Convert relative configured paths into absolute paths.

## `src/data_loader.py`

- **`load_underlying_prices(...)`** — Load and normalize daily SPY OHLCV data.
- **`load_option_chain(...)`** — Load contract-level historical option quotes.
- **`load_volatility_indices(...)`** — Load VIX-family time series.
- **`load_risk_free_rates(...)`** — Load annualized short-rate observations.
- **`load_all_market_data(...)`** — Call all four loaders and return one data bundle.
- **`slice_data_as_of(...)`** — Return only information available by a specified date.

## `src/data_validation.py`

- **`validate_required_columns(...)`** — Check that a dataset contains its required schema.
- **`validate_sorted_unique_dates(...)`** — Reject unsorted or duplicated date keys.
- **`validate_underlying_data(...)`** — Check positive prices, OHLC consistency, and volumes.
- **`validate_option_data(...)`** — Check strikes, expirations, bid/ask ordering, IV, and DTE.
- **`validate_volatility_index_data(...)`** — Check VIX-family dates and nonnegative values.
- **`validate_risk_free_data(...)`** — Check rate dates, missing values, and units.
- **`validate_all_data(...)`** — Call every dataset-specific validation function.
- **`assert_no_future_information(...)`** — Raise an error when information dates exceed the decision date.

## `src/market_features.py`

- **`calculate_log_returns(...)`** — Calculate close-to-close logarithmic returns.
- **`calculate_realized_variance(...)`** — Calculate rolling annualized realized variance.
- **`calculate_realized_volatility(...)`** — Take the square root of realized variance.
- **`calculate_ewma_variance(...)`** — Calculate exponentially weighted variance.
- **`calculate_drawdown(...)`** — Calculate drawdown from the prior running peak.
- **`calculate_vix_term_structure(...)`** — Create VIX9D/VIX and VIX/VIX3M slope features.
- **`calculate_volatility_acceleration(...)`** — Measure recent changes in realized volatility.
- **`build_market_feature_table(...)`** — Call all feature functions and merge by date.
- **`get_feature_row_as_of(...)`** — Return the latest feature row available on a date.

## `src/black_scholes.py`

- **`normal_cdf(...)`** — Standard normal cumulative distribution.
- **`normal_pdf(...)`** — Standard normal probability density.
- **`calculate_d1_d2(...)`** — Calculate Black--Scholes d1 and d2.
- **`black_scholes_price(...)`** — Price a European call or put.
- **`calculate_option_greeks(...)`** — Calculate delta, gamma, theta, vega, and rho.
- **`implied_volatility_bisection(...)`** — Recover IV from an observed option price.
- **`calculate_structure_greeks(...)`** — Aggregate signed Greeks across option legs.

## `src/gbm.py`

- **`estimate_gbm_parameters(...)`** — Estimate annualized drift and volatility from returns.
- **`simulate_gbm_paths(...)`** — Simulate geometric Brownian motion price paths.
- **`gbm_expected_variance(...)`** — Return the constant expected variance under GBM.

## `src/heston.py`

- **`build_initial_parameter_guess(...)`** — Create starting kappa, theta, xi, rho, and v0 values.
- **`validate_heston_parameters(...)`** — Check parameter bounds and numerical stability.
- **`generate_correlated_shocks(...)`** — Generate correlated Brownian price and variance shocks.
- **`full_truncation_variance_step(...)`** — Advance variance one step without allowing negativity.
- **`heston_price_step(...)`** — Advance the underlying one Heston time step.
- **`simulate_heston_paths(...)`** — Simulate joint price and variance paths.
- **`heston_calibration_objective(...)`** — Return the loss minimized during Heston calibration.

## `src/calibration.py`

- **`select_calibration_window(...)`** — Select a historical window ending at the decision date.
- **`calibrate_gbm(...)`** — Select the window and call estimate_gbm_parameters.
- **`calibrate_heston(...)`** — Estimate Heston parameters using historical data only.
- **`rolling_model_calibration(...)`** — Store dated model parameters for diagnostics.
- **`validate_calibration_result(...)`** — Reject invalid or unstable estimates.

## `src/monte_carlo.py`

- **`calculate_path_realized_variance(...)`** — Calculate annualized realized variance for each path.
- **`summarize_forecast_distribution(...)`** — Return mean, median, quantiles, and dispersion.
- **`forecast_realized_variance(...)`** — Dispatch to GBM/Heston simulation and summarize results.
- **`estimate_expected_option_payoff(...)`** — Estimate terminal payoff of a proposed option structure.
- **`calculate_monte_carlo_standard_error(...)`** — Measure simulation uncertainty.

## `src/regime.py`

- **`calculate_expanding_regime_thresholds(...)`** — Calculate historical-only volatility quantiles.
- **`classify_volatility_regime(...)`** — Return low, normal, or high volatility.
- **`classify_term_structure_regime(...)`** — Return contango, flat, or backwardation.
- **`build_combined_regime_label(...)`** — Combine volatility and term-structure states.
- **`classify_current_regime(...)`** — Call the lower-level regime functions and return one point-in-time regime.
- **`regime_adjusted_entry_threshold(...)`** — Modify required signal strength by regime.
- **`regime_risk_multiplier(...)`** — Return a regime-based sizing multiplier.

## `src/signals.py`

- **`calculate_market_implied_variance(...)`** — Estimate forward variance from eligible option quotes.
- **`calculate_variance_risk_premium(...)`** — Compute implied variance minus forecast variance.
- **`calculate_rolling_zscore(...)`** — Standardize the VRP using prior observations only.
- **`generate_raw_trade_direction(...)`** — Return long_vol, short_vol, or flat.
- **`apply_regime_filter(...)`** — Approve or suppress the raw direction.
- **`build_volatility_signal(...)`** — Call implied variance, VRP, z-score, direction, and regime logic.

## `src/option_selection.py`

- **`calculate_days_to_expiration(...)`** — Calculate calendar DTE.
- **`calculate_relative_bid_ask_spread(...)`** — Calculate spread divided by midpoint.
- **`filter_liquid_options(...)`** — Apply DTE, spread, volume, open-interest, and quote filters.
- **`select_target_expiration(...)`** — Choose the eligible expiration nearest target DTE.
- **`find_atm_call_and_put(...)`** — Find matched call and put closest to spot.
- **`find_wing_options(...)`** — Find protective wings for a defined-risk short trade.
- **`build_long_straddle(...)`** — Create the two-leg long-volatility structure.
- **`build_iron_butterfly(...)`** — Create the four-leg defined-risk short-volatility structure.
- **`calculate_structure_entry_value(...)`** — Calculate quoted net debit or credit.
- **`calculate_structure_greeks(...)`** — Call Black--Scholes Greeks and aggregate them.
- **`select_trade_structure(...)`** — Run the complete filtering and contract-selection workflow.

## `src/position_sizing.py`

- **`estimate_strategy_pnl_volatility(...)`** — Estimate recent volatility of strategy returns.
- **`calculate_signal_strength_multiplier(...)`** — Map z-score magnitude into a bounded multiplier.
- **`calculate_volatility_target_size(...)`** — Calculate exposure implied by volatility targeting.
- **`calculate_fractional_kelly_size(...)`** — Calculate an optional capped fractional-Kelly size.
- **`apply_regime_size_multiplier(...)`** — Adjust size for the current regime.
- **`apply_trade_risk_cap(...)`** — Cap size using maximum loss and portfolio equity.
- **`round_to_contract_quantity(...)`** — Convert continuous size into whole contracts.
- **`calculate_position_size(...)`** — Call the sizing components in the approved order.

## `src/risk_management.py`

- **`calculate_portfolio_greeks(...)`** — Aggregate portfolio delta, gamma, theta, and vega.
- **`calculate_position_maximum_loss(...)`** — Calculate defined maximum loss when available.
- **`check_liquidity_limits(...)`** — Evaluate spread, volume, and open-interest limits.
- **`check_position_limits(...)`** — Evaluate count, notional, and concentration limits.
- **`check_greek_limits(...)`** — Evaluate post-trade Greek limits.
- **`check_drawdown_limit(...)`** — Apply the portfolio drawdown circuit breaker.
- **`check_pretrade_risk(...)`** — Call all risk checks and combine rejection reasons.
- **`should_force_exit(...)`** — Check stop, DTE, signal reversal, and portfolio-risk exits.
- **`calculate_risk_reduction_quantity(...)`** — Calculate contracts to close during partial de-risking.

## `src/execution.py`

- **`calculate_mid_price(...)`** — Calculate quote midpoint.
- **`estimate_option_fill_price(...)`** — Apply side-aware bid/ask slippage.
- **`estimate_stock_fill_price(...)`** — Apply stock slippage in basis points.
- **`calculate_option_commission(...)`** — Calculate option commissions.
- **`calculate_stock_commission(...)`** — Calculate hedge commissions.
- **`validate_executable_quotes(...)`** — Reject stale, missing, crossed, or nonpositive quotes.
- **`execute_option_entry(...)`** — Generate leg-level entry fills, cash flow, and costs.
- **`execute_option_exit(...)`** — Generate leg-level closing fills, cash flow, and costs.
- **`execute_stock_hedge(...)`** — Generate a hedge fill and transaction costs.

## `src/delta_hedging.py`

- **`calculate_position_delta(...)`** — Calculate signed delta for one option position.
- **`calculate_portfolio_net_delta(...)`** — Combine option delta with existing hedge shares.
- **`should_rehedge(...)`** — Test whether net delta exceeds the threshold.
- **`calculate_required_hedge_shares(...)`** — Calculate shares required to neutralize delta.
- **`rebalance_delta_hedge(...)`** — Call execution and update the hedge when required.
- **`close_delta_hedge(...)`** — Flatten outstanding hedge shares.

## `src/portfolio.py`

- **`create_initial_portfolio(...)`** — Create a cash-only portfolio state.
- **`open_option_position(...)`** — Add an executed option structure and update cash.
- **`close_option_position(...)`** — Remove a position, realize P&L, and update cash.
- **`update_stock_hedge(...)`** — Update hedge shares, cash, and costs.
- **`mark_option_position(...)`** — Calculate conservative liquidation value.
- **`mark_portfolio_to_market(...)`** — Revalue all options and hedge shares.
- **`calculate_portfolio_equity(...)`** — Return cash plus marked holdings.
- **`update_equity_peak_and_drawdown(...)`** — Update running peak and drawdown.
- **`record_portfolio_snapshot(...)`** — Create a serializable daily audit row.

## `src/backtest.py`

- **`build_trading_calendar(...)`** — Build valid decision dates with all required data.
- **`get_market_state_for_date(...)`** — Build a point-in-time market and quote snapshot.
- **`manage_open_positions(...)`** — Evaluate exits and delta hedges.
- **`evaluate_new_trade(...)`** — Run calibration, forecast, signal, selection, sizing, and risk.
- **`process_trading_day(...)`** — Mark, manage, evaluate entries, and record in order.
- **`run_walk_forward_backtest(...)`** — Iterate chronologically through the trading calendar.
- **`record_trade_decision(...)`** — Store accepted, rejected, and flat decisions.
- **`record_backtest_snapshot(...)`** — Store daily portfolio, exposure, and P&L state.

## `src/attribution.py`

- **`calculate_option_leg_pnl(...)`** — Calculate P&L for each option leg.
- **`calculate_option_structure_pnl(...)`** — Aggregate option-leg P&L.
- **`calculate_hedge_pnl(...)`** — Calculate stock-hedging P&L.
- **`calculate_transaction_cost_drag(...)`** — Aggregate commissions and slippage.
- **`calculate_greek_pnl_approximation(...)`** — Approximate delta, gamma, theta, and vega P&L.
- **`aggregate_pnl_by_regime(...)`** — Summarize performance by entry regime.
- **`build_pnl_attribution(...)`** — Call all attribution functions and return tables.

## `src/metrics.py`

- **`calculate_daily_returns(...)`** — Convert equity levels into daily returns.
- **`annualized_return(...)`** — Calculate geometric annualized return.
- **`annualized_volatility(...)`** — Calculate annualized standard deviation.
- **`sharpe_ratio(...)`** — Calculate annualized excess-return Sharpe ratio.
- **`sortino_ratio(...)`** — Calculate return relative to downside deviation.
- **`maximum_drawdown(...)`** — Calculate the largest peak-to-trough loss.
- **`calmar_ratio(...)`** — Calculate annualized return divided by drawdown.
- **`historical_value_at_risk(...)`** — Calculate empirical VaR.
- **`expected_shortfall(...)`** — Calculate mean loss beyond VaR.
- **`calculate_trade_statistics(...)`** — Calculate win rate, payoff ratio, expectancy, and duration.
- **`build_performance_summary(...)`** — Call all portfolio and trade metrics.

## `src/plots.py`

- **`plot_equity_curve(...)`** — Plot portfolio equity.
- **`plot_drawdown(...)`** — Plot drawdown.
- **`plot_signal_history(...)`** — Plot VRP z-score and entry thresholds.
- **`plot_forecast_vs_realized(...)`** — Plot predicted versus realized variance.
- **`plot_regime_performance(...)`** — Plot results by regime.
- **`plot_pnl_attribution(...)`** — Plot option, hedge, and cost contributions.
- **`plot_parameter_stability(...)`** — Plot dated Heston parameters.
- **`generate_all_figures(...)`** — Call every plotting function.

## `src/run_pipeline.py`

- **`parse_command_line_arguments(...)`** — Read the --config argument.
- **`export_results(...)`** — Write result CSVs, figures, and run metadata.
- **`main(...)`** — Orchestrate configuration, data, validation, features, backtest, metrics, and output.
