# Call Graph

```text
run_pipeline.main
├── config.load_config
│   └── config.validate_config
├── data_loader.load_all_market_data
│   ├── load_underlying_prices
│   ├── load_option_chain
│   ├── load_volatility_indices
│   └── load_risk_free_rates
├── data_validation.validate_all_data
├── market_features.build_market_feature_table
├── backtest.run_walk_forward_backtest
│   ├── build_trading_calendar
│   └── process_trading_day
│       ├── portfolio.mark_portfolio_to_market
│       ├── manage_open_positions
│       │   ├── risk_management.should_force_exit
│       │   ├── execution.execute_option_exit
│       │   └── delta_hedging.rebalance_delta_hedge
│       └── evaluate_new_trade
│           ├── calibration.calibrate_gbm
│           │   └── gbm.estimate_gbm_parameters
│           ├── calibration.calibrate_heston
│           │   ├── heston.build_initial_parameter_guess
│           │   ├── heston.heston_calibration_objective
│           │   └── heston.validate_heston_parameters
│           ├── monte_carlo.forecast_realized_variance
│           │   ├── gbm.simulate_gbm_paths OR heston.simulate_heston_paths
│           │   └── calculate_path_realized_variance
│           ├── regime.classify_current_regime [compose classification helpers]
│           ├── signals.build_volatility_signal
│           │   ├── calculate_market_implied_variance
│           │   ├── calculate_variance_risk_premium
│           │   ├── calculate_rolling_zscore
│           │   ├── generate_raw_trade_direction
│           │   └── apply_regime_filter
│           ├── option_selection.select_trade_structure
│           │   ├── filter_liquid_options
│           │   ├── select_target_expiration
│           │   ├── find_atm_call_and_put
│           │   ├── find_wing_options
│           │   └── build_long_straddle OR build_iron_butterfly
│           ├── position_sizing.calculate_position_size
│           ├── risk_management.check_pretrade_risk
│           ├── execution.execute_option_entry
│           └── portfolio.open_option_position
├── attribution.build_pnl_attribution
├── metrics.build_performance_summary
├── plots.generate_all_figures
└── run_pipeline.export_results
```

Note: implement a small `classify_current_regime` orchestration helper inside `backtest.evaluate_new_trade` or add it to `regime.py` after the lower-level regime functions work.
