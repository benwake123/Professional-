# Minimum Viable Implementation Checklist

Implement this smaller path first. It produces a defensible GBM-versus-Heston research backtest before you add every extension.

## Phase 1 — Data and integrity

1. `load_config`
2. `load_all_market_data`
3. `validate_all_data`
4. `assert_no_future_information`
5. `build_market_feature_table`

## Phase 2 — Mathematical models

6. `black_scholes_price`
7. `calculate_option_greeks`
8. `estimate_gbm_parameters`
9. `simulate_gbm_paths`
10. `build_initial_parameter_guess`
11. `simulate_heston_paths`
12. `calibrate_gbm`
13. `calibrate_heston`
14. `forecast_realized_variance`

## Phase 3 — Signal

15. `classify_current_regime`
16. `calculate_market_implied_variance`
17. `calculate_variance_risk_premium`
18. `calculate_rolling_zscore`
19. `build_volatility_signal`

## Phase 4 — Trade construction

20. `filter_liquid_options`
21. `find_atm_call_and_put`
22. `build_long_straddle`
23. `build_iron_butterfly`
24. `select_trade_structure`
25. `calculate_position_size`
26. `check_pretrade_risk`

## Phase 5 — Trading mechanics

27. `execute_option_entry`
28. `execute_option_exit`
29. `calculate_portfolio_net_delta`
30. `rebalance_delta_hedge`
31. `mark_portfolio_to_market`

## Phase 6 — Research loop

32. `evaluate_new_trade`
33. `process_trading_day`
34. `run_walk_forward_backtest`
35. `build_performance_summary`

## Add only after the MVP works

- Fractional-Kelly sizing
- Greek P&L approximation
- Multiple short-volatility structures
- Monte Carlo expected option payoff
- Partial de-risking
- Parameter-stability plots
- VaR and expected shortfall

The first valid version can trade only long straddles and defined-risk iron butterflies, rebalance delta once daily, and use a single fixed forecast horizon. Complexity should be added only when it answers a specific research question.
