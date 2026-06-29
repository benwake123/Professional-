# Architecture

## Data layer
`config`, `data_loader`, `data_validation`

## Quantitative-model layer
`market_features`, `black_scholes`, `gbm`, `heston`, `calibration`, `monte_carlo`

## Decision layer
`regime`, `signals`, `option_selection`, `position_sizing`, `risk_management`

## Trading-mechanics layer
`execution`, `delta_hedging`, `portfolio`

## Research layer
`backtest`, `attribution`, `metrics`, `plots`, `run_pipeline`

Dependencies flow downward only: the backtest may call model and execution modules, but model modules must never import the backtest. This avoids circular dependencies and keeps unit tests isolated.
