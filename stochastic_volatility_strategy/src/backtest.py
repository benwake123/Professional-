"""
Walk-forward backtesting engine with audit funnel tracking.

Purpose
-------
Drives the strategy day by day. Evaluates trading decisions on every
business day (configurable), recalibrates Heston weekly, reuses the latest
valid Heston parameters on intervening days, and records a decision funnel
for every evaluation date.

Module connections
------------------
Upstream: data, forecast, signal, execution, portfolio, audit modules.
Downstream: ``src.run_pipeline``, ``src.research``, ``tests/test_backtest.py``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from src.audit import DecisionFunnelRow
from src.calibration import calibrate_heston, validate_calibration_result
from src.data_loader import slice_data_as_of
from src.delta_hedging import apply_delta_hedge, close_delta_hedge
from src.execution import execute_option_entry, execute_option_exit
from src.exits import should_exit_position
from src.forecast_engine import select_forecast_for_decision
from src.market_features import build_market_feature_table
from src.option_selection import (
    filter_liquid_options,
    find_atm_call_and_put,
    select_target_expiration,
    select_trade_structure,
)
from src.portfolio import (
    calculate_portfolio_equity,
    close_option_position,
    create_initial_portfolio,
    mark_portfolio_to_market,
    open_option_position,
    record_portfolio_snapshot,
    update_equity_peak_and_drawdown,
)
from src.position_sizing import calculate_position_size
from src.regime import classify_current_regime
from src.risk_management import calculate_position_maximum_loss, check_pretrade_risk
from src.signals import build_volatility_signal, expected_edge_after_costs
from src.types import BacktestResults, MarketDataBundle, PortfolioState, TradeSignal

DateLike = Union[str, date, datetime, pd.Timestamp]
ProgressCallback = Optional[Callable[[int, pd.Timestamp], None]]


def build_trading_calendar(
    underlying: pd.DataFrame, start_date: DateLike, end_date: DateLike
) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (underlying["date"] >= start) & (underlying["date"] <= end)
    return pd.DatetimeIndex(underlying.loc[mask, "date"].sort_values().unique())


def _snap_decision_dates(
    trading_calendar: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str,
) -> set[pd.Timestamp]:
    if frequency.upper() in {"B", "BUSINESS", "D", "DAILY"}:
        return set(trading_calendar)
    decision_set: set[pd.Timestamp] = set()
    for target in pd.date_range(start_date, end_date, freq=frequency):
        candidates = trading_calendar[trading_calendar >= target]
        if not candidates.empty:
            decision_set.add(candidates[0])
    return decision_set or {trading_calendar[0]}


def get_market_state_for_date(
    bundle: MarketDataBundle, as_of_date: pd.Timestamp
) -> dict[str, object]:
    snapshot = slice_data_as_of(bundle, as_of_date)
    underlying_today = snapshot.underlying[snapshot.underlying["date"] == as_of_date]
    spot = (
        float(underlying_today["close"].iloc[-1])
        if not underlying_today.empty
        else float("nan")
    )
    options_today = snapshot.options[snapshot.options["quote_date"] == as_of_date]
    rf_frame = snapshot.risk_free_rates
    if rf_frame is None or rf_frame.empty or "annual_rate" not in rf_frame.columns:
        risk_free = 0.0
    else:
        risk_free = float(rf_frame["annual_rate"].iloc[-1])
    return {
        "snapshot": snapshot,
        "spot": spot,
        "options_today": options_today,
        "risk_free_rate": risk_free,
    }


def manage_open_positions(
    portfolio: PortfolioState,
    options_today: pd.DataFrame,
    spot: float,
    as_of_date: pd.Timestamp,
    current_signal: Optional[TradeSignal],
    exit_config: dict[str, float],
    execution_config: dict[str, float],
    trade_log: list[dict[str, object]],
) -> None:
    for position in list(portfolio.option_positions):
        if not position.is_open:
            continue
        max_loss = calculate_position_maximum_loss(position.structure, position.quantity)
        force, reason = should_exit_position(
            position, as_of_date, current_signal, portfolio, exit_config, max_loss
        )
        if not force:
            continue
        try:
            exit_report = execute_option_exit(
                structure=position.structure,
                quantity_per_leg=position.quantity,
                as_of_date=as_of_date,
                commission_per_contract=float(execution_config["option_commission_per_contract"]),
                slippage_fraction_of_spread=float(
                    execution_config["option_slippage_fraction_of_spread"]
                ),
            )
        except ValueError as exc:
            trade_log.append(
                {
                    "date": as_of_date,
                    "action": "exit_skipped",
                    "position_id": position.position_id,
                    "reason": str(exc),
                }
            )
            continue
        close_option_position(portfolio, position, exit_report)
        trade_log.append(
            {
                "date": as_of_date,
                "action": "exit",
                "position_id": position.position_id,
                "reason": reason,
                "realized_pnl": position.realized_pnl,
                "gross_option_pnl": float(
                    position.entry_execution.gross_cash_flow
                    + exit_report.gross_cash_flow
                ),
                "commission": float(
                    position.entry_execution.commission + exit_report.commission
                ),
                "slippage": float(
                    position.entry_execution.slippage_cost + exit_report.slippage_cost
                ),
            }
        )


def evaluate_new_trade(
    portfolio: PortfolioState,
    bundle_snapshot: MarketDataBundle,
    state: dict[str, object],
    as_of_date: pd.Timestamp,
    config: dict[str, object],
    feature_table: pd.DataFrame,
    edge_history: pd.Series,
    portfolio_equity_history: pd.Series,
    cached_heston_calibration: Optional[dict[str, float]],
    decision_log: list[dict[str, object]],
    forecast_log: list[dict[str, object]],
    trade_log: list[dict[str, object]],
    funnel: DecisionFunnelRow,
    direction_filter: Optional[str] = None,
    zero_cost_mode: bool = False,
    midpoint_fill_mode: bool = False,
) -> Optional[TradeSignal]:
    model_config = config["model"]
    signal_config = config["signal"]
    options_config = config["options"]
    risk_config = config["risk"]
    execution_config = config["execution"]

    options_today = state["options_today"]
    spot = state["spot"]
    risk_free = state["risk_free_rate"]
    underlying = bundle_snapshot.underlying

    funnel.has_required_market_data = not np.isnan(spot)
    if not funnel.has_required_market_data:
        funnel.rejection_reason = "missing_spot"
        return None

    funnel.has_valid_option_chain = not options_today.empty
    if not funnel.has_valid_option_chain:
        funnel.rejection_reason = "empty_option_chain"
        return None

    lookback = int(model_config["lookback_days"])
    history_rows = underlying[underlying["date"] <= as_of_date]
    funnel.has_sufficient_model_history = len(history_rows) >= min(lookback, 30)
    if not funnel.has_sufficient_model_history:
        funnel.rejection_reason = "insufficient_history"
        return None

    weights = signal_config.get(
        "ensemble_weights",
        {"heston": 0.4, "ewma": 0.3, "historical": 0.3},
    )
    try:
        forecast, forecast_diag = select_forecast_for_decision(
            underlying=underlying,
            as_of_date=as_of_date,
            spot=float(spot),
            model_config=model_config,
            ensemble_weights=weights,
            cached_heston_calibration=cached_heston_calibration,
            prefer_model=str(config.get("forecast_model", "ensemble")),
        )
        funnel.model_calibration_succeeded = True
        funnel.forecast_created = True
    except ValueError as exc:
        funnel.rejection_reason = f"forecast_failed:{exc}"
        return None

    forecast_log.append(
        {
            "as_of_date": as_of_date,
            "model_name": forecast.model_name,
            "expected_variance": forecast.expected_variance,
            "median_variance": forecast.median_variance,
            "lower_quantile": forecast.lower_quantile,
            "upper_quantile": forecast.upper_quantile,
            "monte_carlo_standard_error": forecast.monte_carlo_standard_error,
            "heston_used": forecast_diag.get("heston_used"),
            "heston_failure": forecast_diag.get("heston_failure"),
        }
    )

    feature_row = feature_table.iloc[-1] if not feature_table.empty else pd.Series(dtype=float)
    realized_vol_history = (
        feature_table["realized_volatility"].dropna()
        if "realized_volatility" in feature_table.columns
        else pd.Series(dtype=float)
    )
    regime = classify_current_regime(feature_row, realized_vol_history)

    signal = build_volatility_signal(
        as_of_date=as_of_date,
        options_snapshot=options_today,
        forecast=forecast,
        edge_history=edge_history,
        rolling_window=int(signal_config["rolling_zscore_window"]),
        long_threshold=float(signal_config.get("long_vol_z_threshold", 0.75)),
        short_threshold=float(signal_config.get("short_vol_z_threshold", 0.75)),
        regime_label=regime["combined_regime"],
        minimum_dte=int(options_config["minimum_dte"]),
        maximum_dte=int(options_config["maximum_dte"]),
        target_dte=int(options_config.get("target_dte", 30)),
        atm_moneyness_low=float(options_config.get("atm_moneyness_low", 0.98)),
        atm_moneyness_high=float(options_config.get("atm_moneyness_high", 1.02)),
        spot=float(spot),
    )

    if direction_filter and signal.direction not in (direction_filter, "flat"):
        signal = TradeSignal(
            as_of_date=signal.as_of_date,
            direction="flat",
            implied_variance=signal.implied_variance,
            forecast_variance=signal.forecast_variance,
            variance_risk_premium=signal.variance_risk_premium,
            variance_edge=signal.variance_edge,
            zscore=signal.zscore,
            regime=signal.regime,
            entry_threshold=signal.entry_threshold,
            approved_by_regime_filter=signal.approved_by_regime_filter,
            model_name=signal.model_name,
        )

    implied = signal.implied_variance
    decision_log.append(
        {
            "as_of_date": as_of_date,
            "direction": signal.direction,
            "implied_variance": implied,
            "forecast_variance": signal.forecast_variance,
            "variance_edge": signal.variance_edge,
            "variance_risk_premium": signal.variance_risk_premium,
            "zscore": signal.zscore,
            "regime": signal.regime,
            "entry_threshold": signal.entry_threshold,
            "approved": signal.approved_by_regime_filter,
            "model_name": signal.model_name,
        }
    )

    funnel.signal_exceeded_threshold = signal.direction != "flat"
    if signal.direction == "flat":
        funnel.rejection_reason = "signal_flat"
        return signal

    liquid = filter_liquid_options(
        options_today,
        minimum_dte=int(options_config["minimum_dte"]),
        maximum_dte=int(options_config["maximum_dte"]),
        maximum_relative_spread=float(options_config["maximum_relative_spread"]),
        minimum_open_interest=int(options_config["minimum_open_interest"]),
        minimum_volume=int(options_config["minimum_volume"]),
    )
    target_dte = int(options_config.get("target_dte", 30))
    expiration = select_target_expiration(liquid, target_dte)
    funnel.eligible_expiration_found = expiration is not None
    if expiration is None:
        funnel.rejection_reason = "no_eligible_expiration"
        return signal

    atm = find_atm_call_and_put(
        liquid,
        expiration,
        float(spot),
        float(options_config.get("atm_moneyness_low", 0.98)),
        float(options_config.get("atm_moneyness_high", 1.02)),
    )
    funnel.atm_contracts_found = atm is not None
    funnel.liquidity_filter_passed = not liquid.empty
    if atm is None or liquid.empty:
        funnel.rejection_reason = "atm_or_liquidity_failed"
        return signal

    structure = select_trade_structure(
        options_snapshot=options_today,
        spot=float(spot),
        direction=signal.direction,
        options_config=options_config,
        as_of_date=as_of_date,
        risk_free_rate=risk_free,
    )
    if structure is None:
        funnel.rejection_reason = "structure_selection_failed"
        trade_log.append({"date": as_of_date, "action": "entry_skipped", "reason": funnel.rejection_reason})
        return signal

    quantity = calculate_position_size(
        signal=signal,
        structure=structure,
        portfolio_equity=portfolio.equity,
        portfolio_equity_history=portfolio_equity_history,
        risk_config=risk_config,
    )
    funnel.position_size_positive = quantity > 0
    if quantity <= 0:
        funnel.rejection_reason = "position_size_zero"
        trade_log.append(
            {
                "date": as_of_date,
                "action": "entry_skipped",
                "reason": funnel.rejection_reason,
            }
        )
        return signal

    safety_buffer = float(signal_config.get("edge_safety_buffer", 0.0))
    edge_after_costs = expected_edge_after_costs(
        variance_edge=float(signal.variance_edge),
        implied_variance=float(implied),
        quantity=quantity,
        structure_legs=structure.legs,
        commission_per_contract=float(execution_config["option_commission_per_contract"]),
        slippage_fraction_of_spread=float(
            execution_config["option_slippage_fraction_of_spread"]
        ),
        structure_entry_value=float(structure.entry_debit_or_credit or 0.0),
    )
    if edge_after_costs <= safety_buffer:
        funnel.rejection_reason = "insufficient_edge_after_costs"
        trade_log.append(
            {
                "date": as_of_date,
                "action": "entry_skipped",
                "reason": funnel.rejection_reason,
                "edge_after_costs": edge_after_costs,
            }
        )
        return signal

    risk_check = check_pretrade_risk(
        portfolio,
        structure,
        quantity,
        risk_config,
        options_config,
        as_of_date=as_of_date,
    )
    funnel.risk_check_passed = risk_check.approved
    if not risk_check.approved:
        funnel.rejection_reason = "risk_rejected"
        trade_log.append(
            {
                "date": as_of_date,
                "action": "entry_rejected",
                "reasons": risk_check.reasons,
            }
        )
        return signal

    exec_cfg = dict(execution_config)
    if zero_cost_mode:
        exec_cfg["option_commission_per_contract"] = 0.0
        exec_cfg["option_slippage_fraction_of_spread"] = 0.0
    fill_at_midpoint = midpoint_fill_mode

    try:
        entry = execute_option_entry(
            structure=structure,
            quantity_per_leg=quantity,
            as_of_date=as_of_date,
            commission_per_contract=float(exec_cfg["option_commission_per_contract"]),
            slippage_fraction_of_spread=float(exec_cfg["option_slippage_fraction_of_spread"]),
            fill_at_midpoint=fill_at_midpoint,
        )
    except ValueError as exc:
        funnel.rejection_reason = f"execution_failed:{exc}"
        trade_log.append(
            {"date": as_of_date, "action": "entry_skipped", "reason": funnel.rejection_reason}
        )
        return signal

    position = open_option_position(portfolio, structure, quantity, entry, signal)
    mark_portfolio_to_market(portfolio, options_today, float(spot), as_of_date)
    funnel.trade_executed = True
    funnel.rejection_reason = "executed"
    trade_log.append(
        {
            "date": as_of_date,
            "action": "entry",
            "position_id": position.position_id,
            "structure": structure.name,
            "quantity": quantity,
            "direction": signal.direction,
            "net_cash_flow": entry.net_cash_flow,
            "edge_after_costs": edge_after_costs,
        }
    )
    return signal


def process_trading_day(
    as_of_date: pd.Timestamp,
    bundle: MarketDataBundle,
    portfolio: PortfolioState,
    feature_table: pd.DataFrame,
    edge_history: pd.Series,
    portfolio_equity_history: pd.Series,
    is_decision_day: bool,
    config: dict[str, object],
    cached_heston_calibration: Optional[dict[str, float]],
    decision_log: list[dict[str, object]],
    forecast_log: list[dict[str, object]],
    trade_log: list[dict[str, object]],
    funnel_log: list[dict[str, object]],
    daily_log: list[dict[str, object]],
    direction_filter: Optional[str] = None,
    zero_cost_mode: bool = False,
    midpoint_fill_mode: bool = False,
) -> Optional[TradeSignal]:
    state = get_market_state_for_date(bundle, as_of_date)
    spot = state["spot"]
    options_today = state["options_today"]

    if not np.isnan(spot):
        mark_portfolio_to_market(portfolio, options_today, float(spot), as_of_date)

    signal: Optional[TradeSignal] = None
    if is_decision_day:
        funnel = DecisionFunnelRow(decision_date=as_of_date)
        signal = evaluate_new_trade(
            portfolio=portfolio,
            bundle_snapshot=state["snapshot"],
            state=state,
            as_of_date=as_of_date,
            config=config,
            feature_table=feature_table,
            edge_history=edge_history,
            portfolio_equity_history=portfolio_equity_history,
            cached_heston_calibration=cached_heston_calibration,
            decision_log=decision_log,
            forecast_log=forecast_log,
            trade_log=trade_log,
            funnel=funnel,
            direction_filter=direction_filter,
            zero_cost_mode=zero_cost_mode,
            midpoint_fill_mode=midpoint_fill_mode,
        )
        funnel_log.append(funnel.to_dict())

    manage_open_positions(
        portfolio=portfolio,
        options_today=options_today,
        spot=float(spot) if not np.isnan(spot) else 0.0,
        as_of_date=as_of_date,
        current_signal=signal,
        exit_config=config.get("exits", {}),
        execution_config=config["execution"],
        trade_log=trade_log,
    )

    if portfolio.option_positions and not np.isnan(spot):
        hedge_report = apply_delta_hedge(
            portfolio=portfolio,
            reference_price=float(spot),
            as_of_date=as_of_date,
            execution_config=config["execution"],
            hedge_config=config.get("delta_hedge", {"mode": "threshold"}),
        )
        if hedge_report is not None and hedge_report.fills:
            trade_log.append(
                {
                    "date": as_of_date,
                    "action": "hedge",
                    "shares": hedge_report.fills[0].get("shares", 0),
                    "net_cash_flow": hedge_report.net_cash_flow,
                    "gross_cash_flow": hedge_report.gross_cash_flow,
                }
            )

    if not any(p.is_open for p in portfolio.option_positions) and portfolio.stock_shares != 0:
        close_delta_hedge(
            portfolio=portfolio,
            reference_price=float(spot) if not np.isnan(spot) else portfolio.stock_reference_price,
            slippage_bps=float(config["execution"]["stock_slippage_bps"]),
            as_of_date=as_of_date,
        )

    calculate_portfolio_equity(portfolio)
    update_equity_peak_and_drawdown(portfolio)
    daily_log.append(record_portfolio_snapshot(portfolio, as_of_date))
    return signal


def run_walk_forward_backtest(
    bundle: MarketDataBundle,
    config: dict[str, object],
    model_name: str = "ensemble",
    decision_frequency: str | None = None,
    progress_callback: ProgressCallback = None,
    direction_filter: Optional[str] = None,
    zero_cost_mode: bool = False,
    midpoint_fill_mode: bool = False,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
) -> BacktestResults:
    dates_config = config["dates"]
    start = pd.Timestamp(start_date or dates_config["development_start"])
    end = pd.Timestamp(end_date or dates_config["end"])

    decision_freq = decision_frequency or config.get("decision_frequency", "B")
    heston_freq = config.get("heston_calibration_frequency", "W-MON")
    config = dict(config)
    config["forecast_model"] = model_name

    trading_calendar = build_trading_calendar(bundle.underlying, start, end)
    decision_set = _snap_decision_dates(trading_calendar, start, end, decision_freq)
    heston_set = _snap_decision_dates(trading_calendar, start, end, heston_freq)

    portfolio = create_initial_portfolio(float(config["risk"]["initial_capital"]))
    daily_log: list[dict[str, object]] = []
    trade_log: list[dict[str, object]] = []
    decision_log: list[dict[str, object]] = []
    forecast_log: list[dict[str, object]] = []
    funnel_log: list[dict[str, object]] = []
    edge_history = pd.Series([], dtype=float)

    cached_heston: Optional[dict[str, float]] = None
    last_progress = -1

    for idx, as_of_date in enumerate(trading_calendar):
        if progress_callback is not None:
            pct = int(100 * (idx + 1) / len(trading_calendar))
            if pct != last_progress and pct % 5 == 0:
                progress_callback(pct, as_of_date)
                last_progress = pct

        if as_of_date in heston_set:
            try:
                cached_heston = calibrate_heston(
                    slice_data_as_of(bundle, as_of_date).underlying,
                    as_of_date,
                    int(config["model"]["lookback_days"]),
                )
                if not validate_calibration_result(cached_heston):
                    cached_heston = None
            except ValueError:
                cached_heston = None

        feature_table = pd.DataFrame()
        equity_history_series = pd.Series([row["equity"] for row in daily_log], dtype=float)
        if as_of_date in decision_set:
            snapshot = slice_data_as_of(bundle, as_of_date)
            try:
                feature_table = build_market_feature_table(
                    snapshot.underlying, snapshot.volatility_indices
                )
            except Exception:  # noqa: BLE001
                feature_table = pd.DataFrame()

        signal = process_trading_day(
            as_of_date=as_of_date,
            bundle=bundle,
            portfolio=portfolio,
            feature_table=feature_table,
            edge_history=edge_history,
            portfolio_equity_history=equity_history_series,
            is_decision_day=as_of_date in decision_set,
            config=config,
            cached_heston_calibration=cached_heston,
            decision_log=decision_log,
            forecast_log=forecast_log,
            trade_log=trade_log,
            funnel_log=funnel_log,
            daily_log=daily_log,
            direction_filter=direction_filter,
            zero_cost_mode=zero_cost_mode,
            midpoint_fill_mode=midpoint_fill_mode,
        )
        if signal is not None and not np.isnan(signal.variance_edge):
            edge_history = pd.concat(
                [edge_history, pd.Series([signal.variance_edge])], ignore_index=True
            )

    results = BacktestResults(
        daily_history=pd.DataFrame(daily_log),
        trades=pd.DataFrame(trade_log),
        trade_decisions=pd.DataFrame(decision_log),
        forecasts=pd.DataFrame(forecast_log),
        calibrations=pd.DataFrame(),
        decision_funnel=pd.DataFrame(funnel_log),
    )
    results._portfolio = portfolio  # type: ignore[attr-defined]
    return results
