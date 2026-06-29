"""
Shared domain objects for the stochastic-volatility trading project.

Purpose
-------
Dataclasses only - no pricing, forecasting, trading, or backtesting logic.
Centralizing the types here means every module that talks about the same
concept (e.g. an option leg) talks about the same object.

Currently defined:
    - ``MarketDataBundle``  : raw market data container.
    - ``OptionLeg``         : a single option contract inside a structure.
    - ``OptionStructure``   : a multi-leg trade definition.
    - ``ModelForecast``     : one point-in-time volatility-model forecast.
    - ``TradeSignal``       : pre-selection trading decision.
    - ``ExecutionReport``   : modeled fills and costs from execution.
    - ``OptionPosition``    : one open options position.
    - ``PortfolioState``    : mutable simulated portfolio state.
    - ``RiskCheckResult``   : output of pre-trade risk checks.
    - ``BacktestResults``   : principal outputs from the backtest engine.

Module connections
------------------
Downstream (this module is imported by):
    - ``src.data_loader`` / ``src.data_validation`` : ``MarketDataBundle``.
    - ``src.black_scholes``    : ``OptionLeg`` for ``calculate_structure_greeks``.
    - ``src.option_selection`` : ``OptionLeg``, ``OptionStructure``.
    - ``tests/conftest.py``    : builds ``MarketDataBundle`` fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class MarketDataBundle:
    """
    Groups all raw market datasets used by the research pipeline.

    Attributes:
        underlying:
            Daily SPY price and volume data.

        options:
            Historical SPY option-chain data.

        volatility_indices:
            VIX, VIX9D, and any other volatility-index data.

        risk_free_rates:
            Historical risk-free interest-rate observations.
    """

    underlying: pd.DataFrame
    options: pd.DataFrame
    volatility_indices: pd.DataFrame
    risk_free_rates: pd.DataFrame


@dataclass
class OptionLeg:
    """
    One option contract inside a multi-leg structure.

    The ``quantity`` field uses a signed convention:

    - ``quantity > 0`` : long contract
    - ``quantity < 0`` : short contract

    Attributes:
        contract_id:
            Project-internal identifier (e.g. ``"SPY 2024-02-16 470C"``).
        option_type:
            ``"call"`` or ``"put"`` (lower case, matching the loader).
        strike:
            Option strike price in dollars.
        expiration:
            Contract expiration date.
        quantity:
            Signed number of contracts (positive = long, negative = short).
        bid, ask:
            Best bid/ask at the quote time, when known.
        implied_volatility:
            IV implied by the bid/ask midpoint or last trade.
    """

    contract_id: str
    option_type: str
    strike: float
    expiration: date
    quantity: int
    bid: float | None = None
    ask: float | None = None
    implied_volatility: float | None = None


@dataclass
class OptionStructure:
    """
    A complete options trade (e.g. long straddle, iron butterfly).

    Attributes:
        name:
            Human-readable structure name (``"long_straddle"`` etc.).
        legs:
            Component option contracts. Sign of each leg's ``quantity``
            distinguishes long from short.
        direction:
            ``"long_vol"`` or ``"short_vol"``.
        maximum_loss_per_unit:
            Maximum loss per unit when the structure has defined risk.
        entry_debit_or_credit:
            Quoted net debit/credit before execution costs (positive = debit).
        greeks:
            Aggregate delta, gamma, theta, vega, rho per structure.
    """

    name: str
    legs: list[OptionLeg]
    direction: str
    maximum_loss_per_unit: float | None = None
    entry_debit_or_credit: float | None = None
    greeks: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelForecast:
    """One point-in-time volatility-model forecast."""

    as_of_date: date
    model_name: str
    expected_variance: float
    median_variance: float
    lower_quantile: float
    upper_quantile: float
    monte_carlo_standard_error: float
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Strategy's pre-selection trading decision for a single date."""

    as_of_date: date
    direction: str  # 'long_vol', 'short_vol', or 'flat'
    implied_variance: float
    forecast_variance: float
    variance_risk_premium: float
    variance_edge: float
    zscore: float
    regime: str
    entry_threshold: float
    approved_by_regime_filter: bool
    model_name: str


@dataclass
class ExecutionReport:
    """Modeled fills and costs from an execution function."""

    execution_date: date
    action: str  # 'entry', 'exit', 'hedge', 'hedge_close'
    fills: list[dict[str, Any]]
    gross_cash_flow: float
    commission: float
    slippage_cost: float
    net_cash_flow: float


@dataclass
class OptionPosition:
    """One open options position with audit trail back to the originating signal."""

    position_id: str
    structure: OptionStructure
    quantity: int
    entry_date: date
    entry_execution: ExecutionReport
    entry_signal: TradeSignal
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_open: bool = True
    exit_date: date | None = None
    exit_execution: ExecutionReport | None = None


@dataclass
class PortfolioState:
    """Mutable state of the simulated portfolio."""

    initial_capital: float
    cash: float
    stock_shares: int
    stock_reference_price: float
    option_positions: list[OptionPosition]
    equity: float
    equity_peak: float
    drawdown: float
    cumulative_commissions: float = 0.0
    cumulative_slippage: float = 0.0
    cumulative_option_gross_pnl: float = 0.0
    cumulative_option_net_pnl: float = 0.0
    cumulative_hedge_gross_pnl: float = 0.0
    cumulative_hedge_net_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_entry_date: date | None = None


@dataclass
class RiskCheckResult:
    """Output of the pre-trade risk-management process."""

    approved: bool
    reasons: list[str] = field(default_factory=list)
    adjusted_quantity: int | None = None


@dataclass
class BacktestResults:
    """Principal outputs produced by the backtesting engine."""

    daily_history: pd.DataFrame
    trades: pd.DataFrame
    trade_decisions: pd.DataFrame
    forecasts: pd.DataFrame
    calibrations: pd.DataFrame
    decision_funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    accounting_reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejection_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance_summary: dict[str, float] = field(default_factory=dict)
    pnl_attribution: dict[str, pd.DataFrame] = field(default_factory=dict)
