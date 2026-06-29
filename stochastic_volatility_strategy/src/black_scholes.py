"""
Black-Scholes pricing and Greeks for the stochastic-volatility options strategy.

Purpose
-------
Closed-form Black-Scholes pricing for European options on a dividend-paying
underlying, plus the five primary Greeks (delta, gamma, theta, vega, rho)
and a bisection root-finder that recovers implied volatility from an
observed option price. A small ``calculate_structure_greeks`` helper sums
signed Greeks across the legs of a multi-leg structure.

All formulas use the continuous-dividend convention with annualized rate
``r`` and continuous dividend yield ``q``:

    d1 = (ln(S/K) + (r - q + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call price = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    Put price  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

Inputs are scalars, not arrays. The decision layer prices one structure at
a time; vectorizing would be premature here.

Module connections
------------------
Upstream (this module imports from):
    - ``math``                   : ``log``, ``sqrt``, ``exp``.
    - ``scipy.stats.norm``       : standard normal CDF and PDF.
    - ``src.types.OptionLeg``    : used (duck-typed) by
                                   :func:`calculate_structure_greeks`.

Downstream (this module is consumed by):
    - ``src.option_selection``   : marks bid/ask of candidate structures,
                                   then aggregates Greeks via
                                   :func:`calculate_structure_greeks`.
    - ``src.delta_hedging``      : per-leg delta for hedging.
    - ``src.signals``            : may use :func:`implied_volatility_bisection`
                                   to recover IV from observed quotes.
    - ``src.portfolio``          : marks open positions via
                                   :func:`black_scholes_price`.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Iterable, Mapping, Union

import pandas as pd
from scipy.stats import norm


OptionType = str  # ``"call"`` or ``"put"``
DateLike = Union[str, date, datetime, pd.Timestamp]


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return float(norm.cdf(x))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return float(norm.pdf(x))


def calculate_d1_d2(
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return ``(d1, d2)`` from the standard Black-Scholes formulas."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be strictly positive.")
    if time_to_expiration <= 0:
        raise ValueError("time_to_expiration must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    sqrt_t = math.sqrt(time_to_expiration)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiration
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    sigma: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put."""
    normalized_type = _normalize_option_type(option_type)

    if time_to_expiration <= 0:
        return _intrinsic_value(spot, strike, normalized_type)

    d1, d2 = calculate_d1_d2(
        spot, strike, time_to_expiration, risk_free_rate, sigma, dividend_yield
    )
    discount_q = math.exp(-dividend_yield * time_to_expiration)
    discount_r = math.exp(-risk_free_rate * time_to_expiration)

    if normalized_type == "call":
        price = spot * discount_q * normal_cdf(d1) - strike * discount_r * normal_cdf(d2)
    else:
        price = strike * discount_r * normal_cdf(-d2) - spot * discount_q * normal_cdf(-d1)
    return float(price)


def calculate_option_greeks(
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    sigma: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """
    Return delta, gamma, theta, vega, rho for one option contract.

    Conventions:
        - ``theta`` is per-year (multiply by 1/365 for per-calendar-day).
        - ``vega`` is per 1.00 change in volatility (multiply by 0.01 for
          "per 1% vol" if you prefer that scale).
        - ``rho`` is per 1.00 change in the annual risk-free rate.

    If the contract has expired (``time_to_expiration <= 0``) or has zero
    volatility (``sigma <= 0``), every Greek is returned as ``0.0`` because
    the closed-form formulas are singular at that boundary.
    """
    normalized_type = _normalize_option_type(option_type)

    if time_to_expiration <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = calculate_d1_d2(
        spot, strike, time_to_expiration, risk_free_rate, sigma, dividend_yield
    )
    discount_q = math.exp(-dividend_yield * time_to_expiration)
    discount_r = math.exp(-risk_free_rate * time_to_expiration)
    pdf_d1 = normal_pdf(d1)
    sqrt_t = math.sqrt(time_to_expiration)

    common_theta_term = -(spot * pdf_d1 * sigma * discount_q) / (2.0 * sqrt_t)

    if normalized_type == "call":
        delta = discount_q * normal_cdf(d1)
        theta = (
            common_theta_term
            - risk_free_rate * strike * discount_r * normal_cdf(d2)
            + dividend_yield * spot * discount_q * normal_cdf(d1)
        )
        rho = strike * time_to_expiration * discount_r * normal_cdf(d2)
    else:
        delta = discount_q * (normal_cdf(d1) - 1.0)
        theta = (
            common_theta_term
            + risk_free_rate * strike * discount_r * normal_cdf(-d2)
            - dividend_yield * spot * discount_q * normal_cdf(-d1)
        )
        rho = -strike * time_to_expiration * discount_r * normal_cdf(-d2)

    gamma = discount_q * pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * discount_q * pdf_d1 * sqrt_t

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def implied_volatility_bisection(
    observed_price: float,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-6,
    maximum_iterations: int = 200,
    sigma_lower: float = 1e-6,
    sigma_upper: float = 5.0,
) -> float:
    """
    Recover implied volatility from an observed option price by bisection.

    Returns the volatility ``sigma`` such that
    ``black_scholes_price(..., sigma, option_type) ≈ observed_price``.

    Raises :class:`ValueError` if ``observed_price`` is below the option's
    intrinsic value (which would imply a negative time value and therefore
    no real solution).
    """
    if observed_price < 0:
        raise ValueError("observed_price must be nonnegative.")

    normalized_type = _normalize_option_type(option_type)
    if time_to_expiration <= 0:
        intrinsic = _intrinsic_value(spot, strike, normalized_type)
        if not math.isclose(observed_price, intrinsic, abs_tol=tolerance):
            raise ValueError(
                "Cannot back out IV for an expired contract whose observed "
                f"price ({observed_price}) does not equal intrinsic ({intrinsic})."
            )
        return 0.0

    intrinsic = _intrinsic_value(spot, strike, normalized_type)
    if observed_price + tolerance < intrinsic:
        raise ValueError(
            f"observed_price {observed_price} below intrinsic value {intrinsic}; "
            "no Black-Scholes IV solution exists."
        )

    def pricing_residual(sigma: float) -> float:
        return (
            black_scholes_price(
                spot,
                strike,
                time_to_expiration,
                risk_free_rate,
                sigma,
                normalized_type,
                dividend_yield,
            )
            - observed_price
        )

    low_residual = pricing_residual(sigma_lower)
    high_residual = pricing_residual(sigma_upper)

    if low_residual > 0:
        return sigma_lower
    if high_residual < 0:
        return sigma_upper

    lower = sigma_lower
    upper = sigma_upper
    for _ in range(maximum_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = pricing_residual(midpoint)
        if abs(midpoint_residual) < tolerance:
            return midpoint
        if midpoint_residual > 0:
            upper = midpoint
        else:
            lower = midpoint
        if (upper - lower) < tolerance:
            return 0.5 * (lower + upper)
    return 0.5 * (lower + upper)


def calculate_structure_greeks(
    legs: Iterable[Union[Mapping[str, object], object]],
    spot: float,
    as_of_date: DateLike,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    days_per_year: float = 365.0,
) -> dict[str, float]:
    """
    Sum signed Greeks across the legs of a structure.

    Each leg may be a :class:`src.types.OptionLeg`, a dict, or any object
    with attributes ``strike``, ``option_type``, ``expiration``,
    ``quantity``, ``implied_volatility``. The ``quantity`` field is signed:
    positive = long, negative = short.

    Returns a dict ``{"delta", "gamma", "theta", "vega", "rho"}``.
    """
    aggregate = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    as_of = pd.Timestamp(as_of_date)

    for leg in legs:
        strike = _read_leg_field(leg, "strike")
        option_type = _read_leg_field(leg, "option_type")
        expiration = _read_leg_field(leg, "expiration")
        quantity = _read_leg_field(leg, "quantity")
        sigma = _read_leg_field(leg, "implied_volatility")

        if sigma is None:
            raise ValueError("Every leg must have implied_volatility set.")

        time_to_expiration = (pd.Timestamp(expiration) - as_of).days / days_per_year
        if time_to_expiration <= 0:
            continue

        leg_greeks = calculate_option_greeks(
            spot=spot,
            strike=float(strike),
            time_to_expiration=float(time_to_expiration),
            risk_free_rate=risk_free_rate,
            sigma=float(sigma),
            option_type=str(option_type),
            dividend_yield=dividend_yield,
        )
        scale = float(quantity)
        for greek_name, greek_value in leg_greeks.items():
            aggregate[greek_name] += scale * greek_value
    return aggregate


def _normalize_option_type(option_type: OptionType) -> str:
    if not isinstance(option_type, str):
        raise TypeError("option_type must be a string ('call' or 'put').")
    normalized = option_type.strip().lower()
    if normalized not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")
    return normalized


def _intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    if option_type == "call":
        return float(max(spot - strike, 0.0))
    return float(max(strike - spot, 0.0))


def _read_leg_field(leg: object, field_name: str) -> object:
    """Read ``field_name`` from a dict-like or attribute-bearing leg object."""
    if isinstance(leg, Mapping):
        if field_name not in leg:
            raise ValueError(f"Leg dict is missing '{field_name}'.")
        return leg[field_name]
    if hasattr(leg, field_name):
        return getattr(leg, field_name)
    raise ValueError(f"Leg object does not expose '{field_name}'.")
