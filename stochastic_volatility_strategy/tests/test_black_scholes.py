"""
Regression tests for ``src/black_scholes.py``.

Purpose
-------
Pins the three properties of the Black-Scholes implementation that the rest
of the pipeline silently depends on:

    1. ``test_call_put_parity``
       For matched (S, K, T, r, sigma, q):
           C - P == S * exp(-q*T) - K * exp(-r*T)
       If this fails, every signed P&L using mixed structures is wrong.

    2. ``test_call_delta_bounds``
       Call delta must lie in [0, 1] and put delta in [-1, 0] over a
       reasonable grid of moneyness/maturity/volatility. Bounds violations
       are usually caused by a sign or dividend mistake in the formula.

    3. ``test_implied_volatility_recovers_known_input``
       Forward price -> bisection IV -> reprice must round-trip to the
       same volatility within tight tolerance. If this fails the signal
       layer's "market-implied variance" feature is unreliable.

Module connections
------------------
Upstream (this file imports from):
    - ``pytest`` / ``math``                          : runner / math.
    - ``src.black_scholes.{black_scholes_price,
                            calculate_option_greeks,
                            implied_volatility_bisection}`` : units under test.

Downstream:
    - ``pytest tests/test_black_scholes.py``
    - ``python3 tests/test_black_scholes.py`` (also works thanks to the
      ``__main__`` block below).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# sys.path shim so direct ``python3 tests/test_black_scholes.py`` works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.black_scholes import (
    black_scholes_price,
    calculate_option_greeks,
    implied_volatility_bisection,
)


PARITY_CASES = [
    # (spot, strike, T_years, r, sigma, q)
    (100.0, 100.0, 0.5, 0.03, 0.20, 0.0),
    (100.0, 95.0, 1.0, 0.05, 0.30, 0.02),
    (250.0, 260.0, 0.25, 0.04, 0.18, 0.015),
    (450.0, 460.0, 0.10, 0.045, 0.22, 0.01),
]


@pytest.mark.parametrize("spot, strike, time, r, sigma, q", PARITY_CASES)
def test_call_put_parity(
    spot: float, strike: float, time: float, r: float, sigma: float, q: float
) -> None:
    """``C - P == S * exp(-qT) - K * exp(-rT)`` for matched parameters."""
    call_price = black_scholes_price(spot, strike, time, r, sigma, "call", q)
    put_price = black_scholes_price(spot, strike, time, r, sigma, "put", q)

    expected = spot * math.exp(-q * time) - strike * math.exp(-r * time)
    assert math.isclose(call_price - put_price, expected, abs_tol=1e-9, rel_tol=1e-9), (
        f"Put-call parity violated: C-P={call_price - put_price}, "
        f"expected {expected}."
    )


@pytest.mark.parametrize(
    "spot, strike, time, sigma",
    [
        (100.0, 80.0, 0.25, 0.20),
        (100.0, 100.0, 0.5, 0.20),
        (100.0, 120.0, 0.5, 0.30),
        (100.0, 100.0, 1.0, 0.10),
        (100.0, 100.0, 1.0, 0.80),
        (100.0, 60.0, 2.0, 0.15),
    ],
)
def test_call_delta_bounds(spot: float, strike: float, time: float, sigma: float) -> None:
    """Call delta in [0, 1], put delta in [-1, 0] across a reasonable grid."""
    risk_free_rate = 0.04

    call = calculate_option_greeks(spot, strike, time, risk_free_rate, sigma, "call")
    put = calculate_option_greeks(spot, strike, time, risk_free_rate, sigma, "put")

    assert 0.0 <= call["delta"] <= 1.0, f"call delta out of bounds: {call['delta']}"
    assert -1.0 <= put["delta"] <= 0.0, f"put delta out of bounds: {put['delta']}"

    assert call["gamma"] >= 0.0
    assert put["gamma"] >= 0.0
    assert call["vega"] >= 0.0
    assert put["vega"] >= 0.0


@pytest.mark.parametrize(
    "spot, strike, time, sigma_true, option_type",
    [
        (100.0, 100.0, 0.50, 0.20, "call"),
        (100.0, 110.0, 0.25, 0.30, "call"),
        (100.0, 90.0, 0.75, 0.15, "put"),
        (450.0, 460.0, 0.10, 0.18, "call"),
        (450.0, 440.0, 0.10, 0.22, "put"),
    ],
)
def test_implied_volatility_recovers_known_input(
    spot: float, strike: float, time: float, sigma_true: float, option_type: str
) -> None:
    """Forward price under known sigma must roundtrip back to that sigma."""
    risk_free_rate = 0.035

    price = black_scholes_price(spot, strike, time, risk_free_rate, sigma_true, option_type)
    recovered_sigma = implied_volatility_bisection(
        observed_price=price,
        spot=spot,
        strike=strike,
        time_to_expiration=time,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
        tolerance=1e-8,
    )
    assert math.isclose(recovered_sigma, sigma_true, abs_tol=1e-5), (
        f"IV roundtrip failed: recovered {recovered_sigma}, expected {sigma_true}."
    )


if __name__ == "__main__":
    # Allows ``python3 tests/test_black_scholes.py`` to run the file under
    # pytest.
    raise SystemExit(pytest.main([__file__, "-v"]))
