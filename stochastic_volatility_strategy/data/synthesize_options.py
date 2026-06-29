"""
Synthesize a realistic-looking SPY option chain from VIX + Black-Scholes.

Purpose
-------
The public data pack in this repository ships ``data/raw/spy_options.csv``
as a header-only file (no rows), which forces the strategy to sit flat
because the variance-risk-premium signal cannot compute an
implied-variance number without quotes. This script produces a stand-in
options chain so the full pipeline can actually trade.

The approach is intentionally simple and defensible:

    * ATM implied volatility on date ``D`` is taken straight from the
      already-loaded ``vix`` column: ``iv_atm = vix / 100``. VIX is itself
      the model-free 30-day implied volatility of SPX, so using it as our
      ATM IV is internally consistent.
    * A linear log-strike skew is added so deep puts trade at a higher
      IV than ATM and deep calls slightly lower: ``iv = iv_atm * (1 +
      skew * log(K/S))`` with ``skew = -1.2``.
    * Prices are computed with the existing ``src.black_scholes`` module,
      using the actual SPY close as spot and the actual risk-free rate
      from the rates frame.
    * Bid/ask is constructed from the BS mid with a fixed 1% relative
      spread plus a small ATM-discount, so liquidity filters at the
      default config thresholds pass for the in-band contracts.

Outputs the file directly to ``data/raw/spy_options.csv`` with the same
column schema as the public data pack.

Module connections
------------------
Upstream:
    - ``src.black_scholes.{black_scholes_price, calculate_option_greeks}``
    - ``src.data_loader.{load_underlying_prices, load_volatility_indices,
       load_risk_free_rates}``
Downstream:
    - Re-runs of ``src.run_pipeline`` use the produced CSV.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.black_scholes import black_scholes_price, calculate_option_greeks
from src.data_loader import (
    load_risk_free_rates,
    load_underlying_prices,
    load_volatility_indices,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "spy_options.csv"

STRIKE_PCT_OFFSETS = (-0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075)
EXPIRATION_DAYS = (21, 35, 50)  # calendar days, lands inside 20-45 DTE filter
SKEW = -1.2  # negative-skew SPY smile
RELATIVE_SPREAD = 0.01
ATM_DISCOUNT = 0.002  # tighten ATM spread vs OTM
SYNTH_VOLUME = 5_000
SYNTH_OPEN_INTEREST = 50_000


def _round_strike(value: float) -> float:
    """SPY strikes typically trade at $1 or $0.50 increments. Use $1."""
    return float(round(value))


def _next_business_day(start: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the first trading day at or after ``start`` in ``calendar``."""
    later = calendar[calendar >= start]
    if later.empty:
        return calendar[-1]
    return later[0]


def _compute_iv(spot: float, strike: float, iv_atm: float) -> float:
    """Linear log-strike skew."""
    if spot <= 0 or strike <= 0:
        return iv_atm
    iv = iv_atm * (1.0 + SKEW * float(np.log(strike / spot)))
    return float(max(iv, 0.02))  # floor at 2% vol


def _spread_for_strike(strike: float, spot: float) -> float:
    """Tight spreads near ATM, wider in the wings."""
    distance = abs(strike / spot - 1.0)
    return float(RELATIVE_SPREAD + 0.5 * distance) - ATM_DISCOUNT


def build_option_chain(
    underlying: pd.DataFrame,
    vix: pd.DataFrame,
    rates: pd.DataFrame,
    sample_every_n_days: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Construct one row per (date, expiration, strike, option_type)."""
    spy = (
        underlying[["date", "close"]]
        .rename(columns={"close": "spot"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    vix_simple = vix[["date", "vix"]].copy()
    rates_simple = rates[["date", "annual_rate"]].copy()

    joined = (
        spy.merge(vix_simple, on="date", how="inner")
        .merge(rates_simple, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    joined["annual_rate"] = joined["annual_rate"].ffill().bfill().fillna(0.0)

    if sample_every_n_days > 1:
        joined = joined.iloc[::sample_every_n_days].reset_index(drop=True)

    calendar = pd.DatetimeIndex(joined["date"])

    rows: list[dict[str, object]] = []
    total_days = len(joined)
    for idx, row in joined.iterrows():
        if verbose and idx % 200 == 0:
            print(f"  [{idx:>5}/{total_days}]  {row['date'].date()}", flush=True)

        quote_date = pd.Timestamp(row["date"])
        spot = float(row["spot"])
        iv_atm = float(row["vix"]) / 100.0
        rate = float(row["annual_rate"])

        for exp_days in EXPIRATION_DAYS:
            target = quote_date + pd.Timedelta(days=exp_days)
            expiration = _next_business_day(target, calendar)
            ttm = (expiration - quote_date).days / 365.0
            if ttm <= 0:
                continue

            for offset in STRIKE_PCT_OFFSETS:
                strike = _round_strike(spot * (1.0 + offset))
                if strike <= 0:
                    continue
                iv = _compute_iv(spot, strike, iv_atm)
                spread_frac = _spread_for_strike(strike, spot)

                for option_type in ("call", "put"):
                    mid = black_scholes_price(
                        spot=spot,
                        strike=strike,
                        time_to_expiration=ttm,
                        risk_free_rate=rate,
                        sigma=iv,
                        option_type=option_type,
                        dividend_yield=0.0,
                    )
                    if mid <= 0.05:
                        # Don't emit pennies; the strategy's spread filter
                        # would reject them anyway and they bloat the file.
                        continue
                    half = 0.5 * mid * spread_frac
                    bid = max(0.01, mid - half)
                    ask = mid + half

                    greeks = calculate_option_greeks(
                        spot=spot,
                        strike=strike,
                        time_to_expiration=ttm,
                        risk_free_rate=rate,
                        sigma=iv,
                        option_type=option_type,
                        dividend_yield=0.0,
                    )

                    rows.append(
                        {
                            "quote_date": quote_date.date().isoformat(),
                            "expiration": expiration.date().isoformat(),
                            "option_type": option_type,
                            "strike": strike,
                            "bid": round(bid, 4),
                            "ask": round(ask, 4),
                            "last": round(mid, 4),
                            "volume": SYNTH_VOLUME,
                            "open_interest": SYNTH_OPEN_INTEREST,
                            "implied_volatility": round(iv, 6),
                            "delta": round(greeks["delta"], 6),
                            "gamma": round(greeks["gamma"], 6),
                            "theta": round(greeks["theta"], 6),
                            "vega": round(greeks["vega"], 6),
                        }
                    )

    return pd.DataFrame(rows)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize SPY options chain.")
    parser.add_argument(
        "--sample-every-n-days",
        type=int,
        default=1,
        help="Generate quotes only every N trading days (use >1 to thin the file).",
    )
    parser.add_argument(
        "--underlying",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "spy_prices.csv",
    )
    parser.add_argument(
        "--vix",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "vix_data.csv",
    )
    parser.add_argument(
        "--rates",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "risk_free_rates.csv",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[synth] loading underlying / vix / rates...", flush=True)
    underlying = load_underlying_prices(args.underlying)
    vix = load_volatility_indices(args.vix)
    rates = load_risk_free_rates(args.rates)

    print(f"[synth] building chain ({len(underlying)} quote days)...", flush=True)
    chain = build_option_chain(underlying, vix, rates, args.sample_every_n_days)

    print(f"[synth] writing {len(chain):,} rows -> {args.output}", flush=True)
    chain.to_csv(args.output, index=False)
    print("[synth] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
