"""
Shared pytest fixtures for the project's regression test suite.

Purpose
-------
Defines small in-memory dataframes and bundles that match the column
contracts in ``data/README.md`` (and therefore the ones declared in
``src.data_loader``/``src.data_validation``). Tests in
``tests/test_*.py`` import these fixtures by name; pytest resolves them
automatically via this ``conftest.py``.

Module connections
------------------
Upstream (this file imports from):
    - ``pandas``                       : dataframe construction.
    - ``src.types.MarketDataBundle``   : container assembled by the
                                         ``market_bundle`` fixture.

Downstream (this file is used by):
    - Every file under ``tests/`` that needs synthetic data. Currently:
        - ``tests/test_data_validation.py``
        - ``tests/test_no_lookahead.py``
        - ``tests/test_real_data.py`` (uses ``real_paths`` / ``real_market_bundle``)
      As more modules ship, future test files (``tests/test_black_scholes.py``
      etc.) will also pull from these fixtures so the synthetic data stays
      in one place.

Real-data fixtures
------------------
``real_paths`` and ``real_market_bundle`` lazily load the CSVs under
``data/raw/`` (as configured in ``config.json``). Any test that doesn't
request them pays nothing. If the project-level ``config.json`` or the raw
CSVs are missing, the fixtures call ``pytest.skip(...)`` so the rest of the
suite keeps running.

Also exposes :func:`sys.path` adjustment so tests can ``import src.*`` when
pytest is invoked from the project root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_paths  # noqa: E402
from src.data_loader import load_all_market_data  # noqa: E402
from src.types import MarketDataBundle  # noqa: E402


@pytest.fixture
def underlying_frame() -> pd.DataFrame:
    """Clean three-day SPY OHLCV frame that satisfies validate_underlying_data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [470.0, 471.0, 472.0],
            "high": [472.0, 473.0, 473.5],
            "low": [469.5, 470.5, 470.0],
            "close": [471.0, 472.0, 472.5],
            "adjusted_close": [470.5, 471.5, 472.0],
            "volume": [80_000_000, 85_000_000, 90_000_000],
        }
    )


@pytest.fixture
def option_frame() -> pd.DataFrame:
    """Clean option-chain frame that satisfies validate_option_data."""
    return pd.DataFrame(
        {
            "quote_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "expiration": pd.to_datetime(
                ["2024-02-16", "2024-02-16", "2024-02-16", "2024-02-16"]
            ),
            "option_type": ["call", "put", "call", "call"],
            "strike": [470.0, 470.0, 470.0, 475.0],
            "bid": [5.10, 4.90, 5.30, 3.40],
            "ask": [5.30, 5.10, 5.50, 3.60],
            "last": [5.20, 5.00, 5.40, 3.50],
            "volume": [1200, 1100, 800, 900],
            "open_interest": [5500, 5200, 5600, 4800],
            "implied_volatility": [0.18, 0.19, 0.182, 0.175],
        }
    )


@pytest.fixture
def volatility_index_frame() -> pd.DataFrame:
    """Clean VIX-family frame that satisfies validate_volatility_index_data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "vix": [13.5, 13.7, 13.6],
            "vix9d": [12.8, 13.0, 12.9],
            "vix3m": [15.0, 15.1, 15.05],
        }
    )


@pytest.fixture
def risk_free_frame() -> pd.DataFrame:
    """Clean risk-free-rate frame that satisfies validate_risk_free_data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "annual_rate": [0.0533, 0.0532, 0.0531],
        }
    )


@pytest.fixture
def market_bundle(
    underlying_frame: pd.DataFrame,
    option_frame: pd.DataFrame,
    volatility_index_frame: pd.DataFrame,
    risk_free_frame: pd.DataFrame,
) -> MarketDataBundle:
    """Clean MarketDataBundle that satisfies validate_all_data."""
    return MarketDataBundle(
        underlying=underlying_frame,
        options=option_frame,
        volatility_indices=volatility_index_frame,
        risk_free_rates=risk_free_frame,
    )


# ---------------------------------------------------------------------------
# Real-data fixtures (loaded from data/raw/ via the project config).
# These are session-scoped so the CSVs are read only once across the run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_paths() -> dict[str, Path]:
    """Absolute paths to the real CSVs declared in ``config.json``."""
    config_path = PROJECT_ROOT / "config.json"
    if not config_path.is_file():
        pytest.skip(f"config.json is missing at {config_path}")
    config = load_config(config_path)
    resolved = resolve_project_paths(config, PROJECT_ROOT)
    required = ("underlying_csv", "options_csv", "vix_csv", "risk_free_csv")
    for key in required:
        if key not in resolved or not resolved[key].is_file():
            pytest.skip(
                f"Real data file is missing for paths.{key}: "
                f"{resolved.get(key, '(unresolved)')}"
            )
    return resolved


@pytest.fixture(scope="session")
def real_verification() -> dict[str, object]:
    """Ground-truth row counts and date ranges shipped with the data pack."""
    verification_path = PROJECT_ROOT / "data" / "raw" / "verification.json"
    if not verification_path.is_file():
        pytest.skip(f"verification.json is missing at {verification_path}")
    with verification_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def real_market_bundle(real_paths: dict[str, Path]) -> MarketDataBundle:
    """``MarketDataBundle`` built from the real CSVs under ``data/raw/``."""
    return load_all_market_data(real_paths)
