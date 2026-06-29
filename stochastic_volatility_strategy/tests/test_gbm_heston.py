"""
Regression tests for ``src/gbm.py`` and ``src/heston.py``.

Purpose
-------
Pins three structural invariants of the simulation code:

    1. ``test_gbm_path_shape``
       Simulated GBM paths have the right shape and start at S0.
    2. ``test_heston_variance_nonnegative``
       Full-truncation variance never goes negative in any path.
    3. ``test_heston_shock_correlation``
       The empirical correlation between generated shocks matches the
       requested rho to within Monte Carlo tolerance.

Module connections
------------------
Upstream:
    - ``src.gbm``  / ``src.heston``  : units under test.
Downstream:
    - ``pytest tests/test_gbm_heston.py``
    - ``python3 tests/test_gbm_heston.py`` (also works).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.gbm import estimate_gbm_parameters, simulate_gbm_paths
from src.heston import generate_correlated_shocks, simulate_heston_paths


def test_gbm_path_shape() -> None:
    paths = simulate_gbm_paths(
        initial_price=100.0,
        mu=0.05,
        sigma=0.20,
        horizon_days=21,
        n_paths=500,
        random_seed=0,
    )
    assert paths.shape == (500, 22)
    assert np.allclose(paths[:, 0], 100.0)
    assert (paths > 0).all()


def test_gbm_parameter_estimation_recovers_inputs() -> None:
    rng = np.random.default_rng(seed=42)
    sigma_true = 0.20
    daily_sigma = sigma_true / np.sqrt(252)
    sample = rng.normal(loc=0.0, scale=daily_sigma, size=20_000)
    params = estimate_gbm_parameters(sample)
    assert np.isclose(params["sigma"], sigma_true, rtol=0.05)


def test_heston_variance_nonnegative() -> None:
    parameters = {
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.6,
        "rho": -0.7,
        "v0": 0.04,
        "mu": 0.05,
    }
    prices, variances = simulate_heston_paths(
        initial_price=100.0,
        parameters=parameters,
        horizon_days=40,
        n_paths=300,
        random_seed=1,
    )
    assert prices.shape == (300, 41)
    assert variances.shape == (300, 41)
    assert (variances >= 0).all(), "full-truncation variance went negative."
    assert (prices > 0).all()


@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5, 0.9])
def test_heston_shock_correlation(rho: float) -> None:
    z1, z2 = generate_correlated_shocks(n_paths=20_000, n_steps=1, rho=rho, random_seed=7)
    empirical = float(np.corrcoef(z1.flatten(), z2.flatten())[0, 1])
    assert abs(empirical - rho) < 0.03, (
        f"empirical correlation {empirical} vs requested {rho}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
