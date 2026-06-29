"""Apple earnings-volatility research package.

Modules in this package implement the empirical pipeline behind the paper
``Can Earnings Volatility Be Timed? Forecasting Apple's Implied-Realized
Variance Spread with Public Data.``

The pipeline intentionally relies only on ``numpy``, ``pandas`` and
``matplotlib`` plus the Python standard library, so every step is
transparent and reproducible from a single prepared CSV file.
"""

__all__ = [
    "data_loader",
    "features",
    "regression",
    "forecasting",
    "metrics",
    "plots",
]
