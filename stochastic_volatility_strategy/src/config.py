"""
Configuration loading, validation, and path resolution for the
stochastic-volatility options strategy.

Purpose
-------
Single entry point for reading project configuration. Parses ``config.json``
(or any equivalent file the caller hands it), checks that every required
section is present and internally consistent (date ordering, parameter
ranges, etc.), and converts the relative paths declared under ``"paths"``
into absolute :class:`pathlib.Path` objects rooted at the project directory.

The configuration file is expected to be a JSON document with the following
top-level sections (see ``config.json`` at the project root):

    paths, dates, model, signal, options, risk, execution

Module connections
------------------
Upstream (this module imports from):
    - Python standard library only (``json``, ``pathlib``, ``datetime``).
      No project module is imported here, so config.py stays loadable even
      before the rest of the pipeline is implemented.

Downstream (this module is imported / called by):
    - ``src.run_pipeline.main``   : calls ``load_config`` at startup, then
                                    ``resolve_project_paths`` to build the
                                    mapping passed to
                                    ``src.data_loader.load_all_market_data``.
    - ``src.data_loader``         : receives the ``"paths"`` mapping
                                    produced here. Path keys are shared:
                                    ``underlying_csv``, ``options_csv``,
                                    ``vix_csv``, ``risk_free_csv``.
    - Decision-layer modules      : read parameters from the ``model``,
                                    ``signal``, ``options``, ``risk``, and
                                    ``execution`` sections returned by
                                    ``load_config``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS: tuple[str, ...] = (
    "paths",
    "dates",
    "model",
    "signal",
    "options",
    "risk",
    "execution",
)

REQUIRED_DATE_KEYS: tuple[str, ...] = (
    "development_start",
    "validation_start",
    "test_start",
    "end",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Read a JSON configuration file, validate it, and return the parsed dict.

    Parameters
        config_path:
            Path to the JSON configuration file.

    Returns
        dict[str, Any]
            The validated configuration as a plain Python dictionary.

    Raises
        FileNotFoundError:
            If ``config_path`` does not point at an existing file.
        json.JSONDecodeError:
            If the file is not valid JSON.
        ValueError:
            Propagated from :func:`validate_config` when the contents are invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate that the configuration contains every required section, that the
    date sections are properly ordered, and that parameter ranges are sane.

    Parameters
        config:
            Parsed configuration dictionary.

    Raises
        TypeError
            If ``config`` is not a mapping.
        ValueError
            If any required section is missing, any date is unparseable or out of
            order, or any parameter is outside its expected range.
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a JSON object (dict).")

    missing_sections = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    _validate_dates_section(config["dates"])
    _validate_model_section(config["model"])
    _validate_signal_section(config["signal"])
    _validate_options_section(config["options"])
    _validate_risk_section(config["risk"])
    _validate_execution_section(config["execution"])
    _validate_paths_section(config["paths"])


def resolve_project_paths(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Path]:
    """
    Convert each entry in ``config["paths"]`` into an absolute :class:`Path`.

    Absolute paths in the configuration are returned unchanged. Relative paths
    are interpreted relative to ``project_root``.

    Parameters
        config:
            Parsed configuration dictionary. Must contain a ``"paths"`` section.
        project_root:
            Directory used to anchor relative paths (typically the repository
            root).

    Returns
        dict[str, Path]
            Mapping from the original key (e.g. ``"underlying_csv"``) to the
            resolved absolute path.
    """
    root = Path(project_root).resolve()
    paths_section = config.get("paths", {})
    if not isinstance(paths_section, dict):
        raise TypeError("Configuration 'paths' section must be a dict.")

    resolved: dict[str, Path] = {}
    for key, value in paths_section.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        resolved[key] = candidate
    return resolved


def _validate_dates_section(dates: dict[str, Any]) -> None:
    if not isinstance(dates, dict):
        raise TypeError("Configuration 'dates' section must be a dict.")

    missing = [key for key in REQUIRED_DATE_KEYS if key not in dates]
    if missing:
        raise ValueError(f"Missing required date keys: {missing}")

    parsed: dict[str, date] = {}
    for key in REQUIRED_DATE_KEYS:
        raw = dates[key]
        try:
            parsed[key] = date.fromisoformat(str(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ISO date for '{key}': {raw!r}") from exc

    ordered = (
        parsed["development_start"]
        < parsed["validation_start"]
        < parsed["test_start"]
        < parsed["end"]
    )
    if not ordered:
        raise ValueError(
            "Date sections must satisfy "
            "development_start < validation_start < test_start < end."
        )


def _validate_model_section(model: dict[str, Any]) -> None:
    _require_positive_int(model, "model", "lookback_days")
    _require_positive_int(model, "model", "forecast_horizon_days")
    _require_positive_int(model, "model", "simulation_paths")
    if "random_seed" in model and not isinstance(model["random_seed"], int):
        raise ValueError("model.random_seed must be an integer when present.")


def _validate_signal_section(signal: dict[str, Any]) -> None:
    _require_positive_int(signal, "signal", "rolling_zscore_window")
    exit_ = _require_nonnegative_number(signal, "signal", "exit_threshold")
    if "long_vol_z_threshold" in signal or "short_vol_z_threshold" in signal:
        long_thr = _require_nonnegative_number(signal, "signal", "long_vol_z_threshold")
        short_thr = _require_nonnegative_number(signal, "signal", "short_vol_z_threshold")
        if exit_ > min(long_thr, short_thr):
            raise ValueError(
                "signal.exit_threshold must be <= both long and short z thresholds."
            )
    elif "entry_threshold" in signal:
        entry = _require_nonnegative_number(signal, "signal", "entry_threshold")
        if exit_ > entry:
            raise ValueError("signal.exit_threshold must be <= signal.entry_threshold.")
    else:
        raise ValueError(
            "signal section must include long_vol_z_threshold and short_vol_z_threshold "
            "or legacy entry_threshold."
        )
    if "edge_safety_buffer" in signal:
        _require_nonnegative_number(signal, "signal", "edge_safety_buffer")


def _validate_options_section(options: dict[str, Any]) -> None:
    min_dte = _require_positive_int(options, "options", "minimum_dte")
    max_dte = _require_positive_int(options, "options", "maximum_dte")
    if max_dte < min_dte:
        raise ValueError("options.maximum_dte must be >= options.minimum_dte.")
    if "target_dte" in options:
        _require_positive_int(options, "options", "target_dte")
    if "atm_moneyness_low" in options:
        low = _require_nonnegative_number(options, "options", "atm_moneyness_low")
        if low <= 0:
            raise ValueError("options.atm_moneyness_low must be positive.")
    if "atm_moneyness_high" in options:
        high = _require_nonnegative_number(options, "options", "atm_moneyness_high")
        if high <= 0:
            raise ValueError("options.atm_moneyness_high must be positive.")
    if "atm_moneyness_low" in options and "atm_moneyness_high" in options:
        if options["atm_moneyness_high"] < options["atm_moneyness_low"]:
            raise ValueError("options.atm_moneyness_high must be >= options.atm_moneyness_low.")
    _require_fraction_open_interval(options, "options", "maximum_relative_spread")
    _require_nonnegative_number(options, "options", "minimum_open_interest")
    _require_nonnegative_number(options, "options", "minimum_volume")
    _require_fraction_open_interval(options, "options", "wing_width_percent")


def _validate_risk_section(risk: dict[str, Any]) -> None:
    capital = _require_nonnegative_number(risk, "risk", "initial_capital")
    if capital <= 0:
        raise ValueError("risk.initial_capital must be positive.")
    _require_fraction_open_interval(risk, "risk", "target_annual_volatility")
    _require_fraction_half_open(risk, "risk", "maximum_trade_risk_fraction")
    _require_fraction_open_interval(risk, "risk", "maximum_drawdown")
    _require_positive_int(risk, "risk", "maximum_open_positions")
    if "maximum_absolute_delta" in risk:
        _require_nonnegative_number(risk, "risk", "maximum_absolute_delta")


def _validate_execution_section(execution: dict[str, Any]) -> None:
    _require_nonnegative_number(execution, "execution", "option_commission_per_contract")
    fraction = _require_nonnegative_number(
        execution, "execution", "option_slippage_fraction_of_spread"
    )
    if fraction > 1:
        raise ValueError(
            "execution.option_slippage_fraction_of_spread must be in [0, 1]."
        )
    _require_nonnegative_number(execution, "execution", "stock_slippage_bps")


def _validate_paths_section(paths: dict[str, Any]) -> None:
    if not isinstance(paths, dict):
        raise TypeError("Configuration 'paths' section must be a dict.")
    if not paths:
        raise ValueError("Configuration 'paths' section must not be empty.")
    for key, value in paths.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a non-empty string.")


def _require_positive_int(section: dict[str, Any], section_name: str, key: str) -> int:
    value = _require_key(section, section_name, key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{section_name}.{key} must be a positive integer.")
    return value


def _require_nonnegative_number(
    section: dict[str, Any], section_name: str, key: str
) -> float:
    value = _require_key(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{section_name}.{key} must be a nonnegative number.")
    return float(value)


def _require_fraction_open_interval(
    section: dict[str, Any], section_name: str, key: str
) -> float:
    """Require a number in the open interval (0, 1)."""
    value = _require_key(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section_name}.{key} must be a number in (0, 1).")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{section_name}.{key} must be in the open interval (0, 1).")
    return float(value)


def _require_fraction_half_open(
    section: dict[str, Any], section_name: str, key: str
) -> float:
    """Require a number in the half-open interval (0, 1]."""
    value = _require_key(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section_name}.{key} must be a number in (0, 1].")
    if not 0.0 < float(value) <= 1.0:
        raise ValueError(f"{section_name}.{key} must be in (0, 1].")
    return float(value)


def _require_key(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key: {section_name}.{key}")
    return section[key]
