"""Closed whitelist for what the model may see about a case.

Indicators are relative changes between condition B and condition A, |B - A| / A.
Anything not listed here is rejected, so ground-truth labels, intervention settings and holdout
data cannot reach the prompt through a free-form key. Extend the whitelist in ONE place when a
new public measurement is added, and re-freeze the benchmark afterwards.
"""

from __future__ import annotations

import math

from blindspot.agent.schemas import TOOLS

TOP_LEVEL_KEYS = ("pilot_measurements", "indicators", "available_probes")
MEASUREMENT_KEYS = ("recovery_counts", "sources_per_batch")
RECOVERY_LEVELS = ("nominal", "bright")
RECOVERY_ARMS = ("A", "B")
INDICATOR_KEYS = ("background_mad_change", "integrated_amplitude_change", "rms_radius_change")
MAX_BATCHES = 100


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_observation(observation: object) -> dict:
    """Return ``observation`` unchanged if it only contains whitelisted public measurements."""
    if type(observation) is not dict or set(observation) != set(TOP_LEVEL_KEYS):
        raise ValueError(f"Observation must have exactly the keys {TOP_LEVEL_KEYS}")
    if observation["available_probes"] != list(TOOLS):
        raise ValueError("available_probes does not match the probe contract")

    measurements = observation["pilot_measurements"]
    if type(measurements) is not dict or not set(measurements) <= set(MEASUREMENT_KEYS):
        raise ValueError("Unknown key in pilot_measurements")
    if "sources_per_batch" in measurements:
        value = measurements["sources_per_batch"]
        if not _is_int(value) or value < 1:
            raise ValueError("sources_per_batch must be a positive integer")
    counts = measurements.get("recovery_counts", {})
    if type(counts) is not dict or not set(counts) <= set(RECOVERY_LEVELS):
        raise ValueError("Unknown level in recovery_counts")
    for arms in counts.values():
        if type(arms) is not dict or not set(arms) <= set(RECOVERY_ARMS):
            raise ValueError("Unknown arm in recovery_counts")
        for batches in arms.values():
            if type(batches) is not list or len(batches) > MAX_BATCHES:
                raise ValueError("Recovery counts must be a short list")
            if not all(_is_int(c) and c >= 0 for c in batches):
                raise ValueError("Recovery counts must be non-negative integers")

    indicators = observation["indicators"]
    if type(indicators) is not dict or not set(indicators) <= set(INDICATOR_KEYS):
        raise ValueError("Unknown indicator")
    if not all(_is_number(v) and v >= 0 for v in indicators.values()):
        raise ValueError("Indicators are relative changes: finite and non-negative")
    return observation
