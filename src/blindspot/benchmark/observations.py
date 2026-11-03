"""Builds the public observation a model sees, and the holdout probe measurements.

Everything returned by ``build_observation`` is a measurement of conditions A and B; the
condition itself (which factor was changed) is never part of it.
"""

from __future__ import annotations

import numpy as np

from blindspot.agent.observation import validate_observation
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.interventions import NOMINAL, Condition, restore
from blindspot.benchmark.measurement import BATCH_SIZE, count_recovered, make_arm, profile_diagnostics
from blindspot.metrics.completeness import gain_lower_bound

LEVEL_SNR = {"nominal": 6.0, "bright": 9.0}  # nominal injected significance per level
PROFILE_SNR = 12.0
PILOT_BATCHES = 4
HOLDOUT_BATCHES = 4

PROBE_INDICATOR = {
    "probe_sky": "background_mad_change",
    "probe_psf": "rms_radius_change",
    "probe_flux": "integrated_amplitude_change",
}

# Separate pseudo-random streams. They do not guarantee disjoint positions or independent image
# noise: all arms share the real patch, and before/after intentionally share added noise.
_STREAMS = {
    "pilot_a": 1,
    "pilot_b": 2,
    "profile_a": 3,
    "profile_b": 4,
    "holdout_before": 5,
    "noise": 6,
    "holdout_after": 7,
}


def _seeds(case_seed: int, stream: str, n: int) -> list[int]:
    return [int(x) for x in np.random.SeedSequence([case_seed, _STREAMS[stream]]).generate_state(n)]


def _relative_change(reference: float, value: float) -> float:
    if not reference > 0:
        raise ValueError("Reference measurement must be positive")
    return abs(value - reference) / reference


def build_observation(
    raw: np.ndarray, condition: Condition, *, case_seed: int, masked: tuple[str, ...] = ()
) -> dict:
    """Pilot comparison of the nominal condition A against ``condition`` (B).

    ``masked`` probes have their diagnostic indicator omitted, which models a diagnostic that
    was not measured.
    """
    if any(p not in TOOLS for p in masked):
        raise ValueError("masked must list known probes")
    a = make_arm(raw, NOMINAL, noise_seed=0)
    b = make_arm(raw, condition, noise_seed=_seeds(case_seed, "noise", 1)[0])
    seeds_a, seeds_b = (
        _seeds(case_seed, "pilot_a", PILOT_BATCHES),
        _seeds(case_seed, "pilot_b", PILOT_BATCHES),
    )
    counts = {
        level: {
            "A": [count_recovered(a, snr, s, holdout=False) for s in seeds_a],
            "B": [count_recovered(b, snr, s, holdout=False) for s in seeds_b],
        }
        for level, snr in LEVEL_SNR.items()
    }
    amp_a, rad_a = profile_diagnostics(a, PROFILE_SNR, _seeds(case_seed, "profile_a", 1)[0])
    amp_b, rad_b = profile_diagnostics(b, PROFILE_SNR, _seeds(case_seed, "profile_b", 1)[0])
    indicators = {
        "background_mad_change": _relative_change(a.patch["sigma"], b.patch["sigma"]),
        "integrated_amplitude_change": _relative_change(amp_a, amp_b),
        "rms_radius_change": _relative_change(rad_a, rad_b),
    }
    for probe in masked:
        del indicators[PROBE_INDICATOR[probe]]
    return validate_observation(
        {
            "pilot_measurements": {"sources_per_batch": BATCH_SIZE, "recovery_counts": counts},
            "indicators": {k: round(v, 4) for k, v in indicators.items()},
            "available_probes": list(TOOLS),
        }
    )


def recovery_anomaly(observation: dict) -> bool:
    """True when nominal-level recovery in B is lower than in A with statistical support.

    Uses the same conservative interval as the closure rule (Wilson non-overlap), so ordinary
    sampling noise between two identical conditions does not count as an anomaly.
    """
    counts = observation["pilot_measurements"]["recovery_counts"]["nominal"]
    if len(counts["A"]) != len(counts["B"]) or not counts["A"]:
        raise ValueError("Pilot arms must have the same number of batches")
    n = observation["pilot_measurements"]["sources_per_batch"] * len(counts["A"])
    return gain_lower_bound(sum(counts["B"]), sum(counts["A"]), n) > 0


def features_from_observation(observation: dict) -> dict[str, float | None]:
    """Per-probe features for the baseline policies; a masked indicator becomes ``None``."""
    return {p: observation["indicators"].get(PROBE_INDICATOR[p]) for p in TOOLS}


def holdout_measurement(raw: np.ndarray, condition: Condition, probe: str, *, case_seed: int) -> dict:
    """Holdout recovery of B before and after restoring ONE factor.

    Both measurements use the same noise realisation but independent source positions, so restoring
    a factor that was never changed still shows sampling noise, as a real repeat measurement would.
    """
    noise_seed = _seeds(case_seed, "noise", 1)[0]
    seeds_before = _seeds(case_seed, "holdout_before", HOLDOUT_BATCHES)
    seeds_after = _seeds(case_seed, "holdout_after", HOLDOUT_BATCHES)
    snr = LEVEL_SNR["nominal"]
    before_arm = make_arm(raw, condition, noise_seed=noise_seed)
    after_arm = make_arm(raw, restore(condition, probe), noise_seed=noise_seed)
    return {
        "before": sum(count_recovered(before_arm, snr, s, holdout=True) for s in seeds_before),
        "after": sum(count_recovered(after_arm, snr, s, holdout=True) for s in seeds_after),
        "n": HOLDOUT_BATCHES * BATCH_SIZE,
    }


def oracle_measurements(raw: np.ndarray, condition: Condition, *, case_seed: int) -> dict[str, dict]:
    """Holdout measurement for every probe. For scoring and calibration only, never for the agent."""
    return {p: holdout_measurement(raw, condition, p, case_seed=case_seed) for p in TOOLS}
