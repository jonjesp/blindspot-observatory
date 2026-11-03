"""Probe-count scoring. A case counts as closed only when the gain is statistically supported.

A probe closes a case when its holdout recovery gain is at least the threshold AND the
conservative lower bound of the gain (Wilson non-overlap) is above zero. Failing to close is a
failure at its true probe cost, never a fast win.
"""

from __future__ import annotations

from blindspot.agent.schemas import TOOLS
from blindspot.config import EXPERIMENT
from blindspot.metrics.completeness import gain, gain_lower_bound


def closure_accepted(before: int, after: int, n: int, threshold: float | None = None) -> bool:
    threshold = EXPERIMENT.benchmark.gain_threshold if threshold is None else threshold
    return gain(before, after, n) >= threshold and gain_lower_bound(before, after, n) > 0


def score_trace(pilots: list[dict], *, no_intervention: bool, threshold: float | None = None) -> dict:
    """Score the probes executed for one case.

    ``pilots`` holds one ``{"before", "after", "n"}`` holdout measurement per executed probe, in order.
    """
    threshold = EXPERIMENT.benchmark.gain_threshold if threshold is None else threshold
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    if len(pilots) > min(EXPERIMENT.benchmark.max_probes, len(TOOLS)):
        raise ValueError("Trace has more probes than the protocol allows")
    # Validate every measurement first so malformed trailing data cannot hide from the budget.
    gains = [gain(p["before"], p["after"], p["n"]) for p in pilots]
    if no_intervention:
        return {
            "success": not pilots,
            "correct_abstention": not pilots,
            "false_alarm": bool(pilots),
            "probes": len(pilots),
        }
    closed = [closure_accepted(p["before"], p["after"], p["n"], threshold) for p in pilots]
    unsupported = sum(1 for g, c in zip(gains, closed, strict=True) if g >= threshold and not c)
    for i, (g, ok) in enumerate(zip(gains, closed, strict=True), start=1):
        if ok:
            if i != len(pilots):
                raise ValueError("Invalid trace: probes recorded after the case was closed")
            return {
                "success": True,
                "probes": i,
                "gain": g,
                "censored": False,
                "unsupported_gains": unsupported,
            }
    return {"success": False, "probes": len(pilots), "censored": True, "unsupported_gains": unsupported}
