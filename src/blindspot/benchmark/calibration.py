"""Calibration of the case generator on the real images.

Runs every planned case, measures the effect of its private intervention with the oracle probes,
and summarises how well separated the causes are. It never calls the model.
"""

from __future__ import annotations

from statistics import mean

from blindspot import __version__
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.design import plan, plan_hash
from blindspot.benchmark.generator import case_seed, private_case, public_observation
from blindspot.benchmark.observations import oracle_measurements
from blindspot.benchmark.scoring import closure_accepted
from blindspot.datasets.ztf import extract_patch, image_manifest, load_verified_image
from blindspot.metrics.completeness import gain


def _case_row(raw, case: dict) -> dict:
    truth = private_case(case)
    observation = public_observation(raw, case)
    measures = oracle_measurements(raw, truth.condition, case_seed=case_seed(case))
    counts = observation["pilot_measurements"]["recovery_counts"]["nominal"]
    n = sum(1 for _ in counts["A"]) * observation["pilot_measurements"]["sources_per_batch"]
    return {
        "id": case["id"],
        "field": case["field"],
        "kind": case["kind"],
        "causes": list(truth.causes),
        "dominant": truth.dominant,
        "masked": list(truth.masked),
        "pilot_recovery_A": sum(counts["A"]) / n,
        "pilot_recovery_B": sum(counts["B"]) / n,
        "indicators": observation["indicators"],
        "gains": {p: gain(m["before"], m["after"], m["n"]) for p, m in measures.items()},
        "closures": {p: closure_accepted(m["before"], m["after"], m["n"]) for p, m in measures.items()},
    }


def _summarise(rows: list[dict]) -> dict:
    by_kind = {
        k: [r for r in rows if r["kind"] == k] for k in ("no_intervention", "single_cause", "two_causes")
    }
    summary: dict = {}
    none = by_kind["no_intervention"]
    summary["no_intervention"] = {
        "cases": len(none),
        "false_closures": sum(any(r["closures"].values()) for r in none),
        "max_indicator": max(max(r["indicators"].values()) for r in none),
    }
    single = by_kind["single_cause"]
    others = [(r["gains"][p], r["closures"][p]) for r in single for p in TOOLS if p not in r["causes"]]
    summary["single_cause"] = {
        "cases": len(single),
        "pilot_recovery_A": mean(r["pilot_recovery_A"] for r in single),
        "pilot_recovery_B": mean(r["pilot_recovery_B"] for r in single),
        "true_probe_gain_mean": mean(r["gains"][r["causes"][0]] for r in single),
        "true_probe_gain_min": min(r["gains"][r["causes"][0]] for r in single),
        "true_probe_closed": sum(r["closures"][r["causes"][0]] for r in single),
        "other_probe_gain_max": max(g for g, _ in others),
        "other_probe_false_closures": sum(c for _, c in others),
        "other_probe_trials": len(others),
    }
    two = by_kind["two_causes"]
    minor = lambda r: next(c for c in r["causes"] if c != r["dominant"])  # noqa: E731
    summary["two_causes"] = {
        "cases": len(two),
        "dominant_gain_mean": mean(r["gains"][r["dominant"]] for r in two),
        "dominant_closed": sum(r["closures"][r["dominant"]] for r in two),
        "minor_gain_mean": mean(r["gains"][minor(r)] for r in two),
        "minor_closed": sum(r["closures"][minor(r)] for r in two),
        "max_gain_probe_is_dominant": sum(max(r["gains"], key=r["gains"].get) == r["dominant"] for r in two),
    }
    return summary


def calibrate(fits_archive) -> dict:
    """Return the calibration summary and one row per case. Takes about a minute."""
    images = {e["field"]: load_verified_image(fits_archive, e["field"])[0] for e in image_manifest()}
    rows = [
        _case_row(extract_patch(images[c["field"]], c["offset_y"], c["offset_x"]), c) for c in plan()["cases"]
    ]
    return {"version": __version__, "plan_sha256": plan_hash(), "summary": _summarise(rows), "cases": rows}
