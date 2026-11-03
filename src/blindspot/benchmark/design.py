"""Deterministic 36-case spatial plan (18 non-overlapping sites per exposure).

The plan fixes WHERE each case lives and what class it has. Which physical intervention a
case receives is decided later by a private generator and must never reach the model prompt.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from blindspot.config import DETECTOR, EXPERIMENT
from blindspot.provenance.ledger import digest

PATCH_SIZE = DETECTOR.patch_size


def _non_overlapping(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) >= PATCH_SIZE or abs(a[1] - b[1]) >= PATCH_SIZE


def plan() -> dict:
    bench = EXPERIMENT.benchmark
    sites = [(y, x) for y in bench.y_offsets for x in bench.x_offsets]
    classes = [kind for kind, count in bench.class_counts for _ in range(count)]
    cases: list[dict] = []
    for i, exposure in enumerate(EXPERIMENT.exposures):
        # A per-exposure permutation keeps the public case ID from revealing the class order.
        order = np.random.default_rng(np.random.SeedSequence([EXPERIMENT.seed, i])).permutation(len(classes))
        shuffled = [classes[int(k)] for k in order]
        for j, ((dy, dx), kind) in enumerate(zip(sites, shuffled, strict=True), start=1):
            cases.append(
                {
                    "id": f"F{i * bench.cases_per_exposure + j:02d}",
                    "field": exposure,
                    "offset_y": dy,
                    "offset_x": dx,
                    "kind": kind,
                }
            )
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case IDs are not unique")
    for exposure in EXPERIMENT.exposures:
        coords = [(c["offset_y"], c["offset_x"]) for c in cases if c["field"] == exposure]
        if not all(_non_overlapping(a, b) for a, b in combinations(coords, 2)):
            raise ValueError(f"Overlapping patches in {exposure}")
        if not all(_non_overlapping(a, b) for a in coords for b in bench.excluded_offsets):
            raise ValueError(f"Patches in {exposure} overlap previously used sites")
    return {
        "experiment": "benchmark_geometry",
        "seed": EXPERIMENT.seed,
        "note": "Synthetic interventions on authentic images. Cases share only two FITS exposures.",
        "cases": cases,
    }


def plan_hash() -> str:
    return digest(plan())
