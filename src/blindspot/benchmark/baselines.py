"""Baseline probe-ordering policies.

Every policy sees only the public per-probe diagnostic ``features``: the relative change of the
measurement associated with each probe between reference condition A and anomalous condition B.
A value of ``None`` means "not observed" (masked); this is what separates ``elimination`` from
``heuristic``.

* ``random``       - seeded permutation.
* ``heuristic``    - largest observed change first; unobserved probes last.
* ``elimination``  - probes whose diagnostic clearly changed come first; then probes whose
                     diagnostic was NOT observed (they cannot be ruled out); probes whose
                     diagnostic is observed and unchanged are ruled out and come last.
"""

from __future__ import annotations

import math

import numpy as np

from blindspot.agent.schemas import TOOLS
from blindspot.config import EXPERIMENT

METHODS = ("random", "heuristic", "elimination")


def _validate(features: dict) -> None:
    if set(features) != set(TOOLS):
        raise ValueError(f"Features must have exactly the keys {TOOLS}")
    for value in features.values():
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Each feature must be None or a finite non-negative number")


def _by_change(probes: list[str], features: dict) -> list[str]:
    return sorted(probes, key=lambda p: (-features[p], TOOLS.index(p)))


def policy_order(
    method: str,
    features: dict[str, float | None],
    *,
    seed: int = 0,
    tolerance: float | None = None,
) -> tuple[str, ...]:
    """Return the order in which ``method`` would run the three probes."""
    if method not in METHODS:
        raise ValueError(f"Unknown policy: {method}")
    _validate(features)
    if method == "random":
        permutation = np.random.default_rng(seed).permutation(len(TOOLS))
        return tuple(TOOLS[int(i)] for i in permutation)
    observed = [p for p in TOOLS if features[p] is not None]
    unobserved = [p for p in TOOLS if features[p] is None]
    if method == "heuristic":
        return tuple(_by_change(observed, features)) + tuple(unobserved)
    tol = EXPERIMENT.change_tolerance if tolerance is None else tolerance
    if tol < 0:
        raise ValueError("Tolerance must be non-negative")
    changed = [p for p in observed if features[p] > tol]
    ruled_out = [p for p in observed if features[p] <= tol]
    return tuple(_by_change(changed, features)) + tuple(unobserved) + tuple(_by_change(ruled_out, features))


def plan_probes(
    method: str,
    features: dict[str, float | None],
    *,
    anomaly: bool,
    seed: int = 0,
    tolerance: float | None = None,
) -> tuple[str, ...]:
    """Probes a baseline would actually run.

    ``anomaly`` says whether the pilot shows a statistically supported recovery drop (see
    ``observations.recovery_anomaly``). ``heuristic`` and ``elimination`` abstain (return ``()``)
    when it does not; a drop with unchanged diagnostics is still probed. ``random`` never abstains.
    """
    order = policy_order(method, features, seed=seed, tolerance=tolerance)
    if method != "random" and not anomaly:
        return ()
    return order
