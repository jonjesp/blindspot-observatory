"""Private case generator: turns a public plan entry into physical truth and a public observation.

"Private" means isolated from the model prompt, not secret: the generator is deterministic from
the configured seed, so anyone with the code can recompute the causes. Nothing under
``blindspot.agent`` may import this module or ``interventions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.interventions import Condition, apply_causes
from blindspot.benchmark.observations import build_observation
from blindspot.config import EXPERIMENT

_TRUTH_STREAM, _SEED_STREAM, _MASK_STREAM = 7919, 104729, 15485863
MASK_PROBABILITY = 0.5  # chance of one missing diagnostic, independent of class and causes


@dataclass(frozen=True)
class PrivateCase:
    causes: tuple[str, ...]
    masked: tuple[str, ...]
    dominant: str | None
    condition: Condition


def _index(case: dict) -> int:
    case_id = case["id"]
    if not (isinstance(case_id, str) and case_id[:1] == "F" and case_id[1:].isdigit()):
        raise ValueError("Unexpected case id")
    return int(case_id[1:])


def case_seed(case: dict) -> int:
    """Seed for every measurement of this case."""
    return int(np.random.SeedSequence([EXPERIMENT.seed, _SEED_STREAM, _index(case)]).generate_state(1)[0])


def private_case(case: dict) -> PrivateCase:
    rng = np.random.default_rng(np.random.SeedSequence([EXPERIMENT.seed, _TRUTH_STREAM, _index(case)]))
    kind = case["kind"]
    if kind == "no_intervention":
        causes: tuple[str, ...] = ()
    elif kind == "single_cause":
        causes = (TOOLS[int(rng.integers(len(TOOLS)))],)
    elif kind == "two_causes":
        causes = tuple(sorted(TOOLS[int(i)] for i in rng.choice(len(TOOLS), size=2, replace=False)))
    else:
        raise ValueError(f"Unknown case kind: {kind}")
    dominant = causes[int(rng.integers(len(causes)))] if causes else None
    mask_rng = np.random.default_rng(np.random.SeedSequence([EXPERIMENT.seed, _MASK_STREAM, _index(case)]))
    hide = bool(mask_rng.random() < MASK_PROBABILITY)
    masked = (TOOLS[int(mask_rng.integers(len(TOOLS)))],) if hide else ()
    return PrivateCase(
        causes=causes, masked=masked, dominant=dominant, condition=apply_causes(causes, dominant)
    )


def public_observation(raw: np.ndarray, case: dict) -> dict:
    """The only thing the model is allowed to see about ``case``."""
    truth = private_case(case)
    return build_observation(raw, truth.condition, case_seed=case_seed(case), masked=truth.masked)
