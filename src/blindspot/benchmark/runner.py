"""Benchmark runner: one model call per case, the same oracle probes for every policy.

Protocol (see docs/BENCHMARK_PROTOCOL.md)
-----------------------------------------
* A probe is a deterministic function of (image, case): every policy that runs the same probe in
  the same case sees the same measurement, so policies are compared on identical evidence.
* A policy runs its probes in order and stops as soon as a probe closes the case. Failing to
  close costs every probe run.
* Muse: its chosen probes run first. If none closes the case (or its reply was invalid), the
  remaining probes run in the fixed order of ``TOOLS``. An explicit abstention is final.
    * Baselines: ``heuristic`` and ``elimination`` abstain without a supported pilot recovery drop.
    * Hidden case labels are used for scoring only, never to choose or stop an action.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from blindspot.agent.muse import MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.baselines import plan_probes
from blindspot.benchmark.generator import PrivateCase, case_seed, private_case, public_observation
from blindspot.benchmark.observations import features_from_observation, oracle_measurements, recovery_anomaly
from blindspot.benchmark.report import BASELINES, summarise
from blindspot.benchmark.scoring import closure_accepted, score_trace
from blindspot.metrics.completeness import gain


@dataclass(frozen=True)
class PreparedCase:
    """Everything deterministic about a case, computed once and shared by all policies."""

    case: dict
    truth: PrivateCase
    observation: dict
    features: dict
    oracle: dict  # probe -> {"before", "after", "n", "gain", "closed"}


def prepare_case(raw: np.ndarray, case: dict) -> PreparedCase:
    truth = private_case(case)
    observation = public_observation(raw, case)
    measures = oracle_measurements(raw, truth.condition, case_seed=case_seed(case))
    oracle = {
        p: {**m, "gain": gain(m["before"], m["after"], m["n"]), "closed": closure_accepted(**m)}
        for p, m in measures.items()
    }
    return PreparedCase(case, truth, observation, features_from_observation(observation), oracle)


def _execute(order: Iterable[str], prepared: PreparedCase) -> dict:
    no_intervention = prepared.case["kind"] == "no_intervention"
    executed: list[str] = []
    for probe in order:
        if probe in executed or probe not in TOOLS:
            raise ValueError(f"Invalid probe order: {probe}")
        executed.append(probe)
        if prepared.oracle[probe]["closed"]:
            break
    pilots = [{k: prepared.oracle[p][k] for k in ("before", "after", "n")} for p in executed]
    result = score_trace(pilots, no_intervention=no_intervention)
    first = bool(executed) and not no_intervention and prepared.oracle[executed[0]]["closed"]
    return {"executed": executed, **result, "first_probe_closed": first}


def _muse_chosen(record: dict, live: bool) -> list[str]:
    if record["state"] == "reserved_unknown":
        raise RuntimeError("Unresolved reservation in the ledger: inspect it with 'blindspot ledger'")
    if str(record.get("response_id", "offline-")).startswith("offline-") == live:
        raise RuntimeError("Ledger mode mismatch: a dry-run ledger cannot be mixed with a live one")
    return list(record["decision"]["probes"]) if record["state"] == "completed" else []


def evaluate_case(prepared: PreparedCase, *, agent: MuseAgent, client, allow_paid: bool = False) -> dict:
    case = prepared.case
    record = agent.decide(case["id"], prepared.observation, client=client, allow_paid=allow_paid)
    live = not isinstance(client, OfflineMuseClient)
    chosen = _muse_chosen(record, live)
    abstained = record["state"] == "completed" and record["decision"]["decision"] == "abstain"
    order = list(chosen)
    if not abstained:
        order += [p for p in TOOLS if p not in chosen]  # fixed-order continuation
    anomaly = recovery_anomaly(prepared.observation)
    policies = {
        name: _execute(plan_probes(name, prepared.features, anomaly=anomaly, seed=case_seed(case)), prepared)
        for name in BASELINES
    }
    policies["muse"] = _execute(order, prepared)
    used_fallback = len(policies["muse"]["executed"]) > len(chosen)
    return {
        "id": case["id"],
        "field": case["field"],
        "kind": case["kind"],
        "causes": list(prepared.truth.causes),
        "dominant": prepared.truth.dominant,
        "masked": list(prepared.truth.masked),
        "features": prepared.features,
        "recovery_anomaly": anomaly,
        "oracle": prepared.oracle,
        "solvable": any(m["closed"] for m in prepared.oracle.values()),
        "policies": {
            n: {k: v for k, v in t.items() if k != "executed"} | {"executed": t["executed"]}
            for n, t in policies.items()
        },
        "muse": {
            "state": record["state"],
            "decision": record.get("decision"),
            "cost_usd": record["cost_usd"],
            "request_sha256": record["request_sha256"],
            "used_fallback": used_fallback,
            "closed_without_fallback": bool(chosen) and policies["muse"]["success"] and not used_fallback,
        },
    }


def run_benchmark(
    cases: list[dict],
    get_raw: Callable[[dict], np.ndarray],
    *,
    agent: MuseAgent,
    client,
    allow_paid: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run every case; resumable because the agent's ledger answers repeated cases without a call."""
    rows = []
    for case in cases:
        row = evaluate_case(
            prepare_case(get_raw(case), case), agent=agent, client=client, allow_paid=allow_paid
        )
        rows.append(row)
        if progress:
            progress(row)
    return {"cases": rows, "summary": summarise(rows), "ledger": agent.summary()}
