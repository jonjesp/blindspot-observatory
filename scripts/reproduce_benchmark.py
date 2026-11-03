"""Offline reproduction of frozen benchmark measurements from verified FITS inputs.

Inputs comprise stored results, a decision ledger and the original FITS ZIP or directory.
The default scope covers four reference cases. Counts, requests, features and policy traces
must match exactly. Record-only verification leaves pixel reproduction pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml

from blindspot.agent.muse import AGENT, build_request, usage_cost
from blindspot.agent.schemas import TOOLS, validate_decision
from blindspot.benchmark.baselines import plan_probes
from blindspot.benchmark.design import plan
from blindspot.benchmark.freeze import freeze_record
from blindspot.benchmark.generator import case_seed, private_case
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.report import BASELINES, summarise
from blindspot.benchmark.runner import PreparedCase, _execute, prepare_case
from blindspot.benchmark.scoring import closure_accepted
from blindspot.datasets.ztf import extract_patch, load_verified_image
from blindspot.metrics.completeness import gain
from blindspot.provenance.ledger import digest, save_json

DEFAULT_CASES = ("F03", "F12", "F15", "F18")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"Expected a JSON object: {path}")
    digest(value)  # Reject nonfinite JSON values before they reach the comparison.
    return value


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Mismatch: {label}")


def recorded_decision(record: dict) -> dict | None:
    state = record["state"]
    if state == "completed":
        require_equal(record["finish_reason"], "stop", "completed response finish reason")
        decision = validate_decision(record["raw_output"])
        require_equal(decision, record["decision"], "raw versus parsed response")
        return decision
    if state != "invalid_response":
        raise ValueError("Unresolved reservation or unknown response state")
    if record["finish_reason"] == "stop":
        try:
            validate_decision(record["raw_output"])
        except ValueError:
            return None
        raise ValueError("Valid response incorrectly marked invalid")
    return None


def replay_policies(prepared: PreparedCase, decision: dict | None, *, anomaly: bool) -> dict:
    policies = {
        name: _execute(
            plan_probes(name, prepared.features, anomaly=anomaly, seed=case_seed(prepared.case)), prepared
        )
        for name in BASELINES
    }
    chosen = list(decision["probes"]) if decision else []
    order = list(chosen)
    if not decision or decision["decision"] != "abstain":
        order += [p for p in TOOLS if p not in chosen]
    policies["muse"] = _execute(order, prepared)
    return policies


def audit_records(results: dict, ledger: dict) -> dict:
    """Verify stored accounting and source identity; do not claim raw-image reproduction."""
    require_equal(
        ledger["seal_sha256"],
        digest({"model": ledger["model"], "attempts": ledger["attempts"]}),
        "ledger seal",
    )
    require_equal(results["freeze"], freeze_record(), "frozen source/configuration")
    require_equal(results["model"], AGENT.model, "results model")
    require_equal(ledger["model"], results["model"], "ledger model")
    if results["mode"] not in ("live", "offline_dry_run"):
        raise ValueError("Unknown benchmark mode")
    rows = results["cases"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated reference cases")
    require_equal(set(ledger["attempts"]), {r["id"] for r in rows}, "ledger case set")
    definitions = {c["id"]: c for c in plan()["cases"]}
    if results["complete"]:
        require_equal({r["id"] for r in rows}, set(definitions), "complete benchmark case set")
    require_equal(results["summary"], summarise(rows), "aggregate summary")
    spent = 0.0
    for row in rows:
        cid = row["id"]
        definition = definitions[cid]
        truth = private_case(definition)
        record = ledger["attempts"][cid]
        decision = recorded_decision(record)
        require_equal(record["state"], row["muse"]["state"], f"{cid} response state")
        require_equal(decision, row["muse"]["decision"], f"{cid} decision")
        require_equal(record["request_sha256"], row["muse"]["request_sha256"], f"{cid} request hash")
        require_equal(row["field"], definition["field"], f"{cid} field")
        require_equal(row["kind"], definition["kind"], f"{cid} class")
        require_equal(row["causes"], list(truth.causes), f"{cid} causes")
        require_equal(row["masked"], list(truth.masked), f"{cid} masking")
        require_equal(row["dominant"], truth.dominant, f"{cid} dominant cause")
        offline = results["mode"] == "offline_dry_run"
        require_equal(record["response_id"].startswith("offline-"), offline, f"{cid} response mode")
        usage = record["usage"]
        cost = usage_cost(
            usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"], AGENT.prices
        )
        cost = 0.0 if offline else cost
        if not math.isclose(cost, record["cost_usd"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Mismatch: {cid} token cost")
        require_equal(record["cost_usd"], row["muse"]["cost_usd"], f"{cid} recorded cost")
        spent += cost
        for probe, measure in row["oracle"].items():
            counts = {k: measure[k] for k in ("before", "after", "n")}
            require_equal(measure["gain"], gain(**counts), f"{cid}/{probe} gain")
            require_equal(measure["closed"], closure_accepted(**counts), f"{cid}/{probe} closure")
        prepared = PreparedCase(definition, truth, {}, row["features"], row["oracle"])
        traces = replay_policies(prepared, decision, anomaly=row["recovery_anomaly"])
        require_equal(row["policies"], traces, f"{cid} policy traces")
        require_equal(row["solvable"], any(m["closed"] for m in row["oracle"].values()), f"{cid} solvable")
        chosen = decision["probes"] if decision else []
        fallback = len(traces["muse"]["executed"]) > len(chosen)
        require_equal(row["muse"]["used_fallback"], fallback, f"{cid} fallback attribution")
        direct = bool(chosen) and traces["muse"]["success"] and not fallback
        require_equal(row["muse"]["closed_without_fallback"], direct, f"{cid} selected-probe attribution")
    states = [r["state"] for r in ledger["attempts"].values()]
    for name, value in {
        "cases": len(rows),
        "completed": states.count("completed"),
        "invalid_response": states.count("invalid_response"),
        "unresolved": [],
        "spent_usd": round(spent, 6),
    }.items():
        require_equal(results["ledger"][name], value, f"ledger summary: {name}")
    active = [r for r in rows if r["kind"] != "no_intervention"]
    return {
        "status": "verified",
        "reference_cases": len(rows),
        "reference_complete": results["complete"],
        "probe_totals_intervention_cases": {
            p: sum(r["policies"][p]["probes"] for r in active) for p in results["summary"]["policies"]
        },
        "configured_cost_usd": round(spent, 6),
        "summary": results["summary"],
        "scope": "Stored accounting and source hashes, not raw-image reproduction or provider billing.",
    }


def reproduce_cases(results: dict, ledger: dict, fits_path: Path, case_ids: list[str]) -> dict:
    """Repeat numeric measurements and reconstruct request bytes; never call a model."""
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Choose at least one distinct case")
    rows = {r["id"]: r for r in results["cases"]}
    definitions = {c["id"]: c for c in plan()["cases"]}
    if any(cid not in rows for cid in case_ids):
        raise ValueError("Requested case is absent from the reference")
    images, identities = {}, {}
    for field in sorted({definitions[cid]["field"] for cid in case_ids}):
        images[field], metadata = load_verified_image(fits_path, field)
        require_equal(metadata["sha256"], results["freeze"]["fits_sha256"][field], f"{field} FITS identity")
        identities[field] = metadata["sha256"]
    outcomes = []
    for cid in case_ids:
        definition, row = definitions[cid], rows[cid]
        raw = extract_patch(images[definition["field"]], definition["offset_y"], definition["offset_x"])
        prepared = prepare_case(raw, definition)
        record = ledger["attempts"][cid]
        decision = recorded_decision(record)
        request_hash = digest(build_request(cid, prepared.observation))
        policies = replay_policies(prepared, decision, anomaly=recovery_anomaly(prepared.observation))
        matching = {
            "request_hash": request_hash == record["request_sha256"],
            "features": prepared.features == row["features"],
            "oracle": prepared.oracle == row["oracle"],
            "recovery_anomaly": recovery_anomaly(prepared.observation) == row["recovery_anomaly"],
            "policies": policies == row["policies"],
        }
        outcomes.append(
            {
                "id": cid,
                "matched": all(matching.values()),
                "checks": matching,
                "expected_oracle": row["oracle"],
                "actual_oracle": prepared.oracle,
                "expected_request_sha256": record["request_sha256"],
                "actual_request_sha256": request_hash,
            }
        )
        print(f"{cid}: {'MATCH' if all(matching.values()) else 'MISMATCH'}", file=sys.stderr)
    return {
        "status": "reproduced" if all(c["matched"] for c in outcomes) else "mismatch",
        "scope": "Selected cases only; all three probes and four policies per selected case.",
        "whole_reference_reproduced": set(case_ids) == set(rows),
        "reference_cases_total": len(rows),
        "case_ids": case_ids,
        "fits_sha256_verified": identities,
        "cases": outcomes,
        "api_calls": 0,
        "new_model_cost_usd": 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("evidence/benchmark_results.json"))
    parser.add_argument("--ledger", type=Path, default=Path("evidence/benchmark_ledger.json"))
    parser.add_argument("--fits-zip", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--cases", nargs="+")
    scope.add_argument("--all-cases", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    args.out = args.out or Path("evidence/runs") / (
        "record_audit.json" if args.audit_only else "reproduction.json"
    )
    if not args.audit_only and args.fits_zip is None:
        parser.error("--fits-zip is required unless --audit-only is selected")
    protected = [args.results, args.ledger] + ([args.fits_zip] if args.fits_zip else [])
    if args.out.resolve() in {p.resolve() for p in protected}:
        parser.error("--out must not overwrite an input file")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
        "script_sha256": file_hash(Path(__file__)),
        "api_calls": 0,
    }
    try:
        results, ledger = read_json(args.results), read_json(args.ledger)
        report["input_sha256"] = {"results": file_hash(args.results), "ledger": file_hash(args.ledger)}
        report["record_audit"] = audit_records(results, ledger)
        if args.audit_only:
            report.update(status="records_verified_pixels_pending", pixel_reproduction="not_run")
        else:
            cases = (
                [r["id"] for r in results["cases"]] if args.all_cases else args.cases or list(DEFAULT_CASES)
            )
            report["pixel_reproduction"] = reproduce_cases(results, ledger, args.fits_zip, cases)
            report["status"] = report["pixel_reproduction"]["status"]
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr)
    save_json(args.out, report)
    print(json.dumps({"status": report["status"], "report": str(args.out), "api_calls": 0}))
    return 0 if report["status"] in ("reproduced", "records_verified_pixels_pending") else 1


if __name__ == "__main__":
    raise SystemExit(main())
