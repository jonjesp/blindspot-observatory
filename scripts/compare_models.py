"""Prepare, freeze and run one paired Muse Wonder comparison without altering the Spark reference.

Run from the repository root with python -m scripts.compare_models. Preparation makes no API calls.
Running defaults to canned offline replies; only run --live can make paid calls. The reference
source, prompt, numeric evidence and scoring remain fixed. A sealed plan binds settings and inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from blindspot.agent.client import DEFAULT_BASE_URL, make_live_client
from blindspot.agent.muse import build_request, usage_cost
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS, validate_decision
from blindspot.benchmark.freeze import freeze_record
from blindspot.benchmark.generator import private_case
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.report import _policy_block, paired_bootstrap_ci
from blindspot.benchmark.runner import PreparedCase, _execute, prepare_case
from blindspot.config import Prices
from blindspot.datasets.ztf import extract_patch, load_verified_image
from blindspot.provenance.ledger import canonical, digest, run_lock, save_json
from scripts.reproduce_benchmark import audit_records, file_hash, read_json, require_equal

CANDIDATE = {
    "model": "muse-wonder-1.0",
    "prices_usd_per_million_tokens": {"input": 4.0, "cached_input": 0.20, "output": 20.0},
    "pricing_source": "https://dev.meta.ai/docs/pricing-rate-limits",
    "pricing_checked_date": "2026-11-01",
}
ROOT = Path("evidence/runs/comparison")


def sealed(payload: dict) -> dict:
    return {**payload, "seal_sha256": digest(payload)}


def load_sealed(path: Path) -> dict:
    document = read_json(path)
    payload = {k: v for k, v in document.items() if k != "seal_sha256"}
    require_equal(document.get("seal_sha256"), digest(payload), f"integrity: {path}")
    return document


def reserve_cost(request: dict, prices: Prices) -> float:
    """Conservative estimate: all request JSON bytes plus framing allowance, and full output.

    This is a local reservation estimate, not a provider-enforced spending limit.
    """
    prompt_bound = len(canonical(request)) + 256
    return (prompt_bound * prices.input + request["max_completion_tokens"] * prices.output) / 1_000_000


def verify_proof(proof: dict, reference: dict, ledger: dict, input_hashes: dict) -> None:
    require_equal(proof["status"], "reproduced", "completed reproduction")
    require_equal(proof["input_sha256"], input_hashes, "reproduction inputs")
    require_equal(
        proof["script_sha256"],
        file_hash(Path(__file__).with_name("reproduce_benchmark.py")),
        "reproduction script",
    )
    pixels = proof["pixel_reproduction"]
    require_equal(proof["api_calls"], 0, "reproduction API calls")
    require_equal(pixels["api_calls"], 0, "pixel reproduction API calls")
    require_equal(pixels["whole_reference_reproduced"], True, "whole reference reproduced")
    expected = {r["id"]: r for r in reference["cases"]}
    require_equal(len(pixels["cases"]), len(expected), "reproduction length")
    require_equal({c["id"] for c in pixels["cases"]}, set(expected), "reproduced case IDs")
    require_equal(pixels["fits_sha256_verified"], reference["freeze"]["fits_sha256"], "reproduced FITS")
    for case in pixels["cases"]:
        require_equal(case["matched"], True, f"{case['id']} reproduced")
        require_equal(
            set(case["checks"]),
            {"features", "oracle", "policies", "recovery_anomaly", "request_hash"},
            "reproduction check set",
        )
        if not all(value is True for value in case["checks"].values()):
            raise ValueError(f"Incomplete reproduction checks: {case['id']}")
        require_equal(case["actual_oracle"], expected[case["id"]]["oracle"], "reproduced measurements")
        require_equal(
            case["actual_request_sha256"],
            ledger["attempts"][case["id"]]["request_sha256"],
            "reproduced request",
        )


def prepare_comparison(
    results_path: Path, ledger_path: Path, proof_path: Path, fits_path: Path, out: Path, budget: float = 3.0
) -> dict:
    if out.exists():
        raise ValueError("Comparison plan already exists. Use it; preparation never overwrites a frozen plan")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be finite and positive")
    reference, ledger, proof = read_json(results_path), read_json(ledger_path), read_json(proof_path)
    audit_records(reference, ledger)
    require_equal(reference["complete"], True, "complete reference")
    require_equal(reference["mode"], "live", "live reference")
    hashes = {"results": file_hash(results_path), "ledger": file_hash(ledger_path)}
    verify_proof(proof, reference, ledger, hashes)
    from blindspot.benchmark.design import plan

    definitions = {c["id"]: c for c in plan()["cases"]}
    images = {field: load_verified_image(fits_path, field)[0] for field in reference["freeze"]["fits_sha256"]}
    cases = []
    for row in reference["cases"]:
        definition = definitions[row["id"]]
        raw = extract_patch(images[row["field"]], definition["offset_y"], definition["offset_x"])
        prepared = prepare_case(raw, definition)
        request = build_request(row["id"], prepared.observation)
        require_equal(digest(request), row["muse"]["request_sha256"], f"{row['id']} original request")
        require_equal(prepared.oracle, row["oracle"], f"{row['id']} original measurements")
        require_equal(prepared.features, row["features"], f"{row['id']} original features")
        require_equal(
            recovery_anomaly(prepared.observation), row["recovery_anomaly"], f"{row['id']} recovery anomaly"
        )
        request["model"] = CANDIDATE["model"]
        cases.append(
            {
                "definition": definition,
                "reference": row,
                "observation": prepared.observation,
                "request": request,
            }
        )
    prices = Prices(**CANDIDATE["prices_usd_per_million_tokens"])
    estimated_max = sum(reserve_cost(c["request"], prices) for c in cases)
    if estimated_max > budget:
        raise ValueError(
            f"Budget ${budget:.2f} is below the full-run reservation estimate ${estimated_max:.4f}"
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": copy.deepcopy(CANDIDATE),
        "budget_usd": budget,
        "base_url": DEFAULT_BASE_URL,
        "maximum_calls": len(cases),
        "reservation_estimate_usd": estimated_max,
        "reference_inputs_sha256": {**hashes, "reproduction": file_hash(proof_path)},
        "reference_freeze": reference["freeze"],
        "comparison_script_sha256": file_hash(Path(__file__)),
        "reproduction_script_sha256": proof["script_sha256"],
        "reference_summary": reference["summary"],
        "reference_model": reference["model"],
        "cases": cases,
        "criterion": {
            "primary_comparator": "elimination",
            "bootstrap_seed": 1729,
            "bootstrap_samples": 5000,
            "prediction": "Fewer probes with at least reference closure and control abstention",
            "acceptance": "Complete live run; paired CI upper < 0; closure and abstention not worse",
            "scope": "Conditional case bootstrap; one call per case, no population/model-repeat inference",
        },
    }
    document = sealed(payload)
    save_json(out, document)
    return document


def validate_plan(document: dict) -> None:
    require_equal(
        document["seal_sha256"],
        digest({k: v for k, v in document.items() if k != "seal_sha256"}),
        "comparison plan seal",
    )
    require_equal(
        document["comparison_script_sha256"], file_hash(Path(__file__)), "comparison implementation"
    )
    require_equal(
        document["reproduction_script_sha256"],
        file_hash(Path(__file__).with_name("reproduce_benchmark.py")),
        "reproduction implementation",
    )
    require_equal(document["reference_freeze"], freeze_record(), "reference source")
    require_equal(document["candidate"], CANDIDATE, "candidate model and prices")
    require_equal(document["base_url"], DEFAULT_BASE_URL, "API endpoint")
    from blindspot.benchmark.design import plan

    definitions = {c["id"]: c for c in plan()["cases"]}
    require_equal(len(document["cases"]), document["maximum_calls"], "call count")
    require_equal({c["definition"]["id"] for c in document["cases"]}, set(definitions), "case set")
    if len(document["cases"]) != len(definitions):
        raise ValueError("Duplicated or incomplete comparison cases")
    for case in document["cases"]:
        cid = case["definition"]["id"]
        require_equal(case["definition"], definitions[cid], f"{cid} case definition")
        request = build_request(cid, case["observation"])
        require_equal(digest(request), case["reference"]["muse"]["request_sha256"], f"{cid} reference prompt")
        request["model"] = document["candidate"]["model"]
        require_equal(case["request"], request, f"{cid} candidate differs only in model ID")


def open_ledger(path: Path, document: dict, live: bool) -> dict:
    identity = {
        "plan_sha256": document["seal_sha256"],
        "model": document["candidate"]["model"],
        "mode": "live" if live else "offline_dry_run",
    }
    if not path.exists():
        return {**identity, "attempts": {}}
    ledger = load_sealed(path)
    for name, value in identity.items():
        require_equal(ledger[name], value, f"ledger {name}")
    return ledger


def spent_usd(ledger: dict) -> float:
    return sum(r.get("cost_usd", r.get("reserved_usd", 0.0)) for r in ledger["attempts"].values())


def call_case(
    document: dict,
    case: dict,
    ledger: dict,
    path: Path,
    client,
    *,
    live: bool,
    attempt_key: str | None = None,
) -> dict:
    cid, request = case["definition"]["id"], case["request"]
    key = cid if attempt_key is None else attempt_key
    if isinstance(client, OfflineMuseClient) == live:
        raise ValueError("Client and live/offline mode differ")
    previous = ledger["attempts"].get(key)
    if previous is not None:
        require_equal(previous["request_sha256"], digest(request), f"{cid} resumed request")
        if previous["state"] == "reserved_unknown":
            raise RuntimeError(f"Unresolved call {cid}; no automatic retry or rebilling")
        if previous.get("response_model") != document["candidate"]["model"]:
            raise RuntimeError(f"Model identity mismatch for {cid}; stop this comparison")
        if previous.get("usage_exceeded_reservation"):
            raise RuntimeError(f"Usage/reservation mismatch for {cid}; review before continuing")
        return previous
    if len(ledger["attempts"]) >= document["maximum_calls"]:
        raise RuntimeError("Comparison call limit reached")
    prices = Prices(**document["candidate"]["prices_usd_per_million_tokens"])
    reservation = reserve_cost(request, prices) if live else 0.0
    if spent_usd(ledger) + reservation > document["budget_usd"]:
        raise RuntimeError("Budget reservation exceeds the remaining experiment allowance")
    ledger["attempts"][key] = {
        "state": "reserved_unknown",
        "request_sha256": digest(request),
        "reserved_usd": reservation,
        "provider_base_url": document["base_url"] if live else None,
    }
    save_json(path, sealed({k: v for k, v in ledger.items() if k != "seal_sha256"}))
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = time.perf_counter()
    response = client.chat.completions.create(**copy.deepcopy(request))
    elapsed = time.perf_counter() - start
    choice, usage = response.choices[0], response.usage
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    counts = {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "cached_tokens": int(cached),
    }
    cost = usage_cost(**counts, prices=prices) if live else 0.0
    response_model = getattr(response, "model", document["candidate"]["model"] if not live else None)
    record = {
        "state": "invalid_response",
        "request_sha256": digest(request),
        "response_id": response.id,
        "response_model": response_model,
        "provider_base_url": document["base_url"] if live else None,
        "started_utc": started_utc,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finish_reason": choice.finish_reason,
        "raw_output": choice.message.content,
        "usage": counts,
        "cost_usd": cost,
        "reservation_usd": reservation,
        "usage_exceeded_reservation": bool(live and cost > reservation),
        "elapsed_seconds": elapsed,
        "elapsed_scope": "Client HTTP/generation only; not full pipeline",
        "reasoning_tokens": getattr(
            getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None
        ),
    }
    if response_model == document["candidate"]["model"] and choice.finish_reason == "stop":
        try:
            record["decision"] = validate_decision(record["raw_output"])
            record["state"] = "completed"
        except ValueError as exc:
            record["validation_error"] = str(exc)
    ledger["attempts"][key] = record
    save_json(path, sealed({k: v for k, v in ledger.items() if k != "seal_sha256"}))
    if response_model != document["candidate"]["model"]:
        raise RuntimeError(
            "Provider returned a different or unidentified model; result recorded and run stopped"
        )
    if live and cost > reservation:
        raise RuntimeError("Reported usage exceeded its reservation estimate; run stopped")
    return record


def evaluate_candidate(case: dict, record: dict) -> dict:
    row = case["reference"]
    prepared = PreparedCase(
        case["definition"],
        private_case(case["definition"]),
        case["observation"],
        row["features"],
        row["oracle"],
    )
    chosen = record["decision"]["probes"] if record["state"] == "completed" else []
    abstained = record["state"] == "completed" and record["decision"]["decision"] == "abstain"
    order = list(chosen) + ([] if abstained else [p for p in TOOLS if p not in chosen])
    trace = _execute(order, prepared)
    fallback = len(trace["executed"]) > len(chosen)
    return {
        **row,
        "policies": {**row["policies"], "candidate": trace},
        "candidate": {
            "state": record["state"],
            "decision": record.get("decision"),
            "used_fallback": fallback,
            "closed_without_fallback": bool(chosen) and trace["success"] and not fallback,
        },
    }


def comparison_summary(rows: list[dict], document: dict, ledger: dict, *, live: bool) -> dict:
    complete = len(rows) == len(document["cases"])
    active = [r for r in rows if r["kind"] != "no_intervention"]
    policies = {
        name: _policy_block(rows, name)
        for name in ("random", "heuristic", "elimination", "muse", "candidate")
    }
    for name, block in policies.items():
        block["total_probes_intervention_cases"] = sum(r["policies"][name]["probes"] for r in active)
    diffs = {}
    if active:
        for name in ("random", "heuristic", "elimination", "muse"):
            values = [r["policies"]["candidate"]["probes"] - r["policies"][name]["probes"] for r in active]
            diffs[name] = paired_bootstrap_ci(
                values,
                seed=document["criterion"]["bootstrap_seed"],
                n_boot=document["criterion"]["bootstrap_samples"],
            )
    candidate, baseline = policies["candidate"], document["reference_summary"]["policies"]["elimination"]
    accepted = bool(
        live
        and complete
        and diffs
        and diffs["elimination"]["ci95"][1] < 0
        and candidate["intervention_cases"]["k"] >= baseline["intervention_cases"]["k"]
        and candidate["correct_abstentions"]["k"] >= baseline["correct_abstentions"]["k"]
    )
    return {
        "models": {"reference": document["reference_model"], "candidate": document["candidate"]["model"]},
        "mode": "live" if live else "offline_dry_run",
        "complete": complete,
        "primary_gate_passed": accepted,
        "gate_interpretation": "Known-case comparison only; no generalisation or model-repeat variance claim",
        "policies": policies,
        "candidate_minus_comparator": diffs,
        "candidate_valid": sum(r["candidate"]["state"] == "completed" for r in rows),
        "candidate_invalid": sum(r["candidate"]["state"] != "completed" for r in rows),
        "fallback_cases": [r["id"] for r in rows if r["candidate"]["used_fallback"]],
        "closed_without_fallback": sum(r["candidate"]["closed_without_fallback"] for r in active),
        "known_configured_cost_usd": round(
            sum(a.get("cost_usd", 0.0) for a in ledger["attempts"].values()), 6
        ),
        "unknown_reserved_usd": round(
            sum(
                a.get("reserved_usd", 0.0)
                for a in ledger["attempts"].values()
                if a["state"] == "reserved_unknown"
            ),
            6,
        ),
        "budget_accounted_usd": round(spent_usd(ledger), 6),
        "reference_configured_cost_usd": document["reference_summary"]["muse_replies"]["cost_usd"],
        "candidate_request_seconds": sum(a.get("elapsed_seconds", 0.0) for a in ledger["attempts"].values()),
        "reference_request_seconds": None,
        "time_scope": "No historical Spark latency; no end-to-end speed claim",
    }


def run_comparison(document: dict, directory: Path, client, *, live: bool = False) -> dict:
    validate_plan(document)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "wonder" if live else "dry_run"
    ledger_path, result_path = directory / f"{prefix}_ledger.json", directory / f"{prefix}_results.json"
    with run_lock(ledger_path):
        ledger = open_ledger(ledger_path, document, live)
        rows, error = [], None
        try:
            if live:
                require_equal(str(client.base_url).rstrip("/"), document["base_url"], "client endpoint")
                enabled = {model.id for model in client.models.list().data}
                if document["candidate"]["model"] not in enabled:
                    raise RuntimeError("Candidate is unavailable for this account; no paid request made")
            for case in document["cases"]:
                record = call_case(document, case, ledger, ledger_path, client, live=live)
                row = evaluate_candidate(case, record)
                rows.append(row)
                print(
                    f"{row['id']}: candidate probes={row['policies']['candidate']['probes']}", file=sys.stderr
                )
        except Exception as exc:
            # SDK exception bodies can contain credentials or request data.
            error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
            status = getattr(exc, "status_code", None)
            if type(status) is int:
                error = f"{type(exc).__name__} (HTTP {status})"
        summary = comparison_summary(rows, document, ledger, live=live)
        if error:
            summary["primary_gate_passed"] = False
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "live" if live else "offline_dry_run",
            "model": document["candidate"]["model"],
            "plan_sha256": document["seal_sha256"],
            "cases": rows,
            "error": error,
            "summary": summary,
            "ledger_path": str(ledger_path),
        }
        save_json(result_path, report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--fits-zip", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("evidence/benchmark_results.json"))
    p.add_argument("--ledger", type=Path, default=Path("evidence/benchmark_ledger.json"))
    p.add_argument("--reproduction", type=Path, default=Path("evidence/reproduction.json"))
    p.add_argument("--out", type=Path, default=ROOT / "plan.json")
    p.add_argument("--budget", type=float, default=3.0)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, default=ROOT / "plan.json")
    p.add_argument("--directory", type=Path, default=ROOT)
    p.add_argument("--live", action="store_true")
    sub.add_parser("catalog")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            document = prepare_comparison(
                args.results, args.ledger, args.reproduction, args.fits_zip, args.out, args.budget
            )
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "model": document["candidate"]["model"],
                        "cases": len(document["cases"]),
                        "budget_usd": document["budget_usd"],
                        "reservation_estimate_usd": document["reservation_estimate_usd"],
                        "api_calls": 0,
                    }
                )
            )
            return 0
        if args.command == "catalog":
            client = make_live_client()
            print(json.dumps({"models": [m.id for m in client.models.list().data]}))
            return 0
        client = make_live_client() if args.live else OfflineMuseClient()
        report = run_comparison(load_sealed(args.plan), args.directory, client, live=args.live)
        print(json.dumps(report["summary"], indent=2))
        if report["error"]:
            print(report["error"], file=sys.stderr)
        return 0 if report["summary"]["complete"] and not report["error"] else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
