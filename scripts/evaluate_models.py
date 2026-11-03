"""Paired Meta model evaluation with a frozen protocol and shared request accounting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from blindspot.agent.client import DEFAULT_BASE_URL, make_live_client
from blindspot.agent.muse import build_request, usage_cost
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import validate_decision
from blindspot.benchmark.design import plan
from blindspot.benchmark.freeze import freeze_record, git_state
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.runner import prepare_case
from blindspot.config import Prices
from blindspot.datasets.ztf import extract_patch, image_manifest, load_verified_image
from blindspot.provenance.ledger import digest, save_json
from scripts.compare_models import call_case, load_sealed, reserve_cost, run_lock, sealed
from scripts.model_report import write_report
from scripts.reproduce_benchmark import replay_policies, require_equal

MAXIMUM_BUDGET_USD = 10.0
RESERVATION_TOLERANCE_USD = 1e-12
OUTPUT_TOKEN_LIMIT = 8192
ORDER_SEED = 20261031
COMPARISON_CRITERION = {
    "primary_comparator": "elimination",
    "bootstrap_seed": 1729,
    "bootstrap_samples": 10000,
    "familywise_alpha": 0.05,
    "primary_comparisons": 3,
}
MODEL_PROFILES = (
    {
        "name": "Muse Spark 1.1",
        "model": "muse-spark-1.1",
        "reasoning_effort": "xhigh",
        "prices_usd_per_million_tokens": {"input": 1.25, "cached_input": 0.15, "output": 4.25},
    },
    {
        "name": "Muse Spark 1.3",
        "model": "muse-spark-1.3",
        "reasoning_effort": "max",
        "prices_usd_per_million_tokens": {"input": 1.25, "cached_input": 0.15, "output": 4.25},
    },
    {
        "name": "Muse Wonder 1.0",
        "model": "muse-wonder-1.0",
        "reasoning_effort": "max",
        "prices_usd_per_million_tokens": {"input": 4.0, "cached_input": 0.20, "output": 20.0},
    },
)
SOURCE_NAMES = ("evaluate_models.py", "model_report.py", "compare_models.py", "reproduce_benchmark.py")
ROOT = Path("evidence/runs/model-evaluation")


def source_hashes(directory: Path | None = None) -> dict:
    directory = Path(__file__).parent if directory is None else directory
    return {
        name: hashlib.sha256((directory / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in SOURCE_NAMES
    }


def environment_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "PyYAML", "openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def request_for(case: dict, profile: dict) -> dict:
    request = build_request(case["definition"]["id"], case["observation"])
    request.update(
        model=profile["model"],
        reasoning_effort=profile["reasoning_effort"],
        max_completion_tokens=OUTPUT_TOKEN_LIMIT,
        temperature=1.0,
    )
    return request


def reservation_for(case: dict, profile: dict) -> float:
    return reserve_cost(request_for(case, profile), Prices(**profile["prices_usd_per_million_tokens"]))


def validate_protocol(document: dict, *, recorded_sources: Path | None = None) -> None:
    """Validate source identity and the complete contract before execution."""
    require_equal(document["source_sha256"], source_hashes(recorded_sources), "evaluation scripts")
    require_equal(document["freeze"], freeze_record(), "benchmark source and configuration")
    validate_protocol_contract(document)


def validate_protocol_contract(document: dict) -> None:
    """Validate model settings, requests, case geometry, schedule and reservation bounds."""
    require_equal(document["profiles"], list(MODEL_PROFILES), "model profiles")
    require_equal(document["base_url"], DEFAULT_BASE_URL, "Meta endpoint")
    require_equal(document["output_token_limit"], OUTPUT_TOKEN_LIMIT, "output-token limit")
    require_equal(document["criterion"], COMPARISON_CRITERION, "comparison criterion")
    budget = document["budget_usd"]
    if isinstance(budget, bool) or not math.isfinite(budget) or not 0 < budget <= MAXIMUM_BUDGET_USD:
        raise ValueError("Experiment budget must be finite and within the authorized allowance")
    require_equal([case["definition"] for case in document["cases"]], plan()["cases"], "case definitions")
    expected = set()
    for case in document["cases"]:
        for profile in document["profiles"]:
            model, cid = profile["model"], case["definition"]["id"]
            require_equal(
                case["request_sha256"][model], digest(request_for(case, profile)), "request identity"
            )
            expected.add((model, cid))
    schedule = [(s["model"], s["case_id"]) for s in document["schedule"]]
    require_equal(set(schedule), expected, "scheduled requests")
    require_equal(len(schedule), len(expected), "unique scheduled requests")
    bound = math.fsum(
        reservation_for(case, profile) for case in document["cases"] for profile in document["profiles"]
    )
    if not math.isclose(
        document["reservation_bound_usd"], bound, rel_tol=0, abs_tol=RESERVATION_TOLERANCE_USD
    ):
        raise ValueError("Mismatch: campaign reservation bound")
    if bound > budget:
        raise ValueError("The full protocol cannot be reserved within the experiment budget")


def prepare_evaluation(fits_path: Path, directory: Path, budget: float = MAXIMUM_BUDGET_USD) -> dict:
    """Measure the shared cases once and freeze every request before inference."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty evaluation directory; existing experiments are not overwritten")
    if isinstance(budget, bool) or not math.isfinite(budget) or not 0 < budget <= MAXIMUM_BUDGET_USD:
        raise ValueError("Experiment budget must be finite and within the authorized allowance")
    with run_lock(directory / "protocol.json"):
        if (directory / "protocol.json").exists():
            raise ValueError("The experiment protocol already exists")
        images = {e["field"]: load_verified_image(fits_path, e["field"])[0] for e in image_manifest()}
        cases = []
        for definition in plan()["cases"]:
            raw = extract_patch(images[definition["field"]], definition["offset_y"], definition["offset_x"])
            prepared = prepare_case(raw, definition)
            policies = replay_policies(prepared, None, anomaly=recovery_anomaly(prepared.observation))
            case = {
                "definition": definition,
                "observation": prepared.observation,
                "reference": {
                    "id": definition["id"],
                    "kind": definition["kind"],
                    "field": definition["field"],
                    "features": prepared.features,
                    "oracle": prepared.oracle,
                    "policies": {name: policies[name] for name in ("random", "heuristic", "elimination")},
                },
            }
            case["request_sha256"] = {
                profile["model"]: digest(request_for(case, profile)) for profile in MODEL_PROFILES
            }
            cases.append(case)
        order = random.Random(ORDER_SEED)
        shuffled_cases = order.sample(cases, len(cases))
        schedule = []
        for case in shuffled_cases:
            for profile in order.sample(list(MODEL_PROFILES), len(MODEL_PROFILES)):
                schedule.append({"case_id": case["definition"]["id"], "model": profile["model"]})
        document = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": DEFAULT_BASE_URL,
            "profiles": copy.deepcopy(list(MODEL_PROFILES)),
            "documentation": {
                "reasoning": "https://dev.meta.ai/docs/reasoning",
                "pricing": "https://dev.meta.ai/docs/pricing-rate-limits",
                "models": "https://dev.meta.ai/docs/models",
                "checked_date": "2026-11-02",
            },
            "budget_usd": budget,
            "output_token_limit": OUTPUT_TOKEN_LIMIT,
            "order_seed": ORDER_SEED,
            "criterion": copy.deepcopy(COMPARISON_CRITERION),
            "freeze": freeze_record(),
            "environment": environment_versions(),
            "git": git_state(),
            "source_sha256": source_hashes(),
            "cases": cases,
            "schedule": schedule,
            "reservation_bound_usd": math.fsum(
                reservation_for(case, profile) for case in cases for profile in MODEL_PROFILES
            ),
        }
        validate_protocol(document)
        document = sealed(document)
        save_json(directory / "protocol.json", document)
        return document


def validate_ledger(document: dict, ledger: dict, *, live: bool, allow_unresolved: bool = False) -> None:
    require_equal(ledger["protocol_sha256"], document["seal_sha256"], "ledger protocol")
    require_equal(ledger["mode"], "live" if live else "offline_dry_run", "ledger execution mode")
    cases = {case["definition"]["id"]: case for case in document["cases"]}
    profiles = {profile["model"]: profile for profile in document["profiles"]}
    response_ids = []
    for key, record in ledger["attempts"].items():
        model, cid = key.split("/", 1)
        case, profile = cases[cid], profiles[model]
        require_equal(record["request_sha256"], digest(request_for(case, profile)), "ledger request")
        reservation = reservation_for(case, profile) if live else 0.0
        if record["state"] == "reserved_unknown":
            require_equal(record["reserved_usd"], reservation, "pending reservation")
            if allow_unresolved:
                continue
            raise RuntimeError("An unresolved request requires inspection before the experiment can resume")
        require_equal(record["reservation_usd"], reservation, "request reservation")
        require_equal(record["response_model"], model, "returned model identity")
        require_equal(record["provider_base_url"], document["base_url"] if live else None, "provider origin")
        usage = record["usage"]
        cost = usage_cost(**usage, prices=Prices(**profile["prices_usd_per_million_tokens"])) if live else 0.0
        if not math.isclose(record["cost_usd"], cost, rel_tol=0, abs_tol=1e-12) or cost > reservation:
            raise ValueError("Recorded usage does not match its budget reservation")
        elapsed = record["elapsed_seconds"]
        reasoning = record.get("reasoning_tokens")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid recorded request duration")
        if reasoning is not None and (
            type(reasoning) is not int or not 0 <= reasoning <= usage["completion_tokens"]
        ):
            raise ValueError("Invalid reported reasoning-token count")
        if not isinstance(record["response_id"], str) or not record["response_id"]:
            raise ValueError("Missing provider response identifier")
        if live and record["response_id"].startswith("offline-"):
            raise ValueError("An offline response cannot be used as a live observation")
        if record["state"] == "completed":
            require_equal(record["finish_reason"], "stop", "completion status")
            require_equal(record["decision"], validate_decision(record["raw_output"]), "recorded decision")
        elif record["state"] != "invalid_response":
            raise ValueError("Unknown response state")
        response_ids.append(record["response_id"])
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Repeated provider response identifiers require inspection")


def run_evaluation(document: dict, directory: Path, client, *, live: bool = False) -> dict:
    """Execute or resume the sealed schedule; unresolved requests are never retried."""
    validate_protocol(document)
    if isinstance(client, OfflineMuseClient) == live:
        raise ValueError("Client and experiment mode differ")
    path = directory / "ledger.json"
    with run_lock(path):
        ledger = (
            load_sealed(path)
            if path.exists()
            else {
                "protocol_sha256": document["seal_sha256"],
                "mode": "live" if live else "offline_dry_run",
                "availability": {},
                "attempts": {},
            }
        )
        if not path.exists():
            save_json(path, sealed(ledger))
        error = None
        try:
            validate_ledger(document, ledger, live=live)
            if not ledger["availability"]:
                if live:
                    require_equal(str(client.base_url).rstrip("/"), document["base_url"], "client endpoint")
                    catalog = sorted(catalog_model.id for catalog_model in client.models.list().data)
                else:
                    catalog = [profile["model"] for profile in document["profiles"]]
                ledger["catalog_models"] = catalog
                ledger["availability"] = {
                    profile["model"]: profile["model"] in catalog for profile in document["profiles"]
                }
                save_json(
                    path, sealed({field: value for field, value in ledger.items() if field != "seal_sha256"})
                )
            if live:
                require_equal(str(client.base_url).rstrip("/"), document["base_url"], "client endpoint")
            cases = {case["definition"]["id"]: case for case in document["cases"]}
            profiles = {profile["model"]: profile for profile in document["profiles"]}
            for task in document["schedule"]:
                model, cid = task["model"], task["case_id"]
                if not ledger["availability"][model]:
                    continue
                key = f"{model}/{cid}"
                profile, case = profiles[model], cases[cid]
                candidate = {
                    "candidate": profile,
                    "base_url": document["base_url"],
                    "budget_usd": document["budget_usd"],
                    "maximum_calls": len(document["schedule"]),
                }
                request_case = {"definition": case["definition"], "request": request_for(case, profile)}
                record = call_case(candidate, request_case, ledger, path, client, live=live, attempt_key=key)
                validate_ledger(document, ledger, live=live)
                print(f"{profile['name']} | {cid} | {record['state']}", file=sys.stderr)
        except Exception as exc:
            # An SDK failure leaves its reservation intact; exception bodies may contain request data.
            error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
            status = getattr(exc, "status_code", None)
            if type(status) is int:
                error = f"{type(exc).__name__} (HTTP {status})"
        write_report(document, ledger, directory, error=error)
        return {
            "complete": error is None
            and all(ledger["availability"].values())
            and len(ledger["attempts"]) == len(document["schedule"]),
            "error": error,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="measure and freeze the shared experimental protocol")
    prepare.add_argument("--fits-zip", required=True, type=Path)
    prepare.add_argument("--directory", type=Path, default=ROOT)
    prepare.add_argument("--budget", type=float, default=MAXIMUM_BUDGET_USD)
    run = sub.add_parser("run", help="execute the frozen schedule with real model requests")
    run.add_argument("--directory", type=Path, default=ROOT)
    run.add_argument("--live", action="store_true", required=True)
    report = sub.add_parser("report", help="regenerate the Markdown report and case table without API calls")
    report.add_argument("--directory", type=Path, default=ROOT)
    report.add_argument("--recorded-sources", type=Path)
    report.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            document = prepare_evaluation(args.fits_zip, args.directory, args.budget)
            print(
                "Protocol prepared: 36 cases, 3 models, 108 requests; "
                f"reservation bound ${document['reservation_bound_usd']:.4f}."
            )
            return 0
        document = load_sealed(args.directory / "protocol.json")
        if args.command == "report":
            destination = args.output_directory or args.directory
            if args.recorded_sources is not None and destination.resolve() == args.directory.resolve():
                raise ValueError("Recorded-source audits require a separate output directory")
            validate_protocol(document, recorded_sources=args.recorded_sources)
            with run_lock(destination / "ledger.json"):
                ledger = load_sealed(args.directory / "ledger.json")
                validate_ledger(document, ledger, live=ledger["mode"] == "live", allow_unresolved=True)
                write_report(document, ledger, destination)
            return 0
        outcome = run_evaluation(document, args.directory, make_live_client(timeout=900.0), live=True)
        print(f"Report: {args.directory / 'report.md'}")
        if outcome["error"]:
            print(outcome["error"], file=sys.stderr)
        return 0 if outcome["complete"] else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
