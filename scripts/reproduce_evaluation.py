"""Reproduce published evaluation measurements and case tables without model requests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from blindspot.benchmark.freeze import freeze_record
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.runner import prepare_case
from blindspot.datasets.ztf import extract_patch, image_manifest, load_verified_image
from blindspot.provenance.ledger import run_lock, save_json
from scripts.compare_models import load_sealed
from scripts.evaluate_models import (
    environment_versions,
    source_hashes,
    validate_ledger,
    validate_protocol_contract,
)
from scripts.model_report import write_report
from scripts.reproduce_benchmark import replay_policies, require_equal

ROOT = Path("evidence/model_evaluation")
OUTPUT = Path("evidence/runs/publication-reproduction")


def verify_measurements(document: dict, fits_path: Path) -> int:
    """Require exact agreement of every pilot, probe and baseline with the recorded cases."""
    images = {entry["field"]: load_verified_image(fits_path, entry["field"])[0] for entry in image_manifest()}
    for case in document["cases"]:
        definition = case["definition"]
        cid = definition["id"]
        raw = extract_patch(images[definition["field"]], definition["offset_y"], definition["offset_x"])
        prepared = prepare_case(raw, definition)
        require_equal(prepared.observation, case["observation"], f"{cid} pilot observations")
        require_equal(prepared.features, case["reference"]["features"], f"{cid} baseline features")
        require_equal(prepared.oracle, case["reference"]["oracle"], f"{cid} probe measurements")
        policies = replay_policies(prepared, None, anomaly=recovery_anomaly(prepared.observation))
        reference = {
            "id": cid,
            "kind": definition["kind"],
            "field": definition["field"],
            "features": prepared.features,
            "oracle": prepared.oracle,
            "policies": {name: policies[name] for name in ("random", "heuristic", "elimination")},
        }
        require_equal(reference, case["reference"], f"{cid} reference policies")
    return len(document["cases"])


def reproduce_evaluation(directory: Path, fits_path: Path, output: Path) -> dict:
    """Verify immutable inputs against fresh measurements and regenerate matching outputs."""
    source, destination = directory.resolve(), output.resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Reproduction outputs must be separate from the publication directory")
    receipt = destination / "reproduction.json"
    with run_lock(receipt):
        # A receipt certifies the current attempt, even when the output path is reused.
        receipt.unlink(missing_ok=True)
        document = load_sealed(source / "protocol.json")
        ledger = load_sealed(source / "ledger.json")
        validate_protocol_contract(document)
        current = freeze_record()
        for name, value in current.items():
            if name != "implementation_sha256":
                require_equal(document["freeze"][name], value, f"experimental definition: {name}")
        require_equal(len(ledger["attempts"]), len(document["schedule"]), "complete response collection")
        validate_ledger(document, ledger, live=True)
        cases = verify_measurements(document, fits_path)
        with TemporaryDirectory(dir=destination) as staging:
            staged = Path(staging)
            write_report(document, ledger, staged)
            with (source / "case_results.csv").open(encoding="utf-8", newline="") as stream:
                published = list(csv.DictReader(stream))
            with (staged / "case_results.csv").open(encoding="utf-8", newline="") as stream:
                generated = list(csv.DictReader(stream))
            require_equal(generated, published, "published case table")
            for name in ("report.md", "case_results.csv"):
                (staged / name).replace(destination / name)
        result = {
            "status": "reproduced",
            "cases_verified": cases,
            "responses_verified": len(generated),
            "api_calls": 0,
            "protocol_sha256": document["seal_sha256"],
            "ledger_sha256": ledger["seal_sha256"],
            "environment": environment_versions(),
            "implementation": current,
            "report_sources_sha256": source_hashes(),
            "reproducer_sha256": hashlib.sha256(
                Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "scope": "Current-source pixel reproduction; not reconstruction of the original runtime.",
        }
        save_json(receipt, result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT)
    parser.add_argument("--fits-zip", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = reproduce_evaluation(args.directory, args.fits_zip, args.output_directory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Reproduction stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        f"REPRODUCED: {result['cases_verified']} cases, {result['responses_verified']} responses, 0 API calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())