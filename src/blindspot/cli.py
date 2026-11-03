"""Command line interface. Only ``benchmark --live`` contacts the model API."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blindspot.agent.client import make_live_client
from blindspot.agent.muse import AGENT, MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.benchmark.calibration import calibrate
from blindspot.benchmark.design import plan, plan_hash
from blindspot.benchmark.freeze import freeze_record, git_state, verify_freeze, write_freeze
from blindspot.benchmark.report import format_summary
from blindspot.benchmark.runner import run_benchmark
from blindspot.datasets.ztf import extract_patch, image_manifest, load_verified_image
from blindspot.detection.detector import PSF_ENERGY, detect_sources, prepare_patch
from blindspot.injection.psf import inject_sources, sample_positions
from blindspot.metrics.completeness import count_new_matched
from blindspot.provenance.ledger import save_json

RUNS_DIR = Path("evidence") / "runs"

# Smoke-test acceptance bands, calibrated on all 36 planned sites of the two original images.
NOMINAL_SNR_BAND = (1, 20)  # recovered sources of 25 at nominal SNR 5
HIGH_SNR = 12.0
HIGH_SNR_MIN_RECOVERED = 13  # observed minimum over the 36 sites is 17


def _recovered_at(patch: dict, snr: float, seed: int) -> int:
    positions = sample_positions(seed, holdout=False)
    flux = snr * patch["sigma"] / np.sqrt(PSF_ENERGY)
    injected = inject_sources(patch["base"], positions, flux)
    detections = detect_sources(injected, patch["background"], patch["sigma"])
    return count_new_matched(positions, detections, patch["baseline"])


def _smoke(fits_zip: Path) -> tuple[list[dict], list[str]]:
    seed = plan()["seed"]
    details, failures = [], []
    for entry in image_manifest():
        image, _ = load_verified_image(fits_zip, entry["field"])
        site = next(c for c in plan()["cases"] if c["field"] == entry["field"])
        patch = prepare_patch(extract_patch(image, site["offset_y"], site["offset_x"]))
        zero, nominal, high = (_recovered_at(patch, s, seed) for s in (0.0, 5.0, HIGH_SNR))
        details.append(
            {
                "field": entry["field"],
                "site": site["id"],
                "zero_flux": zero,
                "snr5": nominal,
                "snr12": high,
                "injected": 25,
            }
        )
        if zero != 0:
            failures.append(f"{entry['field']}: zero-flux injection recovered {zero} sources (expected 0)")
        if not NOMINAL_SNR_BAND[0] <= nominal <= NOMINAL_SNR_BAND[1]:
            failures.append(f"{entry['field']}: SNR 5 recovery {nominal} outside {NOMINAL_SNR_BAND}")
        if high < HIGH_SNR_MIN_RECOVERED:
            failures.append(f"{entry['field']}: SNR {HIGH_SNR:g} recovery {high} < {HIGH_SNR_MIN_RECOVERED}")
    return details, failures


def _benchmark(args: argparse.Namespace) -> int:
    live = args.live
    if live:
        verify_freeze(args.freeze)
        git = git_state()
        if git["dirty"] and not args.allow_dirty:
            raise RuntimeError(
                "Working tree has uncommitted changes: commit the freeze first (or --allow-dirty)"
            )
        client = make_live_client()
    else:
        client = OfflineMuseClient()
    run_root = RUNS_DIR / ("live" if live else "dry_run")
    ledger = args.ledger or run_root / "benchmark_ledger.json"
    agent = MuseAgent(ledger_path=ledger, budget_usd=args.budget)
    cases = plan()["cases"][: args.limit] if args.limit else plan()["cases"]
    images = {e["field"]: load_verified_image(args.fits_zip, e["field"])[0] for e in image_manifest()}

    def get_raw(case: dict):
        return extract_patch(images[case["field"]], case["offset_y"], case["offset_x"])

    def progress(row: dict) -> None:
        decision = (row["muse"]["decision"] or {}).get("probes")
        shown = "+".join(decision) if decision else ("abstain" if row["muse"]["decision"] else "invalid")
        trace = row["policies"]["muse"]
        print(
            f"{row['id']} {row['kind']:<16} muse={shown:<22} probes={trace['probes']} "
            f"{'ok' if trace['success'] else 'FAIL'}",
            file=sys.stderr,
        )

    if live:
        print(f"LIVE run: up to {len(cases)} paid calls, budget ${agent.budget_usd:.2f}", file=sys.stderr)
    result = run_benchmark(cases, get_raw, agent=agent, client=client, allow_paid=live, progress=progress)
    result.update(
        {
            "mode": "live" if live else "offline_dry_run",
            "complete": len(cases) == len(plan()["cases"]),
            "model": AGENT.model,
            "freeze": freeze_record(),
            "git": git_state(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    save_json(args.out or run_root / "benchmark_results.json", result)
    print(format_summary(result["summary"], dry_run=not live))
    if not live:
        print("\nDRY RUN: canned replies, no model was called. These numbers say nothing about Muse.")
    elif not result["complete"]:
        print("\nINCOMPLETE run (--limit): do not report these numbers.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blindspot", description="BlindSpot Observatory (offline tools)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write the 36-case benchmark geometry")
    p.add_argument("--out", default=str(RUNS_DIR / "benchmark_plan.json"))
    for name, text in (("verify-data", "check FITS checksums"), ("smoke-fits", "physical sanity checks")):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "--fits-zip", required=True, type=Path, help="ZIP or directory with the two FITS files"
        )
    p = sub.add_parser("calibrate", help="measure the case generator on the real images (about a minute)")
    p.add_argument("--fits-zip", required=True, type=Path)
    p.add_argument("--out", default=str(RUNS_DIR / "calibration.json"))
    p = sub.add_parser("freeze", help="fingerprint the benchmark definition before the first paid run")
    freeze_path = str(RUNS_DIR / "benchmark_freeze.json")
    p.add_argument("--out", default=freeze_path)
    p = sub.add_parser("benchmark", help="run the 36-case benchmark (offline dry run unless --live)")
    p.add_argument("--fits-zip", required=True, type=Path)
    p.add_argument(
        "--live", action="store_true", help="call the model (paid); needs MODEL_API_KEY and a freeze"
    )
    p.add_argument("--freeze", default=freeze_path)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--ledger", default=None)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--limit", type=int, default=None, help="only the first N cases (result is flagged incomplete)"
    )
    p.add_argument("--budget", type=float, default=None)
    p = sub.add_parser("ledger", help="summarise an agent ledger (cost, unresolved cases)")
    p.add_argument("--path", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
        payload = plan()
        save_json(args.out, payload)
        print(json.dumps({"status": "PLAN_OK", "cases": len(payload["cases"]), "sha256": plan_hash()}))
        return 0
    if args.command == "ledger":
        print(json.dumps(MuseAgent(ledger_path=args.path).summary(), indent=2))
        return 0
    if args.command in ("freeze", "benchmark"):
        try:
            if args.command == "freeze":
                print(json.dumps(write_freeze(args.out), indent=2))
                return 0
            return _benchmark(args)
        except (RuntimeError, ValueError, PermissionError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "calibrate":
        result = calibrate(args.fits_zip)
        save_json(args.out, result)
        print(json.dumps({"status": "CALIBRATION_DONE", "summary": result["summary"]}, indent=2))
        return 0
    if args.command == "verify-data":
        images = []
        for entry in image_manifest():
            image, meta = load_verified_image(args.fits_zip, entry["field"])
            images.append({"field": entry["field"], "shape": list(image.shape), "sha256": meta["sha256"]})
        print(json.dumps({"status": "FITS_VERIFIED", "images": images}))
        return 0
    details, failures = _smoke(args.fits_zip)
    print(
        json.dumps(
            {"status": "SMOKE_FAILED" if failures else "SMOKE_OK", "cases": details, "failures": failures}
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
