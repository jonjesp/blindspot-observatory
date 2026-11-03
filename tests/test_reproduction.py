"""Checks of the offline reproducer using synthetic image fixtures, never a paid model."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_fits, requires_reference

from blindspot.agent.muse import AGENT, MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.benchmark.design import plan
from blindspot.benchmark.freeze import freeze_record
from blindspot.benchmark.report import summarise
from blindspot.benchmark.runner import evaluate_case, prepare_case
from blindspot.config import DETECTOR
from blindspot.provenance.ledger import digest, save_json

SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_benchmark.py"
spec = importlib.util.spec_from_file_location("reproduce_benchmark", SCRIPT)
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


@pytest.fixture(scope="module")
def prepared():
    raw = np.random.default_rng(19).normal(1000.0, 7.0, (DETECTOR.patch_size,) * 2)
    case = next(c for c in plan()["cases"] if c["id"] == "F03")
    return raw, prepare_case(raw, case)


@pytest.fixture
def records(prepared, tmp_path):
    raw, case = prepared
    path = tmp_path / "original_ledger.json"
    agent = MuseAgent(ledger_path=path)
    row = evaluate_case(case, agent=agent, client=OfflineMuseClient())
    results = {
        "mode": "offline_dry_run",
        "complete": False,
        "model": AGENT.model,
        "freeze": freeze_record(),
        "cases": [row],
        "summary": summarise([row]),
        "ledger": agent.summary(),
    }
    return raw, results, json.loads(path.read_text())


def mock_images(monkeypatch, raw, results):
    def load(_path, field):
        return raw, {"sha256": results["freeze"]["fits_sha256"][field]}

    monkeypatch.setattr(reproduction, "load_verified_image", load)
    monkeypatch.setattr(reproduction, "extract_patch", lambda *args: raw)


def test_record_audit_is_separate_from_pixel_reproduction(records):
    _, results, ledger = records
    audited = reproduction.audit_records(results, ledger)
    assert audited["status"] == "verified" and audited["reference_cases"] == 1
    assert "raw-image reproduction" in audited["scope"]


def test_numeric_roundtrip_uses_no_model_calls(records, monkeypatch):
    raw, results, ledger = records
    mock_images(monkeypatch, raw, results)

    def forbidden(*args, **kwargs):
        raise AssertionError("A reproduction must not ask a model")

    monkeypatch.setattr(MuseAgent, "decide", forbidden)
    report = reproduction.reproduce_cases(results, ledger, Path("unused.zip"), ["F03"])
    assert report["status"] == "reproduced" and report["api_calls"] == 0
    assert report["whole_reference_reproduced"]
    assert all(report["cases"][0]["checks"].values())


def test_measured_count_mismatch_cannot_pass(records, monkeypatch):
    raw, results, ledger = records
    mock_images(monkeypatch, raw, results)
    altered = copy.deepcopy(results)
    altered["cases"][0]["oracle"]["probe_flux"]["after"] += 1
    report = reproduction.reproduce_cases(altered, ledger, Path("unused.zip"), ["F03"])
    assert report["status"] == "mismatch" and not report["cases"][0]["checks"]["oracle"]


def test_request_reconstruction_detects_a_consistent_but_wrong_saved_hash(records, monkeypatch):
    raw, results, ledger = records
    mock_images(monkeypatch, raw, results)
    results, ledger = copy.deepcopy(results), copy.deepcopy(ledger)
    results["cases"][0]["muse"]["request_sha256"] = "0" * 64
    ledger["attempts"]["F03"]["request_sha256"] = "0" * 64
    ledger["seal_sha256"] = digest({"model": ledger["model"], "attempts": ledger["attempts"]})
    assert reproduction.audit_records(results, ledger)["status"] == "verified"
    report = reproduction.reproduce_cases(results, ledger, Path("unused.zip"), ["F03"])
    assert report["status"] == "mismatch"
    assert not report["cases"][0]["checks"]["request_hash"]


def test_changed_source_identity_is_rejected(records):
    _, results, ledger = records
    altered = copy.deepcopy(results)
    altered["freeze"]["implementation_sha256"]["cli.py"] = "0" * 64
    with pytest.raises(ValueError, match="frozen source"):
        reproduction.audit_records(altered, ledger)


def test_changed_ledger_is_rejected(records):
    _, results, ledger = records
    altered = copy.deepcopy(ledger)
    altered["attempts"]["F03"]["cost_usd"] = 1.0
    with pytest.raises(ValueError, match="ledger seal"):
        reproduction.audit_records(results, altered)


@pytest.mark.parametrize("cases", [[], ["F03", "F03"], ["F00"]])
def test_empty_duplicated_or_unknown_case_is_rejected(records, cases):
    _, results, ledger = records
    with pytest.raises(ValueError):
        reproduction.reproduce_cases(results, ledger, Path("unused.zip"), cases)


def test_wrong_image_identity_is_rejected(records, monkeypatch):
    raw, results, ledger = records
    monkeypatch.setattr(reproduction, "load_verified_image", lambda *args: (raw, {"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="FITS identity"):
        reproduction.reproduce_cases(results, ledger, Path("unused.zip"), ["F03"])


def write_inputs(tmp_path, results, ledger):
    a, b = tmp_path / "results.json", tmp_path / "ledger.json"
    save_json(a, results)
    save_json(b, ledger)
    return a, b


def test_audit_only_explicitly_leaves_pixels_pending(records, tmp_path):
    _, results, ledger = records
    a, b = write_inputs(tmp_path, results, ledger)
    out = tmp_path / "report.json"
    code = reproduction.main(["--results", str(a), "--ledger", str(b), "--audit-only", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 0 and report["status"] == "records_verified_pixels_pending"
    assert report["pixel_reproduction"] == "not_run"


def test_cli_all_reference_cases(records, tmp_path, monkeypatch):
    raw, results, ledger = records
    mock_images(monkeypatch, raw, results)
    a, b = write_inputs(tmp_path, results, ledger)
    out = tmp_path / "report.json"
    code = reproduction.main(
        [
            "--results",
            str(a),
            "--ledger",
            str(b),
            "--fits-zip",
            "unused.zip",
            "--all-cases",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert code == 0 and report["pixel_reproduction"]["case_ids"] == ["F03"]


def test_default_audit_output_preserves_a_pixel_report(records, tmp_path, monkeypatch):
    _, results, ledger = records
    a, b = write_inputs(tmp_path, results, ledger)
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    existing = evidence / "reproduction.json"
    existing.write_text('{"status":"reproduced"}')
    code = reproduction.main(["--results", str(a), "--ledger", str(b), "--audit-only"])
    assert code == 0 and (evidence / "runs" / "record_audit.json").exists()
    assert json.loads(existing.read_text())["status"] == "reproduced"


def test_output_cannot_overwrite_an_input(records, tmp_path):
    _, results, ledger = records
    a, b = write_inputs(tmp_path, results, ledger)
    previous = a.read_bytes()
    with pytest.raises(SystemExit) as exc:
        reproduction.main(["--results", str(a), "--ledger", str(b), "--audit-only", "--out", str(a)])
    assert exc.value.code == 2 and a.read_bytes() == previous


def test_failed_reproduction_preserves_published_report_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    archived = evidence / "reproduction.json"
    archived.write_text('{"retained": true}', encoding="utf-8")
    original = archived.read_bytes()
    save_json(evidence / "benchmark_results.json", {})
    save_json(evidence / "benchmark_ledger.json", {})
    assert reproduction.main(["--fits-zip", "unused.zip"]) == 1
    assert archived.read_bytes() == original
    assert json.loads((evidence / "runs" / "reproduction.json").read_text())["status"] == "error"


@requires_fits
@requires_reference
def test_real_fits_reproduce_the_stored_reference(fits_zip, reference_root, tmp_path):
    root = Path(__file__).parents[1]
    a, b = root / "evidence/benchmark_results.json", root / "evidence/benchmark_ledger.json"
    proof = json.loads((root / "evidence/reproduction.json").read_text(encoding="utf-8"))
    script = reference_root / "scripts/reproduce_benchmark.py"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == proof["script_sha256"]
    out = tmp_path / "reproduction.json"
    command = [
        sys.executable,
        "-m",
        "scripts.reproduce_benchmark",
        "--results",
        str(a),
        "--ledger",
        str(b),
        "--fits-zip",
        str(fits_zip.resolve()),
        "--out",
        str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=reference_root,
        env={**os.environ, "PYTHONPATH": str(reference_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "reproduced" and report["api_calls"] == 0
    assert report["pixel_reproduction"]["case_ids"] == list(reproduction.DEFAULT_CASES)
