"""Paired comparison controls, using synthetic records and simulated API clients only."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent import FakePaidClient

from blindspot.agent.muse import AGENT, MuseAgent, build_request
from blindspot.benchmark.design import plan
from blindspot.benchmark.freeze import freeze_record
from blindspot.benchmark.generator import private_case
from blindspot.benchmark.observations import PROBE_INDICATOR
from blindspot.benchmark.report import summarise
from blindspot.benchmark.runner import PreparedCase, evaluate_case
from blindspot.provenance.ledger import digest, save_json
from scripts import compare_models as comparison


def test_default_comparison_paths_preserve_the_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "evidence" / "comparison"
    archive.mkdir(parents=True)
    retained = archive / "wonder_results.json"
    retained.write_text('{"retained": true}', encoding="utf-8")
    received = {}

    def load_plan(path):
        received["plan"] = path
        return {}

    def run_plan(document, directory, client, *, live):
        received["directory"] = directory
        return {"summary": {"complete": True}, "error": None}

    monkeypatch.setattr(comparison, "load_sealed", load_plan)
    monkeypatch.setattr(comparison, "run_comparison", run_plan)
    assert comparison.main(["run"]) == 0
    assert received["plan"] == Path("evidence/runs/comparison/plan.json")
    assert received["directory"] == Path("evidence/runs/comparison")
    assert json.loads(retained.read_text()) == {"retained": True}


def reply(decision="probe", probes=("probe_flux",)):
    return json.dumps(
        {
            "decision": decision,
            "probes": list(probes),
            "justification": "Selected from the supplied measurements.",
        }
    )


class CandidateClient:
    """No sockets: mimics the exact API surface used by the comparison."""

    def __init__(self, raw=None, *, fails=False, model=None, enabled=True):
        self.calls, self.raw, self.fails = 0, raw or reply(), fails
        self.response_model = model or comparison.CANDIDATE["model"]
        self.base_url = comparison.DEFAULT_BASE_URL
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=comparison.CANDIDATE["model"])] if enabled else []
            )
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls += 1
        if self.fails:
            raise TimeoutError("Simulated transport failure")
        return SimpleNamespace(
            id=f"candidate-{self.calls}",
            model=self.response_model,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=self.raw))],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=100,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison_reference")
    agent = MuseAgent(ledger_path=root / "ledger.json")
    prepared, rows = {}, []
    for definition in plan()["cases"]:
        cid, truth = definition["id"], private_case(definition)
        indicators = {name: 0.0 for name in PROBE_INDICATOR.values()}
        if truth.dominant:
            indicators[PROBE_INDICATOR[truth.dominant]] = 0.5
        for probe in truth.masked:
            del indicators[PROBE_INDICATOR[probe]]
        obs = {
            "pilot_measurements": {
                "sources_per_batch": 25,
                "recovery_counts": {"nominal": {"A": [20] * 4, "B": [5 if truth.causes else 20] * 4}},
            },
            "indicators": indicators,
            "available_probes": list(comparison.TOOLS),
        }
        features = {p: indicators.get(PROBE_INDICATOR[p]) for p in comparison.TOOLS}
        oracle = {
            p: {
                "before": 10,
                "after": 70 if p == truth.dominant else 10,
                "n": 100,
                "gain": 0.6 if p == truth.dominant else 0.0,
                "closed": p == truth.dominant,
            }
            for p in comparison.TOOLS
        }
        prepared[cid] = PreparedCase(definition, truth, obs, features, oracle)
        raw = reply("probe", (truth.dominant,)) if truth.causes else reply("abstain", ())
        rows.append(
            evaluate_case(prepared[cid], agent=agent, client=FakePaidClient(raw=raw), allow_paid=True)
        )
    results = {
        "mode": "live",
        "complete": True,
        "model": AGENT.model,
        "freeze": freeze_record(),
        "cases": rows,
        "summary": summarise(rows),
        "ledger": agent.summary(),
    }
    save_json(root / "results.json", results)
    hashes = {
        "results": comparison.file_hash(root / "results.json"),
        "ledger": comparison.file_hash(root / "ledger.json"),
    }
    proof = {
        "status": "reproduced",
        "api_calls": 0,
        "input_sha256": hashes,
        "script_sha256": comparison.file_hash(Path(comparison.__file__).with_name("reproduce_benchmark.py")),
        "pixel_reproduction": {
            "whole_reference_reproduced": True,
            "api_calls": 0,
            "fits_sha256_verified": results["freeze"]["fits_sha256"],
            "cases": [
                {
                    "id": c["id"],
                    "matched": True,
                    "actual_oracle": c["oracle"],
                    "actual_request_sha256": c["muse"]["request_sha256"],
                    "checks": dict.fromkeys(
                        ["features", "oracle", "policies", "recovery_anomaly", "request_hash"], True
                    ),
                }
                for c in rows
            ],
        },
    }
    save_json(root / "proof.json", proof)
    return root, prepared


@pytest.fixture
def document(reference, tmp_path, monkeypatch):
    root, prepared = reference
    monkeypatch.setattr(comparison, "load_verified_image", lambda *args: (np.zeros((128, 128)), {}))
    monkeypatch.setattr(comparison, "extract_patch", lambda *args: np.zeros((128, 128)))
    monkeypatch.setattr(comparison, "prepare_case", lambda raw, case: prepared[case["id"]])
    return comparison.prepare_comparison(
        root / "results.json",
        root / "ledger.json",
        root / "proof.json",
        Path("synthetic.zip"),
        tmp_path / "plan.json",
    )


def test_catalog_failure_does_not_expose_exception_body(document, tmp_path):
    class CatalogError(Exception):
        status_code = 403

    def unavailable_catalog():
        raise CatalogError("sensitive-request-body")

    client = CandidateClient()
    client.models.list = unavailable_catalog
    report = comparison.run_comparison(document, tmp_path, client, live=True)
    assert report["error"] == "CatalogError (HTTP 403)"
    assert "sensitive-request-body" not in json.dumps(report)
    assert client.calls == 0


def test_preparation_changes_only_model_id_and_preserves_source(document):
    assert document["reference_freeze"] == freeze_record()
    assert document["maximum_calls"] == 36 and document["reservation_estimate_usd"] <= 3
    for case in document["cases"]:
        original = build_request(case["definition"]["id"], case["observation"])
        modified = {**original, "model": comparison.CANDIDATE["model"]}
        assert case["request"] == modified
        assert digest(original) == case["reference"]["muse"]["request_sha256"]


def test_plan_mutation_is_rejected(document):
    changed = copy.deepcopy(document)
    changed["budget_usd"] = 100
    with pytest.raises(ValueError, match="plan seal"):
        comparison.validate_plan(changed)


def test_prepare_cannot_overwrite_plan(reference, tmp_path):
    root, _ = reference
    out = tmp_path / "plan.json"
    out.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        comparison.prepare_comparison(
            root / "results.json", root / "ledger.json", root / "proof.json", Path("unused.zip"), out
        )
    assert out.read_text() == "keep"


def test_live_run_resumes_without_duplicate_paid_calls(document, tmp_path):
    client = CandidateClient()
    first = comparison.run_comparison(document, tmp_path / "live", client, live=True)
    second = comparison.run_comparison(document, tmp_path / "live", client, live=True)
    assert first["summary"]["complete"] and second["summary"]["complete"]
    assert client.calls == 36 and first["cases"] == second["cases"]


def test_unavailable_model_stops_before_paid_requests(document, tmp_path):
    client = CandidateClient(enabled=False)
    report = comparison.run_comparison(document, tmp_path, client, live=True)
    assert client.calls == 0 and not report["summary"]["complete"]
    assert not report["summary"]["primary_gate_passed"]


def test_invalid_control_is_not_a_correct_abstention(document, tmp_path):
    case = next(c for c in document["cases"] if c["definition"]["kind"] == "no_intervention")
    ledger = comparison.open_ledger(tmp_path / "l.json", document, True)
    record = comparison.call_case(
        document, case, ledger, tmp_path / "l.json", CandidateClient(raw="bad json"), live=True
    )
    row = comparison.evaluate_candidate(case, record)
    assert record["state"] == "invalid_response"
    assert row["policies"]["candidate"]["probes"] == 3
    assert not row["policies"]["candidate"]["correct_abstention"]


def test_failed_call_keeps_reservation_and_is_not_retried(document, tmp_path):
    case = document["cases"][0]
    path = tmp_path / "l.json"
    ledger = comparison.open_ledger(path, document, True)
    client = CandidateClient(fails=True)
    with pytest.raises(TimeoutError):
        comparison.call_case(document, case, ledger, path, client, live=True)
    assert comparison.spent_usd(comparison.load_sealed(path)) > 0
    with pytest.raises(RuntimeError, match="Unresolved"):
        comparison.call_case(document, case, ledger, path, client, live=True)
    assert client.calls == 1


def test_budget_blocks_before_call(document, tmp_path):
    changed = {**document, "budget_usd": 1e-9}
    client = CandidateClient()
    ledger = comparison.open_ledger(tmp_path / "l.json", changed, True)
    with pytest.raises(RuntimeError, match="Budget"):
        comparison.call_case(changed, changed["cases"][0], ledger, tmp_path / "l.json", client, live=True)
    assert client.calls == 0


def test_response_model_identity_mismatch_is_not_retried(document, tmp_path):
    path = tmp_path / "l.json"
    ledger = comparison.open_ledger(path, document, True)
    client = CandidateClient(model="different-model")
    with pytest.raises(RuntimeError, match="different"):
        comparison.call_case(document, document["cases"][0], ledger, path, client, live=True)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        comparison.call_case(document, document["cases"][0], ledger, path, client, live=True)
    assert client.calls == 1 and comparison.spent_usd(ledger) > 0


def test_run_lock_excludes_concurrent_runs(tmp_path):
    path = tmp_path / "ledger.json"
    with comparison.run_lock(path), pytest.raises(RuntimeError, match="lock exists"):
        with comparison.run_lock(path):
            raise AssertionError("Concurrent run entered")
    assert not path.with_suffix(".json.lock").exists()


def test_offline_ledger_cannot_be_reused_live(document, tmp_path):
    path = tmp_path / "l.json"
    ledger = comparison.open_ledger(path, document, False)
    save_json(path, comparison.sealed(ledger))
    with pytest.raises(ValueError, match="mode"):
        comparison.open_ledger(path, document, True)


@pytest.mark.parametrize("failure", [None, "offline", "partial", "closure", "control", "tie"])
def test_primary_gate_requires_strict_advantage_and_no_lost_success(document, tmp_path, failure):
    rows = []
    for case in document["cases"]:
        row = copy.deepcopy(case["reference"])
        trace = copy.deepcopy(row["policies"]["elimination"])
        if row["kind"] != "no_intervention":
            row["policies"]["elimination"]["probes"] = 2
            trace["probes"] = 2 if failure == "tie" else 1
        row["policies"]["candidate"] = trace
        row["candidate"] = {"state": "completed", "used_fallback": False, "closed_without_fallback": True}
        rows.append(row)
    if failure == "partial":
        rows = rows[:-1]
    elif failure in ("closure", "control"):
        selected = next(r for r in rows if (r["kind"] == "no_intervention") == (failure == "control"))
        selected["policies"]["candidate"]["success"] = False
    ledger = comparison.open_ledger(tmp_path / "l.json", document, True)
    summary = comparison.comparison_summary(rows, document, ledger, live=failure != "offline")
    assert summary["primary_gate_passed"] == (failure is None)


def test_openai_sdk_wire_request_uses_mock_transport(document, tmp_path):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "mock-response",
                "object": "chat.completion",
                "created": 0,
                "model": comparison.CANDIDATE["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply()},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        client = openai.OpenAI(
            api_key="test-only", base_url=comparison.DEFAULT_BASE_URL, max_retries=0, http_client=http_client
        )
        path = tmp_path / "l.json"
        record = comparison.call_case(
            document,
            document["cases"][0],
            comparison.open_ledger(path, document, True),
            path,
            client,
            live=True,
        )
    assert record["state"] == "completed" and len(captured) == 1
    assert record["started_utc"].endswith("+00:00") and record["recorded_utc"].endswith("+00:00")
    assert captured[0].url.path == "/v1/chat/completions"
    assert json.loads(captured[0].content) == document["cases"][0]["request"]