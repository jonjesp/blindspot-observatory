"""Multi-model accounting and reporting checks with deterministic, network-free clients."""

import copy
import csv
import json
import math
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_fits

from blindspot.agent.client import DEFAULT_BASE_URL
from blindspot.benchmark.generator import private_case
from blindspot.benchmark.runner import PreparedCase
from blindspot.provenance.ledger import save_json
from scripts import evaluate_models as evaluation
from scripts import reproduce_evaluation as reproduction
from scripts.compare_models import call_case, load_sealed, run_lock, sealed


class EvaluationClient:
    """Test responses use known oracle outcomes; this class never opens a socket."""

    def __init__(
        self,
        protocol,
        *,
        fails=False,
        returned_model=None,
        finish="stop",
        unavailable=(),
        control_alarm=False,
        follow_elimination=False,
    ):
        self.calls = []
        self.fails, self.returned_model, self.finish = fails, returned_model, finish
        self.control_alarm, self.follow_elimination = control_alarm, follow_elimination
        self.base_url = DEFAULT_BASE_URL
        self.cases = {c["definition"]["id"]: c for c in protocol["cases"]}
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[
                    SimpleNamespace(id=profile["model"])
                    for profile in protocol["profiles"]
                    if profile["model"] not in unavailable
                ]
            )
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        if self.fails:
            raise TimeoutError("Simulated request timeout")
        cid = json.loads(request["messages"][1]["content"])["case_id"]
        case = self.cases[cid]
        controls = case["definition"]["kind"] == "no_intervention"
        best = max(case["reference"]["oracle"], key=lambda probe: case["reference"]["oracle"][probe]["gain"])
        probes = [] if controls else [best]
        if self.control_alarm and controls:
            probes = ["probe_sky"]
        if self.follow_elimination:
            probes = case["reference"]["policies"]["elimination"]["executed"][:2]
        decision = "ambiguous" if len(probes) == 2 else "probe" if probes else "abstain"
        raw = json.dumps(
            {
                "decision": decision,
                "probes": probes,
                "justification": "Deterministic fixture for the accounting checks.",
            }
        )
        return SimpleNamespace(
            id=f"evaluation-response-{len(self.calls)}",
            model=request["model"] if self.returned_model is None else self.returned_model,
            choices=[SimpleNamespace(finish_reason=self.finish, message=SimpleNamespace(content=raw))],
            usage=SimpleNamespace(
                prompt_tokens=200,
                completion_tokens=100,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
            ),
        )


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    evidence = Path(__file__).parents[1] / "evidence/comparison/plan.json"
    cases = json.loads(evidence.read_text())["cases"]
    prepared = {
        case["definition"]["id"]: PreparedCase(
            case["definition"],
            private_case(case["definition"]),
            case["observation"],
            case["reference"]["features"],
            case["reference"]["oracle"],
        )
        for case in cases
    }
    monkeypatch.setattr(evaluation, "load_verified_image", lambda *args: (np.zeros((128, 128)), {}))
    monkeypatch.setattr(evaluation, "extract_patch", lambda *args: np.zeros((128, 128)))
    monkeypatch.setattr(evaluation, "prepare_case", lambda raw, definition: prepared[definition["id"]])
    return evaluation.prepare_evaluation(Path("fixture.zip"), tmp_path / "evaluation")


def test_protocol_is_balanced_and_freezes_supported_reasoning(protocol):
    assert len(protocol["schedule"]) == 108
    assert protocol["reservation_bound_usd"] <= 10
    assert [profile["reasoning_effort"] for profile in protocol["profiles"]] == ["xhigh", "max", "max"]
    assert protocol["output_token_limit"] == 8192
    assert {profile["model"] for profile in protocol["profiles"]} == {
        "muse-spark-1.1",
        "muse-spark-1.3",
        "muse-wonder-1.0",
    }
    case = protocol["cases"][0]
    requests = [evaluation.request_for(case, profile) for profile in protocol["profiles"]]
    assert requests[0]["messages"] == requests[1]["messages"] == requests[2]["messages"]
    assert "oracle" not in json.loads(requests[0]["messages"][1]["content"])


def test_live_collection_is_complete_and_resumes_without_duplicate_calls(protocol, tmp_path):
    directory = tmp_path / "evaluation"
    client = EvaluationClient(protocol)
    outcome = evaluation.run_evaluation(protocol, directory, client, live=True)
    assert outcome == {"complete": True, "error": None}
    assert len(client.calls) == 108
    ledger = load_sealed(directory / "ledger.json")
    assert len(ledger["attempts"]) == 108
    assert all(record["provider_base_url"] == DEFAULT_BASE_URL for record in ledger["attempts"].values())
    assert all(record["reasoning_tokens"] == 60 for record in ledger["attempts"].values())
    retry = EvaluationClient(protocol)
    assert evaluation.run_evaluation(protocol, directory, retry, live=True)["complete"]
    assert retry.calls == []
    table = list(csv.DictReader((directory / "case_results.csv").open(encoding="utf-8")))
    assert len(table) == 108
    report = (directory / "report.md").read_text()
    assert "98.33%" in report
    assert "Muse Wonder 1.0" in report
    assert "private reasoning text is not collected" in report


def test_interrupted_request_cannot_be_rebilled_on_resume(protocol, tmp_path):
    directory = tmp_path / "evaluation"
    first = EvaluationClient(protocol, fails=True)
    outcome = evaluation.run_evaluation(protocol, directory, first, live=True)
    assert not outcome["complete"]
    assert len(first.calls) == 1
    ledger = load_sealed(directory / "ledger.json")
    assert list(ledger["attempts"].values())[0]["state"] == "reserved_unknown"
    retry = EvaluationClient(protocol)
    result = evaluation.run_evaluation(protocol, directory, retry, live=True)
    assert "unresolved" in result["error"]
    assert retry.calls == []
    assert "Unresolved requests: 1" in (directory / "report.md").read_text()


@pytest.mark.parametrize("returned_model", ["unrequested-model", ""])
def test_provider_identity_mismatch_stops_further_inference(protocol, tmp_path, returned_model):
    directory = tmp_path / "evaluation"
    client = EvaluationClient(protocol, returned_model=returned_model)
    outcome = evaluation.run_evaluation(protocol, directory, client, live=True)
    assert not outcome["complete"]
    assert len(client.calls) == 1
    retry = EvaluationClient(protocol)
    evaluation.run_evaluation(protocol, directory, retry, live=True)
    assert retry.calls == []


def test_global_budget_is_shared_between_models(protocol, tmp_path):
    case = protocol["cases"][0]
    ledger = {"attempts": {}}
    client = EvaluationClient(protocol)
    first, second = protocol["profiles"][0], protocol["profiles"][1]
    allowance = evaluation.reservation_for(case, first) + 1e-6
    assert evaluation.reservation_for(case, second) < allowance
    settings = {"base_url": DEFAULT_BASE_URL, "budget_usd": allowance, "maximum_calls": 108}
    call_case(
        {**settings, "candidate": first},
        {"definition": case["definition"], "request": evaluation.request_for(case, first)},
        ledger,
        tmp_path / "shared.json",
        client,
        live=True,
        attempt_key="spark/F01",
    )
    with pytest.raises(RuntimeError, match="Budget reservation"):
        call_case(
            {**settings, "candidate": second},
            {"definition": case["definition"], "request": evaluation.request_for(case, second)},
            ledger,
            tmp_path / "shared.json",
            client,
            live=True,
            attempt_key="spark-later/F01",
        )
    assert len(client.calls) == 1
    assert len(ledger["attempts"]) == 1


def test_unavailable_model_is_reported_without_synthetic_results(protocol, tmp_path):
    missing = protocol["profiles"][0]["model"]
    client = EvaluationClient(protocol, unavailable=(missing,))
    result = evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)
    assert not result["complete"]
    assert len(client.calls) == 72
    assert all(request["model"] != missing for request in client.calls)
    report = (tmp_path / "evaluation/report.md").read_text()
    assert "Unavailable" in report
    assert "Not evaluated" in report


def test_truncated_responses_are_counted_without_retries_or_superiority_claim(protocol, tmp_path):
    client = EvaluationClient(protocol, finish="length")
    assert evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)["complete"]
    assert len(client.calls) == 108
    report = (tmp_path / "evaluation/report.md").read_text()
    assert "Met for this case set" not in report
    ledger = load_sealed(tmp_path / "evaluation/ledger.json")
    assert all(record["state"] == "invalid_response" for record in ledger["attempts"].values())


def test_changed_criterion_is_rejected_before_any_inference(protocol, tmp_path):
    altered = copy.deepcopy(protocol)
    altered["criterion"]["familywise_alpha"] = 0.9
    client = EvaluationClient(protocol)
    with pytest.raises(ValueError, match="criterion"):
        evaluation.run_evaluation(altered, tmp_path / "evaluation", client, live=True)
    assert client.calls == []


def test_resealed_accounting_change_is_rejected_before_resume(protocol, tmp_path):
    directory = tmp_path / "evaluation"
    evaluation.run_evaluation(protocol, directory, EvaluationClient(protocol), live=True)
    ledger = load_sealed(directory / "ledger.json")
    first = next(iter(ledger["attempts"].values()))
    first["cost_usd"] = 0
    save_json(directory / "ledger.json", sealed({k: v for k, v in ledger.items() if k != "seal_sha256"}))
    retry = EvaluationClient(protocol)
    result = evaluation.run_evaluation(protocol, directory, retry, live=True)
    assert not result["complete"]
    assert retry.calls == []
    assert "usage" in result["error"]


def test_non_meta_endpoint_is_rejected_before_inference(protocol, tmp_path):
    client = EvaluationClient(protocol)
    client.base_url = "https://unrelated.invalid/v1"
    assert not evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)["complete"]
    assert client.calls == []


def test_concurrent_evaluation_is_excluded(protocol, tmp_path):
    directory = tmp_path / "evaluation"
    client = EvaluationClient(protocol)
    with run_lock(directory / "ledger.json"):
        with pytest.raises(RuntimeError, match="lock"):
            evaluation.run_evaluation(protocol, directory, client, live=True)
    assert client.calls == []


@pytest.mark.parametrize("budget", [0, -1, 11, float("nan"), float("inf")])
def test_invalid_campaign_budget_is_rejected_before_reading_images(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        evaluation.prepare_evaluation(Path("absent.zip"), tmp_path / "evaluation", budget)


def test_existing_protocol_is_not_overwritten(protocol, tmp_path):
    path = tmp_path / "evaluation/protocol.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="empty"):
        evaluation.prepare_evaluation(Path("unused.zip"), path.parent)
    assert path.read_bytes() == original


def test_competing_preparation_cannot_replace_an_existing_protocol(protocol, tmp_path, monkeypatch):
    target = tmp_path / "racing"

    @contextmanager
    def competing_writer(path):
        with run_lock(path):
            save_json(path, {"owner": "concurrent-preparation"})
            yield

    monkeypatch.setattr(evaluation, "run_lock", competing_writer)
    with pytest.raises(ValueError, match="already exists"):
        evaluation.prepare_evaluation(Path("fixture.zip"), target)
    assert json.loads((target / "protocol.json").read_text()) == {"owner": "concurrent-preparation"}


def test_insufficient_allowance_does_not_reduce_the_frozen_conditions(protocol, tmp_path):
    target = tmp_path / "insufficient"
    with pytest.raises(ValueError, match="full protocol"):
        evaluation.prepare_evaluation(Path("fixture.zip"), target, budget=1)
    assert not (target / "protocol.json").exists()


def test_catalog_failure_preserves_an_inspectable_ledger_without_exposing_error_body(protocol, tmp_path):
    class CatalogError(Exception):
        status_code = 403

    def unavailable():
        raise CatalogError("private-response-body")

    directory = tmp_path / "evaluation"
    client = EvaluationClient(protocol)
    client.models.list = unavailable
    result = evaluation.run_evaluation(protocol, directory, client, live=True)
    assert result["error"] == "CatalogError (HTTP 403)"
    assert client.calls == []
    assert load_sealed(directory / "ledger.json")["attempts"] == {}
    report = (directory / "report.md").read_text()
    assert "HTTP 403" in report
    assert "private-response-body" not in report


def test_lost_control_abstentions_prevent_a_superiority_claim(protocol, tmp_path):
    client = EvaluationClient(protocol, control_alarm=True)
    assert evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)["complete"]
    report = (tmp_path / "evaluation/report.md").read_text()
    assert "Met for this case set" not in report


def test_tying_elimination_does_not_establish_superiority(protocol, tmp_path):
    client = EvaluationClient(protocol, follow_elimination=True)
    assert evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)["complete"]
    report = (tmp_path / "evaluation/report.md").read_text()
    assert "Met for this case set" not in report
    assert "+0.0000 [+0.0000, +0.0000]" in report


def test_provider_text_is_safe_in_spreadsheet_cells(protocol, tmp_path):
    client = EvaluationClient(protocol, returned_model="=1+1")
    evaluation.run_evaluation(protocol, tmp_path / "evaluation", client, live=True)
    with (tmp_path / "evaluation/case_results.csv").open(encoding="utf-8") as stream:
        record = next(csv.DictReader(stream))
    assert record["returned_model"] == "'=1+1"


def test_sdk_transmits_profiles_and_captures_provider_metadata_without_network(protocol, tmp_path):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    observed = []

    def handle(request):
        body = json.loads(request.content)
        observed.append(body)
        assert request.url.host == "api.meta.ai"
        assert body["max_completion_tokens"] == 8192
        assert body["response_format"]["type"] == "json_schema"
        assert body["store"] is False
        return httpx.Response(
            200,
            json={
                "id": f"wire-response-{len(observed)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "decision": "abstain",
                                    "probes": [],
                                    "justification": "Wire-format fixture without network access.",
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                    "prompt_tokens_details": {"cached_tokens": 50},
                    "completion_tokens_details": {"reasoning_tokens": 60},
                },
            },
        )

    case, ledger = protocol["cases"][0], {"attempts": {}}
    settings = {"base_url": DEFAULT_BASE_URL, "budget_usd": 10.0, "maximum_calls": 108}
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = openai.OpenAI(
            api_key="test-only",
            base_url=DEFAULT_BASE_URL,
            max_retries=0,
            http_client=transport,
        )
        for profile in protocol["profiles"]:
            record = call_case(
                {**settings, "candidate": profile},
                {"definition": case["definition"], "request": evaluation.request_for(case, profile)},
                ledger,
                tmp_path / "wire.json",
                client,
                live=True,
                attempt_key=f"{profile['model']}/F01",
            )
            assert record["response_model"] == profile["model"]
            assert record["reasoning_tokens"] == 60
            assert record["provider_base_url"] == DEFAULT_BASE_URL
    assert [body["reasoning_effort"] for body in observed] == ["xhigh", "max", "max"]


@pytest.mark.parametrize("direction", [math.inf, -math.inf])
def test_reservation_roundoff_is_accepted_with_an_absolute_tolerance(protocol, direction):
    rounded = copy.deepcopy(protocol)
    rounded["reservation_bound_usd"] = math.nextafter(protocol["reservation_bound_usd"], direction)
    evaluation.validate_protocol(rounded)


@pytest.mark.parametrize("difference", [5e-12, 1e-6, math.inf, math.nan])
def test_reservation_discrepancies_above_tolerance_are_rejected(protocol, difference):
    altered = copy.deepcopy(protocol)
    altered["reservation_bound_usd"] += difference
    with pytest.raises(ValueError, match="campaign reservation bound"):
        evaluation.validate_protocol(altered)


@pytest.fixture(scope="module")
def recorded_sources(tmp_path_factory):
    root = Path(__file__).parents[1]
    checkout = tmp_path_factory.mktemp("recorded_evaluation")
    source = checkout / "scripts"
    source.mkdir()
    for name in evaluation.SOURCE_NAMES:
        shutil.copyfile(root / "scripts" / name, source / name)
    return source


@requires_fits
def test_published_evaluation_can_be_audited_without_credentials_or_input_changes(
    fits_zip,
    tmp_path,
    monkeypatch,
):
    archive = Path(__file__).parents[1] / "evidence/model_evaluation"
    names = ["protocol.json", "ledger.json", "case_results.csv"]
    original = {name: (archive / name).read_bytes() for name in names}

    def forbidden_client(*args, **kwargs):
        raise AssertionError("An offline publication audit must not construct a model client")

    monkeypatch.setattr(evaluation, "make_live_client", forbidden_client)
    output = tmp_path / "audit"
    assert (
        reproduction.main(
            [
                "--directory",
                str(archive),
                "--fits-zip",
                str(fits_zip),
                "--output-directory",
                str(output),
            ]
        )
        == 0
    )
    with (output / "case_results.csv").open(encoding="utf-8") as stream:
        generated = list(csv.DictReader(stream))
    with (archive / "case_results.csv").open(encoding="utf-8") as stream:
        supplied = list(csv.DictReader(stream))
    assert generated == supplied
    assert len(generated) == 108
    assert sum(row["response_status"] == "completed" for row in generated) == 105
    assert {name: (archive / name).read_bytes() for name in names} == original


def test_recorded_source_audit_cannot_overwrite_the_publication_evidence(recorded_sources):
    archive = Path(__file__).parents[1] / "evidence/model_evaluation"
    assert (
        evaluation.main(
            [
                "report",
                "--directory",
                str(archive),
                "--recorded-sources",
                str(recorded_sources),
            ]
        )
        == 2
    )


def test_recorded_source_option_is_not_available_for_paid_execution(recorded_sources):
    with pytest.raises(SystemExit) as stopped:
        evaluation.main(["run", "--live", "--recorded-sources", str(recorded_sources)])
    assert stopped.value.code == 2


def test_original_source_hashes_remain_required_for_archival_audits(protocol, recorded_sources, tmp_path):
    document = protocol
    different = tmp_path / "different_sources"
    shutil.copytree(recorded_sources, different)
    with (different / "evaluate_models.py").open("a") as stream:
        stream.write("\n# altered source\n")
    with pytest.raises(ValueError, match="evaluation scripts"):
        evaluation.validate_protocol(document, recorded_sources=different)


def test_matching_source_directory_is_accepted(protocol, recorded_sources):
    evaluation.validate_protocol(protocol, recorded_sources=recorded_sources)


def test_paid_execution_rejects_archived_sources_before_any_request(tmp_path):
    archive = Path(__file__).parents[1] / "evidence/model_evaluation"
    document = load_sealed(archive / "protocol.json")
    client = EvaluationClient(document)
    with pytest.raises(ValueError, match="evaluation scripts"):
        evaluation.run_evaluation(document, tmp_path, client, live=True)
    assert client.calls == []


def test_publication_csv_retains_original_bytes_when_added_to_git(tmp_path):
    root = Path(__file__).parents[1]
    archive = Path("evidence/model_evaluation/case_results.csv")
    destination = tmp_path / archive
    destination.parent.mkdir(parents=True)
    shutil.copyfile(root / ".gitattributes", tmp_path / ".gitattributes")
    shutil.copyfile(root / archive, destination)
    original = destination.read_bytes()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-c", "core.autocrlf=true", "add", ".gitattributes", archive.as_posix()],
        cwd=tmp_path,
        check=True,
    )
    staged = subprocess.run(
        ["git", "show", f":{archive.as_posix()}"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout
    assert staged == original
