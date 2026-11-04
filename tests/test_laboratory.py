"""Verify laboratory replay, model accounting, and loopback request restrictions."""

import base64
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from conftest import requires_fits

from blindspot.config import DETECTOR
from scripts.compare_models import load_sealed
from scripts.evaluate_models import request_for
from scripts.laboratory import LaboratoryServer, RunQueue
from scripts.laboratory_imaging import recovery_matches
from scripts.laboratory_run import Laboratory, LaboratoryError

MODELS = ("muse-spark-1.1", "muse-spark-1.3", "muse-wonder-1.0")


@pytest.mark.parametrize(("model", "cid"), list(zip(MODELS, ("F01", "F12", "F36"), strict=True)))
@requires_fits
def test_replay_matches_published_trace(fits_zip, model, cid):
    client = Mock()
    lab = Laboratory(fits_zip, client_factory=client)
    stages = Mock()
    result = lab.run(cid, model, "replay", stages)
    with Path("evidence/model_evaluation/case_results.csv").open(encoding="utf-8", newline="") as stream:
        published = next(
            row for row in csv.DictReader(stream) if row["model"] == model and row["case_id"] == cid
        )
    assert result["api_calls"] == 0
    assert result["response"]["state"] == published["response_status"]
    assert "|".join((result["decision"] or {}).get("probes", [])) == published["selected_probes"]
    assert "|".join(result["model_trace"]["executed"]) == published["executed_probes"]
    assert str(bool(result["continuation"])) == published["continuation_used"]
    assert str(result["model_trace"]["success"]) == (published["closed"] or published["correct_abstention"])
    assert [call.args[0] for call in stages.call_args_list] == ["image", "measurement", "decision", "scoring"]
    image = stages.call_args_list[1].kwargs["image"]
    assert image["width"] == image["height"] == DETECTOR.patch_size
    assert len(base64.b64decode(image["gray8"], validate=True)) == DETECTOR.patch_size**2
    assert stages.call_args_list[3].kwargs["model_reply"]["decision"] == result["decision"]
    assert all(
        0 <= row < image["height"] and 0 <= column < image["width"]
        for row, column in stages.call_args_list[2].kwargs["original_detections"]
    )
    if cid == "F01":
        views = result["holdout_views"]
        assert sum(len(batch["matches"]) for batch in views["before"]) == 17
        assert sum(len(batch["matches"]) for batch in views["after_by_probe"]["probe_psf"]) == 65
        assert sum(len(batch["injected"]) for batch in views["before"]) == 100
        assert sum(len(batch["injected"]) for batch in views["after_by_probe"]["probe_psf"]) == 100
    client.assert_not_called()


@pytest.mark.parametrize("model", MODELS)
@requires_fits
def test_live_request_and_ledger(fits_zip, tmp_path, monkeypatch, model):
    monkeypatch.setattr("scripts.laboratory_run.LIVE_RUNS", tmp_path)
    decision = {"decision": "probe", "probes": ["probe_psf"], "justification": "Supported recovery drop."}
    usage = SimpleNamespace(prompt_tokens=250, completion_tokens=100)
    choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(decision)))
    complete = Mock(
        return_value=SimpleNamespace(id="fixture-response", model=model, choices=[choice], usage=usage)
    )
    client = SimpleNamespace(
        base_url="https://api.meta.ai/v1/",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        ),
    )
    lab = Laboratory(fits_zip, 0.25, client_factory=lambda: nullcontext(client))
    result = lab.run("F01", model, "live", Mock())
    complete.assert_called_once_with(**request_for(lab.cases["F01"], lab.profiles[model]))
    assert result["api_calls"] == 1
    assert result["decision"] == decision
    assert result["model_trace"]["executed"] == ["probe_psf"]
    record = load_sealed(lab.ledger_path)["attempts"][f"{model}/F01"]
    assert 0 < record["cost_usd"] <= record["reservation_usd"] <= 0.25
    with pytest.raises(LaboratoryError, match="already ran"):
        lab.authorize_live("F01", model)


@requires_fits
def test_unknown_provider_outcome_keeps_reservation(fits_zip, tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.laboratory_run.LIVE_RUNS", tmp_path)
    complete = Mock(side_effect=TimeoutError("sensitive provider payload"))
    client = SimpleNamespace(
        base_url="https://api.meta.ai/v1/",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        ),
    )
    model = MODELS[1]
    lab = Laboratory(fits_zip, 0.25, client_factory=lambda: nullcontext(client))
    with pytest.raises(LaboratoryError, match="paused") as exc:
        lab.run("F01", model, "live", Mock())
    assert "sensitive provider payload" not in str(exc.value)
    assert load_sealed(lab.ledger_path)["attempts"][f"{model}/F01"]["state"] == "reserved_unknown"
    with pytest.raises(LaboratoryError, match="disabled"):
        lab.authorize_live("F02", model)
    complete.assert_called_once()


def test_overlay_matching_filters_baseline_and_resolves_distance_ties():
    positions = np.array([[10, 10], [10, 12], [28, 30]])
    detections = np.array([[10, 11], [28, 30]])
    baseline = np.array([[28, 30]])

    assert recovery_matches(positions, detections, baseline) == [(0, 0)]


def test_live_budget_rejected_before_client_initialization():
    client = Mock()
    lab = Laboratory(Path("unused.zip"), 0.0001, client_factory=client)
    with pytest.raises(LaboratoryError, match="exceeds"):
        lab.authorize_live("F01", MODELS[1])
    client.assert_not_called()


def test_local_server_serves_catalog_and_rejects_unsafe_requests():
    lab = Laboratory(Path("unused.zip"))
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        LaboratoryServer(0, RunQueue(lab, executor)) as server,
    ):
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            with urlopen(f"{server.origin}/api/catalog", timeout=5) as response:
                catalog = json.load(response)
            assert len(catalog["cases"]) == 36
            assert len(catalog["models"]) == 3
            assert catalog["live_enabled"] is False
            unsafe = Request(f"{server.origin}/api/catalog", headers={"Origin": "https://example.invalid"})
            with pytest.raises(HTTPError, match="403"):
                urlopen(unsafe, timeout=5)
            payload = json.dumps({"case_id": "F01", "model": MODELS[1], "mode": "live"}).encode()
            unconfirmed = Request(f"{server.origin}/api/runs", payload, {"Content-Type": "application/json"})
            with pytest.raises(HTTPError, match="400"):
                urlopen(unconfirmed, timeout=5)
        finally:
            server.shutdown()
            thread.join()
