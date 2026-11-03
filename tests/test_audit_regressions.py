"""Regressions for hidden-label execution, missingness leakage, freeze and invalid inputs."""

from dataclasses import replace

import numpy as np
import pytest

from blindspot.agent.muse import MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark import freeze, interventions
from blindspot.benchmark.design import plan
from blindspot.benchmark.generator import PrivateCase, private_case
from blindspot.benchmark.interventions import NOMINAL
from blindspot.benchmark.report import summarise
from blindspot.benchmark.runner import PreparedCase, evaluate_case
from blindspot.config import CASE_KINDS
from blindspot.detection.detector import detect_sources


@pytest.fixture
def quiet_case():
    observation = {
        "pilot_measurements": {
            "sources_per_batch": 25,
            "recovery_counts": {"nominal": {"A": [15] * 4, "B": [15] * 4}},
        },
        "indicators": {},
        "available_probes": list(TOOLS),
    }
    oracle = {p: {"before": 60, "after": 60, "n": 100, "gain": 0.0, "closed": False} for p in TOOLS}
    return PreparedCase(
        {"id": "F01", "field": "E4_r", "kind": "no_intervention"},
        PrivateCase((), (), None, NOMINAL),
        observation,
        {p: None for p in TOOLS},
        oracle,
    )


@pytest.mark.parametrize("kwargs", [{"raw": "invalid json"}, {"raw": "", "finish": "length"}])
def test_invalid_control_reply_is_not_a_correct_abstention(quiet_case, tmp_path, kwargs):
    row = evaluate_case(
        quiet_case,
        agent=MuseAgent(ledger_path=tmp_path / "invalid.json"),
        client=OfflineMuseClient(**kwargs),
    )
    assert row["muse"]["state"] == "invalid_response"
    assert not row["policies"]["muse"]["success"]
    assert not row["policies"]["muse"]["correct_abstention"]
    assert row["policies"]["muse"]["executed"] == list(TOOLS)


@pytest.mark.parametrize("close_first", [False, True])
def test_actions_and_stop_rule_ignore_hidden_class(quiet_case, tmp_path, close_first):
    if close_first:
        oracle = {
            **quiet_case.oracle,
            "probe_sky": {"before": 20, "after": 80, "n": 100, "gain": 0.6, "closed": True},
        }
        quiet_case = replace(quiet_case, oracle=oracle)
    orders = []
    for kind in CASE_KINDS:
        candidate = replace(quiet_case, case={**quiet_case.case, "kind": kind})
        row = evaluate_case(
            candidate,
            agent=MuseAgent(ledger_path=tmp_path / f"{kind}.json"),
            client=OfflineMuseClient(),
        )
        orders.append(row["policies"]["muse"]["executed"])
    assert all(order == orders[0] for order in orders)
    assert len(orders[0]) == (1 if close_first else 3)


def test_explicit_valid_abstention_still_succeeds_on_a_control(quiet_case, tmp_path):
    row = evaluate_case(
        quiet_case,
        agent=MuseAgent(ledger_path=tmp_path / "abstain.json"),
        client=OfflineMuseClient(
            raw='{"decision":"abstain","probes":[],"justification":"No recovery drop is supported."}'
        ),
    )
    assert row["policies"]["muse"]["correct_abstention"]
    assert row["policies"]["muse"]["probes"] == 0


def test_masking_does_not_depend_on_hidden_class_or_number_of_causes():
    masks = []
    for case in plan()["cases"]:
        alternatives = [private_case({**case, "kind": kind}).masked for kind in CASE_KINDS]
        assert all(mask == alternatives[0] for mask in alternatives)
        masks.append(alternatives[0])
    assert any(masks) and not all(masks)
    assert len({probe for mask in masks for probe in mask}) == len(TOOLS)


def test_an_absent_diagnostic_does_not_identify_the_cause():
    masked = [private_case(c) for c in plan()["cases"] if private_case(c).masked]
    assert any(not t.causes for t in masked)
    assert any(t.causes and not set(t.masked) <= set(t.causes) for t in masked)


def test_freeze_detects_runtime_intervention_change(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    freeze.write_freeze(path)
    monkeypatch.setitem(interventions.STRONG, "probe_flux", 0.01)
    with pytest.raises(RuntimeError, match="experimental_parameters_sha256"):
        freeze.verify_freeze(path)


def test_freeze_hashes_source_and_resources(tmp_path, monkeypatch):
    package = tmp_path / "blindspot"
    module = package / "benchmark" / "freeze.py"
    module.parent.mkdir(parents=True)
    module.write_text('"""test fixture"""\n')
    detector = package / "detector.py"
    detector.write_bytes(b"THRESHOLD = 5\r\n")
    monkeypatch.setattr(freeze, "__file__", str(module))
    before = freeze.implementation_hashes()
    detector.write_bytes(b"THRESHOLD = 5\n")
    assert freeze.implementation_hashes() == before
    path = tmp_path / "freeze.json"
    freeze.write_freeze(path)
    detector.write_text("THRESHOLD = 7\n")
    with pytest.raises(RuntimeError, match="implementation_sha256"):
        freeze.verify_freeze(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_background_is_rejected(bad):
    image = np.zeros((128, 128))
    background = np.zeros_like(image)
    background[60, 60] = bad
    with pytest.raises(ValueError, match="background"):
        detect_sources(image, background, 1.0)


def test_fallback_closure_is_not_attributed_to_selected_probes(quiet_case, tmp_path):
    oracle = {
        **quiet_case.oracle,
        "probe_flux": {"before": 20, "after": 80, "n": 100, "gain": 0.6, "closed": True},
    }
    prepared = replace(quiet_case, case={**quiet_case.case, "kind": "single_cause"}, oracle=oracle)
    row = evaluate_case(
        prepared,
        agent=MuseAgent(ledger_path=tmp_path / "fallback.json"),
        client=OfflineMuseClient(),
    )
    assert row["policies"]["muse"]["success"]
    assert row["muse"]["used_fallback"]
    assert not row["muse"]["closed_without_fallback"]
    summary = summarise([row])
    assert summary["muse_replies"]["closed_without_fallback"]["k"] == 0
