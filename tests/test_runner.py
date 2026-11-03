"""Benchmark execution, continuation and summary tests."""

import json

import numpy as np
import pytest

from blindspot.agent.muse import MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark import freeze
from blindspot.benchmark.baselines import plan_probes
from blindspot.benchmark.design import plan
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.report import format_summary, paired_bootstrap_ci, summarise
from blindspot.benchmark.runner import evaluate_case, prepare_case, run_benchmark
from blindspot.config import DETECTOR


def reply(decision, probes, text="Chosen from the indicator pattern shown."):
    return json.dumps({"decision": decision, "probes": probes, "justification": text})


def pick(kind, n=1):
    return [c for c in plan()["cases"] if c["kind"] == kind][:n]


@pytest.fixture(scope="module")
def raw():
    return np.random.default_rng(3).normal(1000.0, 7.0, (DETECTOR.patch_size, DETECTOR.patch_size))


@pytest.fixture(scope="module")
def prepared(raw):
    cases = pick("no_intervention") + pick("single_cause", 2) + pick("two_causes")
    return [prepare_case(raw, c) for c in cases]


def run(prepared, client, tmp_path, name="l.json"):
    agent = MuseAgent(ledger_path=tmp_path / name)
    return [evaluate_case(p, agent=agent, client=client) for p in prepared], agent


def test_prepared_cases_are_solvable_and_deterministic(prepared, raw):
    for p in prepared:
        assert p.case["kind"] == "no_intervention" or any(m["closed"] for m in p.oracle.values())
        assert prepare_case(raw, p.case).oracle == p.oracle


def test_correct_first_choice_costs_one_probe(prepared, tmp_path):
    for p in prepared:
        if p.case["kind"] == "no_intervention":
            continue
        best = max(p.oracle, key=lambda k: p.oracle[k]["gain"])
        row = evaluate_case(
            p,
            agent=MuseAgent(ledger_path=tmp_path / f"{p.case['id']}.json"),
            client=OfflineMuseClient(raw=reply("probe", [best])),
        )
        assert row["policies"]["muse"]["executed"] == [best] and row["policies"]["muse"]["probes"] == 1
        assert row["policies"]["muse"]["success"] and row["policies"]["muse"]["first_probe_closed"]
        assert not row["muse"]["used_fallback"]
        assert row["muse"]["closed_without_fallback"]


def test_wrong_first_choice_falls_back_in_fixed_order_and_pays_for_it(prepared, tmp_path):
    p = next(x for x in prepared if x.case["kind"] == "single_cause")
    wrong = next(t for t in TOOLS if not p.oracle[t]["closed"])
    row = evaluate_case(
        p,
        agent=MuseAgent(ledger_path=tmp_path / "l.json"),
        client=OfflineMuseClient(raw=reply("probe", [wrong])),
    )
    executed = row["policies"]["muse"]["executed"]
    assert executed[0] == wrong and executed[1:] == [t for t in TOOLS if t != wrong][: len(executed) - 1]
    assert row["policies"]["muse"]["success"] and row["policies"]["muse"]["probes"] >= 2
    assert row["muse"]["used_fallback"] and not row["policies"]["muse"]["first_probe_closed"]


def test_abstention_is_final(prepared, tmp_path):
    rows, _ = run(prepared, OfflineMuseClient(raw=reply("abstain", [])), tmp_path)
    for row in rows:
        trace = row["policies"]["muse"]
        assert trace["probes"] == 0 and trace["executed"] == []
        assert trace["success"] == (row["kind"] == "no_intervention")


def test_invalid_reply_runs_the_fixed_order_and_is_flagged(prepared, tmp_path):
    p = next(x for x in prepared if x.case["kind"] == "two_causes")
    row = evaluate_case(
        p, agent=MuseAgent(ledger_path=tmp_path / "l.json"), client=OfflineMuseClient(finish="length", raw="")
    )
    assert row["muse"]["state"] == "invalid_response" and row["muse"]["decision"] is None
    assert row["policies"]["muse"]["executed"][0] == TOOLS[0] and row["muse"]["used_fallback"]


def test_no_intervention_false_alarm_uses_the_same_fallback(prepared, tmp_path):
    p = next(x for x in prepared if x.case["kind"] == "no_intervention")
    row = evaluate_case(
        p,
        agent=MuseAgent(ledger_path=tmp_path / "l.json"),
        client=OfflineMuseClient(raw=reply("probe", ["probe_sky"])),
    )
    trace = row["policies"]["muse"]
    assert trace["executed"] == list(TOOLS) and trace["false_alarm"] and not trace["success"]
    assert row["muse"]["used_fallback"]


def test_baselines_see_identical_evidence_and_abstain_only_when_nothing_changed(prepared, tmp_path):
    rows, _ = run(prepared, OfflineMuseClient(), tmp_path)
    for row, source in zip(rows, prepared, strict=True):
        assert row["oracle"] == source.oracle
        for name in ("heuristic", "elimination"):
            if row["kind"] == "no_intervention":
                assert row["policies"][name]["probes"] == 0 and row["policies"][name]["success"]
        assert row["policies"]["random"]["probes"] >= 1 or row["kind"] == "no_intervention"


def test_baselines_abstain_only_without_a_recovery_anomaly():
    quiet = {"probe_sky": 0.01, "probe_psf": 0.02, "probe_flux": 0.0}
    assert plan_probes("heuristic", quiet, anomaly=False) == ()
    assert plan_probes("elimination", quiet, anomaly=False) == ()
    assert (
        len(plan_probes("heuristic", quiet, anomaly=True)) == 3
    )  # a drop with unchanged diagnostics is probed
    assert len(plan_probes("random", quiet, anomaly=False, seed=1)) == 3  # random never abstains
    assert plan_probes("elimination", {**quiet, "probe_flux": None}, anomaly=True)[0] == "probe_flux"


def test_recovery_anomaly_needs_statistical_support():
    def obs(a, b):
        counts = {"nominal": {"A": a, "B": b}}
        return {"pilot_measurements": {"sources_per_batch": 25, "recovery_counts": counts}}

    assert recovery_anomaly(obs([18, 17, 19, 16], [7, 6, 8, 7]))
    assert not recovery_anomaly(obs([18, 17, 19, 16], [17, 18, 16, 19]))
    assert not recovery_anomaly(obs([17, 17, 17, 17], [16, 16, 16, 16]))  # a small gap is not an anomaly
    with pytest.raises(ValueError):
        recovery_anomaly(obs([1, 2], [1]))


def test_run_is_resumable_and_makes_no_repeat_calls(prepared, tmp_path, raw):
    cases = [p.case for p in prepared[:3]]
    client = OfflineMuseClient(raw=reply("probe", ["probe_psf"]))
    agent = MuseAgent(ledger_path=tmp_path / "l.json")
    first = run_benchmark(cases, lambda c: raw, agent=agent, client=client)
    again = run_benchmark(cases, lambda c: raw, agent=agent, client=client)
    assert client.calls == 3 and again["cases"] == first["cases"]


def test_unresolved_reservation_stops_the_run(prepared, tmp_path):
    agent = MuseAgent(ledger_path=tmp_path / "l.json")
    with pytest.raises(ConnectionError):
        agent.decide(prepared[1].case["id"], prepared[1].observation, client=OfflineMuseClient(fails=True))
    with pytest.raises(RuntimeError, match="Unresolved"):
        evaluate_case(prepared[1], agent=agent, client=OfflineMuseClient())


def test_dry_run_and_live_ledgers_cannot_be_mixed(prepared, tmp_path):
    agent = MuseAgent(ledger_path=tmp_path / "l.json")
    evaluate_case(prepared[1], agent=agent, client=OfflineMuseClient())
    from test_agent import FakePaidClient

    with pytest.raises(RuntimeError, match="mismatch"):
        evaluate_case(prepared[1], agent=agent, client=FakePaidClient(), allow_paid=True)


def test_summary_counts_and_wilson_intervals(prepared, tmp_path):
    rows, _ = run(prepared, OfflineMuseClient(raw=reply("abstain", [])), tmp_path)
    s = summarise(rows)
    muse = s["policies"]["muse"]
    assert muse["intervention_cases"]["k"] == 0 and muse["intervention_cases"]["n"] == 3
    assert muse["correct_abstentions"]["k"] == 1 and muse["mean_probes"] == 0
    assert s["muse_replies"] == {
        "valid": 4,
        "invalid": 0,
        "used_fallback": 0,
        "cost_usd": 0.0,
        "closed_without_fallback": {"k": 0, "n": 3, "rate": 0.0, "ci95": [0.0, 0.561]},
    }
    assert "muse" in format_summary(s)


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    one = paired_bootstrap_ci([1, 0, -1, 1, 0, 1], seed=4)
    assert one == paired_bootstrap_ci([1, 0, -1, 1, 0, 1], seed=4)
    assert one["ci95"][0] <= one["mean"] <= one["ci95"][1]
    assert paired_bootstrap_ci([2, 2, 2])["ci95"] == [2.0, 2.0]
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])


def test_freeze_roundtrip_and_detects_changes(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    record = freeze.write_freeze(path)
    assert freeze.verify_freeze(path) == record == freeze.freeze_record()
    monkeypatch.setattr(freeze, "SYSTEM_PROMPT", "a different prompt")
    with pytest.raises(RuntimeError, match="request_contract_sha256"):
        freeze.verify_freeze(path)
    with pytest.raises(RuntimeError, match="not found"):
        freeze.verify_freeze(tmp_path / "missing.json")
