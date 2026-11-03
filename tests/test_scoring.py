"""Probe-trace closure and abstention scoring tests."""

import json
from pathlib import Path

import pytest

from blindspot.benchmark.scoring import closure_accepted, score_trace

EVIDENCE = Path(__file__).parents[1] / "evidence" / "E19_audit.json"


def p(before, after, n=100):
    return {"before": before, "after": after, "n": n}


def test_supported_gain_closes_the_case():
    assert score_trace([p(5, 27)], no_intervention=False) == {
        "success": True,
        "probes": 1,
        "gain": 0.22,
        "censored": False,
        "unsupported_gains": 0,
    }


def test_unsupported_gain_is_not_a_closure():
    result = score_trace([p(39, 52)], no_intervention=False)
    assert result["success"] is False and result["unsupported_gains"] == 1 and result["probes"] == 1


def test_failure_costs_every_probe_run():
    result = score_trace([p(2, 4), p(2, 5), p(2, 6)], no_intervention=False)
    assert result["success"] is False and result["probes"] == 3 and result["censored"]


def test_probes_after_closure_are_invalid():
    with pytest.raises(ValueError, match="after the case was closed"):
        score_trace([p(3, 41), p(3, 50)], no_intervention=False)


def test_trace_longer_than_protocol_is_invalid():
    with pytest.raises(ValueError):
        score_trace([p(0, 0)] * 4, no_intervention=False)


def test_malformed_measurement_is_rejected_even_after_closure():
    with pytest.raises(ValueError):
        score_trace([p(3, 41), {"before": 1, "after": 500, "n": 100}], no_intervention=False)


def test_abstention_scoring():
    assert score_trace([], no_intervention=True)["correct_abstention"] is True
    assert score_trace([p(0, 0)], no_intervention=True)["false_alarm"] is True
    assert score_trace([], no_intervention=False) == {
        "success": False,
        "probes": 0,
        "censored": True,
        "unsupported_gains": 0,
    }


def test_closure_rule_agrees_with_the_frozen_e19_audit():
    """The scoring rule reproduces the accept/reject outcome of every holdout recorded in E19."""
    cases = json.loads(EVIDENCE.read_text(encoding="utf-8"))["cases"]
    holdouts = [c["hybrid_holdout"] for c in cases if c.get("hybrid_holdout")]
    assert len(holdouts) >= 5
    for h in holdouts:
        assert closure_accepted(h["before"], h["after"], h["n"]) == h["acceptable"]