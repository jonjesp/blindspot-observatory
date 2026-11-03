"""Decision and public-observation contract validation tests."""

import json

import pytest

from blindspot.agent.observation import validate_observation
from blindspot.agent.schemas import TOOLS, validate_decision

TEXT = "Noise rose while the flux stayed stable."


def raw(decision="probe", probes=("probe_sky",), text=TEXT, **extra):
    return json.dumps({"decision": decision, "probes": list(probes), "justification": text, **extra})


def test_valid_decisions():
    assert validate_decision(raw())["probes"] == ["probe_sky"]
    assert validate_decision(raw("ambiguous", ("probe_sky", "probe_psf")))["decision"] == "ambiguous"
    assert validate_decision(raw("abstain", ()))["probes"] == []


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        raw("probe", ()),
        raw("probe", ("probe_sky", "probe_psf")),
        raw("ambiguous", ("probe_sky", "probe_sky")),
        raw("abstain", ("probe_sky",)),
        raw("probe", ("probe_other",)),
        raw("guess"),
        raw(text="short"),
        raw(text="x" * 351),
        raw(extra_key=1),
        json.dumps({"decision": "probe", "probes": ["probe_sky"]}),
        json.dumps(["probe"]),
    ],
)
def test_invalid_decisions_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_decision(bad)


def observation(**overrides):
    base = {
        "pilot_measurements": {
            "sources_per_batch": 25,
            "recovery_counts": {"nominal": {"A": [8, 12], "B": [2, 3]}},
        },
        "indicators": {"background_mad_change": 0.9, "rms_radius_change": 0.1},
        "available_probes": list(TOOLS),
    }
    base.update(overrides)
    return base


def test_valid_observation_and_masked_indicator():
    validate_observation(observation())
    validate_observation(observation(indicators={}))  # masked diagnostics are simply omitted


@pytest.mark.parametrize(
    "bad",
    [
        observation(indicators={"flux_factor_applied": 0.5}),  # leaks the intervention
        observation(indicators={"noise_multiplier": 1.5}),
        observation(indicators={"background_mad_change": float("nan")}),
        observation(indicators={"background_mad_change": True}),
        observation(pilot_measurements={"true_cause": 1}),
        observation(pilot_measurements={"recovery_counts": {"holdout": {"A": [1]}}}),
        observation(pilot_measurements={"recovery_counts": {"nominal": {"C": [1]}}}),
        observation(pilot_measurements={"recovery_counts": {"nominal": {"A": [-1]}}}),
        observation(pilot_measurements={"recovery_counts": {"nominal": {"A": [1.5]}}}),
        observation(available_probes=["probe_sky"]),
        {**observation(), "extra": 1},
        "text",
    ],
)
def test_unlisted_or_malformed_observation_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_observation(bad)