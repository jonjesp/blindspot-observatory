"""Deterministic intervention and observation generation tests."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_fits

from blindspot.agent.observation import INDICATOR_KEYS, validate_observation
from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.design import plan
from blindspot.benchmark.generator import MASK_PROBABILITY, case_seed, private_case
from blindspot.benchmark.interventions import MINOR, NOMINAL, STRONG, apply_causes, restore
from blindspot.benchmark.observations import (
    PROBE_INDICATOR,
    build_observation,
    features_from_observation,
    holdout_measurement,
    oracle_measurements,
)
from blindspot.benchmark.scoring import closure_accepted
from blindspot.config import DETECTOR
from blindspot.metrics.completeness import gain

SEED = 424242


@pytest.fixture(scope="module")
def raw():
    return np.random.default_rng(11).normal(1000.0, 7.0, (DETECTOR.patch_size, DETECTOR.patch_size))


def test_condition_validation_and_restore_touches_only_one_factor():
    both = apply_causes(("probe_sky", "probe_flux"), dominant="probe_sky")
    assert both.noise_multiplier == STRONG["probe_sky"] and both.flux_scale == MINOR["probe_flux"]
    undone = restore(both, "probe_sky")
    assert undone.noise_multiplier == NOMINAL.noise_multiplier and undone.flux_scale == both.flux_scale
    assert apply_causes(()) == NOMINAL
    for bad in (
        lambda: apply_causes(("probe_sky", "probe_psf")),
        lambda: apply_causes(("probe_x",)),
        lambda: apply_causes(("probe_sky", "probe_sky"), dominant="probe_sky"),
        lambda: apply_causes(tuple(TOOLS), dominant="probe_sky"),
        lambda: restore(NOMINAL, "probe_x"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_observation_is_valid_deterministic_and_does_not_name_interventions(raw):
    condition = apply_causes(("probe_sky",))
    one = build_observation(raw, condition, case_seed=SEED)
    assert one == build_observation(raw, condition, case_seed=SEED)
    validate_observation(one)
    text = json.dumps(one)
    for forbidden in ("noise_multiplier", "psf_scale", "flux_scale", "cause", "intervention", "holdout"):
        assert forbidden not in text
    assert len(text) < 1500  # keeps the prompt, and its worst-case cost, small


def test_no_intervention_shows_no_anomaly(raw):
    obs = build_observation(raw, NOMINAL, case_seed=SEED)
    assert max(obs["indicators"].values()) < 0.10
    counts = obs["pilot_measurements"]["recovery_counts"]["nominal"]
    assert abs(sum(counts["A"]) - sum(counts["B"])) < 0.12 * sum(counts["A"])


def test_each_cause_moves_its_own_indicator(raw):
    sky = build_observation(raw, apply_causes(("probe_sky",)), case_seed=SEED)["indicators"]
    flux = build_observation(raw, apply_causes(("probe_flux",)), case_seed=SEED)["indicators"]
    assert sky["background_mad_change"] > 0.2 > flux["background_mad_change"]
    assert flux["integrated_amplitude_change"] > 0.1 > sky["integrated_amplitude_change"]


def test_masking_removes_only_the_requested_indicator(raw):
    obs = build_observation(raw, apply_causes(("probe_flux",)), case_seed=SEED, masked=("probe_flux",))
    assert PROBE_INDICATOR["probe_flux"] not in obs["indicators"] and len(obs["indicators"]) == 2
    features = features_from_observation(obs)
    assert features["probe_flux"] is None and features["probe_sky"] is not None
    assert set(features) == set(TOOLS) and set(PROBE_INDICATOR.values()) == set(INDICATOR_KEYS)
    with pytest.raises(ValueError):
        build_observation(raw, NOMINAL, case_seed=SEED, masked=("probe_x",))


def test_true_probe_closes_and_other_probes_do_not(raw):
    condition = apply_causes(("probe_psf",))
    measures = oracle_measurements(raw, condition, case_seed=SEED)
    assert all(m["n"] == 100 for m in measures.values())
    assert closure_accepted(**measures["probe_psf"])
    assert not closure_accepted(**measures["probe_sky"]) and not closure_accepted(**measures["probe_flux"])
    others = [gain(**measures[p]) for p in ("probe_sky", "probe_flux")]
    assert max(abs(g) for g in others) < 0.15
    assert any(g != 0 for g in others)  # independent positions: sampling noise, never an exact identity


def test_dominant_cause_gains_more_than_the_minor_one(raw):
    condition = apply_causes(("probe_sky", "probe_flux"), dominant="probe_flux")
    dominant = holdout_measurement(raw, condition, "probe_flux", case_seed=SEED)
    minor = holdout_measurement(raw, condition, "probe_sky", case_seed=SEED)
    assert gain(**dominant) > gain(**minor) + 0.1


def test_private_cases_follow_the_plan_and_are_reproducible():
    cases = plan()["cases"]
    expected = {"no_intervention": 0, "single_cause": 1, "two_causes": 2}
    for case in cases:
        truth = private_case(case)
        assert truth == private_case(case)
        assert len(truth.causes) == expected[case["kind"]]
        assert (truth.dominant in truth.causes) == bool(truth.causes)
        assert set(truth.masked) <= set(TOOLS) and len(truth.masked) <= 1
    assert len({case_seed(c) for c in cases}) == len(cases)
    masked = sum(bool(private_case(c).masked) for c in cases if c["kind"] == "single_cause")
    singles = sum(c["kind"] == "single_cause" for c in cases)
    assert 0 < masked < singles and MASK_PROBABILITY == 0.5  # both masked and unmasked cases exist


def test_model_side_code_cannot_import_private_benchmark_modules():
    """Firewall: nothing under blindspot.agent may import the generator, interventions or measurements."""
    agent_dir = Path(__file__).parents[1] / "src" / "blindspot" / "agent"
    for path in agent_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            assert not any(n.startswith("blindspot.benchmark") for n in names), f"{path.name} imports {names}"


@requires_fits
def test_generator_calibration_on_real_images(fits_zip):
    from blindspot.benchmark.calibration import calibrate

    summary = calibrate(fits_zip)["summary"]
    single, two, none = summary["single_cause"], summary["two_causes"], summary["no_intervention"]
    assert single["true_probe_closed"] >= 0.9 * single["cases"]
    assert single["other_probe_false_closures"] <= 1
    assert two["max_gain_probe_is_dominant"] >= 0.85 * two["cases"]
    assert two["dominant_closed"] >= 0.9 * two["cases"]
    assert none["false_closures"] <= 1 and none["max_indicator"] < 0.10