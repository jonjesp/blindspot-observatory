"""Diagnostic-probe baseline ordering tests."""

import pytest

from blindspot.agent.schemas import TOOLS
from blindspot.benchmark.baselines import METHODS, policy_order

CLEAR = {"probe_sky": 0.9, "probe_psf": 0.05, "probe_flux": 0.3}
MASKED = {"probe_sky": 0.02, "probe_psf": 0.03, "probe_flux": None}


def test_every_policy_returns_a_permutation():
    for method in METHODS:
        for features in (CLEAR, MASKED):
            assert sorted(policy_order(method, features, seed=3)) == sorted(TOOLS)


def test_heuristic_follows_largest_change():
    assert policy_order("heuristic", CLEAR) == ("probe_sky", "probe_flux", "probe_psf")


def test_elimination_is_not_the_reverse_of_the_heuristic():
    for features in (CLEAR, {"probe_sky": 0.5, "probe_psf": 0.4, "probe_flux": 0.1}):
        assert policy_order("elimination", features) != tuple(reversed(policy_order("heuristic", features)))


def test_masked_diagnostic_is_probed_first_when_everything_else_is_unchanged():
    assert policy_order("elimination", MASKED)[0] == "probe_flux"
    assert policy_order("heuristic", MASKED)[-1] == "probe_flux"  # the heuristic cannot reason about it


def test_changed_probes_still_precede_unobserved_ones():
    features = {"probe_sky": 0.8, "probe_psf": 0.01, "probe_flux": None}
    assert policy_order("elimination", features) == ("probe_sky", "probe_flux", "probe_psf")


def test_random_is_seeded():
    assert policy_order("random", CLEAR, seed=1) == policy_order("random", CLEAR, seed=1)
    assert len({policy_order("random", CLEAR, seed=s) for s in range(30)}) > 1


@pytest.mark.parametrize(
    "bad",
    [
        {"probe_sky": 1.0},
        {**CLEAR, "probe_psf": -0.1},
        {**CLEAR, "probe_psf": float("nan")},
        {**CLEAR, "probe_psf": True},
        {**CLEAR, "probe_other": 1.0},
    ],
)
def test_invalid_features_are_rejected(bad):
    with pytest.raises(ValueError):
        policy_order("heuristic", bad)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        policy_order("oracle", CLEAR)