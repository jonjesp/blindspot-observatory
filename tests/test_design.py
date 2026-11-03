"""Benchmark geometry and deterministic case-plan tests."""

from collections import Counter
from itertools import combinations

from blindspot.benchmark.design import PATCH_SIZE, plan, plan_hash
from blindspot.config import EXPERIMENT

# Pinned geometry. If the configuration changes on purpose, update this value in the same commit.
PLAN_SHA256 = "96fadc5a7c7f92016810dcf5c6155e9de4855d02b8e625493c5455a4b09ba7ff"


def test_plan_size_and_class_balance_per_exposure():
    cases = plan()["cases"]
    assert len(cases) == 36 and len({c["id"] for c in cases}) == 36
    expected = dict(EXPERIMENT.benchmark.class_counts)
    for exposure in EXPERIMENT.exposures:
        assert Counter(c["kind"] for c in cases if c["field"] == exposure) == expected


def test_sites_are_disjoint_from_each_other_and_from_earlier_experiments():
    cases = plan()["cases"]
    for exposure in EXPERIMENT.exposures:
        sites = [(c["offset_y"], c["offset_x"]) for c in cases if c["field"] == exposure]
        for a, b in list(combinations(sites, 2)) + [
            (s, e) for s in sites for e in EXPERIMENT.benchmark.excluded_offsets
        ]:
            assert abs(a[0] - b[0]) >= PATCH_SIZE or abs(a[1] - b[1]) >= PATCH_SIZE


def test_plan_is_deterministic_and_pinned():
    assert plan() == plan()
    assert plan_hash() == PLAN_SHA256


def test_public_fields_do_not_expose_physical_interventions():
    assert set(plan()["cases"][0]) == {"id", "field", "offset_y", "offset_x", "kind"}