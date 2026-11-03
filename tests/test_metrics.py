"""Source-matching and completeness-statistics tests."""

import numpy as np
import pytest

from blindspot.metrics.completeness import count_new_matched, gain, gain_lower_bound, wilson


def test_wilson_matches_known_values():
    lo, hi = wilson(27, 100)
    assert lo == pytest.approx(0.1927, abs=5e-4) and hi == pytest.approx(0.3643, abs=5e-4)
    assert wilson(0, 100)[0] == 0.0 and wilson(100, 100)[1] == 1.0
    assert wilson(0, 100)[1] == pytest.approx(0.0370, abs=5e-4)


def test_numpy_integers_are_accepted_but_not_floats_or_bools():
    assert wilson(np.int64(30), 100) == wilson(30, 100)
    assert gain(np.int64(3), np.int64(9), 100) == pytest.approx(0.06)
    for bad in (3.0, True):
        with pytest.raises(ValueError):
            wilson(bad, 100)


def test_invalid_counts_are_rejected():
    for args in ((-1, 10), (11, 10), (1, 0)):
        with pytest.raises(ValueError):
            wilson(*args)
    with pytest.raises(ValueError):
        gain(5, 101, 100)


def test_gain_lower_bound_is_conservative():
    assert gain_lower_bound(5, 27, 100) > 0
    assert gain_lower_bound(39, 52, 100) < 0 < gain(39, 52, 100)


def test_matching_edge_cases():
    pos = np.array([[20, 20], [40, 40]])
    none = np.empty((0, 2), dtype=int)
    assert count_new_matched(pos, none, none) == 0
    assert count_new_matched(none, pos, none) == 0
    assert count_new_matched(pos, np.array([[20, 21]]), none) == 1
    assert count_new_matched(pos, np.array([[20, 21]]), np.array([[20, 20]])) == 0  # already in baseline
    assert count_new_matched(np.array([[20, 20], [20, 22]]), np.array([[20, 21]]), none) == 1  # one-to-one
    with pytest.raises(ValueError):
        count_new_matched(pos, np.array([1, 2]), none)