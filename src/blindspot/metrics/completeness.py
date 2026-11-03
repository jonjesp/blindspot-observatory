"""One-to-one source matching and binomial statistics."""

from __future__ import annotations

import math
from numbers import Integral
from statistics import NormalDist

import numpy as np
from scipy.spatial import cKDTree

from blindspot.config import DETECTOR


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer count")
    return int(value)


def count_new_matched(
    positions: np.ndarray,
    detections: np.ndarray,
    baseline: np.ndarray,
    match_radius: float = DETECTOR.match_radius,
) -> int:
    """Count injected sources matched one-to-one to a detection that was not already in the base image.

    Detections within ``match_radius`` of a baseline detection are discarded first, so a real
    object cannot be credited as a recovered injection. Matching is greedy by distance.
    """
    if any(x.ndim != 2 or x.shape[1] != 2 for x in (positions, detections, baseline)):
        raise ValueError("Coordinates must have shape (n, 2)")
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError("Match radius must be positive")
    if not len(detections) or not len(positions):
        return 0
    if len(baseline):
        separation, _ = cKDTree(baseline).query(detections, distance_upper_bound=match_radius)
        detections = detections[~np.isfinite(separation)]
    if not len(detections):
        return 0
    links = cKDTree(detections).query_ball_point(positions, match_radius)
    pairs = sorted(
        (float(np.linalg.norm(positions[i] - detections[j])), i, j)
        for i, neighbours in enumerate(links)
        for j in neighbours
    )
    used_sources: set[int] = set()
    used_detections: set[int] = set()
    for _, i, j in pairs:
        if i not in used_sources and j not in used_detections:
            used_sources.add(i)
            used_detections.add(j)
    return len(used_sources)


def wilson(k: int, n: int, coverage: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    k, n = _count(k, "k"), _count(n, "n")
    if n < 1 or not 0 <= k <= n:
        raise ValueError("Binomial counts out of range")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be in (0, 1)")
    z = NormalDist().inv_cdf((1 + coverage) / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def gain(before: int, after: int, n: int) -> float:
    """Recovery gain (after - before) / n."""
    before, after, n = _count(before, "before"), _count(after, "after"), _count(n, "n")
    if n <= 0 or not (0 <= before <= n and 0 <= after <= n):
        raise ValueError("Counts out of range")
    return (after - before) / n


def gain_lower_bound(before: int, after: int, n: int, coverage: float = 0.95) -> float:
    """Conservative lower bound on the gain: Wilson lower(after) - Wilson upper(before).

    This is a non-overlap rule for two marginal binomial intervals, not a calibrated interval
    for correlated, repeated-image measurements. It ignores pairing; its coverage must not be
    claimed as guaranteed when positions or image noise are reused.
    """
    gain(before, after, n)  # validates the counts
    return wilson(after, n, coverage)[0] - wilson(before, n, coverage)[1]
