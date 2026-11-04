"""Render FITS cutouts and trace the detector matches used in holdout recovery."""

from __future__ import annotations

import base64

import numpy as np
from scipy.spatial import cKDTree

from blindspot.benchmark.generator import case_seed
from blindspot.benchmark.interventions import restore
from blindspot.benchmark.measurement import Arm, _inject, make_arm
from blindspot.benchmark.observations import HOLDOUT_BATCHES, LEVEL_SNR, _seeds
from blindspot.benchmark.runner import PreparedCase
from blindspot.config import DETECTOR
from blindspot.detection.detector import detect_sources
from blindspot.injection.psf import sample_positions
from blindspot.metrics.completeness import count_new_matched


def display_pixels(image: np.ndarray) -> dict:
    """Encode grayscale pixels stretched between the 1st and 99.5th percentiles."""
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("The cutout contains no finite pixel values")
    low, high = np.percentile(finite, [1, 99.5])
    span = high - low if high > low else 1.0
    intensity = np.clip((image.astype(np.float64) - low) / span, 0, 1)
    pixels = np.rint(np.where(np.isfinite(intensity), intensity, 0) * 255).astype(np.uint8)
    height, width = image.shape
    return {"width": width, "height": height, "gray8": base64.b64encode(pixels).decode("ascii")}


def recovery_matches(
    positions: np.ndarray, detections: np.ndarray, baseline: np.ndarray
) -> list[tuple[int, int]]:
    """Trace the frozen greedy matcher, with indices into the original coordinate arrays."""
    if not len(detections) or not len(positions):
        return []
    radius = DETECTOR.match_radius
    candidates = np.arange(len(detections))
    if len(baseline):
        separation, _ = cKDTree(baseline).query(detections, distance_upper_bound=radius)
        candidates = candidates[~np.isfinite(separation)]
    if not len(candidates):
        return []

    links = cKDTree(detections[candidates]).query_ball_point(positions, radius)
    # Original indices preserve the frozen matcher's tie ordering after filtering.
    pairs = sorted(
        (float(np.linalg.norm(positions[i] - detections[j])), i, j)
        for i, neighbours in enumerate(links)
        for j in candidates[neighbours].tolist()
    )
    matches: dict[int, int] = {}
    used_detections: set[int] = set()
    for _, source, detection in pairs:
        if source not in matches and detection not in used_detections:
            matches[source] = detection
            used_detections.add(detection)
    return list(matches.items())


def _holdout_batch(arm: Arm, seed: int) -> dict:
    positions = sample_positions(seed, holdout=True)
    injected = _inject(arm, LEVEL_SNR["nominal"], positions)
    baseline = arm.patch["baseline"]
    detections = detect_sources(injected, arm.patch["background"], arm.patch["sigma"])
    matches = recovery_matches(positions, detections, baseline)
    if len(matches) != count_new_matched(positions, detections, baseline):
        raise ValueError("Holdout markers disagree with the detector match count")
    return {
        **display_pixels(injected),
        "injected": positions.tolist(),
        "detections": detections.tolist(),
        "matches": matches,
    }


def holdout_views(raw: np.ndarray, prepared: PreparedCase, probes: list[str]) -> dict:
    """Generate one image and its detector coordinates for each inspected holdout batch."""
    if not probes:
        return {}
    seed = case_seed(prepared.case)
    noise_seed = _seeds(seed, "noise", 1)[0]
    condition = prepared.truth.condition
    before_arm = make_arm(raw, condition, noise_seed=noise_seed)
    before = [_holdout_batch(before_arm, batch) for batch in _seeds(seed, "holdout_before", HOLDOUT_BATCHES)]
    after_seeds = _seeds(seed, "holdout_after", HOLDOUT_BATCHES)
    after_by_probe = {}
    for probe in probes:
        arm = make_arm(raw, restore(condition, probe), noise_seed=noise_seed)
        after = [_holdout_batch(arm, batch) for batch in after_seeds]
        reference = prepared.oracle[probe]
        for name, batches in (("before", before), ("after", after)):
            if sum(len(batch["matches"]) for batch in batches) != reference[name]:
                raise ValueError(f"{probe}: {name} markers disagree with the verified holdout total")
            if sum(len(batch["injected"]) for batch in batches) != reference["n"]:
                raise ValueError(f"{probe}: {name} markers disagree with the verified injection total")
        after_by_probe[probe] = after
    return {"before": before, "after_by_probe": after_by_probe}
