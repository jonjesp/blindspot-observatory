"""Deterministic measurements on one arm (a patch under one condition)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blindspot.benchmark.interventions import Condition
from blindspot.config import DETECTOR
from blindspot.detection.detector import PSF_ENERGY, detect_sources, prepare_patch
from blindspot.injection.psf import PSF_RADIUS, PSF_SIGMA, gaussian_psf, inject_sources, sample_positions
from blindspot.metrics.completeness import count_new_matched


@dataclass(frozen=True)
class Arm:
    patch: dict  # output of prepare_patch for this condition
    condition: Condition
    sigma_ref: float  # noise of the reference condition A; injected fluxes are defined against it
    psf: np.ndarray  # PSF used for injection (the detector always keeps the nominal one)


def make_arm(raw: np.ndarray, condition: Condition, *, noise_seed: int) -> Arm:
    """Build the patch under ``condition``; extra noise is added BEFORE noise and background are measured."""
    reference = prepare_patch(raw)
    patch = reference
    if condition.noise_multiplier > 0:
        extra = np.random.default_rng(noise_seed).normal(
            0.0, condition.noise_multiplier * reference["sigma"], raw.shape
        )
        patch = prepare_patch(raw + extra)
    psf = gaussian_psf(PSF_SIGMA * condition.psf_scale)
    return Arm(patch=patch, condition=condition, sigma_ref=reference["sigma"], psf=psf)


def _inject(arm: Arm, snr: float, positions: np.ndarray) -> np.ndarray:
    flux = snr * arm.sigma_ref / np.sqrt(PSF_ENERGY) * arm.condition.flux_scale
    return inject_sources(arm.patch["base"], positions, flux, arm.psf)


def count_recovered(arm: Arm, snr: float, seed: int, *, holdout: bool) -> int:
    """Recovered sources out of ``sources_per_batch`` for one seeded batch at nominal ``snr``."""
    positions = sample_positions(seed, holdout=holdout)
    injected = _inject(arm, snr, positions)
    detections = detect_sources(injected, arm.patch["background"], arm.patch["sigma"])
    return count_new_matched(positions, detections, arm.patch["baseline"])


def profile_diagnostics(arm: Arm, snr: float, seed: int) -> tuple[float, float]:
    """Integrated amplitude and RMS radius of the batch-median source stamp.

    Measured on the injected image after background subtraction, so noise limits the precision
    exactly as it would for a real measurement; the injection parameters are not read. The median
    (not the mean) keeps a few sources sitting on bright real objects from dominating.
    """
    positions = sample_positions(seed, holdout=False)
    residual = _inject(arm, snr, positions) - arm.patch["background"]
    r = PSF_RADIUS
    stamp = np.median([residual[y - r : y + r + 1, x - r : x + r + 1] for y, x in positions], axis=0)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    weights = np.clip(stamp, 0.0, None)
    if weights.sum() <= 0 or stamp.sum() <= 0:
        raise ValueError("No measurable source profile in this arm")
    rms_radius = float(np.sqrt((weights * (xx * xx + yy * yy)).sum() / weights.sum()))
    return float(stamp.sum()), rms_radius


BATCH_SIZE = DETECTOR.sources_per_batch
