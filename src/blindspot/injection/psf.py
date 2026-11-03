"""Gaussian PSF and artificial source insertion."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from blindspot.config import DETECTOR

SIZE = DETECTOR.patch_size
PSF_RADIUS = DETECTOR.psf_radius
PSF_SIGMA = DETECTOR.psf_sigma


def gaussian_psf(sigma: float = PSF_SIGMA, radius: int = PSF_RADIUS) -> np.ndarray:
    """Return a normalised (sum == 1) circular Gaussian kernel of side ``2 * radius + 1``."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("PSF sigma must be positive and finite")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
        raise ValueError("PSF radius must be a positive integer")
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return kernel / kernel.sum()


PSF = gaussian_psf()


def inject_sources(base: np.ndarray, positions: np.ndarray, flux: float, psf: np.ndarray = PSF) -> np.ndarray:
    """Add one scaled PSF stamp per integer ``(y, x)`` position to a copy of ``base``."""
    if base.shape != (SIZE, SIZE) or not np.all(np.isfinite(base)):
        raise ValueError("Invalid base image: expected a finite square patch")
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.dtype.kind not in "iu":
        raise ValueError("Positions must be an integer array of shape (n, 2)")
    if not np.isfinite(flux) or flux < 0:
        raise ValueError("Flux must be finite and non-negative")
    if (
        psf.shape != PSF.shape
        or not np.all(np.isfinite(psf))
        or np.any(psf < 0)
        or not np.isclose(psf.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid injection PSF: needs the standard shape, non-negative, unit sum")
    image = base.copy()
    for y, x in positions:
        y, x = int(y), int(x)
        if not (PSF_RADIUS <= y < SIZE - PSF_RADIUS and PSF_RADIUS <= x < SIZE - PSF_RADIUS):
            raise ValueError("Source stamp would cross the patch border")
        image[y - PSF_RADIUS : y + PSF_RADIUS + 1, x - PSF_RADIUS : x + PSF_RADIUS + 1] += flux * psf
    return image


def sample_positions(seed: int, *, holdout: bool) -> np.ndarray:
    """Grid of ``sources_per_batch`` positions with seeded integer jitter.

    Training and holdout use disjoint centre grids. PSF footprints can overlap and all batches
    reuse the same underlying image; this is not an independent-pixel or independent-image split.
    """
    grid = np.array(DETECTOR.holdout_grid if holdout else DETECTOR.train_grid)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    jitter = np.random.default_rng(seed).integers(
        -DETECTOR.position_jitter, DETECTOR.position_jitter + 1, size=(DETECTOR.sources_per_batch, 2)
    )
    return np.column_stack((yy.ravel(), xx.ravel())) + jitter
