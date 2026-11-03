"""Simplified matched-filter detection. This is NOT the ZTF production pipeline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, maximum_filter, median_filter

from blindspot.config import DETECTOR
from blindspot.injection.psf import PSF, PSF_RADIUS, SIZE

PSF_ENERGY = float(np.sum(PSF * PSF))


def detection_threshold(noise: float) -> float:
    """Matched-filter response threshold, assuming white noise of standard deviation ``noise``."""
    return DETECTOR.detection_snr * noise * np.sqrt(PSF_ENERGY)


def detect_sources(image: np.ndarray, background: np.ndarray, noise: float) -> np.ndarray:
    """Return ``(y, x)`` positions of local response maxima above the threshold."""
    if image.shape != (SIZE, SIZE) or background.shape != image.shape:
        raise ValueError("Wrong image or background shape")
    if (
        not np.isfinite(noise)
        or noise <= 0
        or not np.all(np.isfinite(image))
        or not np.all(np.isfinite(background))
    ):
        raise ValueError("Invalid image, background or noise level")
    response = convolve(image - background, PSF, mode="constant", cval=0.0)
    peaks = maximum_filter(response, size=DETECTOR.peak_filter_size, mode="constant", cval=-np.inf)
    chosen = (response >= detection_threshold(noise)) & (response == peaks)
    edge = PSF_RADIUS + 3
    chosen[:edge, :] = chosen[-edge:, :] = False
    chosen[:, :edge] = chosen[:, -edge:] = False
    return np.argwhere(chosen)


def prepare_patch(raw: np.ndarray) -> dict:
    """Estimate background and noise for a patch and detect the sources already present.

    The background is a median filter of the raw patch; noise is the robust (MAD) standard
    deviation of the residual. The background is NOT re-estimated after injection.
    """
    if raw.shape != (SIZE, SIZE) or not np.all(np.isfinite(raw)):
        raise ValueError("Invalid FITS patch: wrong size or non-finite values")
    if np.max(raw) == np.min(raw):
        raise ValueError("FITS patch has no dynamic range")
    background = median_filter(raw, size=DETECTOR.background_median_size, mode="reflect")
    residual = raw - background
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Local noise could not be measured")
    base = np.array(raw, dtype=np.float64, copy=True)
    return {
        "base": base,
        "background": background,
        "sigma": noise,
        "baseline": detect_sources(base, background, noise),
    }
