"""Point-source injection and matched-filter detection tests."""

import numpy as np
import pytest

from blindspot.config import DETECTOR
from blindspot.detection.detector import PSF_ENERGY, detect_sources, prepare_patch
from blindspot.injection.psf import PSF, PSF_RADIUS, SIZE, gaussian_psf, inject_sources, sample_positions
from blindspot.metrics.completeness import count_new_matched

SIGMA = 7.0


def _white_patch(rng):
    return rng.normal(1000.0, SIGMA, (SIZE, SIZE)), np.full((SIZE, SIZE), 1000.0)


def _recovery(snr: float, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    flux = snr * SIGMA / np.sqrt(PSF_ENERGY)
    hits = total = 0
    for _ in range(trials):
        image, background = _white_patch(rng)
        positions = sample_positions(int(rng.integers(1 << 30)), holdout=False)
        injected = inject_sources(image, positions, flux)
        hits += count_new_matched(
            positions,
            detect_sources(injected, background, SIGMA),
            detect_sources(image, background, SIGMA),
        )
        total += len(positions)
    return hits / total


def test_psf_is_normalised_and_symmetric():
    assert PSF.shape == (2 * PSF_RADIUS + 1,) * 2
    assert PSF.sum() == pytest.approx(1.0)
    assert np.allclose(PSF, PSF.T) and np.allclose(PSF, PSF[::-1, ::-1])


def test_invalid_psf_arguments():
    with pytest.raises(ValueError):
        gaussian_psf(sigma=0)
    with pytest.raises(ValueError):
        gaussian_psf(radius=True)


def test_injection_conserves_flux_and_leaves_input_untouched():
    base = np.zeros((SIZE, SIZE))
    positions = np.array([[30, 30], [90, 70]])
    out = inject_sources(base, positions, 100.0)
    assert out.sum() == pytest.approx(200.0)
    assert base.sum() == 0.0


def test_injection_rejects_border_and_bad_inputs():
    base = np.zeros((SIZE, SIZE))
    with pytest.raises(ValueError):
        inject_sources(base, np.array([[2, 60]]), 1.0)
    with pytest.raises(ValueError):
        inject_sources(base, np.array([[60.0, 60.0]]), 1.0)
    with pytest.raises(ValueError):
        inject_sources(base, np.array([[60, 60]]), -1.0)


@pytest.mark.parametrize("holdout", [False, True])
def test_positions_are_inbounds_and_stamps_never_overlap(holdout):
    for seed in range(200):
        pos = sample_positions(seed, holdout=holdout)
        assert pos.shape == (DETECTOR.sources_per_batch, 2)
        assert pos.min() >= PSF_RADIUS and pos.max() < SIZE - PSF_RADIUS
        spacing = np.abs(pos[:, None, :] - pos[None, :, :]).max(axis=2) + np.eye(len(pos)) * 999
        assert spacing.min() > 2 * PSF_RADIUS


def test_train_and_holdout_positions_are_disjoint():
    train = {tuple(p) for s in range(50) for p in sample_positions(s, holdout=False)}
    holdout = {tuple(p) for s in range(50) for p in sample_positions(s, holdout=True)}
    assert not train & holdout


def test_positions_are_reproducible():
    assert np.array_equal(sample_positions(7, holdout=False), sample_positions(7, holdout=False))
    assert not np.array_equal(sample_positions(7, holdout=False), sample_positions(8, holdout=False))


def test_detector_has_no_false_positives_on_pure_white_noise():
    rng = np.random.default_rng(0)
    found = sum(
        len(detect_sources(*_white_patch(rng)[:1], np.full((SIZE, SIZE), 1000.0), SIGMA)) for _ in range(40)
    )
    assert found == 0


def test_detector_calibration_on_white_noise():
    """Matched-filter 5-sigma threshold: ~54 % at SNR 5, fully recovered at SNR 8, almost none at SNR 3."""
    assert _recovery(3.0, 25, seed=1) < 0.15
    assert 0.40 < _recovery(5.0, 25, seed=2) < 0.68
    assert _recovery(8.0, 25, seed=3) > 0.98


def test_zero_flux_injection_recovers_nothing():
    rng = np.random.default_rng(5)
    image, background = _white_patch(rng)
    positions = sample_positions(1, holdout=False)
    injected = inject_sources(image, positions, 0.0)
    baseline = detect_sources(image, background, SIGMA)
    assert count_new_matched(positions, detect_sources(injected, background, SIGMA), baseline) == 0


def test_prepare_patch_rejects_unusable_patches():
    with pytest.raises(ValueError):
        prepare_patch(np.zeros((SIZE, SIZE)))
    bad = np.random.default_rng(0).normal(size=(SIZE, SIZE))
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        prepare_patch(bad)
    with pytest.raises(ValueError):
        prepare_patch(np.ones((64, 64)))
