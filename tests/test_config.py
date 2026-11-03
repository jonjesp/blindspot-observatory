"""Detector and experiment configuration validation tests."""

import pytest

from blindspot import config


def test_configs_load_and_agree():
    assert config.DETECTOR.sources_per_batch == len(config.DETECTOR.train_grid) ** 2
    assert config.EXPERIMENT.benchmark.cases_per_exposure == 18
    assert sum(n for _, n in config.EXPERIMENT.benchmark.class_counts) == 18
    assert config.EXPERIMENT.agent.model == "muse-spark-1.3"


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.setattr(config, "_read_yaml", lambda name: {"patch_size_px": 128})
    with pytest.raises(ValueError, match="missing key"):
        config.load_detector_config()


def test_overlapping_grid_is_rejected(monkeypatch):
    real = config._read_yaml("detector.yaml")
    real["train_grid_px"] = [20, 25, 60, 80, 100]
    monkeypatch.setattr(config, "_read_yaml", lambda name: real)
    with pytest.raises(ValueError, match="overlapping"):
        config.load_detector_config()


def test_bad_type_is_rejected(monkeypatch):
    real = config._read_yaml("detector.yaml")
    real["psf_sigma_px"] = "wide"
    monkeypatch.setattr(config, "_read_yaml", lambda name: real)
    with pytest.raises(ValueError, match="psf_sigma_px"):
        config.load_detector_config()


@pytest.mark.parametrize(
    "key,value",
    [
        ("background_median_size_px", -1),
        ("peak_filter_size_px", 0),
        ("position_jitter_px", -2),
        ("patch_size_px", 0),
    ],
)
def test_invalid_detector_geometry_is_rejected(monkeypatch, key, value):
    raw = config._read_yaml("detector.yaml")
    raw[key] = value
    monkeypatch.setattr(config, "_read_yaml", lambda name: raw)
    with pytest.raises(ValueError):
        config.load_detector_config()


@pytest.mark.parametrize("coordinate", [1.5, True, "2"])
def test_excluded_offsets_require_integer_coordinates(monkeypatch, coordinate):
    raw = config._read_yaml("experiment.yaml")
    raw["benchmark"]["excluded_offsets_px"] = [[coordinate, 0]]
    monkeypatch.setattr(config, "_read_yaml", lambda name: raw)
    with pytest.raises(ValueError, match="integer"):
        config.load_experiment_config()
