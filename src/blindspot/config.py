"""Single source of truth for detector and experiment settings.

Values live in the packaged YAML files ``configs/detector.yaml`` and
``configs/experiment.yaml``. They are loaded and validated once at import time;
every other module reads them from here instead of repeating constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from numbers import Real
from typing import Any

import yaml

CASE_KINDS = ("no_intervention", "single_cause", "two_causes")


def _read_yaml(name: str) -> dict:
    text = files("blindspot.configs").joinpath(name).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{name}: top level must be a mapping")
    return data


def _get(mapping: dict, key: str, kind: type, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"{where}: missing key '{key}'")
    value = mapping[key]
    if kind is float:
        valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    elif kind is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = isinstance(value, kind)
    if not valid:
        raise ValueError(f"{where}: '{key}' must be of type {kind.__name__}")
    return float(value) if kind is float else value


def _int_list(mapping: dict, key: str, where: str) -> tuple[int, ...]:
    values = _get(mapping, key, list, where)
    if not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(f"{where}: '{key}' must be a non-empty list of integers")
    return tuple(values)


@dataclass(frozen=True)
class DetectorConfig:
    patch_size: int
    psf_sigma: float
    psf_radius: int
    match_radius: float
    detection_snr: float
    background_median_size: int
    peak_filter_size: int
    sources_per_batch: int
    train_grid: tuple[int, ...]
    holdout_grid: tuple[int, ...]
    position_jitter: int


@dataclass(frozen=True)
class Prices:
    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class AgentConfig:
    model: str
    reasoning_effort: str
    max_completion_tokens: int
    max_calls: int
    budget_usd: float
    prices: Prices


@dataclass(frozen=True)
class BenchmarkConfig:
    cases_per_exposure: int
    class_counts: tuple[tuple[str, int], ...]
    gain_threshold: float
    max_probes: int
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    excluded_offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    exposures: tuple[str, ...]
    benchmark: BenchmarkConfig
    change_tolerance: float
    agent: AgentConfig


def load_detector_config() -> DetectorConfig:
    raw, where = _read_yaml("detector.yaml"), "detector.yaml"
    cfg = DetectorConfig(
        patch_size=_get(raw, "patch_size_px", int, where),
        psf_sigma=_get(raw, "psf_sigma_px", float, where),
        psf_radius=_get(raw, "psf_radius_px", int, where),
        match_radius=_get(raw, "match_radius_px", float, where),
        detection_snr=_get(raw, "detection_snr_threshold", float, where),
        background_median_size=_get(raw, "background_median_size_px", int, where),
        peak_filter_size=_get(raw, "peak_filter_size_px", int, where),
        sources_per_batch=_get(raw, "sources_per_batch", int, where),
        train_grid=_int_list(raw, "train_grid_px", where),
        holdout_grid=_int_list(raw, "holdout_grid_px", where),
        position_jitter=_get(raw, "position_jitter_px", int, where),
    )
    if min(cfg.psf_sigma, cfg.match_radius, cfg.detection_snr) <= 0 or cfg.psf_radius < 1:
        raise ValueError(f"{where}: PSF, match radius and SNR threshold must be positive")
    if cfg.patch_size < 1 or cfg.position_jitter < 0:
        raise ValueError(f"{where}: patch size must be positive and position jitter non-negative")
    if any(size < 1 or size % 2 == 0 for size in (cfg.background_median_size, cfg.peak_filter_size)):
        raise ValueError(f"{where}: filter sizes must be positive and odd")
    for name, grid in (("train_grid_px", cfg.train_grid), ("holdout_grid_px", cfg.holdout_grid)):
        if len(grid) ** 2 != cfg.sources_per_batch:
            raise ValueError(f"{where}: {name} must define sources_per_batch grid points")
        if min(grid) - cfg.position_jitter < cfg.psf_radius:
            raise ValueError(f"{where}: {name} places a source stamp outside the patch")
        if max(grid) + cfg.position_jitter >= cfg.patch_size - cfg.psf_radius:
            raise ValueError(f"{where}: {name} places a source stamp outside the patch")
        steps = [b - a for a, b in zip(grid, grid[1:], strict=False)]
        if steps and min(steps) - 2 * cfg.position_jitter < 2 * cfg.psf_radius + 1:
            raise ValueError(f"{where}: {name} allows overlapping source stamps")
    return cfg


def load_experiment_config() -> ExperimentConfig:
    raw, where = _read_yaml("experiment.yaml"), "experiment.yaml"
    bench_raw = _get(raw, "benchmark", dict, where)
    bw = f"{where}:benchmark"
    counts_raw = _get(bench_raw, "class_counts", dict, bw)
    if set(counts_raw) != set(CASE_KINDS) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts_raw.values()
    ):
        raise ValueError(f"{bw}: class_counts must give a non-negative integer for {CASE_KINDS}")
    excluded = _get(bench_raw, "excluded_offsets_px", list, bw)
    if any(
        not isinstance(p, list) or len(p) != 2 or any(type(value) is not int for value in p) for p in excluded
    ):
        raise ValueError(f"{bw}: excluded_offsets_px must be a list of integer [y, x] pairs")
    benchmark = BenchmarkConfig(
        cases_per_exposure=_get(bench_raw, "cases_per_exposure", int, bw),
        class_counts=tuple((kind, counts_raw[kind]) for kind in CASE_KINDS),
        gain_threshold=_get(bench_raw, "gain_threshold", float, bw),
        max_probes=_get(bench_raw, "max_probes", int, bw),
        y_offsets=_int_list(bench_raw, "y_offsets_px", bw),
        x_offsets=_int_list(bench_raw, "x_offsets_px", bw),
        excluded_offsets=tuple((int(p[0]), int(p[1])) for p in excluded),
    )
    if not 0 < benchmark.gain_threshold <= 1 or not 1 <= benchmark.max_probes <= 3:
        raise ValueError(f"{bw}: invalid gain_threshold or max_probes")
    if sum(counts_raw.values()) != benchmark.cases_per_exposure:
        raise ValueError(f"{bw}: class_counts must sum to cases_per_exposure")
    if len(benchmark.y_offsets) * len(benchmark.x_offsets) != benchmark.cases_per_exposure:
        raise ValueError(f"{bw}: offset grid must contain cases_per_exposure sites")

    agent_raw = _get(raw, "agent", dict, where)
    aw = f"{where}:agent"
    price_raw = _get(agent_raw, "prices_usd_per_million_tokens", dict, aw)
    prices = Prices(
        input=_get(price_raw, "input", float, aw),
        cached_input=_get(price_raw, "cached_input", float, aw),
        output=_get(price_raw, "output", float, aw),
    )
    agent = AgentConfig(
        model=_get(agent_raw, "model", str, aw),
        reasoning_effort=_get(agent_raw, "reasoning_effort", str, aw),
        max_completion_tokens=_get(agent_raw, "max_completion_tokens", int, aw),
        max_calls=_get(agent_raw, "max_calls", int, aw),
        budget_usd=_get(agent_raw, "budget_usd", float, aw),
        prices=prices,
    )
    if agent.max_calls < 1 or agent.budget_usd <= 0 or agent.max_completion_tokens < 1:
        raise ValueError(f"{aw}: max_calls, budget_usd and max_completion_tokens must be positive")
    if min(prices.input, prices.cached_input, prices.output) < 0:
        raise ValueError(f"{aw}: prices must be non-negative")

    exposures = _get(raw, "exposures", list, where)
    if not exposures or any(not isinstance(e, str) or not e.strip() for e in exposures):
        raise ValueError(f"{where}: exposures must be a non-empty list of names")
    if len(set(exposures)) != len(exposures):
        raise ValueError(f"{where}: exposure names must be distinct")
    tolerance = _get(_get(raw, "baselines", dict, where), "change_tolerance", float, where)
    if tolerance < 0:
        raise ValueError(f"{where}: change_tolerance must be non-negative")
    return ExperimentConfig(
        seed=_get(raw, "seed", int, where),
        exposures=tuple(exposures),
        benchmark=benchmark,
        change_tolerance=tolerance,
        agent=agent,
    )


DETECTOR = load_detector_config()
EXPERIMENT = load_experiment_config()
