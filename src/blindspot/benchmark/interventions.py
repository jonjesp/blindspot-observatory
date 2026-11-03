"""Physical interventions that turn the reference condition A into an anomalous condition B.

This module belongs to the PRIVATE side of the benchmark. Nothing under ``blindspot.agent`` may
import it (a test enforces this): the model must never see which factor was changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from blindspot.agent.schemas import TOOLS


@dataclass(frozen=True)
class Condition:
    """Observing condition. The defaults are the nominal reference condition A."""

    noise_multiplier: float = 0.0  # extra Gaussian noise, in units of the reference noise
    psf_scale: float = 1.0  # >= 1: width of the injected PSF relative to the detector's PSF
    flux_scale: float = 1.0  # in (0, 1]: injected flux relative to nominal

    def __post_init__(self) -> None:
        if not (math.isfinite(self.noise_multiplier) and self.noise_multiplier >= 0):
            raise ValueError("noise_multiplier must be finite and non-negative")
        if not (math.isfinite(self.psf_scale) and self.psf_scale >= 1):
            raise ValueError("psf_scale must be finite and >= 1")
        if not (math.isfinite(self.flux_scale) and 0 < self.flux_scale <= 1):
            raise ValueError("flux_scale must be in (0, 1]")


NOMINAL = Condition()

# Which condition field each probe restores.
FACTOR = {"probe_sky": "noise_multiplier", "probe_psf": "psf_scale", "probe_flux": "flux_scale"}
# Degradation magnitudes. A single cause (or the dominant one of two) is STRONG; the other cause of
# a two-cause case is MINOR, so restoring the dominant factor recovers clearly more than the other.
STRONG = {"probe_sky": 0.8, "probe_psf": 1.3, "probe_flux": 0.8}
MINOR = {"probe_sky": 0.3, "probe_psf": 1.1, "probe_flux": 0.93}


def apply_causes(causes: tuple[str, ...], dominant: str | None = None) -> Condition:
    """Condition B in which each listed cause is active.

    One cause is always strong. With two causes, ``dominant`` selects the strong one.
    """
    if len(set(causes)) != len(causes) or any(c not in TOOLS for c in causes):
        raise ValueError("Causes must be distinct known probes")
    if len(causes) > 2:
        raise ValueError("At most two simultaneous causes are supported")
    if len(causes) == 2 and dominant not in causes:
        raise ValueError("Two causes need a dominant one among them")
    if len(causes) < 2 and dominant not in (None, *causes):
        raise ValueError("dominant must be one of the causes")
    strengths = {c: (STRONG if len(causes) < 2 or c == dominant else MINOR)[c] for c in causes}
    return replace(NOMINAL, **{FACTOR[c]: v for c, v in strengths.items()})


def restore(condition: Condition, probe: str) -> Condition:
    """Undo ONE factor, leaving the others in place. This is what a probe does."""
    if probe not in FACTOR:
        raise ValueError(f"Unknown probe: {probe}")
    return replace(condition, **{FACTOR[probe]: getattr(NOMINAL, FACTOR[probe])})
