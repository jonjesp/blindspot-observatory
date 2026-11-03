"""Pre-run freeze: fingerprint everything that defines the benchmark, and refuse to run if it changed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.resources import files
from pathlib import Path

from blindspot import __version__
from blindspot.agent.muse import AGENT, SYSTEM_PROMPT
from blindspot.agent.schemas import DECISION_FORMAT
from blindspot.benchmark import generator, interventions, observations
from blindspot.benchmark.design import plan_hash
from blindspot.datasets.ztf import image_manifest
from blindspot.provenance.ledger import digest, save_json

CONFIG_FILES = ("detector.yaml", "experiment.yaml")


def _config_sha256(name: str) -> str:
    text = files("blindspot.configs").joinpath(name).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def implementation_hashes() -> dict[str, str]:
    """Fingerprint packaged implementation and resources, with platform-independent newlines.

    The source tree is included rather than only a version string: a new clean Git commit can
    otherwise change the generator or scoring while an old freeze still passes validation.
    """
    package = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".json")
    }


def freeze_record() -> dict:
    """Everything that must not change between the freeze commit and the paid run."""
    return {
        "version": __version__,
        "implementation_sha256": implementation_hashes(),
        "experimental_parameters_sha256": digest(
            {
                "strong": interventions.STRONG,
                "minor": interventions.MINOR,
                "factors": interventions.FACTOR,
                "mask_probability": generator.MASK_PROBABILITY,
                "level_snr": observations.LEVEL_SNR,
                "profile_snr": observations.PROFILE_SNR,
                "pilot_batches": observations.PILOT_BATCHES,
                "holdout_batches": observations.HOLDOUT_BATCHES,
            }
        ),
        "configs_sha256": {name: _config_sha256(name) for name in CONFIG_FILES},
        "plan_sha256": plan_hash(),
        "request_contract_sha256": digest(
            {
                "system_prompt": SYSTEM_PROMPT,
                "response_format": DECISION_FORMAT,
                "model": AGENT.model,
                "reasoning_effort": AGENT.reasoning_effort,
                "max_completion_tokens": AGENT.max_completion_tokens,
            }
        ),
        "fits_sha256": {e["field"]: e["sha256"] for e in image_manifest()},
    }


def write_freeze(path: str | Path) -> dict:
    record = freeze_record()
    save_json(path, record)
    return record


def verify_freeze(path: str | Path) -> dict:
    """Raise ``RuntimeError`` naming what changed since the freeze file was written."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Freeze file not found: {path}. Run 'blindspot freeze' and commit it first")
    frozen, current = json.loads(path.read_text(encoding="utf-8")), freeze_record()
    changed = sorted(k for k in current if frozen.get(k) != current[k])
    if changed:
        raise RuntimeError(
            f"Benchmark changed since the freeze ({', '.join(changed)}). Re-freeze deliberately"
        )
    return current


def git_state() -> dict:
    """Best-effort commit hash and dirty flag of the working directory (None when unavailable)."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}
