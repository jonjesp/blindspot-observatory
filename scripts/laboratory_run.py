"""Execute a single laboratory case with the published measurement and request contracts."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import numpy as np

from blindspot.agent.client import DEFAULT_BASE_URL, make_live_client
from blindspot.benchmark.observations import recovery_anomaly
from blindspot.benchmark.runner import prepare_case
from blindspot.config import EXPERIMENT
from blindspot.datasets.ztf import extract_patch, load_verified_image
from blindspot.detection.detector import prepare_patch
from blindspot.metrics.completeness import gain_lower_bound
from blindspot.provenance.ledger import digest, run_lock
from scripts.compare_models import call_case, load_sealed, spent_usd
from scripts.evaluate_models import (
    MAXIMUM_BUDGET_USD,
    request_for,
    reservation_for,
    validate_ledger,
    validate_protocol_contract,
)
from scripts.laboratory_imaging import display_pixels, holdout_views
from scripts.reproduce_benchmark import recorded_decision, replay_policies, require_equal

PUBLICATION = Path("evidence/model_evaluation")
LIVE_RUNS = Path("evidence/runs/laboratory")


class LaboratoryError(RuntimeError):
    """An execution failure with a message suitable for the local interface."""


def _probe_measurements(oracle: dict, executed: list[str]) -> dict:
    return {
        name: {
            **oracle[name],
            "lower_bound": gain_lower_bound(**{key: oracle[name][key] for key in ("before", "after", "n")}),
        }
        for name in dict.fromkeys(executed)
    }


class Laboratory:
    """Load sealed cases and execute replay or explicitly enabled paid requests."""

    def __init__(self, archive: Path, budget_usd: float = 0, *, client_factory=make_live_client):
        if not math.isfinite(budget_usd) or not 0 <= budget_usd <= MAXIMUM_BUDGET_USD:
            raise ValueError(f"Live session budget must be between 0 and {MAXIMUM_BUDGET_USD:g} USD")
        self.archive = archive.resolve()
        self.protocol = load_sealed(PUBLICATION / "protocol.json")
        validate_protocol_contract(self.protocol)
        self.recorded = load_sealed(PUBLICATION / "ledger.json")
        validate_ledger(self.protocol, self.recorded, live=True)
        self.cases = {case["definition"]["id"]: case for case in self.protocol["cases"]}
        self.profiles = {profile["model"]: profile for profile in self.protocol["profiles"]}
        self.budget_usd = budget_usd
        self.client_factory = client_factory
        self.ledger_path = LIVE_RUNS / uuid4().hex / "ledger.json"
        self.live_ledger = {"protocol_sha256": self.protocol["seal_sha256"], "mode": "live", "attempts": {}}
        self.live_blocked = False

    def catalog(self) -> dict:
        attempts = self.live_ledger["attempts"].copy()
        return {
            "cases": [
                {key: case["definition"][key] for key in ("id", "field", "offset_x", "offset_y")}
                for case in self.cases.values()
            ],
            "models": [
                {
                    "model": model,
                    "name": profile["name"],
                    "reasoning_effort": profile["reasoning_effort"],
                    "archived_cases": [
                        cid for cid in self.cases if f"{model}/{cid}" in self.recorded["attempts"]
                    ],
                    "reservations_usd": {
                        cid: reservation_for(case, profile) for cid, case in self.cases.items()
                    },
                }
                for model, profile in self.profiles.items()
            ],
            "live_enabled": self.budget_usd > 0 and not self.live_blocked,
            "budget_usd": self.budget_usd,
            "remaining_usd": max(0.0, self.budget_usd - spent_usd({"attempts": attempts})),
            "live_attempts": list(attempts),
        }

    def cutout(self, cid: str) -> tuple[np.ndarray, dict]:
        definition = self.cases[cid]["definition"]
        field = definition["field"]
        image, metadata = load_verified_image(self.archive, field)
        return extract_patch(image, definition["offset_y"], definition["offset_x"]), metadata

    def authorize_live(self, cid: str, model: str) -> None:
        if not self.budget_usd or self.live_blocked:
            raise LaboratoryError("Live execution is disabled for this session.")
        if f"{model}/{cid}" in self.live_ledger["attempts"]:
            raise LaboratoryError("This model and case already ran in the current session.")
        reservation = reservation_for(self.cases[cid], self.profiles[model])
        if spent_usd(self.live_ledger) + reservation > self.budget_usd:
            raise LaboratoryError("The request reservation exceeds the remaining session budget.")

    def _live_record(self, key: str, case: dict, profile: dict, request: dict) -> dict:
        self.authorize_live(case["definition"]["id"], profile["model"])
        candidate = {
            "candidate": profile,
            "base_url": DEFAULT_BASE_URL,
            "budget_usd": self.budget_usd,
            "maximum_calls": len(self.protocol["schedule"]),
        }
        try:
            with self.client_factory() as client:
                if str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
                    raise LaboratoryError("Live execution requires the endpoint in the sealed protocol.")
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with run_lock(self.ledger_path):
                    record = call_case(
                        candidate,
                        {"definition": case["definition"], "request": request},
                        self.live_ledger,
                        self.ledger_path,
                        client,
                        live=True,
                        attempt_key=key,
                    )
                    validate_ledger(self.protocol, self.live_ledger, live=True)
        except Exception as exc:
            if key in self.live_ledger["attempts"]:
                self.live_blocked = True
                raise LaboratoryError(
                    "The provider attempt needs review. Live execution is paused for this session. "
                    f"Inspect {self.ledger_path.as_posix()}."
                ) from None
            if isinstance(exc, LaboratoryError):
                raise
            raise LaboratoryError(
                "The live client could not be initialized. Check the agent dependency and API environment."
            ) from None
        return record

    def run(self, cid: str, model: str, mode: str, progress: Callable[..., None]) -> dict:
        case, profile = self.cases[cid], self.profiles[model]
        key = f"{model}/{cid}"
        progress("image")
        try:
            raw, metadata = self.cutout(cid)
        except (OSError, ValueError) as exc:
            raise LaboratoryError(
                "The selected FITS image could not be verified. Check the local archive."
            ) from exc
        image = {
            **display_pixels(raw),
            "sha256": metadata["sha256"],
            "field": metadata["field"],
            "case_id": cid,
        }
        progress("measurement", image=image)
        prepared = prepare_case(raw, case["definition"])
        require_equal(prepared.observation, case["observation"], "pilot measurements")
        require_equal(prepared.features, case["reference"]["features"], "baseline features")
        require_equal(prepared.oracle, case["reference"]["oracle"], "holdout measurements")
        measured_case = {**case, "observation": prepared.observation}
        request = request_for(measured_case, profile)
        request_sha256 = digest(request)
        require_equal(request_sha256, case["request_sha256"][model], "model request digest")
        progress(
            "decision",
            observation=prepared.observation,
            original_detections=prepare_patch(raw)["baseline"].tolist(),
        )
        record = (
            self._live_record(key, measured_case, profile, request)
            if mode == "live"
            else self.recorded["attempts"][key]
        )
        decision = recorded_decision(record)
        progress("scoring", model_reply={"decision": decision})
        policies = replay_policies(prepared, decision, anomaly=recovery_anomaly(prepared.observation))
        baseline, trace = policies["elimination"], policies["muse"]
        require_equal(baseline, case["reference"]["policies"]["elimination"], "elimination policy")
        chosen = decision["probes"] if decision else []
        probes = list(dict.fromkeys(trace["executed"] + baseline["executed"]))
        return {
            "definition": case["definition"],
            "mode": mode,
            "model": model,
            "request_sha256": request_sha256,
            "decision": decision,
            "response": {
                key: record.get(key)
                for key in (
                    "state",
                    "response_model",
                    "finish_reason",
                    "usage",
                    "reasoning_tokens",
                    "elapsed_seconds",
                    "cost_usd",
                )
            },
            "api_calls": int(mode == "live"),
            "model_trace": trace,
            "baseline_trace": baseline,
            "continuation": trace["executed"][len(chosen) :],
            "measurements": _probe_measurements(prepared.oracle, probes),
            "holdout_views": holdout_views(raw, prepared, probes),
            "gain_threshold": EXPERIMENT.benchmark.gain_threshold,
        }
