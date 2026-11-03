"""Strict contract for an experimental decision. A decision is never a measurement."""

from __future__ import annotations

import json

TOOLS = ("probe_sky", "probe_psf", "probe_flux")
PROBES_PER_DECISION = {"probe": 1, "ambiguous": 2, "abstain": 0}
JUSTIFICATION_RANGE = (12, 350)

DECISION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "blindspot_select_probe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(PROBES_PER_DECISION)},
                "probes": {"type": "array", "items": {"type": "string", "enum": list(TOOLS)}},
                "justification": {"type": "string"},
            },
            "required": ["decision", "probes", "justification"],
            "additionalProperties": False,
        },
    },
}


def validate_decision(raw: str) -> dict:
    """Parse and validate a model reply; raise ``ValueError`` for anything outside the contract."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reply is not valid JSON") from exc
    if type(data) is not dict or set(data) != {"decision", "probes", "justification"}:
        raise ValueError("Decision does not match the schema")
    decision, probes, text = data["decision"], data["probes"], data["justification"]
    if type(decision) is not str or decision not in PROBES_PER_DECISION:
        raise ValueError("Unknown decision type")
    if type(probes) is not list or len(probes) != PROBES_PER_DECISION[decision]:
        raise ValueError("Number of probes does not match the decision")
    if any(type(p) is not str or p not in TOOLS for p in probes) or len(set(probes)) != len(probes):
        raise ValueError("Duplicate or unauthorised probe")
    low, high = JUSTIFICATION_RANGE
    if type(text) is not str or not low <= len(text.strip()) <= high:
        raise ValueError("Justification missing or out of length range")
    return data
