"""Muse decision agent: strict request, cost cap and a resumable, sealed ledger.

Safety properties
-----------------
* Only validated, whitelisted observations reach the prompt (see ``observation``).
* A paid client needs ``allow_paid=True`` and a remaining budget; the offline client is free.
* The reservation is written to the ledger BEFORE the call, so a crash or network failure leaves
  a ``reserved_unknown`` record and the case is never silently re-billed.
* Repeating a case returns the stored record; if the request changed, it raises.
* The ledger seal detects accidental edits. It is not keyed and is not tamper-proof.
"""

from __future__ import annotations

import json
from pathlib import Path

from blindspot.agent.observation import validate_observation
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import DECISION_FORMAT, validate_decision
from blindspot.config import EXPERIMENT, Prices
from blindspot.provenance.ledger import digest, run_lock, save_json

AGENT = EXPERIMENT.agent

SYSTEM_PROMPT = (
    "You are a research agent auditing the completeness of a source-detection pipeline on "
    "astronomical images. You receive only numeric measurements from a pilot comparison between a "
    "reference condition A and an anomalous condition B. Choose which reversible probe to run: "
    "probe_sky, probe_psf or probe_flux. Answer 'probe' with exactly one probe when one mechanism "
    "is clearly most plausible, 'ambiguous' with two distinct probes when two mechanisms are "
    "plausible, and 'abstain' with no probes when the data show no anomaly. Never invent "
    "measurements. You cannot see hidden causes or holdout data. Keep the justification between "
    "12 and 350 characters."
)


def build_request(case_id: str, observation: dict) -> dict:
    """Pure function: the exact request that will be hashed and sent."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    validate_observation(observation)
    user = json.dumps({"case_id": case_id, **observation}, sort_keys=True, ensure_ascii=False)
    return {
        "model": AGENT.model,
        "reasoning_effort": AGENT.reasoning_effort,
        "store": False,
        "response_format": DECISION_FORMAT,
        "max_completion_tokens": AGENT.max_completion_tokens,
        "messages": [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }


def usage_cost(prompt_tokens: int, completion_tokens: int, cached_tokens: int, prices: Prices) -> float:
    """Dollar cost of one call from token counts."""
    if min(prompt_tokens, completion_tokens, cached_tokens) < 0 or cached_tokens > prompt_tokens:
        raise ValueError("Invalid token counts")
    return (
        (prompt_tokens - cached_tokens) * prices.input
        + cached_tokens * prices.cached_input
        + completion_tokens * prices.output
    ) / 1_000_000


def worst_case_cost(request: dict, prices: Prices) -> float:
    """Upper bound used for reservation: every request byte billed as a token, full output."""
    prompt_bytes = len(json.dumps(request["messages"], ensure_ascii=False).encode("utf-8"))
    return (prompt_bytes * prices.input + request["max_completion_tokens"] * prices.output) / 1_000_000


class MuseAgent:
    def __init__(
        self,
        *,
        ledger_path: str | Path,
        max_calls: int | None = None,
        budget_usd: float | None = None,
    ):
        self.path = Path(ledger_path)
        self.max_calls = AGENT.max_calls if max_calls is None else max_calls
        self.budget_usd = AGENT.budget_usd if budget_usd is None else budget_usd
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= AGENT.max_calls:
            raise ValueError(f"max_calls must be an integer in [1, {AGENT.max_calls}]")
        if not 0 < self.budget_usd <= AGENT.budget_usd:
            raise ValueError(f"budget_usd must be in (0, {AGENT.budget_usd}]")

    def _load(self) -> dict:
        if not self.path.exists():
            return {"model": AGENT.model, "attempts": {}}
        ledger = json.loads(self.path.read_text(encoding="utf-8"))
        seal = digest({"model": ledger.get("model"), "attempts": ledger.get("attempts")})
        if ledger.get("seal_sha256") != seal:
            raise RuntimeError("Ledger integrity check failed")
        if ledger.get("model") != AGENT.model or type(ledger.get("attempts")) is not dict:
            raise RuntimeError("Ledger belongs to a different model or is malformed")
        return ledger

    def _save(self, ledger: dict) -> None:
        ledger["seal_sha256"] = digest({"model": ledger["model"], "attempts": ledger["attempts"]})
        save_json(self.path, ledger)

    @staticmethod
    def _spent(ledger: dict) -> float:
        """Spent so far; unresolved reservations count at their worst case."""
        return sum(float(a.get("cost_usd", a.get("reserved_usd", 0.0))) for a in ledger["attempts"].values())

    def summary(self) -> dict:
        ledger = self._load()
        states = [a["state"] for a in ledger["attempts"].values()]
        return {
            "cases": len(states),
            "completed": states.count("completed"),
            "invalid_response": states.count("invalid_response"),
            "unresolved": sorted(
                c for c, a in ledger["attempts"].items() if a["state"] == "reserved_unknown"
            ),
            "spent_usd": round(self._spent(ledger), 6),
            "budget_usd": self.budget_usd,
        }

    def decide(self, case_id: str, observation: dict, *, client, allow_paid: bool = False) -> dict:
        """Record one decision under an exclusive reservation-to-response transaction."""
        with run_lock(self.path):
            return self._decide(case_id, observation, client=client, allow_paid=allow_paid)

    def _decide(self, case_id: str, observation: dict, *, client, allow_paid: bool) -> dict:
        request = build_request(case_id, observation)
        request_hash = digest(request)
        offline = isinstance(client, OfflineMuseClient)
        if not offline and not allow_paid:
            raise PermissionError("Paid calls are disabled: pass allow_paid=True to use a live client")
        if not hasattr(getattr(client, "chat", None), "completions"):
            raise TypeError("client must expose chat.completions.create")

        ledger = self._load()
        previous = ledger["attempts"].get(case_id)
        if previous is not None:
            if previous.get("request_sha256") != request_hash:
                raise RuntimeError("Same case with a different request: do not reuse this ledger")
            return previous
        if len(ledger["attempts"]) >= self.max_calls:
            raise RuntimeError("Total call limit reached")
        reserved = 0.0 if offline else worst_case_cost(request, AGENT.prices)
        if self._spent(ledger) + reserved > self.budget_usd:
            raise RuntimeError("Budget exhausted: the worst-case cost of this call exceeds what is left")

        ledger["attempts"][case_id] = {
            "state": "reserved_unknown",
            "request_sha256": request_hash,
            "reserved_usd": reserved,
        }
        self._save(ledger)  # persisted before any network I/O

        response = client.chat.completions.create(**request)
        choice = response.choices[0]
        usage = response.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        prompt, completion = int(usage.prompt_tokens), int(usage.completion_tokens)
        record = {
            "state": "invalid_response",
            "request_sha256": request_hash,
            "response_id": response.id,
            "finish_reason": choice.finish_reason,
            "raw_output": choice.message.content,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": int(cached)},
            "cost_usd": 0.0 if offline else usage_cost(prompt, completion, int(cached), AGENT.prices),
        }
        if choice.finish_reason == "stop":
            try:
                record["decision"] = validate_decision(record["raw_output"])
                record["state"] = "completed"
            except ValueError:
                pass
        ledger["attempts"][case_id] = record
        self._save(ledger)
        return record
