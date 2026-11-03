"""Decision-agent accounting, permissions and client-contract tests."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from blindspot.agent import client as live
from blindspot.agent.muse import AGENT, MuseAgent, build_request, usage_cost, worst_case_cost
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS
from blindspot.provenance.ledger import digest, save_json

OBS = {
    "pilot_measurements": {"recovery_counts": {"nominal": {"A": [8, 12], "B": [2, 3]}}},
    "indicators": {"background_mad_change": 0.9},
    "available_probes": list(TOOLS),
}
GOOD = json.dumps(
    {"decision": "probe", "probes": ["probe_flux"], "justification": "Signal fell while noise held."}
)


class FakePaidClient:
    """Stands in for the live SDK: same call shape, controllable usage, no network."""

    def __init__(self, raw=GOOD, prompt=1000, completion=500, cached=0, finish="stop", fails=False):
        self.calls, self.fails = 0, fails
        self._reply = dict(raw=raw, prompt=prompt, completion=completion, cached=cached, finish=finish)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls += 1
        if self.fails:
            raise TimeoutError("simulated timeout")
        r = self._reply
        return SimpleNamespace(
            id=f"resp-{self.calls}",
            choices=[SimpleNamespace(finish_reason=r["finish"], message=SimpleNamespace(content=r["raw"]))],
            usage=SimpleNamespace(
                prompt_tokens=r["prompt"],
                completion_tokens=r["completion"],
                prompt_tokens_details=SimpleNamespace(cached_tokens=r["cached"]),
            ),
        )


@pytest.fixture
def agent(tmp_path):
    return MuseAgent(ledger_path=tmp_path / "ledger.json")


def test_request_is_pure_and_pinned_to_the_contract():
    request = build_request("C1", OBS)
    assert request == build_request("C1", OBS)
    assert request["store"] is False and request["model"] == AGENT.model
    assert request["max_completion_tokens"] == AGENT.max_completion_tokens
    assert json.loads(request["messages"][1]["content"])["case_id"] == "C1"
    with pytest.raises(ValueError):
        build_request("", OBS)


def test_offline_client_is_free_and_idempotent(agent):
    client = OfflineMuseClient()
    first = agent.decide("C1", OBS, client=client)
    again = agent.decide("C1", OBS, client=client)
    assert first["state"] == "completed" and first["cost_usd"] == 0.0
    assert client.calls == 1 and again == first


def test_live_client_requires_explicit_permission(agent):
    paid = FakePaidClient()
    with pytest.raises(PermissionError):
        agent.decide("C1", OBS, client=paid)
    assert paid.calls == 0


def test_cost_is_recorded_from_usage(agent):
    record = agent.decide("C1", OBS, client=FakePaidClient(prompt=1000, completion=500), allow_paid=True)
    assert record["cost_usd"] == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    assert agent.summary()["spent_usd"] == pytest.approx(record["cost_usd"], abs=1e-6)


def test_cached_tokens_are_billed_at_the_cached_rate():
    prices = AGENT.prices
    assert usage_cost(1000, 0, 400, prices) == pytest.approx(
        (600 * prices.input + 400 * prices.cached_input) / 1e6
    )
    with pytest.raises(ValueError):
        usage_cost(10, 0, 20, prices)


def test_budget_blocks_the_call_before_it_is_made(tmp_path):
    tiny = MuseAgent(ledger_path=tmp_path / "l.json", budget_usd=1e-6)
    paid = FakePaidClient()
    with pytest.raises(RuntimeError, match="Budget"):
        tiny.decide("C1", OBS, client=paid, allow_paid=True)
    assert paid.calls == 0 and not (tmp_path / "l.json").exists()


def test_budget_accumulates_across_cases(tmp_path):
    reserve = worst_case_cost(build_request("C1", OBS), AGENT.prices)
    capped = MuseAgent(ledger_path=tmp_path / "l.json", budget_usd=reserve + 1e-3)
    heavy = dict(prompt=2000, completion=3000)  # spends about the whole reservation
    capped.decide("C1", OBS, client=FakePaidClient(**heavy), allow_paid=True)
    with pytest.raises(RuntimeError, match="Budget"):
        capped.decide("C2", OBS, client=FakePaidClient(**heavy), allow_paid=True)


def test_call_limit(tmp_path):
    limited = MuseAgent(ledger_path=tmp_path / "l.json", max_calls=1)
    limited.decide("C1", OBS, client=OfflineMuseClient())
    with pytest.raises(RuntimeError, match="call limit"):
        limited.decide("C2", OBS, client=OfflineMuseClient())


def test_failure_leaves_a_reservation_and_is_never_retried(agent, tmp_path):
    with pytest.raises(TimeoutError):
        agent.decide("C1", OBS, client=FakePaidClient(fails=True), allow_paid=True)
    summary = agent.summary()
    assert summary["unresolved"] == ["C1"] and summary["spent_usd"] > 0  # worst case stays reserved
    retry = FakePaidClient()
    stored = agent.decide("C1", OBS, client=retry, allow_paid=True)
    assert retry.calls == 0 and stored["state"] == "reserved_unknown"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"finish": "length", "raw": ""},
        {"raw": "not json"},
        {"raw": json.dumps({"decision": "probe", "probes": [], "justification": "x" * 20})},
    ],
)
def test_invalid_replies_are_recorded_and_still_billed(agent, kwargs):
    record = agent.decide("C1", OBS, client=FakePaidClient(**kwargs), allow_paid=True)
    assert record["state"] == "invalid_response" and "decision" not in record and record["cost_usd"] > 0


def test_changed_request_for_same_case_raises(agent):
    agent.decide("C1", OBS, client=OfflineMuseClient())
    other = {**OBS, "indicators": {"background_mad_change": 0.5}}
    with pytest.raises(RuntimeError, match="different request"):
        agent.decide("C1", other, client=OfflineMuseClient())


def test_ledger_accidental_edit_is_detected(agent, tmp_path):
    agent.decide("C1", OBS, client=OfflineMuseClient())
    path = tmp_path / "ledger.json"
    ledger = json.loads(path.read_text())
    ledger["attempts"]["C1"]["decision"]["decision"] = "abstain"
    path.write_text(json.dumps(ledger))
    with pytest.raises(RuntimeError, match="integrity"):
        agent.decide("C2", OBS, client=OfflineMuseClient())


def test_seal_is_not_tamper_proof_by_design(agent, tmp_path):
    """Documented limitation: anyone can recompute an unkeyed digest."""
    agent.decide("C1", OBS, client=OfflineMuseClient())
    path = tmp_path / "ledger.json"
    ledger = json.loads(path.read_text())
    ledger["attempts"]["C1"]["decision"]["decision"] = "abstain"
    ledger["seal_sha256"] = digest({"model": ledger["model"], "attempts": ledger["attempts"]})
    save_json(path, ledger)
    assert agent.decide("C1", OBS, client=OfflineMuseClient())["decision"]["decision"] == "abstain"


def test_leaky_observation_never_reaches_the_client(agent):
    spy = OfflineMuseClient()
    leaky = {**OBS, "indicators": {"flux_factor_applied": 0.5}}
    with pytest.raises(ValueError):
        agent.decide("C1", leaky, client=spy)
    assert spy.calls == 0


def test_agent_limits_cannot_exceed_the_frozen_configuration(tmp_path):
    with pytest.raises(ValueError):
        MuseAgent(ledger_path=tmp_path / "l.json", max_calls=AGENT.max_calls + 1)
    with pytest.raises(ValueError):
        MuseAgent(ledger_path=tmp_path / "l.json", budget_usd=AGENT.budget_usd * 10)


def test_client_without_chat_api_is_rejected(agent):
    with pytest.raises(TypeError):
        agent.decide("C1", OBS, client=object(), allow_paid=True)


def test_live_factory_requires_a_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        live.make_live_client()


def test_live_example_import_and_missing_opt_in_make_no_request(monkeypatch):
    def forbidden_client():
        raise AssertionError("Import and argument validation must not construct a paid client")

    monkeypatch.setattr(live, "make_live_client", forbidden_client)
    example = Path(__file__).parents[1] / "examples" / "live_test.py"
    namespace = runpy.run_path(str(example))
    with pytest.raises(SystemExit) as stopped:
        namespace["main"]([])
    assert stopped.value.code == 2


def test_live_example_explicit_opt_in_uses_one_mock_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakePaidClient()
    monkeypatch.setattr(live, "make_live_client", lambda: client)
    example = Path(__file__).parents[1] / "examples" / "live_test.py"
    namespace = runpy.run_path(str(example))
    assert namespace["main"](["--live"]) == 0
    assert client.calls == 1


def test_interleaved_writer_is_rejected_before_request_and_can_resume(agent):
    second = OfflineMuseClient()

    class InterleavedClient(OfflineMuseClient):
        def _create(self, **request):
            with pytest.raises(RuntimeError, match="Run lock exists"):
                MuseAgent(ledger_path=agent.path).decide("C2", OBS, client=second)
            assert second.calls == 0
            return super()._create(**request)

    agent.decide("C1", OBS, client=InterleavedClient())
    assert not agent.path.with_suffix(".json.lock").exists()
    agent.decide("C2", OBS, client=second)
    assert agent.summary()["completed"] == 2
    assert second.calls == 1


def test_failed_request_releases_lock_but_retains_reservation(agent):
    with pytest.raises(TimeoutError):
        agent.decide("C1", OBS, client=FakePaidClient(fails=True), allow_paid=True)
    assert not agent.path.with_suffix(".json.lock").exists()
    assert agent.summary()["unresolved"] == ["C1"]
