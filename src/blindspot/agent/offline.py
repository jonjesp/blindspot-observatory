"""Network-free stand-in for the model client, for tests and dry runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

DEFAULT_REPLY = json.dumps(
    {
        "decision": "ambiguous",
        "probes": ["probe_sky", "probe_psf"],
        "justification": "Two competing mechanisms need contrastive probes.",
    }
)


class OfflineMuseClient:
    """Returns a canned completion. It cannot send network requests and costs nothing."""

    def __init__(self, *, finish: str = "stop", raw: str | None = None, fails: bool = False):
        self.calls = 0
        self.finish = finish
        self.raw = DEFAULT_REPLY if raw is None else raw
        self.fails = fails
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_request):
        self.calls += 1
        if self.fails:
            raise ConnectionError("Simulated network failure")
        return SimpleNamespace(
            id=f"offline-{self.calls}",
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish,
                    message=SimpleNamespace(content=self.raw),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=40),
        )
