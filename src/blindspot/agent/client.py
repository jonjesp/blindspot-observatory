"""Factory for the live model client (optional dependency: ``pip install '.[agent]'``)."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.meta.ai/v1"


def make_live_client(*, timeout: float = 120.0):
    """Build an OpenAI-compatible client from ``MODEL_API_KEY`` / ``MODEL_BASE_URL``.

    SDK-level retries are disabled: a silent retry could bill the same decision twice and would
    not appear in the ledger.
    """
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (see .env.example)")
    base_url = os.environ.get("MODEL_BASE_URL", "").strip() or DEFAULT_BASE_URL
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the agent extra: pip install -e '.[agent]'") from exc
    return OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=timeout)
