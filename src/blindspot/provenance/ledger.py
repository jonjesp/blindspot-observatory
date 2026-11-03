"""Atomic JSON records with SHA-256 content digests.

The digests detect accidental edits and make runs comparable. They are not keyed, so they do
not protect against someone who deliberately recomputes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def save_json(path: str | Path, obj: dict) -> None:
    """Write JSON atomically (temp file, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(path: Path):
    """Hold an exclusive process lock for one file transaction."""
    path = path.resolve()
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Run lock exists: {lock}. Inspect the previous run before removing it") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink()


def verified_record(payload: dict) -> dict:
    return {"payload": payload, "sha256": digest(payload)}


def verify_record(record: dict) -> dict:
    if set(record) != {"payload", "sha256"} or digest(record["payload"]) != record["sha256"]:
        raise ValueError("Record altered or malformed")
    return record["payload"]
