"""Atomic JSON persistence and cross-process transaction-lock tests."""

import subprocess
import sys

import pytest

from blindspot.provenance import ledger


def test_process_lock_rejects_an_independent_writer(tmp_path):
    path = tmp_path / "ledger.json"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from blindspot.provenance.ledger import run_lock\n"
        "try:\n"
        "    with run_lock(Path(sys.argv[1])): sys.exit(1)\n"
        "except RuntimeError:\n"
        "    sys.exit(0)\n"
    )
    with ledger.run_lock(path):
        result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr.decode()
    assert not path.with_suffix(".json.lock").exists()


def test_failed_atomic_replace_preserves_original_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    ledger.save_json(path, {"retained": True})
    original = path.read_bytes()

    def unavailable_replace(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(ledger.os, "replace", unavailable_replace)
    with pytest.raises(OSError):
        ledger.save_json(path, {"retained": False})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]