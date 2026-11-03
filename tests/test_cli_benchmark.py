"""Command-line benchmark, freeze and artifact-routing tests."""

import json

import pytest
from conftest import requires_fits

from blindspot import cli


def test_freeze_command_writes_the_record(tmp_path, capsys):
    out = tmp_path / "freeze.json"
    assert cli.main(["freeze", "--out", str(out)]) == 0
    assert set(json.loads(out.read_text())) >= {"plan_sha256", "request_contract_sha256", "configs_sha256"}


def test_live_run_refuses_without_a_freeze(tmp_path, capsys):
    code = cli.main(["benchmark", "--live", "--fits-zip", "x.zip", "--freeze", str(tmp_path / "none.json")])
    assert code == 2 and "Freeze file not found" in capsys.readouterr().err


def test_live_run_refuses_a_dirty_tree_and_a_missing_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify_freeze", lambda path: {})
    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "abc", "dirty": True})
    assert cli.main(["benchmark", "--live", "--fits-zip", "x.zip"]) == 2
    assert "uncommitted" in capsys.readouterr().err
    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert cli.main(["benchmark", "--live", "--fits-zip", "x.zip"]) == 2
    assert "MODEL_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("custom_paths", [False, True])
def test_default_artifacts_have_stable_paths_and_separate_modes(tmp_path, monkeypatch, live, custom_paths):
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    archived = evidence / "benchmark_results.json"
    archived.write_text('{"archived": true}', encoding="utf-8")
    monkeypatch.setattr(cli, "verify_freeze", lambda path: {})
    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(cli, "make_live_client", object)
    monkeypatch.setattr(cli, "load_verified_image", lambda *args: (None, {}))
    monkeypatch.setattr(cli, "format_summary", lambda *args, **kwargs: "mock runner, no API")
    captured = {}

    def fake_run(cases, get_raw, *, agent, client, allow_paid, progress):
        captured["ledger"] = agent.path
        return {"cases": [], "summary": {}}

    monkeypatch.setattr(cli, "run_benchmark", fake_run)
    argv = ["benchmark", "--fits-zip", "unused.zip"] + (["--live"] if live else [])
    root = tmp_path / "evidence" / "runs" / ("live" if live else "dry_run")
    out, ledger = root / "benchmark_results.json", root / "benchmark_ledger.json"
    if custom_paths:
        out, ledger = tmp_path / "selected_results.json", tmp_path / "selected_ledger.json"
        argv += ["--out", str(out), "--ledger", str(ledger)]
    assert cli.main(argv) == 0
    assert out.exists()
    assert captured["ledger"].resolve() == ledger
    assert json.loads(archived.read_text()) == {"archived": True}
    assert not (tmp_path / "output").exists()


def test_calibration_defaults_to_local_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    archived = evidence / "calibration.json"
    archived.write_text('{"archived": true}', encoding="utf-8")
    payload = {"summary": {"cases": 36}}
    monkeypatch.setattr(cli, "calibrate", lambda path: payload)
    assert cli.main(["calibrate", "--fits-zip", "unused.zip"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "CALIBRATION_DONE"
    assert json.loads((evidence / "runs" / "calibration.json").read_text()) == payload
    assert json.loads(archived.read_text()) == {"archived": True}


def test_default_freeze_path_is_stable(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    archived = evidence / "benchmark_freeze.json"
    archived.write_text('{"archived": true}', encoding="utf-8")
    assert cli.main(["freeze"]) == 0
    capsys.readouterr()
    assert (evidence / "runs" / "benchmark_freeze.json").is_file()
    assert json.loads(archived.read_text()) == {"archived": True}


@requires_fits
def test_dry_run_on_real_images_is_flagged_and_incomplete_when_limited(fits_zip, tmp_path, capsys):
    out = tmp_path / "r.json"
    code = cli.main(
        [
            "benchmark",
            "--fits-zip",
            str(fits_zip),
            "--limit",
            "5",
            "--out",
            str(out),
            "--ledger",
            str(tmp_path / "dry.json"),
        ]
    )
    assert code == 0 and "DRY RUN" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["mode"] == "offline_dry_run" and result["complete"] is False and len(result["cases"]) == 5
