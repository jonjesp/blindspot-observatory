"""Command-line planning, ledger and FITS validation tests."""

import json

from conftest import requires_fits

from blindspot.agent.muse import MuseAgent
from blindspot.agent.offline import OfflineMuseClient
from blindspot.agent.schemas import TOOLS
from blindspot.cli import main


def test_plan_command(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert main(["plan", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_OK"
    assert len(json.loads(out.read_text())["cases"]) == 36


def test_plan_defaults_to_local_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["plan"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_OK"
    out = tmp_path / "evidence" / "runs" / "benchmark_plan.json"
    assert len(json.loads(out.read_text())["cases"]) == 36
    assert not (tmp_path / "output").exists()


def test_ledger_command(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    obs = {"pilot_measurements": {}, "indicators": {}, "available_probes": list(TOOLS)}
    MuseAgent(ledger_path=path).decide("C1", obs, client=OfflineMuseClient())
    assert main(["ledger", "--path", str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["completed"] == 1 and summary["unresolved"] == []


@requires_fits
def test_verify_and_smoke_on_real_images(fits_zip, capsys):
    assert main(["verify-data", "--fits-zip", str(fits_zip)]) == 0
    capsys.readouterr()
    assert main(["smoke-fits", "--fits-zip", str(fits_zip)]) == 0
    cases = json.loads(capsys.readouterr().out)["cases"]
    # Numerical regression: identical to the pre-refactor engine on the first site of each exposure.
    assert {c["field"]: c["snr5"] for c in cases} == {"E4_r": 5, "E5_g": 11}
    assert all(c["zero_flux"] == 0 for c in cases)


@requires_fits
def test_all_planned_sites_are_usable(fits_zip):
    from blindspot.benchmark.design import plan
    from blindspot.datasets.ztf import extract_patch, load_verified_image
    from blindspot.detection.detector import prepare_patch

    images = {f: load_verified_image(fits_zip, f)[0] for f in ("E4_r", "E5_g")}
    for case in plan()["cases"]:
        patch = prepare_patch(extract_patch(images[case["field"]], case["offset_y"], case["offset_x"]))
        assert patch["sigma"] > 0
