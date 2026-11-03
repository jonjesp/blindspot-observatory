"""Published-evaluation reproduction, input preservation and rejection tests."""

import json
import shutil
from pathlib import Path

import pytest

from blindspot.provenance.ledger import save_json
from scripts import reproduce_evaluation as reproduction
from scripts.compare_models import load_sealed, sealed


@pytest.fixture
def publication(tmp_path):
    original = Path(__file__).parents[1] / "evidence/model_evaluation"
    destination = tmp_path / "publication"
    destination.mkdir()
    for name in ("protocol.json", "ledger.json", "case_results.csv"):
        shutil.copyfile(original / name, destination / name)
    return destination


def test_current_source_replay_validates_records_and_preserves_inputs(publication, tmp_path, monkeypatch):
    original = {path.name: path.read_bytes() for path in publication.iterdir()}
    verified = []

    def verify_cases(document, fits_path):
        verified.append(len(document["cases"]))
        return len(document["cases"])

    monkeypatch.setattr(reproduction, "verify_measurements", verify_cases)
    result = reproduction.reproduce_evaluation(publication, Path("unused.zip"), tmp_path / "outputs")
    assert verified == [36]
    assert result["status"] == "reproduced"
    assert result["responses_verified"] == 108
    assert result["api_calls"] == 0
    assert {path.name: path.read_bytes() for path in publication.iterdir()} == original


def test_measurement_disagreement_stops_reproduction(publication, tmp_path, monkeypatch):
    def mismatched_measurements(*args):
        raise ValueError("Mismatch: F01 probe measurements")

    monkeypatch.setattr(reproduction, "verify_measurements", mismatched_measurements)
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="F01 probe measurements"):
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), output)
    assert not (output / "reproduction.json").exists()
    assert not (output / "report.md").exists()


def test_changed_scientific_definition_cannot_be_replayed(publication, tmp_path):
    path = publication / "protocol.json"
    document = load_sealed(path)
    document.pop("seal_sha256")
    document["freeze"]["plan_sha256"] = "0" * 64
    save_json(path, sealed(document))
    with pytest.raises(ValueError, match="experimental definition"):
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), tmp_path / "outputs")


def test_publication_table_mismatch_cannot_produce_success(publication, tmp_path, monkeypatch):
    monkeypatch.setattr(reproduction, "verify_measurements", lambda *args: 36)
    (publication / "case_results.csv").write_text("model\nincorrect\n", encoding="utf-8")
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="published case table"):
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), output)
    assert not (output / "reproduction.json").exists()
    assert not (output / "report.md").exists()
    assert not (output / "case_results.csv").exists()


def test_failed_rerun_invalidates_previous_success(publication, tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    with monkeypatch.context() as mocked:
        mocked.setattr(reproduction, "verify_measurements", lambda *args: 36)
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), output)
    assert (output / "reproduction.json").is_file()
    previous_table = (output / "case_results.csv").read_bytes()

    status = reproduction.main(
        [
            "--directory",
            str(publication),
            "--fits-zip",
            str(tmp_path / "missing.zip"),
            "--output-directory",
            str(output),
        ]
    )

    assert status == 2
    assert not (output / "reproduction.json").exists()
    assert (output / "case_results.csv").read_bytes() == previous_table


@pytest.mark.parametrize("nested", [False, True])
def test_reproduction_cannot_write_inside_publication(publication, nested):
    output = publication / "generated" if nested else publication
    original = (publication / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="separate"):
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), output)
    assert (publication / "protocol.json").read_bytes() == original


def test_tampered_ledger_is_rejected_before_measurements(publication, tmp_path):
    path = publication / "ledger.json"
    document = json.loads(path.read_text())
    document["mode"] = "offline_dry_run"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        reproduction.reproduce_evaluation(publication, Path("unused.zip"), tmp_path / "outputs")
