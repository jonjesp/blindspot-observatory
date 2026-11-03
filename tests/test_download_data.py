"""Offline tests for the IRSA download helper (no network access)."""

import hashlib
import io

import pytest

from scripts.download_data import fetch_image, image_url


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _entry(payload: bytes, name: str = "ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits") -> dict:
    return {"field": "T", "filename": name, "sha256": hashlib.sha256(payload).hexdigest()}


def test_urls_match_the_irsa_documented_examples():
    assert image_url("ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits") == (
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0411/467847/"
        "ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits"
    )
    assert image_url("ztf_20180325251921_000314_zg_c08_o_q2_sciimg.fits").endswith(
        "/sci/2018/0325/251921/ztf_20180325251921_000314_zg_c08_o_q2_sciimg.fits"
    )


def test_rejects_names_that_are_not_ztf_science_images():
    with pytest.raises(ValueError):
        image_url("something_else.fits")


def test_download_verifies_and_then_uses_the_cache(tmp_path):
    payload = b"pixels" * 100
    entry = _entry(payload)
    assert fetch_image(entry, tmp_path, opener=lambda url: _Response(payload)) == "downloaded"
    assert (tmp_path / entry["filename"]).read_bytes() == payload
    assert fetch_image(entry, tmp_path, opener=lambda url: pytest.fail("must not download")) == "cached"


def test_checksum_mismatch_is_rejected_and_leaves_no_file(tmp_path):
    entry = _entry(b"expected")
    with pytest.raises(ValueError, match="mismatch"):
        fetch_image(entry, tmp_path, opener=lambda url: _Response(b"different"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5, "4"])
def test_invalid_retry_count_fails_before_io(tmp_path, attempts):
    destination = tmp_path / "images"
    with pytest.raises(ValueError, match="attempts must be a positive integer"):
        fetch_image(_entry(b"pixels"), destination, attempts=attempts)
    assert not destination.exists()


def test_transient_download_failure_retries_and_removes_partial_file(tmp_path):
    payload = b"pixels"
    calls, pauses = [], []

    def open_image(url):
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("simulated download timeout")
        return _Response(payload)

    assert fetch_image(_entry(payload), tmp_path, opener=open_image, sleep=pauses.append) == "downloaded"
    assert len(calls) == 2 and pauses == [2]
    assert not list(tmp_path.glob("*.part"))


def test_failed_forced_download_preserves_existing_image(tmp_path):
    entry = _entry(b"original")
    target = tmp_path / entry["filename"]
    target.write_bytes(b"original")
    with pytest.raises(ValueError, match="mismatch"):
        fetch_image(entry, tmp_path, force=True, opener=lambda url: _Response(b"incorrect"))
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.part"))
