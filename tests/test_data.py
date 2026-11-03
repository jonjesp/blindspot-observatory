"""FITS decoding, image identity and patch extraction tests."""

import zipfile

import numpy as np
import pytest

from blindspot.datasets import ztf
from blindspot.datasets.ztf import (
    _parse_card_value,
    decode_primary_fits,
    extract_patch,
    image_manifest,
    load_verified_image,
)


def card(text: str) -> bytes:
    return text.ljust(80).encode("ascii")


def make_fits(shape=(8, 8), extra=(), bitpix=-32) -> bytes:
    cards = [
        card("SIMPLE  =                    T"),
        card(f"BITPIX  = {bitpix:>20}"),
        card("NAXIS   =                    2"),
        card(f"NAXIS1  = {shape[1]:>20}"),
        card(f"NAXIS2  = {shape[0]:>20}"),
        *[card(c) for c in extra],
        card("END"),
    ]
    header = b"".join(cards).ljust(2880 * ((len(cards) * 80 + 2879) // 2880))
    data = np.arange(shape[0] * shape[1], dtype=">f4").tobytes()
    return header + data


def test_decode_roundtrip():
    image, header = decode_primary_fits(make_fits(), min_side=8)
    assert image.shape == (8, 8) and image[1, 0] == 8.0 and header["NAXIS1"] == 8


def test_header_strings_may_contain_slashes_and_quotes():
    assert _parse_card_value("'a/b'   / comment") == "a/b"
    assert _parse_card_value("'it''s'") == "it's"
    assert _parse_card_value("   42 / note") == 42
    assert _parse_card_value("1.5") == 1.5
    _, header = decode_primary_fits(make_fits(extra=["FILTER  = 'zr/x'  / band"]), min_side=8)
    assert header["FILTER"] == "zr/x"


def test_bscale_bzero_applied():
    image, _ = decode_primary_fits(
        make_fits(extra=["BSCALE  =                  2.0", "BZERO   =                 10.0"]), min_side=8
    )
    assert image[0, 1] == 12.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:2000],  # truncated header
        lambda b: b[:-8],  # truncated data
        lambda b: b.replace(b"BITPIX  =                  -32", b"BITPIX  =                   16"),
        lambda b: b"\xff" + b[1:],  # non-ASCII header
    ],
)
def test_malformed_fits_fails_closed(mutate):
    with pytest.raises(ValueError):
        decode_primary_fits(mutate(make_fits()), min_side=8)


def test_cutouts_below_minimum_side_are_rejected():
    with pytest.raises(ValueError):
        decode_primary_fits(make_fits())


def test_extract_patch_bounds_and_centre():
    image = np.arange(300 * 300, dtype=float).reshape(300, 300)
    patch = extract_patch(image, 0, 0, size=64)
    assert patch.shape == (64, 64) and patch[32, 32] == image[150, 150]
    assert np.array_equal(extract_patch(image, 10, -20, size=64)[32, 32:33], image[160, 130:131])
    with pytest.raises(ValueError):
        extract_patch(image, 200, 0, size=64)


def test_manifest_is_well_formed():
    entries = image_manifest()
    assert [e["field"] for e in entries] == ["E4_r", "E5_g"]
    assert all(len(e["sha256"]) == 64 and e["filename"].endswith("_sciimg.fits") for e in entries)


def test_wrong_file_with_correct_name_is_refused(tmp_path):
    entry = image_manifest()[0]
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(entry["filename"], make_fits())
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_image(archive, entry["field"])
    with pytest.raises(ValueError, match="Unknown field"):
        load_verified_image(archive, "nope")
    with pytest.raises(ValueError):
        load_verified_image(tmp_path / "data.tar", "E4_r")


def test_oversized_zip_member_is_rejected_before_decompression(tmp_path, monkeypatch):
    entry = image_manifest()[0]
    archive = tmp_path / "oversized.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(entry["filename"], b"1234")
    monkeypatch.setattr(ztf, "MAX_FITS_BYTES", 3)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Oversized members must not be decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_read)
    with pytest.raises(ValueError, match="size limit"):
        load_verified_image(archive, entry["field"])


def test_oversized_directory_input_is_rejected(tmp_path, monkeypatch):
    entry = image_manifest()[0]
    (tmp_path / entry["filename"]).write_bytes(b"1234")
    monkeypatch.setattr(ztf, "MAX_FITS_BYTES", 3)
    with pytest.raises(ValueError, match="size limit"):
        load_verified_image(tmp_path, entry["field"])


def test_extreme_fits_dimensions_fail_without_integer_overflow():
    payload = make_fits(extra=[f"NAXIS1  = {2**62:>20}", f"NAXIS2  = {2**62:>20}"])
    with pytest.raises(ValueError, match="Truncated FITS data"):
        decode_primary_fits(payload, min_side=8)
