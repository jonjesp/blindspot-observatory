"""Strict, checksum-verified reader for the original uncompressed 2-D ZTF science images."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from blindspot.config import DETECTOR

MIN_FRAME_SIDE_PX = 1600  # full-frame ZTF quadrants are ~3000 px; rejects stray cut-outs
MAX_FITS_BYTES = 128 * 1024 * 1024


def image_manifest() -> list[dict]:
    text = files("blindspot.data").joinpath("manifest.json").read_text(encoding="utf-8")
    return json.loads(text)["images"]


def _parse_card_value(text: str):
    """Parse the value part of an 80-char FITS card (quote-aware; handles '/' inside strings)."""
    text = text.strip()
    if text.startswith("'"):
        out, i = [], 1
        while i < len(text):
            if text[i] == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                break
            out.append(text[i])
            i += 1
        return "".join(out).rstrip()
    value = text.split("/")[0].strip()
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def decode_primary_fits(data: bytes, *, min_side: int = MIN_FRAME_SIDE_PX) -> tuple[np.ndarray, dict]:
    """Decode a primary-HDU 2-D float32 (BITPIX=-32) FITS image from raw bytes."""
    cards: list[str] = []
    offset = 0
    while True:
        if len(data) - offset < 2880:
            raise ValueError("Truncated FITS header")
        block = data[offset : offset + 2880]
        offset += 2880
        try:
            fresh = [block[i : i + 80].decode("ascii") for i in range(0, 2880, 80)]
        except UnicodeDecodeError as exc:
            raise ValueError("FITS header is not ASCII") from exc
        cards.extend(fresh)
        if any(card[:8].strip() == "END" for card in fresh):
            break
    header: dict = {}
    for card in cards:
        key = card[:8].strip()
        if key == "END":
            break
        if card[8:10] == "= ":
            header[key] = _parse_card_value(card[10:])
    if header.get("SIMPLE") not in ("T", True) or header.get("BITPIX") != -32 or header.get("NAXIS") != 2:
        raise ValueError("A primary 2-D FITS image with BITPIX=-32 is required")
    shape = (header.get("NAXIS2"), header.get("NAXIS1"))
    if any(isinstance(n, bool) or not isinstance(n, int) or n < min_side for n in shape):
        raise ValueError("Invalid FITS image shape")
    count = math.prod(shape)
    if len(data) - offset < count * 4:
        raise ValueError("Truncated FITS data")
    raw = np.frombuffer(data, dtype=">f4", count=count, offset=offset).reshape(shape)
    image = np.asarray(raw, dtype=np.float64)
    image = image * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    return image, header


def load_verified_image(archive: str | Path, field: str) -> tuple[np.ndarray, dict]:
    """Load one manifest image from a directory or ZIP, verifying its SHA-256 first."""
    known = {entry["field"]: entry for entry in image_manifest()}
    if field not in known:
        raise ValueError(f"Unknown field: {field}")
    entry = known[field]
    archive = Path(archive)
    if archive.is_dir():
        with (archive / entry["filename"]).open("rb") as handle:
            data = handle.read(MAX_FITS_BYTES + 1)
    elif archive.suffix.lower() == ".zip":
        with ZipFile(archive) as zf:
            names = [
                x.filename
                for x in zf.infolist()
                if not x.is_dir() and Path(x.filename).name == entry["filename"]
            ]
            if len(names) != 1:
                raise ValueError("FITS file missing or duplicated inside the ZIP")
            if zf.getinfo(names[0]).file_size > MAX_FITS_BYTES:
                raise ValueError("FITS file exceeds the supported size limit")
            with zf.open(names[0]) as handle:
                data = handle.read(MAX_FITS_BYTES + 1)
    else:
        raise ValueError("Provide a directory of FITS files or a ZIP archive")
    if len(data) > MAX_FITS_BYTES:
        raise ValueError("FITS file exceeds the supported size limit")
    actual = hashlib.sha256(data).hexdigest()
    if actual != entry["sha256"]:
        raise ValueError("Wrong FITS file: SHA-256 mismatch")
    image, header = decode_primary_fits(data)
    return image, {"field": field, "sha256": actual, "shape": list(image.shape), "fits_header": header}


def extract_patch(
    image: np.ndarray, offset_y: int, offset_x: int, size: int = DETECTOR.patch_size
) -> np.ndarray:
    """Cut a ``size`` x ``size`` patch centred ``(offset_y, offset_x)`` pixels from the image centre."""
    cy, cx = image.shape[0] // 2 + offset_y, image.shape[1] // 2 + offset_x
    y0, x0 = cy - size // 2, cx - size // 2
    if y0 < 0 or x0 < 0 or y0 + size > image.shape[0] or x0 + size > image.shape[1]:
        raise ValueError("Patch falls outside the image")
    return image[y0 : y0 + size, x0 : x0 + size]
