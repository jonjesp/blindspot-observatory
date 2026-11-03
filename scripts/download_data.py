"""Checksum-verified retrieval of the two public ZTF science images from IRSA.

The default destination is data/ztf/. Image identities are specified in the packaged manifest.
Checksum-mismatched downloads are rejected without replacing existing image files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

from blindspot.datasets.ztf import image_manifest

IRSA_SCI_ROOT = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/"
CHUNK = 1 << 20
ACKNOWLEDGEMENT = (
    "ZTF data: please acknowledge ZTF and cite Masci et al., 'The Zwicky Transient Facility: "
    "Data Processing, Products, and Archive' (PASP). Full text: docs/DATA.md#acknowledging-ztf."
)

Opener = Callable[[str], "object"]


def image_url(filename: str) -> str:
    """IRSA URL of a ``..._sciimg.fits`` file: ``sci/YYYY/MMDD/fracday/<filename>``."""
    parts = filename.split("_")
    if len(parts) < 3 or parts[0] != "ztf" or not parts[1].isdigit() or len(parts[1]) < 9:
        raise ValueError(f"Not a ZTF science-image filename: {filename}")
    filefracday = parts[1]
    return f"{IRSA_SCI_ROOT}{filefracday[:4]}/{filefracday[4:8]}/{filefracday[8:]}/{filename}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_url(url: str):
    request = Request(url, headers={"User-Agent": "blindspot-observatory/0.4 (data download)"})
    return urlopen(request, timeout=120)


def _stream(opener: Opener, url: str, partial: Path) -> str:
    """Write the response to ``partial`` and return its SHA-256."""
    digest = hashlib.sha256()
    with opener(url) as response, partial.open("wb") as handle:
        for block in iter(lambda: response.read(CHUNK), b""):
            digest.update(block)
            handle.write(block)
    return digest.hexdigest()


def fetch_image(
    entry: dict,
    dest: Path,
    *,
    opener: Opener = _open_url,
    force: bool = False,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Download one manifest image into ``dest``; return ``"cached"`` or ``"downloaded"``.

    Network errors (resets, timeouts) are retried with growing pauses. A checksum mismatch is
    never retried: it means the bytes differ, not that the connection failed.
    """
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    target = dest / entry["filename"]
    if target.exists() and not force:
        if sha256_file(target) == entry["sha256"]:
            return "cached"
        raise ValueError(f"{target} exists but its SHA-256 differs; remove it or pass --force")
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = image_url(entry["filename"])
    try:
        for attempt in range(1, attempts + 1):
            try:
                actual = _stream(opener, url, partial)
                break
            except OSError as exc:
                if attempt == attempts:
                    raise
                pause = 2**attempt
                note = f"{exc.__class__.__name__}; retry {attempt}/{attempts - 1} in {pause}s"
                print(f"  {note}", file=sys.stderr)
                sleep(pause)
        if actual != entry["sha256"]:
            raise ValueError(
                f"{entry['filename']}: SHA-256 mismatch (got {actual[:16]}..., "
                f"expected {entry['sha256'][:16]}...). IRSA may have reprocessed this file."
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=Path("data") / "ztf")
    parser.add_argument("--force", action="store_true", help="download again even if present")
    parser.add_argument("--print-urls", action="store_true", help="only print the IRSA URLs and exit")
    args = parser.parse_args(argv)
    if args.print_urls:
        for entry in image_manifest():
            print(f"{entry['filename']}\t{image_url(entry['filename'])}")
        return 0
    for entry in image_manifest():
        try:
            state = fetch_image(entry, args.dest, force=args.force)
        except (OSError, ValueError) as exc:
            print(f"error: {entry['field']}: {exc}", file=sys.stderr)
            print(
                "The download failed. Open the URL printed by `--print-urls` in a browser and save the "
                f"file into {args.dest}/, or point BLINDSPOT_FITS_ZIP at an existing ZIP or directory. "
                "The loader verifies the SHA-256 either way.",
                file=sys.stderr,
            )
            return 1
        print(f"{entry['field']}: {state} -> {args.dest / entry['filename']}")
    print(ACKNOWLEDGEMENT)
    print(f"\nNext: set BLINDSPOT_FITS_ZIP={args.dest} (a directory is accepted) and run the tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
