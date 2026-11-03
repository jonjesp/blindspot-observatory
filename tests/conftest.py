"""Shared FITS-input and historical-source test configuration."""

import os
from pathlib import Path

import pytest

FITS_ZIP = os.environ.get("BLINDSPOT_FITS_ZIP")
REFERENCE_ROOT = os.environ.get("BLINDSPOT_REFERENCE_ROOT")

requires_fits = pytest.mark.skipif(
    not FITS_ZIP or not Path(FITS_ZIP).exists(),
    reason="set BLINDSPOT_FITS_ZIP to a ZIP/directory with the two ZTF FITS files",
)

requires_reference = pytest.mark.skipif(
    not REFERENCE_ROOT or not Path(REFERENCE_ROOT).is_dir(),
    reason="set BLINDSPOT_REFERENCE_ROOT to the original frozen source directory",
)


@pytest.fixture
def fits_zip() -> Path:
    return Path(FITS_ZIP)


@pytest.fixture
def reference_root() -> Path:
    return Path(REFERENCE_ROOT).resolve()