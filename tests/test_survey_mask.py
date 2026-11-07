# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Regression tests for native Pan-STARRS survey masks."""

import hashlib
import json
from io import BytesIO
from urllib.parse import urlencode

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from proper_transient_finder import survey


def _header(size: int) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs.to_header()


def _payload(mask: np.ndarray, header: fits.Header) -> bytes:
    stream = BytesIO()
    fits.writeto(stream, mask, header)
    return stream.getvalue()


MASK_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?" + urlencode(
    {"red": "/test.mask.fits"}
)


def test_native_mask_aligns_pixels_and_masks_outside_native_image(monkeypatch):
    header = _header(4)
    native_header = header.copy()
    native_header["CRPIX1"] -= 1
    native = np.zeros((4, 4), dtype=np.uint16)
    native[2, 1] = 128
    monkeypatch.setattr(survey, "_download_url", lambda *args: _payload(native, native_header))

    mask, raw_url = survey._panstarrs_native_mask(MASK_URL, header, (4, 4), 30, 1_000_000)

    assert raw_url == "https://ps1images.stsci.edu/test.mask.fits"
    assert np.all(mask[:, 0] != 0)
    np.testing.assert_array_equal(mask[:, 1:], native[:, :3])
    assert mask[2, 2] == 128


def test_native_mask_rejects_non_aligned_wcs(monkeypatch):
    header = _header(4)
    native_header = header.copy()
    native_header["CRPIX1"] += 0.5
    monkeypatch.setattr(
        survey, "_download_url", lambda *args: _payload(np.zeros((4, 4), np.uint16), native_header)
    )
    with pytest.raises(ValueError, match="not pixel-aligned"):
        survey._panstarrs_native_mask(MASK_URL, header, (4, 4), 30, 1_000_000)


def test_native_mask_upgrades_verified_legacy_cache_without_redownloading_image(
    tmp_path, monkeypatch
):
    request = survey.TemplateRequest(
        tmp_path / "science.fits", "panstarrs", "r", "r", "r", 180, 20, 0.01, None
    )
    identity = request.to_dict()
    for field in ("science_fits", "science_filter", "native_science_filter"):
        identity.pop(field)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    legacy = tmp_path / f"{key}.fits"
    header = _header(4)
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((4, 4)), header),
            fits.ImageHDU(np.ones((4, 4)), name="VAR"),
            fits.ImageHDU(np.full((4, 4), np.nan), name="MASK"),
        ]
    ).writeto(legacy)
    legacy_hash = hashlib.sha256(legacy.read_bytes()).hexdigest()
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"sha256": legacy_hash, "mask_url": MASK_URL}), encoding="utf-8"
    )
    native = np.zeros((4, 4), np.uint16)
    native[1, 2] = 8
    downloads = []

    def download(url, timeout, limit):
        downloads.append(url)
        return _payload(native, header)

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(
        survey, "_discover_template", lambda *args: pytest.fail("Legacy image must be reused")
    )
    upgraded = survey.fetch_survey_template(request, tmp_path)
    assert upgraded.fits_path != legacy
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == legacy_hash
    assert len(downloads) == 1
    with fits.open(upgraded.fits_path) as hdus:
        np.testing.assert_array_equal(hdus["MASK"].data, native)
        np.testing.assert_array_equal(hdus["VAR"].data, np.ones((4, 4)))
    assert survey.fetch_survey_template(request, tmp_path).fits_path == upgraded.fits_path
    assert len(downloads) == 1
