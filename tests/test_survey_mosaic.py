# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Tests for the standalone tiled survey-reference preparer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _science(path: Path, size: int, scale_arcsec: float) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "r"
    header["BUNIT"] = "adu"
    fits.writeto(path, np.ones((size, size)), header, checksum=True)


def _fake_fetch(tmp_path: Path):
    from proper_transient_finder.survey import SurveyTemplate

    def fetch(request, cache_dir, *, timeout, max_download_mb):
        key = hashlib.sha256(repr(request).encode()).hexdigest()
        path = tmp_path / f"tile-{key}.fits"
        if not path.exists():
            # Ten arcsec pixels keep the fixture small while retaining the
            # request's true angular size and WCS geometry.
            size = max(8, int(np.ceil(request.size_deg * 360))) + 4
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
            wcs.wcs.crval = [request.ra_deg, request.dec_deg]
            wcs.wcs.cdelt = [-10 / 3600, 10 / 3600]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            header = wcs.to_header()
            header["FILTER"] = request.template_filter
            header["BUNIT"] = "adu"
            data = np.full((size, size), 7.0)
            fits.HDUList(
                [
                    fits.PrimaryHDU(data, header),
                    fits.ImageHDU(np.full((size, size), 4.0), name="VAR"),
                    fits.ImageHDU(np.zeros((size, size), dtype=np.uint8), name="MASK"),
                ]
            ).writeto(path, checksum=True)
        return SurveyTemplate(path, {"native_bunit": "adu", "image_id": path.name})

    return fetch


def test_plan_uses_centred_crop_for_small_single_tile_overrun(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import plan_survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 1000, 1)

    plan = plan_survey_mosaic(science, "r", survey="panstarrs")

    assert plan.mode == "crop"
    assert plan.crop_bounds is not None
    assert plan.size_deg > plan.tile_limit_deg


def test_preparer_crops_small_overrun_without_editing_input(tmp_path, monkeypatch) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 1000, 1)
    original = science.read_bytes()
    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", _fake_fetch(tmp_path))

    report = survey_mosaic.prepare_survey_mosaic(
        science,
        tmp_path / "prepared",
        template_filter="r",
        survey="panstarrs",
        cache_dir=tmp_path / "cache",
    )

    assert report["plan"]["mode"] == "crop"
    assert science.read_bytes() == original
    assert fits.getdata(report["science_fits"]).shape[0] < 1000
    assert Path(report["reference_fits"]).is_file()


def test_preparer_tiles_and_mosaics_large_field(tmp_path, monkeypatch) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 180, 10)
    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", _fake_fetch(tmp_path))

    report = survey_mosaic.prepare_survey_mosaic(
        science,
        tmp_path / "prepared",
        template_filter="r",
        survey="panstarrs",
        cache_dir=tmp_path / "cache",
    )

    assert report["plan"]["mode"] == "mosaic"
    assert report["acquisition"]["tiles"] == 4
    with fits.open(report["reference_fits"], memmap=False) as hdus:
        assert hdus[0].header["MOSAIC"]
        assert "MASK" in hdus and "VAR" in hdus
        assert np.isfinite(hdus[0].data).any()
        assert np.all(hdus["MASK"].data[np.isfinite(hdus[0].data)] == 0)
