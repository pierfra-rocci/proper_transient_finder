# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Tests for the standalone tiled survey-reference preparer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def _science(
    path: Path, size: int, scale_arcsec: float, *, wcs_in_empty_primary: bool = False
) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "r"
    header["BUNIT"] = "adu"
    if wcs_in_empty_primary:
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(np.ones((size, size)), name="SCI"),
            ]
        ).writeto(path, checksum=True)
    else:
        fits.writeto(path, np.ones((size, size)), header, checksum=True)


def _tile(path: Path, data: np.ndarray, variance: np.ndarray | None) -> None:
    _science(path, data.shape[0], 1)
    header = fits.getheader(path)
    hdus = [fits.PrimaryHDU(data, header)]
    if variance is not None:
        hdus.append(fits.ImageHDU(variance, name="VAR"))
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)


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
    _science(science, 1000, 1, wcs_in_empty_primary=True)

    plan = plan_survey_mosaic(science, "r", survey="panstarrs")

    assert plan.mode == "crop"
    assert plan.crop_bounds is not None
    assert plan.size_deg > plan.tile_limit_deg


@pytest.mark.parametrize("wcs_in_empty_primary", [False, True])
def test_preparer_crops_small_overrun_without_editing_input(
    tmp_path, monkeypatch, wcs_in_empty_primary
) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 1000, 1, wcs_in_empty_primary=wcs_in_empty_primary)
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
    header, (height, width) = survey_mosaic._science_header_and_shape(Path(report["science_fits"]))
    original_header, _ = survey_mosaic._science_header_and_shape(science)
    original_center = WCS(original_header).pixel_to_world(499.5, 499.5)
    cropped_center = WCS(header).pixel_to_world((width - 1) / 2, (height - 1) / 2)
    assert original_center.separation(cropped_center).arcsec < 1
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


@pytest.mark.parametrize("science_has_wcs", [False, True])
def test_crop_preserves_primary_and_science_extension_wcs(tmp_path, science_has_wcs) -> None:
    from proper_transient_finder.survey_mosaic import _crop_science, _science_header_and_shape

    source, cropped = tmp_path / "source.fits", tmp_path / "cropped.fits"
    _science(source, 100, 1, wcs_in_empty_primary=True)
    with fits.open(source, mode="update") as hdus:
        if science_has_wcs:
            header = hdus[0].header.copy()
            header.update(hdus["SCI"].header)
            hdus["SCI"].header.update(WCS(header).to_header())
        hdus.append(fits.ImageHDU(np.full((100, 100), 4.0), name="VAR"))
        hdus.append(fits.ImageHDU(np.zeros((100, 100), dtype=np.uint8), name="MASK"))

    _crop_science(source, cropped, (10, 90, 20, 90))

    original_header, _ = _science_header_and_shape(source)
    cropped_header, shape = _science_header_and_shape(cropped)
    original_pixels = np.array([[20, 10], [45, 31], [89, 89]], dtype=float)
    cropped_pixels = original_pixels - [20, 10]
    np.testing.assert_allclose(
        WCS(cropped_header).all_pix2world(cropped_pixels, 0),
        WCS(original_header).all_pix2world(original_pixels, 0),
        rtol=0,
        atol=1e-10,
    )
    assert shape == (80, 70)
    with fits.open(cropped) as hdus:
        assert hdus[0].header["CRPIX1"] == pytest.approx(original_header["CRPIX1"] - 20)
        assert hdus[0].header["CRPIX2"] == pytest.approx(original_header["CRPIX2"] - 10)
        assert hdus["VAR"].data.shape == hdus["MASK"].data.shape == shape
        if science_has_wcs:
            assert hdus["SCI"].header["CRPIX1"] == pytest.approx(original_header["CRPIX1"] - 20)
            assert hdus["SCI"].header["CRPIX2"] == pytest.approx(original_header["CRPIX2"] - 10)


def test_mosaic_variance_uses_only_valid_science_contributors(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import _write_mosaic

    science = tmp_path / "science.fits"
    _science(science, 32, 1)
    partial = np.full((80, 80), 10.0)
    partial[25:55, 25:55] = np.nan
    first, second = tmp_path / "partial.fits", tmp_path / "valid.fits"
    _tile(first, partial, np.ones((80, 80)))
    _tile(second, np.full((80, 80), 10.0), np.full((80, 80), 100.0))

    destination = tmp_path / "mosaic.fits"
    report = _write_mosaic([first, second], fits.getheader(science), (32, 32), destination)

    assert report["variance_plane"]
    with fits.open(destination) as hdus:
        row, column = np.array(hdus[0].data.shape) // 2
        assert hdus[0].data[row, column] == pytest.approx(10)
        assert hdus["VAR"].data[row, column] == pytest.approx(100)
        assert hdus["MASK"].data[row, column] == 0


def test_mosaic_rejects_missing_variance_for_valid_science(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import _write_mosaic

    science = tmp_path / "science.fits"
    _science(science, 32, 1)
    variance = np.ones((80, 80))
    variance[25:55, 25:55] = np.nan
    first, second = tmp_path / "partial.fits", tmp_path / "valid.fits"
    _tile(first, np.full((80, 80), 10.0), variance)
    _tile(second, np.full((80, 80), 10.0), np.full((80, 80), 100.0))

    destination = tmp_path / "mosaic.fits"
    with pytest.raises(ValueError, match="variance is missing or invalid"):
        _write_mosaic([first, second], fits.getheader(science), (32, 32), destination)
    assert not destination.exists()


def test_mosaic_reports_when_not_all_tiles_have_variance(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import _write_mosaic

    science = tmp_path / "science.fits"
    _science(science, 32, 1)
    first, second = tmp_path / "with-var.fits", tmp_path / "without-var.fits"
    _tile(first, np.full((80, 80), 10.0), np.ones((80, 80)))
    _tile(second, np.full((80, 80), 10.0), None)

    destination = tmp_path / "mosaic.fits"
    report = _write_mosaic([first, second], fits.getheader(science), (32, 32), destination)

    assert not report["variance_plane"]
    with fits.open(destination) as hdus:
        assert "VAR" not in hdus
