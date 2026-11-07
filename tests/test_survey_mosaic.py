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


def test_preparer_overwrite_is_opt_in_and_preserves_unrelated_files(tmp_path, monkeypatch) -> None:
    import json

    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "science_for_reference.fits").write_bytes(b"old science")
    (output / "survey_reference.fits").write_bytes(b"old reference")
    (output / "survey_mosaic.json").write_text('{"state":"old"}', encoding="utf-8")
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", _fake_fetch(tmp_path))

    options = {"template_filter": "r", "survey": "panstarrs", "cache_dir": tmp_path / "cache"}
    with pytest.raises(FileExistsError, match="overwrite"):
        survey_mosaic.prepare_survey_mosaic(science, output, **options)
    assert (output / "survey_reference.fits").read_bytes() == b"old reference"

    report = survey_mosaic.prepare_survey_mosaic(science, output, overwrite=True, **options)
    assert report["science_fits"] == str(output / "science_for_reference.fits")
    assert report["reference_fits"] == str(output / "survey_reference.fits")
    assert fits.getdata(report["reference_fits"]).shape
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert json.loads((output / "survey_mosaic.json").read_text(encoding="utf-8")) == report
    assert not list(output.glob(".survey-mosaic-*"))


def test_failed_preparer_overwrite_preserves_previous_products(tmp_path, monkeypatch) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    output = tmp_path / "prepared"
    output.mkdir()
    old_files = {
        "science_for_reference.fits": b"old science",
        "survey_reference.fits": b"old reference",
        "survey_mosaic.json": b'{"state":"completed"}',
    }
    for name, payload in old_files.items():
        (output / name).write_bytes(payload)

    def fail(*args, **kwargs):
        raise TimeoutError("mock survey timeout")

    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", fail)
    with pytest.raises(TimeoutError, match="mock survey timeout"):
        survey_mosaic.prepare_survey_mosaic(
            science,
            output,
            template_filter="r",
            survey="panstarrs",
            cache_dir=tmp_path / "cache",
            overwrite=True,
        )
    assert {name: (output / name).read_bytes() for name in old_files} == old_files
    assert not list(output.glob(".survey-mosaic-*"))


def test_preparer_overwrite_marks_failed_promotion(tmp_path, monkeypatch) -> None:
    import json

    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "survey_reference.fits").write_bytes(b"old reference")
    (output / "survey_mosaic.json").write_text('{"state":"completed"}', encoding="utf-8")
    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", _fake_fetch(tmp_path))
    replace = Path.replace

    def fail_reference_replace(path, target):
        if Path(target) == output / "survey_reference.fits":
            raise PermissionError("mock locked reference")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_reference_replace)
    with pytest.raises(PermissionError, match="mock locked reference"):
        survey_mosaic.prepare_survey_mosaic(
            science,
            output,
            template_filter="r",
            survey="panstarrs",
            cache_dir=tmp_path / "cache",
            overwrite=True,
        )
    report = json.loads((output / "survey_mosaic.json").read_text(encoding="utf-8"))
    assert report["state"] == "failed"
    assert report["output_written"] is False
    assert (output / "survey_reference.fits").read_bytes() == b"old reference"


def test_preparer_cli_overwrite_reuses_output_directory(tmp_path, monkeypatch, capsys) -> None:
    import json

    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "survey_mosaic.json").write_text('{"state":"failed"}', encoding="utf-8")
    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", _fake_fetch(tmp_path))
    assert (
        survey_mosaic.main(
            [
                "--science",
                str(science),
                "--output",
                str(output),
                "--template-filter",
                "r",
                "--cache",
                str(tmp_path / "cache"),
                "--overwrite",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["output_written"] is True
    assert report["reference_fits"] == str(output / "survey_reference.fits")


def test_preparer_overwrite_refuses_directory_at_product_path(tmp_path) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    output = tmp_path / "prepared"
    (output / "survey_reference.fits").mkdir(parents=True)
    marker = output / "survey_reference.fits" / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="regular file"):
        survey_mosaic.prepare_survey_mosaic(
            science,
            output,
            template_filter="r",
            cache_dir=tmp_path / "cache",
            overwrite=True,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("wcs_in_empty_primary", [False, True])
def test_opt_in_border_crop_plans_before_download_and_preserves_input(
    tmp_path, monkeypatch, wcs_in_empty_primary
) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1, wcs_in_empty_primary=wcs_in_empty_primary)
    original = science.read_bytes()
    options = {
        "template_filter": "r",
        "survey": "panstarrs",
        "cache_dir": tmp_path / "cache",
        "border_crop_fraction": 0.05,
    }
    preflight = survey_mosaic.prepare_survey_mosaic(
        science, tmp_path / "prepared", check_only=True, **options
    )
    assert preflight["plan"]["border_crop_bounds"] == (5, 95, 5, 95)
    assert not (tmp_path / "prepared").exists()
    fake_fetch = _fake_fetch(tmp_path)

    def fetch(request, cache_dir, **kwargs):
        _, shape = survey_mosaic._science_header_and_shape(request.science_fits)
        assert shape == (90, 90)
        return fake_fetch(request, cache_dir, **kwargs)

    monkeypatch.setattr(survey_mosaic, "fetch_survey_template", fetch)
    result = survey_mosaic.prepare_survey_mosaic(science, tmp_path / "prepared", **options)
    assert science.read_bytes() == original
    assert result["plan"] == preflight["plan"]
    assert survey_mosaic._science_header_and_shape(Path(result["science_fits"]))[1] == (90, 90)
    assert any("cropped" in warning for warning in result["warnings"])


def test_border_crop_does_not_trigger_additional_automatic_crop(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import plan_survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 1100, 1)
    plan = plan_survey_mosaic(science, "r", survey="panstarrs", border_crop_fraction=0.05)

    assert plan.tile_limit_deg < plan.size_deg < plan.tile_limit_deg * 1.05
    assert plan.mode == "mosaic"
    assert plan.crop_bounds is None
    assert plan.border_crop_bounds == (55, 1045, 55, 1045)


def test_cli_preflight_reports_opt_in_crop_and_missing_limit(tmp_path, capsys) -> None:
    import json

    from proper_transient_finder.survey_mosaic import main

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    assert (
        main(
            [
                "--science",
                str(science),
                "--output",
                str(tmp_path / "output"),
                "--template-filter",
                "r",
                "--crop-border-percent",
                "--max-missing-fraction",
                "0.1",
                "--check-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["plan"]["border_crop_bounds"] == [5, 95, 5, 95]
    assert report["max_missing_fraction"] == 0.1
    assert any("single-cutout coverage" in warning for warning in report["warnings"])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fraction", [-0.01, 0.251, float("nan")])
def test_border_crop_rejects_invalid_fraction_before_writes(tmp_path, fraction) -> None:
    from proper_transient_finder.survey_mosaic import prepare_survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    with pytest.raises(ValueError, match="border_crop_fraction"):
        prepare_survey_mosaic(
            science, tmp_path / "output", template_filter="r", border_crop_fraction=fraction
        )
    assert not (tmp_path / "output").exists()


def test_partial_mosaic_coverage_enforces_exact_fraction_and_masks_gaps(tmp_path) -> None:
    from proper_transient_finder.survey_mosaic import _write_mosaic

    science = tmp_path / "science.fits"
    _science(science, 32, 1)
    data = np.full((80, 80), 10.0)
    data[39:41, 39:41] = np.nan
    tile = tmp_path / "tile.fits"
    _tile(tile, data, np.ones((80, 80)))
    header = fits.getheader(science)

    with pytest.raises(ValueError, match="complete science footprint"):
        _write_mosaic([tile], header, (32, 32), tmp_path / "strict.fits")
    report = _write_mosaic(
        [tile], header, (32, 32), tmp_path / "partial.fits", max_missing_fraction=0.1
    )
    fraction = report["missing_science_fraction"]
    assert 0 < fraction < 0.1
    assert report["missing_science_pixels"] / 1024 == fraction
    with fits.open(tmp_path / "partial.fits") as hdus:
        assert np.any(hdus["MASK"].data != 0)
        assert np.all(np.isnan(hdus[0].data[hdus["MASK"].data != 0]))
    _write_mosaic(
        [tile], header, (32, 32), tmp_path / "boundary.fits", max_missing_fraction=fraction
    )
    with pytest.raises(ValueError, match="permitted maximum"):
        _write_mosaic(
            [tile],
            header,
            (32, 32),
            tmp_path / "rejected.fits",
            max_missing_fraction=fraction - 1 / 1024,
        )
    assert not (tmp_path / "rejected.fits").exists()


@pytest.mark.parametrize("missing", [10, 11])
def test_partial_mosaic_enforces_ten_percent_boundary(tmp_path, monkeypatch, missing) -> None:
    from proper_transient_finder import survey_mosaic

    science = tmp_path / "science.fits"
    tile = tmp_path / "tile.fits"
    _science(science, 10, 1)
    _tile(tile, np.ones((10, 10)), None)
    header = fits.getheader(science)
    monkeypatch.setattr(survey_mosaic, "_mosaic_wcs", lambda wcs, shape, scale: (wcs, shape))
    calls = 0

    def project(source, output_wcs, *, shape_out, order):
        nonlocal calls
        calls += 1
        data = np.array(source[0], dtype=float, copy=True)
        if calls == 1:
            data.flat[:missing] = np.nan
        return data, np.ones(shape_out)

    monkeypatch.setattr(survey_mosaic, "reproject_interp", project)
    destination = tmp_path / "result.fits"
    if missing == 10:
        report = survey_mosaic._write_mosaic(
            [tile], header, (10, 10), destination, max_missing_fraction=0.1
        )
        assert report["missing_science_fraction"] == 0.1
        with fits.open(destination) as hdus:
            assert np.count_nonzero(hdus["MASK"].data) == 10
    else:
        with pytest.raises(ValueError, match="11.00%"):
            survey_mosaic._write_mosaic(
                [tile], header, (10, 10), destination, max_missing_fraction=0.1
            )
        assert not destination.exists()


@pytest.mark.parametrize("fraction", [-0.01, 0.101, float("nan")])
def test_missing_coverage_rejects_invalid_fraction_before_writes(tmp_path, fraction) -> None:
    from proper_transient_finder.survey_mosaic import prepare_survey_mosaic

    science = tmp_path / "science.fits"
    _science(science, 100, 1)
    with pytest.raises(ValueError, match="max_missing_fraction"):
        prepare_survey_mosaic(
            science, tmp_path / "output", template_filter="r", max_missing_fraction=fraction
        )
    assert not (tmp_path / "output").exists()


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
        _write_mosaic(
            [first, second],
            fits.getheader(science),
            (32, 32),
            destination,
            max_missing_fraction=0.1,
        )
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
