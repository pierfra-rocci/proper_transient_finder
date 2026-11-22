# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Regression tests for the CDS HiPS survey-template backend."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from proper_transient_finder import survey


def science_image(
    tmp_path,
    *,
    shape: tuple[int, int] = (64, 64),
    crval: tuple[float, float] = (180.0, 20.0),
    name: str = "science.fits",
    band: str = "r",
):
    """Write a small WCS science FITS and return its path and WCS."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header.update(FILTER=band, BUNIT="adu")
    path = tmp_path / name
    fits.writeto(path, np.ones(shape), header)
    return path, wcs


def payload(data: np.ndarray, header: fits.Header) -> bytes:
    stream = BytesIO()
    fits.writeto(stream, data, header)
    return stream.getvalue()


def shifted_header(wcs):
    header = wcs.to_header()
    header["CRPIX1"] += 0.5
    return header


@pytest.mark.parametrize(
    ("survey_name", "band", "asinh", "declination"),
    [
        ("panstarrs", "r", True, 20.0),
        ("panstarrs", "g", False, 20.0),
        ("skymapper", "r", False, -45.0),
    ],
)
def test_hips_template_linearizes_only_panstarrs_asinh_bands(
    tmp_path, monkeypatch, survey_name, band, asinh, declination
):
    science, wcs = science_image(tmp_path, crval=(180.0, declination))
    request = survey.prepare_template_request(science, band, survey=survey_name)
    monkeypatch.setattr(
        survey,
        "_download_url",
        lambda *args: payload(np.ones((64, 64)), wcs.to_header()),
    )
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    with fits.open(template.fits_path) as hdus:
        data = hdus[0].data
        assert bool(hdus[0].header["HIPSASIN"]) is asinh
    if asinh:
        expected = np.exp(0.4 * np.log(10)) - np.exp(-0.4 * np.log(10))
        np.testing.assert_allclose(data, expected)
    else:
        np.testing.assert_allclose(data, 1.0)
    assert template.metadata["asinh_linearized"] is asinh
    assert template.metadata["variance_source"] == "none"


def test_hips_template_masks_non_finite_linearized_pixels(tmp_path, monkeypatch):
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    image = np.ones((64, 64))
    image[0, 0] = np.nan
    image[1, 1] = np.inf
    monkeypatch.setattr(survey, "_download_url", lambda *args: payload(image, wcs.to_header()))
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    with fits.open(template.fits_path) as hdus:
        assert hdus["MASK"].data[0, 0]
        assert hdus["MASK"].data[1, 1]
        assert not hdus["MASK"].data[2, 2]
        assert np.isfinite(hdus[0].data[2, 2])


@pytest.mark.parametrize(
    ("mutate", "shape", "message"),
    [
        (lambda header, wcs: header, (32, 32), "shape"),
        (lambda header, wcs: shifted_header(wcs), (64, 64), "pixel grid"),
    ],
)
def test_hips_template_rejects_a_different_grid(tmp_path, monkeypatch, mutate, shape, message):
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    returned = mutate(wcs.to_header(), wcs)
    monkeypatch.setattr(survey, "_download_url", lambda *args: payload(np.ones(shape), returned))
    with pytest.raises(ValueError, match=message):
        survey.fetch_survey_template(request, tmp_path / "cache")


def test_hips_download_falls_back_to_the_second_service_host(tmp_path, monkeypatch):
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if len(calls) == 1:
            raise URLError("first service unavailable")
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    assert len(calls) == 2
    assert "alaskybis.cds.unistra.fr" in calls[0]
    assert "alasky.u-strasbg.fr" in calls[1]
    assert template.metadata["query_url"] == calls[1]


def test_hips_download_retries_all_service_hosts_before_giving_up(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey="CDS/P/PanSTARRS/DR1/r")
    calls = []
    delays = []

    def download(url, timeout, limit):
        calls.append(url)
        raise URLError("service unavailable")

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", delays.append)
    with pytest.raises(URLError, match="every service host"):
        survey.fetch_survey_template(request, tmp_path / "cache")
    hips_calls = [url for url in calls if "/hips-image-services/" in url]
    hosts = [urlsplit(url).hostname for url in hips_calls]
    expected = [urlsplit(base).hostname for base in survey.HIPS_SERVICE_URLS]
    assert hosts == expected * survey.SERVICE_ATTEMPTS
    assert delays == [survey.HIPS_RETRY_DELAY_SECONDS] * (survey.SERVICE_ATTEMPTS - 1)


def test_survey_options_deliver_a_grid_aligned_template_without_reprojection():
    options = survey._survey_options(config={"plots": False})
    assert options.reproject is False
    assert options.reproject_units == "flux-per-pixel"
    aligned = survey._survey_options(config={"align": True})
    assert aligned.align is True
    assert aligned.reproject is False
    with pytest.raises(ValueError, match="kernel"):
        survey._survey_options(config={"method": "proper"})


@pytest.mark.parametrize(
    ("identifier", "band"),
    [
        ("CDS/P/DESI-Legacy-Surveys/DR10/r", "r"),
        ("CDS/P/NEOWISER/W1", "W1"),
    ],
)
def test_explicit_hips_identifier_accepts_documented_multi_character_bands(
    tmp_path, identifier, band
):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, band, survey=identifier)
    assert request.hips_id == identifier
    assert request.template_filter == band.lower()


@pytest.mark.parametrize(
    ("identifier", "band"),
    [
        ("PanSTARRS/DR1/i", "r"),
        ("CDS/P/NEOWISER/W2", "W1"),
    ],
)
def test_explicit_hips_identifier_band_must_match_template_filter(tmp_path, identifier, band):
    science, _ = science_image(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        survey.prepare_template_request(science, band, survey=identifier)


@pytest.mark.parametrize("band", ["", "W 1", "W/1"])
def test_explicit_hips_identifier_requires_a_clean_band_label(tmp_path, band):
    science, _ = science_image(tmp_path)
    with pytest.raises(ValueError, match="band label"):
        survey.prepare_template_request(science, band, survey="CDS/P/DESI-Legacy-Surveys/DR10/r")


def test_hips_service_urls_are_restricted_to_approved_hosts():
    for host in survey.SURVEY_HOSTS:
        survey._validate_url(f"https://{host}/hips2fits")
    for base in survey.HIPS_SERVICE_URLS:
        assert urlsplit(base).hostname in survey.SURVEY_HOSTS
    with pytest.raises(ValueError, match="approved HTTPS"):
        survey._validate_url("https://alasky.cds.unistra.fr.example.org/hips2fits")


def test_moc_coverage_precheck_accepts_covered_explicit_identifier(tmp_path, monkeypatch):
    identifier = "CDS/P/DESI-Legacy-Surveys/DR10/r"
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey=identifier)
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "/MocServer/query" in url:
            return json.dumps([{"ID": identifier}]).encode()
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    assert "/MocServer/query" in calls[0]
    query = parse_qs(urlsplit(calls[0]).query)
    assert query["ID"] == [identifier]
    assert query["intersect"] == ["enclosed"]
    assert query["get"] == ["record"]
    assert query["fields"] == ["ID"]
    assert query["stc"][0].startswith("Polygon ")
    assert template.metadata["coverage_check"]["status"] == "covered"
    assert template.metadata["coverage_check"]["covered"] is True
    assert template.metadata["coverage_check"]["polygon_vertices"] > 4
    assert "hips2fits" in calls[-1]


def test_moc_coverage_precheck_stops_before_download_when_not_covered(tmp_path, monkeypatch):
    identifier = "CDS/P/DESI-Legacy-Surveys/DR10/r"
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey=identifier)
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        return b"[]"

    monkeypatch.setattr(survey, "_download_url", download)
    with pytest.raises(ValueError, match="coverage pre-check failed"):
        survey.fetch_survey_template(request, tmp_path / "cache")
    assert calls and all("/MocServer/query" in url for url in calls)
    assert not list((tmp_path / "cache").glob("*.hips.fits"))


def test_moc_coverage_precheck_warns_and_continues_when_service_is_unavailable(
    tmp_path, monkeypatch
):
    identifier = "CDS/P/DESI-Legacy-Surveys/DR10/r"
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey=identifier)
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "/MocServer/query" in url:
            raise URLError("MOC service unavailable")
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    coverage = template.metadata["coverage_check"]
    assert coverage["status"] == "unavailable"
    assert coverage["covered"] is None
    assert "MOC service unavailable" in coverage["error"]
    assert "coverage pre-check was unavailable" in " ".join(template.metadata["warnings"])
    assert "hips2fits" in calls[-1]


def test_moc_coverage_precheck_retries_all_service_hosts_before_giving_up(tmp_path, monkeypatch):
    identifier = "CDS/P/DESI-Legacy-Surveys/DR10/r"
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey=identifier)
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "/MocServer/query" in url:
            raise URLError("MOC service unavailable")
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    hosts = [urlsplit(url).hostname for url in calls if "/MocServer/query" in url]
    expected = [urlsplit(base).hostname for base in survey.MOC_SERVICE_URLS]
    assert hosts == expected * survey.SERVICE_ATTEMPTS
    assert template.metadata["coverage_check"]["status"] == "unavailable"


def test_moc_coverage_precheck_can_be_explicitly_disabled(tmp_path, monkeypatch):
    identifier = "CDS/P/DESI-Legacy-Surveys/DR10/r"
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey=identifier)
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        assert "/MocServer/query" not in url
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache", check_moc_coverage=False)
    coverage = template.metadata["coverage_check"]
    assert len(calls) == 1
    assert coverage["status"] == "skipped"
    assert coverage["covered"] is None
    assert "explicitly disabled" in " ".join(template.metadata["warnings"])


def test_named_provider_skips_the_moc_coverage_precheck(tmp_path, monkeypatch):
    science, wcs = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r", survey="panstarrs")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    assert len(calls) == 1
    assert "hips2fits" in calls[0]
    assert template.metadata["coverage_check"]["status"] == "not_required"
    assert template.metadata["coverage_check"]["covered"] is None


def test_survey_results_directory_is_replaced_only_with_overwrite(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    stale = results / "survey_reference.json"
    stale.write_text("{}", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("stop before acquisition")

    monkeypatch.setattr(survey, "fetch_survey_template", fail)
    options = {
        "template_filter": "r",
        "cache_dir": tmp_path / "cache",
        "template_units": "adu",
        "check_moc_coverage": False,
    }
    with pytest.raises(FileExistsError, match="--overwrite"):
        survey.compare_survey_images(science, results, **options)
    assert stale.read_text(encoding="utf-8") == "{}"
    with pytest.raises(RuntimeError, match="stop before acquisition"):
        survey.compare_survey_images(science, results, config={"overwrite": True}, **options)
    report = json.loads(stale.read_text(encoding="utf-8"))
    assert report["state"] == "failed"
    assert report["warnings"]


def provider_payload(
    *,
    shape: tuple[int, int] = (64, 64),
    crval: tuple[float, float] = (180.0, 20.0),
    pixel_scale_arcsec: float = 1.0,
):
    """Build a provider-like cutout on its own tangent grid."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return payload(np.ones(shape), wcs.to_header())


def test_panstarrs_fallback_reprojects_after_every_hips2fits_host_fails(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/ps1filenames.py" in url:
            return (
                b"projcell,subcell,ra,dec,filter,mjd,type,filename,shortname,badflag\n"
                b"1806,4,180.0,20.0,r,0.0,stack,"
                b"/rings.v3.skycell/1806/004/test.stk.r.fits,"
                b"test.stk.r.fits,0\n"
            )
        return provider_payload()

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    template = survey.fetch_survey_template(request, tmp_path / "cache")

    hips_hosts = [urlsplit(base).hostname for base in survey.HIPS_SERVICE_URLS]
    attempts = len(hips_hosts) * survey.SERVICE_ATTEMPTS
    attempted_hosts = [urlsplit(url).hostname for url in calls[:attempts]]
    assert attempted_hosts == hips_hosts * survey.SERVICE_ATTEMPTS
    assert len(calls) == attempts + 2
    cutout_url = calls[-1]
    query = parse_qs(urlsplit(cutout_url).query)
    assert urlsplit(cutout_url).hostname == "ps1images.stsci.edu"
    assert query["format"] == ["fits"]
    assert query["red"] == ["/rings.v3.skycell/1806/004/test.stk.r.fits"]
    assert template.metadata["service"] == "Pan-STARRS cutout"
    assert template.metadata["fallback"]["survey"] == "Pan-STARRS"
    assert template.metadata["delivered_band"] == "r"
    assert template.metadata["asinh_linearized"] is False
    assert template.metadata["acquisition_only"] is True
    assert template.metadata["photometric_subtraction_allowed"] is False
    warnings = " ".join(template.metadata["warnings"])
    assert "provider-native cutout" in warnings
    assert "acquisition-only" in warnings
    with fits.open(template.fits_path) as hdus:
        assert hdus[0].data.shape == (64, 64)
        assert hdus[0].header["FILTER"] == "r"
        np.testing.assert_allclose(hdus[0].data, 1.0)


def test_skymapper_siap_fallback_reprojects_downloaded_fits(tmp_path, monkeypatch):
    science, wcs = science_image(tmp_path, crval=(180.0, -45.0))
    request = survey.prepare_template_request(science, "r", survey="skymapper")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/public/siap/dr4/query" in url:
            return b"get_fits,band\nhttps://api.skymapper.nci.org.au/public/siap/dr4/get_image/test.fits,r\n"
        return payload(np.ones((64, 64)), wcs.to_header())

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    template = survey.fetch_survey_template(request, tmp_path / "cache")

    assert any("/public/siap/dr4/query" in url for url in calls)
    assert calls[-1].startswith("https://api.skymapper.nci.org.au/")
    query = parse_qs(urlsplit(calls[-2]).query)
    assert query["BAND"] == ["r"]
    assert query["FORMAT"] == ["image/fits"]
    assert template.metadata["fallback"]["survey"] == "SkyMapper"
    assert template.metadata["acquisition_only"] is True


def test_panstarrs_fallback_masks_pixels_outside_the_cutout(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")

    def download(url, timeout, limit):
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/ps1filenames.py" in url:
            return b"filter,filename\nr,/rings.v3.skycell/1806/004/test.stk.r.fits\n"
        return provider_payload(shape=(32, 32), crval=(181.0, 20.0))

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    with fits.open(template.fits_path) as hdus:
        assert np.isnan(hdus[0].data).all()
        assert hdus["MASK"].data.all()


def test_panstarrs_fallback_is_acquisition_only_and_cannot_be_subtracted(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    results = tmp_path / "results"

    def download(url, timeout, limit):
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/ps1filenames.py" in url:
            return b"filter,filename\nr,/rings.v3.skycell/1806/004/test.stk.r.fits\n"
        return provider_payload()

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    with pytest.raises(survey.FallbackAcquisitionOnlyError, match="acquisition-only"):
        survey.compare_survey_images(
            science,
            results,
            template_filter="r",
            survey="panstarrs",
            cache_dir=tmp_path / "cache",
            config={"plots": False},
        )
    report = json.loads((results / "survey_reference.json").read_text(encoding="utf-8"))
    assert report["state"] == "acquisition_only"
    assert report["template"]["acquisition_only"] is True
    assert report["template"]["photometric_subtraction_allowed"] is False
    assert Path(report["reference_fits"]).is_file()
    assert not (results / "analysis").exists()


def test_cached_panstarrs_fallback_without_new_flags_stays_acquisition_only(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    cache = tmp_path / "cache"

    def download(url, timeout, limit):
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/ps1filenames.py" in url:
            return b"filter,filename\nr,/rings.v3.skycell/1806/004/test.stk.r.fits\n"
        return provider_payload()

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    survey.fetch_survey_template(request, cache)
    manifest = next(cache.glob("*.hips.json"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata.pop("acquisition_only")
    metadata.pop("photometric_subtraction_allowed")
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    cached = survey.fetch_survey_template(request, cache)
    assert cached.metadata["acquisition_only"] is True
    assert cached.metadata["photometric_subtraction_allowed"] is False


def test_survey_batch_summary_lists_acquisition_only_templates(tmp_path, monkeypatch):
    first, _ = science_image(tmp_path, name="science_1.fits")
    second, _ = science_image(tmp_path, name="science_2.fits")
    results = tmp_path / "results"

    def download(url, timeout, limit):
        if "/hips-image-services/" in url:
            raise URLError("CDS hips2fits unavailable")
        if "/ps1filenames.py" in url:
            return b"filter,filename\nr,/rings.v3.skycell/1806/004/test.stk.r.fits\n"
        return provider_payload()

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    report = survey.compare_survey_batch(
        [first, second],
        results,
        template_filter="r",
        survey="panstarrs",
        cache_dir=tmp_path / "cache",
        config={"plots": False},
    )
    assert report.state == "completed_with_errors"
    assert [item.state for item in report.items] == ["acquisition_only", "acquisition_only"]
    status = report.survey_status
    assert status["acquisition_only_count"] == 2
    assert {entry["name"] for entry in status["acquisition_only"]} == {
        "science-0001",
        "science-0002",
    }
    summary = json.loads((results / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["survey_status"]["acquisition_only_count"] == 2
    assert summary["counts"]["acquisition_only"] == 2
    assert all(Path(entry["reference_fits"]).is_file() for entry in status["acquisition_only"])
    assert not any((item.output_dir / "analysis").exists() for item in report.items)


def test_provider_fallback_is_not_used_for_explicit_identifiers(tmp_path, monkeypatch):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "W1", survey="CDS/P/NEOWISER/W1")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        raise URLError("service unavailable")

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", lambda *_: None)
    with pytest.raises(URLError, match="every service host"):
        survey.fetch_survey_template(request, tmp_path / "cache")
    assert all("ps1images" not in url and "skymapper" not in url for url in calls)


@pytest.mark.parametrize(
    ("survey_name", "bands"),
    [
        ("panstarrs", {"g", "r", "i", "z", "y"}),
        ("skymapper", {"g", "r", "i"}),
    ],
)
def test_provider_fallback_maps_supported_bands(tmp_path, survey_name, bands):
    declination = -45.0 if survey_name == "skymapper" else 20.0
    science, _ = science_image(tmp_path, crval=(180.0, declination))
    for band in bands:
        request = survey.prepare_template_request(science, band, survey=survey_name)
        assert survey._provider_fallback(request) == survey_name


def test_science_units_declaration_fills_a_missing_bunit():
    header = fits.Header({"FILTER": "r"})
    assert survey._science_unit(header, "adu") == "adu"
    assert survey._science_unit(header, "electron") == "electron"


def test_science_units_declaration_must_agree_with_bunit():
    with pytest.raises(ValueError, match="disagree"):
        survey._science_unit(fits.Header({"BUNIT": "electron"}), "adu")
    assert survey._science_unit(fits.Header({"BUNIT": "ADU"}), "adu") == "adu"


def test_missing_science_units_requires_a_declaration():
    with pytest.raises(ValueError, match="--science-units"):
        survey._science_unit(fits.Header({"FILTER": "r"}), None)


@pytest.mark.parametrize(
    ("survey_name", "band"),
    [("panstarrs", "u"), ("panstarrs", "v"), ("skymapper", "z")],
)
def test_provider_fallback_is_unavailable_for_unmapped_bands(tmp_path, survey_name, band):
    science, _ = science_image(tmp_path)
    request = survey.prepare_template_request(science, "r")
    request = survey.replace(request, survey=survey_name, template_filter=band)
    assert survey._provider_fallback(request) is None


def test_stc_polygon_keeps_longitudes_continuous_across_ra_zero(tmp_path):
    science, _ = science_image(tmp_path, crval=(0.01, 20.0))
    request = survey.prepare_template_request(
        science, "r", survey="CDS/P/DESI-Legacy-Surveys/DR10/r"
    )
    wcs, _, shape = survey._science_grid(request)
    polygon = survey._stc_polygon(wcs, shape)
    values = [float(token) for token in polygon.split()[1:]]
    right_ascension = values[::2]
    assert len(right_ascension) == len(values) // 2
    assert np.ptp(right_ascension) < 1.0
