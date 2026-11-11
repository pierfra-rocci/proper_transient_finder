# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Regression tests for the CDS HiPS survey-template backend."""

from __future__ import annotations

import json
from io import BytesIO
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
    request = survey.prepare_template_request(science, "r")
    calls = []
    delays = []

    def download(url, timeout, limit):
        calls.append(url)
        raise URLError("service unavailable")

    monkeypatch.setattr(survey, "_download_url", download)
    monkeypatch.setattr(survey, "sleep", delays.append)
    with pytest.raises(URLError, match="every service host"):
        survey.fetch_survey_template(request, tmp_path / "cache")
    hosts = [urlsplit(url).hostname for url in calls]
    expected = [urlsplit(base).hostname for base in survey.HIPS_SERVICE_URLS]
    assert hosts == expected * survey.SERVICE_ATTEMPTS
    assert delays == [survey.HIPS_RETRY_DELAY_SECONDS] * (survey.SERVICE_ATTEMPTS - 1)


def test_survey_options_deliver_a_grid_aligned_template_without_reprojection():
    options = survey._survey_options(config={"plots": False})
    assert options.reproject is False
    assert options.reproject_units == "flux-per-pixel"
    with pytest.raises(ValueError, match="kernel"):
        survey._survey_options(config={"align": True})


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


def test_moc_coverage_precheck_retries_all_service_hosts_before_giving_up(
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
