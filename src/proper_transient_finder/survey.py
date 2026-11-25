# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Opt-in HiPS survey reference acquisition around the comparison pipeline.

Templates are requested from the CDS ``hips2fits`` service on the exact science
pixel grid (WCS and shape). If CDS is unavailable, supported named providers
can supply a FITS cutout or tile mosaic that is reprojected locally. The module
only prepares cache-owned FITS copies; the science input is never edited and
the ordinary comparison pipeline performs PSF matching, photometry, and
candidate vetting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import ssl
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from ._version import __version__
from .astro import _prepare_results_root, _science_hdu

if TYPE_CHECKING:
    from .api import BatchResult, PipelineResult
    from .astro import PipelineConfig

# HiPS identifiers accepted by CDS hips2fits.  SkyMapper DR4 publishes only
# g, i and r HiPS maps; Pan-STARRS DR1 publishes all five broad bands.
SURVEY_HIPS = {
    "panstarrs": {
        "g": "PanSTARRS/DR1/g",
        "r": "PanSTARRS/DR1/r",
        "i": "PanSTARRS/DR1/i",
        "z": "PanSTARRS/DR1/z",
        "y": "PanSTARRS/DR1/y",
    },
    "skymapper": {
        "g": "CDS/P/Skymapper/DR4/g",
        "r": "CDS/P/Skymapper/DR4/r",
        "i": "CDS/P/Skymapper/DR4/i",
    },
}
SURVEY_FILTERS = {name: "".join(bands) for name, bands in SURVEY_HIPS.items()}
# Conservative nominal coverage used only to select a default provider.
SURVEY_COVERAGE = {"panstarrs": (-30.0, 90.0), "skymapper": (-90.0, 0.0)}
SURVEY_HOSTS = {
    "alasky.cds.unistra.fr",
    "alaskybis.cds.unistra.fr",
    "alasky.u-strasbg.fr",
    "alaskybis.u-strasbg.fr",
    "ps1images.stsci.edu",
    "api.skymapper.nci.org.au",
}
HIPS_SERVICE_URLS = (
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alasky.u-strasbg.fr/hips-image-services/hips2fits",
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
)
# One pass through every service host is retried at most this many times.
SERVICE_ATTEMPTS = 3
# Pause between HiPS retry cycles so a brief service outage can recover.
HIPS_RETRY_DELAY_SECONDS = 0.5
MOC_SERVICE_URLS = (
    "https://alaskybis.cds.unistra.fr/MocServer/query",
    "https://alasky.u-strasbg.fr/MocServer/query",
    "https://alasky.cds.unistra.fr/MocServer/query",
)
# Provider-native cutout services are used only after every CDS hips2fits host
# fails.  Their grids are reprojected locally onto the science WCS.
PANSTARRS_CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
PANSTARRS_FILENAMES_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PANSTARRS_IMAGE_URL = "https://ps1images.stsci.edu"
SKYMAPPER_SIAP_URL = "https://api.skymapper.nci.org.au/public/siap/dr4/query"
PROVIDER_FALLBACK_FILTERS = {
    "panstarrs": frozenset({"g", "r", "i", "z", "y"}),
    "skymapper": frozenset({"g", "r", "i"}),
}
SURVEY_WARNING = (
    "HiPS survey templates are coadded, resampled products rather than original "
    "detector counts. Flux scales, bandpasses and noise are approximate, and "
    "candidates require independent confirmation."
)
PROVIDER_FALLBACK_WARNING = (
    "Provider-native survey cutouts are experimental fallback templates: cross-provider "
    "bandpasses and calibrations are not certified automatically."
)


DEFAULT_MAX_DOWNLOAD_MB = 512
_GRID_TOLERANCE_DEG = 1e-5
_TEMPLATE_CACHE_VERSION = 3
_PANSTARRS_SAMPLE_SPACING_DEG = 0.2
_BAND_TOKEN = re.compile(r"[A-Za-z]\d?")
_EXPLICIT_BAND_MAX_LENGTH = 16
_MOC_RESPONSE_LIMIT = 1_000_000


@dataclass(frozen=True)
class TemplateRequest:
    """Validated science pixel grid and explicitly selected HiPS template band."""

    science_fits: Path
    survey: str
    template_filter: str
    science_filter: str
    native_science_filter: str
    ra_deg: float
    dec_deg: float
    size_deg: float
    science_mjd: float | None
    hips_id: str = ""
    width: int = 0
    height: int = 0
    grid_sha256: str = ""

    def to_dict(self) -> dict:
        values = asdict(self)
        values["science_fits"] = str(self.science_fits)
        return values


def normalize_filter(value: str) -> str:
    """Recognize explicit band aliases, without claiming identical bandpasses."""
    label = str(value).strip().lower()
    for prefix in ("sloan-", "sdss-", "ps1-", "panstarrs-", "skymapper-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    if not label or label == "unknown":
        raise ValueError("Science FILTER is missing; provide science_filter explicitly")
    return label


def _science_header_shape(path: Path) -> tuple[fits.Header, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdus:
        image = _science_hdu(hdus)
        if image.data is None or image.data.ndim != 2:
            raise ValueError("Science input must contain one two-dimensional image")
        header = hdus[0].header.copy()
        header.update(image.header)
        shape = tuple(int(value) for value in image.data.shape)
    return header, shape


def _safe_wcs(header: fits.Header) -> WCS:
    """Apply Astropy WCS fixes without emitting deprecated-keyword warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header)


def _celestial_wcs(header: fits.Header, *, purpose: str) -> WCS:
    wcs = _safe_wcs(header)
    if wcs.pixel_n_dim != 2 or not wcs.has_celestial:
        raise ValueError(f"{purpose} requires valid two-dimensional celestial WCS")
    return wcs.celestial


def _grid_sha256(wcs: WCS, shape: tuple[int, int]) -> str:
    header = wcs.to_header(relax=True)
    payload = {
        "header": header.tostring(sep="\n", endcard=True, padding=False),
        "shape": list(shape),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _resolve_hips_id(survey: str, band: str) -> str:
    if "/" in survey:
        identifier = survey.strip()
        if (
            not identifier
            or not identifier.isascii()
            or not identifier.isprintable()
            or any(character.isspace() for character in identifier)
            or any(character in "*?," for character in identifier)
        ):
            raise ValueError("Explicit HiPS identifier must be a single printable ASCII identifier")
        trailing = identifier.rstrip("/").rsplit("/", 1)[-1]
        if _BAND_TOKEN.fullmatch(trailing) and trailing.casefold() != band:
            raise ValueError("HiPS identifier band does not match template_filter")
        return identifier
    mapping = SURVEY_HIPS.get(survey)
    if mapping is None or band not in mapping:
        raise ValueError(
            "Requested survey/filter is not available as a HiPS template; "
            "choose a supported provider/band or pass an explicit HiPS identifier"
        )
    return mapping[band]


def _select_survey(boundary, survey: str, band: str) -> str:
    if "/" in survey:
        return survey.strip()
    if survey not in {"auto", *SURVEY_FILTERS}:
        raise ValueError("Survey must be auto, panstarrs, skymapper, or a HiPS identifier")
    available = []
    for provider, (minimum, maximum) in SURVEY_COVERAGE.items():
        if band not in SURVEY_HIPS[provider]:
            continue
        if np.min(boundary.dec.deg) >= minimum and np.max(boundary.dec.deg) <= maximum:
            available.append(provider)
    selected = available[0] if survey == "auto" and available else survey
    if selected not in available:
        raise ValueError("Requested survey/filter does not cover the complete science footprint")
    return selected


def prepare_template_request(
    science_fits: str | Path,
    template_filter: str,
    *,
    survey: str = "auto",
    science_filter: str | None = None,
) -> TemplateRequest:
    """Read only the science input and describe the HiPS request without network use.

    The returned request asks the service for exactly the science WCS and shape,
    so the whole footprint is covered by a single HiPS cutout; no local tiling or
    mosaic is planned.  Nothing is downloaded or written here.
    """
    path = Path(science_fits).expanduser().resolve()
    header, shape = _science_header_shape(path)
    wcs = _celestial_wcs(header, purpose="Survey templates")
    height, width = shape
    center = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2).icrs
    positions = np.linspace(0, 1, 17)
    columns = np.concatenate(
        (positions * (width - 1), positions * (width - 1), np.zeros(17), np.full(17, width - 1))
    )
    rows = np.concatenate(
        (np.zeros(17), np.full(17, height - 1), positions * (height - 1), positions * (height - 1))
    )
    boundary = wcs.pixel_to_world(columns, rows).icrs
    radius = float(np.max(center.separation(boundary).deg))
    if not np.isfinite(radius) or not 0 < radius < 1:
        raise ValueError("Science WCS footprint is invalid or too large for survey cutouts")
    survey_value = str(survey).strip()
    band = str(template_filter).strip().lower()
    if "/" in survey_value:
        if (
            not band
            or len(band) > _EXPLICIT_BAND_MAX_LENGTH
            or not band.isascii()
            or not band.isprintable()
            or any(character.isspace() or character in "/\\" for character in band)
        ):
            raise ValueError("Specify one explicit HiPS band label")
    elif len(band) != 1:
        raise ValueError("Specify one supported template filter")
    selected = _select_survey(boundary, survey_value, band)
    hips_id = _resolve_hips_id(selected, band)
    native = str(header.get("FILTER", "")).strip()
    label = normalize_filter(science_filter if science_filter is not None else native)
    mjd = header.get("MJD-OBS")
    if mjd is None and header.get("DATE-OBS"):
        from astropy.time import Time

        mjd = Time(header["DATE-OBS"]).mjd
    if mjd is not None and not np.isfinite(float(mjd)):
        raise ValueError("Science observation epoch must be finite")
    return TemplateRequest(
        path,
        selected,
        band,
        label,
        native,
        float(center.ra.deg),
        float(center.dec.deg),
        max(2 * radius * 1.1, 32 / 3600),
        None if mjd is None else float(mjd),
        hips_id,
        width,
        height,
        _grid_sha256(wcs, shape),
    )


@dataclass(frozen=True)
class SurveyTemplate:
    """Locally cached HiPS FITS on the science grid and auditable metadata."""

    fits_path: Path
    metadata: dict


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in SURVEY_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Survey downloads require an approved HTTPS survey host")


class _SurveyRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def _download_url(url: str, timeout: float, limit: int) -> bytes:
    _validate_url(url)
    request = Request(url, headers={"User-Agent": f"ProperTransientFinder/{__version__}"})
    try:
        with build_opener(_SurveyRedirect()).open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > limit:
                raise ValueError("Survey response exceeds the download size limit")
            payload = response.read(limit + 1)
    except URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        raise URLError(
            f"Survey TLS certificate verification failed: {error.reason}. "
            "Check the server certificate chain and your trusted CA certificates, or set "
            "SSL_CERT_FILE to a trusted CA bundle. See USER_GUIDE.md. "
            "Do not disable TLS verification."
        ) from error
    if len(payload) > limit:
        raise ValueError("Survey response exceeds the download size limit")
    return payload


def _science_grid(request: TemplateRequest) -> tuple[WCS, fits.Header, tuple[int, int]]:
    """Return the science celestial WCS, a hips2fits WCS header and the shape."""
    header, shape = _science_header_shape(Path(request.science_fits))
    wcs = _celestial_wcs(header, purpose="Survey templates")
    hips_header = wcs.to_header(relax=True)
    if hips_header.get("CTYPE1") == "RA---TPV" and "PV1_0" not in hips_header:
        # hips2fits rejects TPV headers without distortion coefficients.
        hips_header["CTYPE1"] = "RA---TAN"
        hips_header["CTYPE2"] = "DEC--TAN"
    hips_header["NAXIS1"] = shape[1]
    hips_header["NAXIS2"] = shape[0]
    return wcs, hips_header, shape


def _linearize_asinh(image: np.ndarray) -> np.ndarray:
    """Undo Pan-STARRS asinh scaling as used by the CDS HiPS maps."""
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = image * 0.4 * np.log(10)
        return np.exp(exponent) - np.exp(-exponent)


def _uses_asinh(hips_id: str) -> bool:
    label = hips_id.strip().casefold().rstrip("/")
    return "panstarrs" in label and not label.endswith("/g")


def _download_hips(request: TemplateRequest, timeout: float, limit: int) -> tuple[bytes, str, str]:
    """Request one HiPS cutout covering the full science grid, with host fallback."""
    _, wcs_header, shape = _science_grid(request)
    hips_id = request.hips_id or _resolve_hips_id(request.survey, request.template_filter)
    parameters = {
        "hips": hips_id,
        "width": shape[1],
        "height": shape[0],
        "coordsys": "icrs",
        "format": "fits",
        "wcs": json.dumps(dict(wcs_header), sort_keys=True),
    }
    failures = []
    for attempt in range(SERVICE_ATTEMPTS):
        if attempt:
            sleep(HIPS_RETRY_DELAY_SECONDS)
        for base in HIPS_SERVICE_URLS:
            url = base + "?" + urlencode(parameters)
            try:
                return _download_url(url, timeout, limit), url, hips_id
            except OSError as error:
                failures.append(f"{urlsplit(base).hostname}: {error}")
    raise URLError("HiPS download failed for every service host: " + "; ".join(failures))


def _provider_fallback(request: TemplateRequest) -> str | None:
    """Return the provider-native fallback name for a named provider and band."""
    if "/" in request.survey:
        return None
    if request.survey not in PROVIDER_FALLBACK_FILTERS:
        return None
    if request.template_filter not in PROVIDER_FALLBACK_FILTERS[request.survey]:
        return None
    return request.survey


def _panstarrs_filenames_url(
    request: TemplateRequest, *, ra_deg: float | None = None, dec_deg: float | None = None
) -> str:
    parameters = {
        "ra": f"{request.ra_deg if ra_deg is None else ra_deg:.8f}",
        "dec": f"{request.dec_deg if dec_deg is None else dec_deg:.8f}",
        "filters": request.template_filter,
        "type": "stack",
        "sep": ",",
    }
    return PANSTARRS_FILENAMES_URL + "?" + urlencode(parameters)


def _panstarrs_cutout_url(
    request: TemplateRequest,
    filename: str,
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    size_pixels: int | None = None,
) -> str:
    pixels = size_pixels or int(np.ceil(request.size_deg * 3600.0 / 0.25))
    pixels = min(max(pixels, 32), 6000)
    parameters = {
        "size": str(pixels),
        "format": "fits",
        "ra": f"{request.ra_deg if ra_deg is None else ra_deg:.8f}",
        "dec": f"{request.dec_deg if dec_deg is None else dec_deg:.8f}",
        "red": filename,
    }
    return PANSTARRS_CUTOUT_URL + "?" + urlencode(parameters)


def _skymapper_url(request: TemplateRequest) -> str:
    size = f"{request.size_deg:.8f}"
    parameters = {
        "POS": f"{request.ra_deg:.8f},{request.dec_deg:.8f}",
        "SIZE": size,
        "BAND": request.template_filter,
        "FORMAT": "image/fits",
        "INTERSECT": "COVERS",
        "RESPONSEFORMAT": "CSV",
        "VERB": "3",
    }
    return SKYMAPPER_SIAP_URL + "?" + urlencode(parameters)


def _panstarrs_image_records(
    payload: bytes, band: str, *, fallback_ra: float, fallback_dec: float
) -> list[tuple[str, float, float]]:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise URLError("Pan-STARRS filename service returned no images")
    reader = csv.DictReader(StringIO(text), skipinitialspace=True)
    if reader.fieldnames is None:
        raise URLError("Pan-STARRS filename service returned an invalid table")
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    records = {}
    for row in reader:
        normalized = {
            str(key).strip().casefold(): str(value or "").strip()
            for key, value in row.items()
        }
        if normalized.get("filter", "").casefold() != band.casefold():
            continue
        filename = normalized.get("filename", "")
        if not re.fullmatch(
            r"/rings\.v3\.skycell/[0-9]{4}/[0-9]{3}/[A-Za-z0-9_.-]+\.fits", filename
        ):
            continue
        try:
            ra_deg = float(normalized.get("ra", fallback_ra))
            dec_deg = float(normalized.get("dec", fallback_dec))
        except ValueError:
            ra_deg, dec_deg = fallback_ra, fallback_dec
        if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
            ra_deg, dec_deg = fallback_ra, fallback_dec
        records.setdefault(filename, (filename, ra_deg, dec_deg))
    if not records:
        raise URLError(f"Pan-STARRS filename service returned no stack image for band {band}")
    return list(records.values())


def _panstarrs_filename(payload: bytes, band: str) -> str:
    """Return the first matching file (kept for callers needing a single tile)."""
    records = _panstarrs_image_records(payload, band, fallback_ra=0.0, fallback_dec=0.0)
    return records[0][0]


def _panstarrs_query_positions(request: TemplateRequest) -> list[tuple[float, float]]:
    """Sample the whole science rectangle to discover every overlapping PS1 skycell."""
    science_wcs, _, shape = _science_grid(request)
    height, width = shape
    if request.size_deg <= 0.12:
        return [(request.ra_deg, request.dec_deg)]

    center_row = (height - 1) / 2
    center_column = (width - 1) / 2
    left = science_wcs.pixel_to_world(0, center_row).icrs
    right = science_wcs.pixel_to_world(width - 1, center_row).icrs
    top = science_wcs.pixel_to_world(center_column, 0).icrs
    bottom = science_wcs.pixel_to_world(center_column, height - 1).icrs
    width_deg = float(left.separation(right).deg)
    height_deg = float(top.separation(bottom).deg)
    x_intervals = max(1, int(np.ceil(width_deg / _PANSTARRS_SAMPLE_SPACING_DEG)))
    y_intervals = max(1, int(np.ceil(height_deg / _PANSTARRS_SAMPLE_SPACING_DEG)))
    columns = np.linspace(0, width - 1, x_intervals + 1)
    rows = np.linspace(0, height - 1, y_intervals + 1)
    sample_columns, sample_rows = np.meshgrid(columns, rows)
    sky = science_wcs.pixel_to_world(sample_columns.ravel(), sample_rows.ravel()).icrs
    positions = {
        (round(float(ra), 8), round(float(dec), 8))
        for ra, dec in zip(sky.ra.deg, sky.dec.deg, strict=True)
    }
    positions.add((request.ra_deg, request.dec_deg))
    return sorted(positions)


def _skymapper_fits_urls(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise URLError("SkyMapper SIAP service returned no cutouts")
    reader = csv.DictReader(StringIO(text), skipinitialspace=True)
    if reader.fieldnames is None:
        raise URLError("SkyMapper SIAP service returned an invalid table")
    reader.fieldnames = [name.strip().casefold() for name in reader.fieldnames]
    urls = []
    for row in reader:
        normalized = {str(key).strip().casefold(): str(value).strip() for key, value in row.items()}
        for key in ("get_fits", "get_image", "access_url"):
            url = normalized.get(key)
            if url:
                parsed = urlsplit(url)
                if parsed.scheme != "https" or parsed.hostname not in SURVEY_HOSTS:
                    continue
                if url not in urls:
                    urls.append(url)
                # The documented direct get_image endpoint accepts FORMAT=fits
                # and is an alternate route when a returned get_fits link fails.
                if key == "get_fits" and parsed.path.rstrip("/").endswith("/get_fits"):
                    query = [
                        (name, value)
                        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
                        if name.casefold() != "format"
                    ]
                    query.append(("FORMAT", "fits"))
                    image_path = parsed.path.rstrip("/")[: -len("get_fits")] + "get_image"
                    alternate = urlunsplit(
                        (parsed.scheme, parsed.netloc, image_path, urlencode(query), parsed.fragment)
                    )
                    if alternate not in urls:
                        urls.append(alternate)
    if not urls:
        raise URLError("SkyMapper SIAP service returned no downloadable FITS cutout")
    return urls


def _skymapper_fits_url(payload: bytes) -> str:
    """Return the preferred SkyMapper FITS URL for callers needing one link."""
    return _skymapper_fits_urls(payload)[0]


def _download_provider_fallback(
    request: TemplateRequest, timeout: float, limit: int
) -> tuple[list[tuple[bytes, bool]], list[str], str]:
    """Fetch a provider-native FITS image or a set of panels after HiPS fails.

    PS1's `fitscut.cgi` service limits each request to 6000 pixels. Querying a
    single central skycell therefore clips large science footprints. Sample the
    full science rectangle, retrieve every overlapping stack skycell, and mosaic
    those panels locally. If the cutout CGI is unavailable, the documented full
    skycell FITS URL is an alternate download path for that tile.
    """
    provider = _provider_fallback(request)
    if provider is None:
        raise ValueError("Requested survey/band has no provider-native fallback")
    if provider == "panstarrs":
        return _download_panstarrs_fallback(request, timeout, limit)
    failures = []
    for attempt in range(SERVICE_ATTEMPTS):
        if attempt:
            sleep(HIPS_RETRY_DELAY_SECONDS)
        try:
            siaps_url = _skymapper_url(request)
            listing = _download_url(siaps_url, timeout, limit)
            cutout_urls = _skymapper_fits_urls(listing)
            link_failures = []
            for cutout_url in cutout_urls:
                try:
                    payload = _download_url(cutout_url, timeout, limit)
                    return [(payload, False)], [cutout_url], "SkyMapper"
                except OSError as error:
                    link_failures.append(f"{cutout_url}: {error}")
            raise URLError("all SkyMapper FITS download links failed: " + "; ".join(link_failures))
        except OSError as error:
            failures.append(f"{provider}: {error}")
    raise URLError(f"{provider} provider fallback failed: " + "; ".join(failures))


def _download_panstarrs_fallback(
    request: TemplateRequest, timeout: float, limit: int
) -> tuple[list[tuple[bytes, bool]], list[str], str]:
    """Fetch a full-footprint PS1 mosaic, using direct skycell files as backup."""
    images = {}
    for ra_deg, dec_deg in _panstarrs_query_positions(request):
        filenames_url = _panstarrs_filenames_url(request, ra_deg=ra_deg, dec_deg=dec_deg)
        listing = _download_url(filenames_url, timeout, limit)
        for record in _panstarrs_image_records(
            listing,
            request.template_filter,
            fallback_ra=ra_deg,
            fallback_dec=dec_deg,
        ):
            images.setdefault(record[0], record)

    panels = []
    image_urls = []
    panel_size = int(
        np.ceil(
            max(request.size_deg, 2 * _PANSTARRS_SAMPLE_SPACING_DEG) * 3600.0 / 0.25
        )
    )
    for filename, ra_deg, dec_deg in images.values():
        cutout_url = _panstarrs_cutout_url(
            request,
            filename,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            size_pixels=panel_size,
        )
        try:
            panel = _download_url(cutout_url, timeout, limit)
            linearized = False
        except OSError as cutout_error:
            image_url = PANSTARRS_IMAGE_URL + filename
            try:
                panel = _download_url(image_url, timeout, limit)
                linearized = True
                cutout_url = image_url
            except OSError as direct_error:
                raise URLError(
                    f"PS1 cutout and full-skycell downloads failed: "
                    f"{cutout_error}; {direct_error}"
                ) from direct_error
        panels.append((panel, linearized))
        image_urls.append(cutout_url)
    if not panels:
        raise URLError("Pan-STARRS returned no images overlapping the science field")
    return panels, image_urls, "Pan-STARRS"


def _reproject_provider_cutout(
    data: np.ndarray, header: fits.Header, science_wcs: WCS, shape: tuple[int, int]
) -> np.ndarray:
    """Reproject a provider cutout onto the science grid and mask blank pixels."""
    from reproject import reproject_interp

    cutout_wcs = _safe_wcs(header)
    if cutout_wcs.pixel_n_dim != 2 or not cutout_wcs.has_celestial:
        raise ValueError("Provider fallback cutout has no usable two-dimensional celestial WCS")
    if data.ndim != 2:
        raise ValueError("Provider fallback cutout is not a two-dimensional image")
    projected, footprint = reproject_interp(
        (data, cutout_wcs.celestial), science_wcs, shape_out=shape
    )
    projected = np.asarray(projected, dtype=float)
    # Provider services may encode uncovered pixels as zero.
    covered = (np.asarray(footprint) > 0) & (projected != 0)
    return np.where(covered, projected, np.nan)


def _linearize_panstarrs_stack(image: np.ndarray, header: fits.Header) -> np.ndarray:
    """Convert a directly downloaded PS1 stack from its documented asinh scale."""
    bsoften = header.get("BSOFTEN")
    boffset = header.get("BOFFSET")
    if bsoften is None or boffset is None:
        raise ValueError("Downloaded PS1 skycell is missing BSOFTEN/BOFFSET scaling metadata")
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = image * 0.4 * np.log(10)
        return float(boffset) + float(bsoften) * (np.exp(exponent) - np.exp(-exponent))


def _reproject_provider_mosaic(
    panels: list[tuple[np.ndarray, fits.Header, bool]],
    science_wcs: WCS,
    shape: tuple[int, int],
) -> np.ndarray:
    """Reproject and combine provider cutouts, preserving blank pixels as NaN."""
    from reproject import reproject_interp

    total = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=np.uint16)
    for image, header, is_panstarrs_stack in panels:
        panel_wcs = _safe_wcs(header)
        if panel_wcs.pixel_n_dim != 2 or not panel_wcs.has_celestial:
            raise ValueError("Provider fallback cutout has no usable two-dimensional celestial WCS")
        source = (
            _linearize_panstarrs_stack(image, header) if is_panstarrs_stack else image
        )
        projected, footprint = reproject_interp(
            (source, panel_wcs.celestial), science_wcs, shape_out=shape
        )
        projected = np.asarray(projected, dtype=float)
        valid = (np.asarray(footprint) > 0) & np.isfinite(projected) & (projected != 0)
        total[valid] += projected[valid]
        count[valid] += 1
    mosaic = np.full(shape, np.nan, dtype=float)
    covered = count > 0
    mosaic[covered] = total[covered] / count[covered]
    return mosaic


def _stc_polygon(wcs: WCS, shape: tuple[int, int], samples: int = 33) -> str:
    """Return the science footprint as a closed ICRS STC-S polygon."""
    height, width = shape
    positions = np.linspace(0.0, 1.0, samples)
    edges = (
        (positions, np.zeros(samples)),
        (np.ones(samples), positions),
        (positions[::-1], np.ones(samples)),
        (np.zeros(samples), positions[::-1]),
    )
    columns = np.concatenate([edge[0] for edge in edges]) * (width - 1)
    rows = np.concatenate([edge[1] for edge in edges]) * (height - 1)
    points = np.column_stack((columns, rows))
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
    points = points[keep]
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    sky = wcs.pixel_to_world(points[:, 0], points[:, 1]).icrs
    center = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2).icrs
    # Keep longitudes continuous for footprints crossing the RA=0 seam.
    right_ascension = center.ra.deg + ((sky.ra.deg - center.ra.deg + 180.0) % 360.0 - 180.0)
    declination = sky.dec.deg
    if not np.all(np.isfinite(right_ascension)) or not np.all(np.isfinite(declination)):
        raise ValueError("Science WCS does not define a finite footprint polygon")
    vertices = " ".join(
        f"{longitude:.6f} {latitude:.6f}"
        for longitude, latitude in zip(right_ascension, declination, strict=True)
    )
    return "Polygon " + vertices


def _moc_record_ids(payload: bytes) -> list[str]:
    """Extract dataset identifiers from a MOCServer JSON record response."""
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MOC coverage response is not valid JSON") from error
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("MOC coverage response is not a JSON list")
    identifiers = []
    for item in data:
        if isinstance(item, str):
            identifiers.append(item)
        elif isinstance(item, dict):
            for key in ("ID", "id"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    identifiers.append(value)
                    break
    return identifiers


def _check_hips_coverage(request: TemplateRequest, timeout: float, *, enabled: bool = True) -> dict:
    """Ask the CDS MOCServer whether an explicit HiPS contains the science footprint."""
    hips_id = request.hips_id or _resolve_hips_id(request.survey, request.template_filter)
    result = {
        "service": "CDS MOCServer",
        "hips_id": hips_id,
        "intersect": "enclosed",
    }
    if "/" not in request.survey:
        return {**result, "status": "not_required", "covered": None}
    if not enabled:
        return {
            **result,
            "status": "skipped",
            "covered": None,
            "reason": "disabled by request",
        }
    wcs, _, shape = _science_grid(request)
    polygon = _stc_polygon(wcs, shape)
    parameters = {
        "ID": hips_id,
        "stc": polygon,
        "intersect": "enclosed",
        "get": "record",
        "fields": "ID",
        "fmt": "json",
    }
    failures = []
    for _ in range(SERVICE_ATTEMPTS):
        for base in MOC_SERVICE_URLS:
            url = base + "?" + urlencode(parameters)
            try:
                identifiers = _moc_record_ids(_download_url(url, timeout, _MOC_RESPONSE_LIMIT))
            except (OSError, ValueError) as error:
                failures.append(f"{urlsplit(base).hostname}: {error}")
                continue
            covered = any(
                identifier == hips_id or identifier.casefold() == hips_id.casefold()
                for identifier in identifiers
            )
            checked = {
                **result,
                "status": "covered" if covered else "not_covered",
                "covered": covered,
                "query_url": url,
                "polygon_vertices": (len(polygon.split()) - 1) // 2,
            }
            if identifiers and not covered:
                checked["matched_ids"] = identifiers[:10]
            return checked
    return {
        **result,
        "status": "unavailable",
        "covered": None,
        "error": "; ".join(failures) or "no MOC service response",
    }


def _fits_image(payload: bytes) -> tuple[np.ndarray, fits.Header]:
    with fits.open(BytesIO(payload), memmap=False) as hdus:
        image = _science_hdu(hdus)
        header = hdus[0].header.copy()
        header.update(image.header)
        return np.array(image.data, dtype=float, copy=True), header


def _grid_matches(header: fits.Header, science_wcs: WCS, shape: tuple[int, int]) -> bool:
    returned = _safe_wcs(header)
    if returned.pixel_n_dim != 2 or not returned.has_celestial:
        return False
    height, width = shape
    rows = np.array([0, 0, height - 1, height - 1, (height - 1) / 2])
    columns = np.array([0, width - 1, 0, width - 1, (width - 1) / 2])
    expected = science_wcs.all_pix2world(columns, rows, 0)
    actual = returned.celestial.all_pix2world(columns, rows, 0)
    return bool(
        np.all(np.isfinite(actual))
        and np.allclose(actual, expected, atol=_GRID_TOLERANCE_DEG, rtol=0)
    )


def _cached_template(path: Path, manifest: Path) -> SurveyTemplate | None:
    if not path.is_file() or not manifest.is_file():
        return None
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("Cached survey FITS checksum mismatch; use a fresh cache directory")
    if metadata.get("backend") == "provider-native-cutout" or metadata.get("fallback") is not None:
        metadata["acquisition_only"] = False
        metadata["photometric_subtraction_allowed"] = True
    return SurveyTemplate(path, metadata)


def _write_template_cache(path: Path, manifest: Path, hdus: fits.HDUList, metadata: dict) -> None:
    from .astro import atomic_json

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        hdus.writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    with path.open("rb") as stream:
        metadata["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    atomic_json(manifest, metadata)


def fetch_survey_template(
    request: TemplateRequest,
    cache_dir: str | Path,
    *,
    timeout: float = 30,
    max_download_mb: float = DEFAULT_MAX_DOWNLOAD_MB,
    check_moc_coverage: bool = True,
) -> SurveyTemplate:
    """Download a HiPS template or provider fallback, or reuse a cached template.

    The returned image is already projected onto the requested science WCS and
    shape.  Pan-STARRS asinh values are linearized, non-finite pixels are masked,
    and no variance plane is invented.  Explicit HiPS identifiers are checked
    against the CDS MOCServer coverage map before downloading unless
    ``check_moc_coverage`` is false.  The service image is cached with a
    checksum-verified JSON manifest.
    """
    if not np.isfinite(timeout) or timeout <= 0 or not 0 < max_download_mb <= 512:
        raise ValueError("Use finite positive timeout and download limit (at most 512 MB)")
    if not isinstance(check_moc_coverage, bool):
        raise ValueError("check_moc_coverage must be a boolean")
    science_wcs, _, shape = _science_grid(request)
    hips_id = request.hips_id or _resolve_hips_id(request.survey, request.template_filter)
    identity = request.to_dict()
    identity.pop("science_fits")
    identity.pop("native_science_filter")
    identity.pop("science_filter")
    identity.update(
        cache_version=_TEMPLATE_CACHE_VERSION,
        backend="cds-hips2fits",
        hips_id=hips_id,
        width=shape[1],
        height=shape[0],
        grid_sha256=_grid_sha256(science_wcs, shape),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir).expanduser().resolve()
    path, manifest = cache / f"{key}.hips.fits", cache / f"{key}.hips.json"
    cached = _cached_template(path, manifest)
    if cached is not None:
        return cached
    coverage = _check_hips_coverage(request, timeout, enabled=check_moc_coverage)
    if coverage["status"] == "not_covered":
        raise ValueError(
            f"CDS MOC coverage pre-check failed for {hips_id}: the survey MOC does not "
            "enclose the complete science footprint, or the identifier is not registered "
            "with the CDS MOCServer. Choose another HiPS identifier, a named provider, "
            "or a local reference."
        )
    limit = int(max_download_mb * 1_000_000)
    fallback = None
    try:
        payload, query_url, hips_id = _download_hips(request, timeout, limit)
        service = "CDS hips2fits"
    except OSError as hips_error:
        if _provider_fallback(request) is None:
            raise
        try:
            panels, source_urls, survey_name = _download_provider_fallback(
                request, timeout, limit
            )
        except OSError as fallback_error:
            raise URLError(
                f"{hips_error}; the provider-native fallback also failed: {fallback_error}"
            ) from fallback_error
        query_url = source_urls[0]
        fallback = {
            "survey": survey_name,
            "reason": "CDS hips2fits hosts did not respond",
            "reprojected": True,
            "source_urls": source_urls,
            "images_used": len(panels),
            "strategy": (
                "skycell-mosaic"
                if survey_name == "Pan-STARRS"
                else "siap-fits-link-retry"
            ),
        }
        service = f"{survey_name} cutout"
        if survey_name == "Pan-STARRS":
            reprojected_panels = []
            for panel_payload, is_full_stack in panels:
                panel_data, panel_header = _fits_image(panel_payload)
                reprojected_panels.append((panel_data, panel_header, is_full_stack))
            data = _reproject_provider_mosaic(reprojected_panels, science_wcs, shape)
            header = _science_grid(request)[1]
        else:
            data, header = _fits_image(panels[0][0])
            data = _reproject_provider_cutout(data, header, science_wcs, shape)
    else:
        data, header = _fits_image(payload)
        if data.shape != shape:
            raise ValueError(
                f"Downloaded HiPS template shape {data.shape} does not match the science grid {shape}"
            )
        if not _grid_matches(header, science_wcs, shape):
            raise ValueError("Downloaded HiPS template does not match the science pixel grid")
    native_filter = "" if fallback is not None else str(header.get("FILTER", "")).strip()
    native_bunit = "" if fallback is not None else str(header.get("BUNIT", "")).strip()
    if fallback is not None:
        fallback["valid_fraction"] = float(np.isfinite(data).mean())
        fallback["asinh_linearized"] = any(is_full_stack for _, is_full_stack in panels)
        if request.size_deg > 0.12 and fallback["valid_fraction"] < 0.5:
            raise ValueError(
                f"{survey_name} fallback covers only {fallback['valid_fraction']:.1%} "
                "of the science grid; at least 50% must be valid"
            )
    delivered_band = request.template_filter
    asinh = fallback is None and _uses_asinh(hips_id)
    if asinh:
        data = _linearize_asinh(data)
    valid = np.isfinite(data)
    data = np.where(valid, data, np.nan)
    mask = (~valid).astype(np.uint8)
    for keyword in (
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "EXTEND",
        "BSCALE",
        "BZERO",
        "CHECKSUM",
        "DATASUM",
    ):
        header.pop(keyword, None)
    header.update(science_wcs.to_header(relax=True))
    header["FILTER"] = delivered_band
    header["BUNIT"] = "relative-flux"
    header["HIPS"] = (hips_id, "Requested HiPS template identifier")
    header["HIPSASIN"] = (asinh, "Pan-STARRS asinh values were linearized")
    if fallback is not None:
        header["PROVIDER"] = (fallback["survey"], "Provider-native fallback survey")
        header.add_history(
            f"{fallback['survey']} provider fallback cutout reprojected onto "
            "the science pixel grid."
        )
    else:
        header.add_history("HiPS template projected onto the science pixel grid by CDS hips2fits.")
    warnings = [
        SURVEY_WARNING,
        "HiPS products have no variance plane; noise is estimated from the image background.",
        "Template chronology is not certified; a HiPS coadd may include the science epoch.",
    ]
    if fallback is not None:
        warnings += [
            f"CDS hips2fits did not respond; the template is a {fallback['survey']} "
            f"provider-native cutout instead of {hips_id}.",
            f"The {fallback['survey']} cutout was reprojected onto the science grid.",
            "Blank provider pixels were masked before PSF matching.",
            PROVIDER_FALLBACK_WARNING,
        ]
    if asinh:
        warnings.append(
            "Pan-STARRS HiPS values were linearized from asinh scaling; the flux zero point "
            "is not calibrated."
        )
    if coverage["status"] == "unavailable":
        warnings.append(
            "CDS MOC coverage pre-check was unavailable "
            f"({coverage.get('error', 'no response')}); acquisition continued and the "
            "downloaded HiPS grid and pixels were validated separately."
        )
    if coverage["status"] == "skipped":
        warnings.append(
            "CDS MOC coverage pre-check was explicitly disabled; HiPS coverage was not "
            "verified before download."
        )
    if fallback is not None:
        warnings.append(
            "The provider cutout does not declare FILTER; the fallback survey name is used "
            "as the band label."
        )
    elif not native_filter:
        warnings.append(
            "The HiPS header does not declare FILTER; the requested band label was applied."
        )
    metadata = {
        **identity,
        "query_url": query_url,
        "service": service,
        "backend": "provider-native-cutout" if fallback is not None else "cds-hips2fits",
        "image_id": hips_id,
        "product": "provider-cutout" if fallback is not None else "hips",
        "fallback": fallback,
        "acquisition_only": False,
        "photometric_subtraction_allowed": True,
        "delivered_band": delivered_band,
        "mjd": None,
        "reference_mjd": None,
        "native_reference_filter": native_filter or delivered_band,
        "native_bunit": native_bunit,
        "asinh_linearized": asinh
        or bool(fallback is not None and fallback.get("asinh_linearized")),
        "shape": [shape[0], shape[1]],
        "variance_source": "none",
        "coverage_check": coverage,
        "warnings": warnings,
    }
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(data, header),
            fits.ImageHDU(mask, name="MASK"),
        ]
    )
    _write_template_cache(path, manifest, hdus, metadata)
    return SurveyTemplate(path, metadata)


def _count_unit(value: str) -> str:
    label = str(value).strip().lower()
    if label in {"adu", "dn", "count", "counts"}:
        return "adu"
    if label in {"electron", "electrons", "e-"}:
        return "electron"
    raise ValueError(
        "Survey workflow requires declared detector-count units (adu or electron); "
        "calibrate other science or template units first."
    )


def _science_unit(header: fits.Header, declared: str | None) -> str:
    """Resolve the science count unit from the header or an explicit declaration."""
    header_label = str(header.get("BUNIT", "")).strip()
    if declared is None:
        try:
            return _count_unit(header_label)
        except ValueError as error:
            raise ValueError(
                "The science FITS header does not declare detector-count units; pass "
                "science_units (--science-units adu|electron) to declare them"
            ) from error
    unit = _count_unit(declared)
    if header_label and _count_unit(header_label) != unit:
        raise ValueError("Declared science units disagree with the science header BUNIT")
    return unit


def _normalized_copy(
    source: Path, destination: Path, band: str, unit: str, gain: float | None = None
) -> None:
    with fits.open(source, memmap=False) as hdus:
        copied = fits.HDUList([hdu.copy() for hdu in hdus])
    image = _science_hdu(copied)
    header = copied[0].header.copy()
    header.update(image.header)
    image.header["FILTER"] = band
    image.header["BUNIT"] = unit
    if gain is not None:
        image.header["GAIN"] = gain
    elif "GAIN" not in header and "CELL.GAIN" in header:
        image.header["GAIN"] = header["CELL.GAIN"]
    if "SATURATE" not in header and "CELL.SATURATION" in header:
        image.header["SATURATE"] = header["CELL.SATURATION"]
    image.header.add_history("Survey workflow normalized declared units and band labels only.")
    with NamedTemporaryFile(dir=destination.parent, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        copied.writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(destination)
    finally:
        copied.close()
        temporary.unlink(missing_ok=True)


def _survey_options(
    config=None,
    template_units=None,
    template_gain=None,
    timeout=30,
    max_download_mb=DEFAULT_MAX_DOWNLOAD_MB,
    threshold_sigma=5,
    check_moc_coverage=True,
):
    from .api import _config_values
    from .astro import PipelineConfig

    options = PipelineConfig(**_config_values(config))
    if options.method != "kernel":
        raise ValueError("Survey reference workflow requires the kernel method")
    if template_gain is not None and (not np.isfinite(template_gain) or template_gain <= 0):
        raise ValueError("Template gain must be finite and positive")
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("Detection threshold must be finite and positive")
    if not np.isfinite(timeout) or timeout <= 0 or not 0 < max_download_mb <= 512:
        raise ValueError("Use finite positive timeout and download limit (at most 512 MB)")
    if not isinstance(check_moc_coverage, bool):
        raise ValueError("check_moc_coverage must be a boolean")
    if template_units is not None:
        _count_unit(template_units)
    # The HiPS template is delivered on the science grid, so a second
    # reprojection would only resample an already resampled product.
    return replace(options, reproject=False, reproject_units="flux-per-pixel")


def compare_survey_images(
    science_fits: str | Path,
    output_dir: str | Path,
    *,
    template_filter: str,
    survey: str = "auto",
    cache_dir: str | Path = "survey_cache",
    config: PipelineConfig | Mapping[str, Any] | None = None,
    science_filter: str | None = None,
    science_units: str | None = None,
    template_units: str | None = None,
    template_gain: float | None = None,
    timeout: float = 30,
    max_download_mb: float = DEFAULT_MAX_DOWNLOAD_MB,
    check_moc_coverage: bool = True,
    threshold_sigma: float = 5,
    check_only: bool = False,
) -> PipelineResult | dict:
    """Acquire a HiPS template and call the unchanged comparison API.

    The template is projected onto the science pixel grid by the service, so the
    pipeline runs without an extra reprojection.  Check-only validates local
    WCS/options without downloads or writes; it does not certify service
    availability, calibration, overlap or detectability.  HiPS pixels are not
    original detector counts: cache-owned copies are labeled with the verified
    science count unit so the pipeline can fit one global stellar flux scale.
    ``science_units`` may declare that unit when the science header omits BUNIT;
    it never converts pixels.  Input FITS files are never edited.  Provider-native
    fallback templates are experimental and can be used by the normal subtraction
    pipeline; their cross-provider calibration remains unverified.
    """
    from .api import _config_values, compare_images
    from .astro import atomic_json

    if science_units is not None:
        _count_unit(science_units)
    options = _survey_options(
        config,
        template_units,
        template_gain,
        timeout,
        max_download_mb,
        threshold_sigma,
        check_moc_coverage,
    )
    request = prepare_template_request(
        science_fits,
        template_filter,
        survey=survey,
        science_filter=science_filter if science_filter is not None else options.science_filter,
    )
    root = Path(output_dir).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    if root == cache or root in cache.parents or cache in root.parents:
        raise ValueError("Survey cache and results directories must be separate")
    if root == request.science_fits.parent or root in request.science_fits.parents:
        raise ValueError("Survey results must not contain the science input")
    _prepare_results_root(
        root,
        "Survey results directory",
        overwrite=options.overwrite,
        assume_yes=options.assume_yes,
        protected=(options.ref_mask, options.new_mask, options.ref_psf, options.new_psf),
        clear=not check_only,
    )
    values = request.to_dict()
    cross_filter = request.science_filter != request.template_filter
    values.update(
        comparison_mode="experimental cross-filter" if cross_filter else "same-band",
        warnings=[SURVEY_WARNING],
    )
    if check_only:
        return {
            "state": "checked",
            "output_written": False,
            "network_used": False,
            "request": values,
            "not_checked": [
                "Remote availability",
                "Template calibration and noise",
                "Overlap/registration",
                "PSF/photometry",
                "Detectability",
            ],
        }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "survey_reference.json"
    atomic_json(report_path, {**values, "state": "acquiring"})
    try:
        template = fetch_survey_template(
            request,
            cache,
            timeout=timeout,
            max_download_mb=max_download_mb,
            check_moc_coverage=check_moc_coverage,
        )
        values.update(
            template=template.metadata,
            reference_fits=str(template.fits_path),
            warnings=template.metadata["warnings"],
        )
        reference_band = template.metadata.get("delivered_band") or request.template_filter
        cross_filter = request.science_filter.casefold() != str(reference_band).casefold()
        values.update(
            comparison_mode="experimental cross-filter" if cross_filter else "same-band",
            delivered_band=reference_band,
        )
        with fits.open(request.science_fits, memmap=False) as hdus:
            header = hdus[0].header.copy()
            header.update(_science_hdu(hdus).header)
        science_unit = _science_unit(header, science_units)
        reference_unit = science_unit if template_units is None else _count_unit(template_units)
        if science_unit != reference_unit:
            raise ValueError(
                "Science and declared template units differ; calibrate compatible inputs"
            )
        reference_gain = 1.0 if template_gain is None else template_gain
        options = replace(
            options,
            reproject=False,
            reproject_units="flux-per-pixel",
            cross_filter=cross_filter,
            gain=None,
            science_filter=None,
        )
        cache.mkdir(parents=True, exist_ok=True)
        values["science_sha256"] = hashlib.sha256(request.science_fits.read_bytes()).hexdigest()
        science_key = hashlib.sha256(
            json.dumps(
                {
                    "sha256": values["science_sha256"],
                    "band": request.science_filter,
                    "units": science_unit,
                    "science_gain": _config_values(config).get("gain"),
                    "template_gain": reference_gain,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        reference_hash = hashlib.sha256(template.fits_path.read_bytes()).hexdigest()
        reference_key = hashlib.sha256((reference_hash + science_key).encode()).hexdigest()
        prepared_science = cache / f"science_{science_key[:32]}.fits"
        prepared_reference = cache / f"reference_{reference_key[:32]}.fits"
        _normalized_copy(
            request.science_fits,
            prepared_science,
            request.science_filter,
            science_unit,
            _config_values(config).get("gain"),
        )
        _normalized_copy(
            template.fits_path,
            prepared_reference,
            reference_band,
            reference_unit,
            reference_gain,
        )
        values.update(
            template=template.metadata,
            normalized_unit=science_unit,
            reference_gain=reference_gain,
            absolute_flux_calibration=False,
            reference_fits=str(template.fits_path),
            warnings=[
                *template.metadata["warnings"],
                "The survey template is labeled with the science count unit only to allow "
                "one global stellar flux scale to be fitted; it is not an absolute calibration.",
            ],
        )
        result = compare_images(
            prepared_reference,
            prepared_science,
            root / "analysis",
            config=options,
            threshold_sigma=threshold_sigma,
        )
        result.metadata["survey_reference"] = values
        atomic_json(result.output_dir / "provenance.json", result.metadata)
        if options.plots:
            create_survey_diagnostic(result, values, root / "survey_diagnostic.png")
        atomic_json(report_path, {**values, "state": "completed"})
        return result
    except Exception as error:
        state = getattr(error, "state", "failed")
        atomic_json(report_path, {**values, "state": state, "error": str(error)})
        if options.plots:
            _failure_diagnostic(values, root / "survey_diagnostic.png", error)
        raise


def _failure_diagnostic(details: dict, path: Path, error: Exception) -> None:
    from .astro import LOGGER

    try:
        create_survey_diagnostic(None, details, path, str(error))
    except Exception as plot_error:
        LOGGER.warning("Unable to save survey failure diagnostic: %s", plot_error)


def create_survey_diagnostic(
    result: PipelineResult | None, details: dict, path: Path, error: str | None = None
) -> None:
    """Save an unconfirmed-candidate review sheet or an explicitly labeled failure sheet."""
    import textwrap

    from astropy.visualization import ImageNormalize, ZScaleInterval
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import CenteredNorm
    from matplotlib.figure import Figure

    figure = Figure(figsize=(15, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    figure.suptitle(
        f"Survey reference: {details.get('comparison_mode', 'not analyzed')} / UNCONFIRMED",
        fontsize=13,
    )
    if result is not None:
        panels = [
            (result.products["matched_reference"], "Matched survey reference", "gray"),
            (result.products["matched_new"], "Matched science", "gray"),
            (result.difference, "Difference: science - reference", "RdBu_r"),
            (result.score, "Empirical score, not variability significance", "RdBu_r"),
            (~result.mask, "Usable pixels after all masks", "binary"),
        ]
        common = ImageNormalize(
            np.concatenate((panels[0][0].ravel(), panels[1][0].ravel())),
            interval=ZScaleInterval(),
        )
        for axis, (image, title, cmap) in zip(list(axes.flat)[:5], panels, strict=True):
            norm = common if cmap == "gray" else None
            if cmap == "RdBu_r":
                norm = CenteredNorm(
                    vcenter=0, halfrange=max(float(np.nanpercentile(np.abs(image), 99)), 1e-12)
                )
            artist = axis.imshow(image, origin="lower", cmap=cmap, norm=norm)
            axis.set(title=title, xlabel="Science-grid X [pixels]", ylabel="Y [pixels]")
            figure.colorbar(artist, ax=axis, shrink=0.75)
        for axis in (axes[0, 2], axes[1, 0]):
            axis.scatter(
                result.candidates["x_centroid"],
                result.candidates["y_centroid"],
                s=55,
                facecolors="none",
                edgecolors="lime",
                label="Unconfirmed",
            )
            axis.legend(fontsize=8)
    else:
        for axis in list(axes.flat)[:5]:
            axis.set_axis_off()
        axes[0, 1].text(
            0.5,
            0.5,
            "FAILED\nNo valid subtraction shown",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=15,
            color="firebrick",
        )
    template = details.get("template", {})
    lines = [
        Path(details.get("science_fits", "unknown")).name,
        f"Survey/HiPS: {details.get('hips_id') or details.get('survey', 'unknown')}",
        f"Science band: {details.get('science_filter', 'unknown')}",
        f"Reference band: {details.get('template_filter', 'unknown')}",
        f"Native science band: {details.get('native_science_filter', 'unknown')}",
        f"Native template band: {template.get('native_reference_filter', 'unknown')}",
        f"Center: {details.get('ra_deg', 'unknown')}, {details.get('dec_deg', 'unknown')}",
        f"Science MJD: {details.get('science_mjd')}",
        f"Reference MJD: {template.get('reference_mjd')}",
    ]
    if result is not None:
        quality = result.metadata.get("quality_summary", {})
        lines += [
            f"Accepted: {len(result.candidates)} / rejected: {len(result.rejected_candidates)}",
            f"Quality: {quality.get('status', 'unknown')}",
            f"Registration RMS: {quality.get('metrics', {}).get('registration_rms_pixels')} px",
        ]
    if error:
        lines += ["FAILURE:", error]
    lines += ["WARNINGS:", *details.get("warnings", [])]
    if result is not None:
        lines += result.metadata.get("quality_summary", {}).get("warnings", [])
    axes[1, 2].set_axis_off()
    axes[1, 2].text(
        0,
        1,
        "\n".join(textwrap.fill(str(line), width=46) for line in lines),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
    )
    figure.savefig(path, dpi=130)


def load_science_list(list_path: str | Path) -> list[Path]:
    """Load text lines or a JSON array of science paths, relative to the list file."""
    path = Path(list_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    values = (
        json.loads(text)
        if path.suffix.lower() == ".json"
        else [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    )
    if not isinstance(values, list) or not values:
        raise ValueError("Science list must be a nonempty array or text list of FITS paths")
    paths = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Each science list entry must be a nonempty FITS path string")
        candidate = Path(value).expanduser()
        paths.append((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    return paths


def compare_survey_batch(
    science_images: Iterable[str | Path],
    output_dir: str | Path,
    *,
    template_filter: str,
    survey: str = "auto",
    cache_dir: str | Path = "survey_cache",
    fail_fast: bool = False,
    check_only: bool = False,
    **options: Any,
) -> BatchResult:
    """Sequential HiPS survey comparisons with existing lightweight batch result types."""
    from .api import BatchItemResult, BatchResult
    from .astro import atomic_json

    settings = _survey_options(
        **{
            key: value
            for key, value in options.items()
            if key not in {"science_filter", "science_units"}
        }
    )
    if options.get("science_units") is not None:
        _count_unit(options["science_units"])
    if isinstance(science_images, (str, Path)):
        raise ValueError("Pass a list of science images, not a single path")
    images = [Path(path).expanduser().resolve() for path in science_images]
    if not images:
        raise ValueError("Survey batch must contain science images")
    root, cache = Path(output_dir).expanduser().resolve(), Path(cache_dir).expanduser().resolve()
    if root == cache or root in cache.parents or cache in root.parents:
        raise ValueError("Survey cache and batch results must be separate")
    if any(root == path or root in path.parents for path in images):
        raise ValueError("Batch results must not contain science inputs")
    _prepare_results_root(
        root,
        "Survey batch results directory",
        overwrite=settings.overwrite,
        assume_yes=settings.assume_yes,
        protected=(
            *images,
            settings.ref_mask,
            settings.new_mask,
            settings.ref_psf,
            settings.new_psf,
        ),
        clear=not check_only,
    )
    items = [
        BatchItemResult(
            f"science-{index:04d}", "pending", root / f"science-{index:04d}", science_fits=path
        )
        for index, path in enumerate(images, start=1)
    ]
    if not check_only:
        root.mkdir(parents=True, exist_ok=True)
    acquisition_only: list[dict] = []

    def report(state):
        result = BatchResult(
            tuple(items),
            state,
            root,
            check_only,
            survey_status=None
            if check_only
            else {
                "acquisition_only_count": len(acquisition_only),
                "acquisition_only": list(acquisition_only),
            },
        )
        if not check_only:
            atomic_json(root / "batch_summary.json", result.to_dict())
        return result

    report("running")
    try:
        for index, item in enumerate(items):
            started = perf_counter()
            items[index] = replace(item, state="running")
            report("running")
            try:
                result = compare_survey_images(
                    item.science_fits,
                    item.output_dir,
                    template_filter=template_filter,
                    survey=survey,
                    cache_dir=cache,
                    check_only=check_only,
                    **options,
                )
                details = result if check_only else result.metadata["survey_reference"]
                items[index] = replace(
                    item,
                    state="checked" if check_only else "completed",
                    accepted_candidates=None if check_only else len(result.candidates),
                    rejected_candidates=None if check_only else len(result.rejected_candidates),
                    preflight=details if check_only else None,
                    reference_fits=None if check_only else Path(details["reference_fits"]),
                    elapsed_seconds=perf_counter() - started,
                )
                del result
            except Exception as error:
                items[index] = replace(
                    item,
                    state="failed",
                    error=str(error),
                    error_type=type(error).__name__,
                    elapsed_seconds=perf_counter() - started,
                )
                if not check_only:
                    item.output_dir.mkdir(exist_ok=True)
                    report_path = item.output_dir / "survey_reference.json"
                    details = {
                        "science_fits": str(item.science_fits),
                        "survey": survey,
                        "template_filter": template_filter,
                        "state": "failed",
                        "error": str(error),
                        "warnings": [SURVEY_WARNING],
                    }
                    if not report_path.exists():
                        atomic_json(report_path, details)
                    details = json.loads(report_path.read_text(encoding="utf-8"))
                    if settings.plots and not (item.output_dir / "survey_diagnostic.png").exists():
                        _failure_diagnostic(
                            details, item.output_dir / "survey_diagnostic.png", error
                        )
                if fail_fast:
                    for remaining in range(index + 1, len(items)):
                        items[remaining] = replace(items[remaining], state="skipped")
                    return report("stopped")
            report("running")
    except BaseException:
        for index, item in enumerate(items):
            if item.state in {"pending", "running"}:
                items[index] = replace(
                    item, state="interrupted" if item.state == "running" else "skipped"
                )
        report("interrupted")
        raise
    return report(
        "completed_with_errors"
        if any(item.state in {"failed", "acquisition_only"} for item in items)
        else "checked"
        if check_only
        else "completed"
    )
