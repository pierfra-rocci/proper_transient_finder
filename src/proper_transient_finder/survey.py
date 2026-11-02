# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Opt-in survey reference acquisition around the existing comparison pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .astro import _science_hdu

if TYPE_CHECKING:
    from .api import BatchResult, PipelineResult
    from .astro import PipelineConfig

SURVEY_FILTERS = {"panstarrs": "grizy", "skymapper": "uvgriz"}
SURVEY_HOSTS = {"ps1images.stsci.edu", "api.skymapper.nci.org.au"}
SURVEY_WARNING = (
    "Survey/instrument bandpasses can differ even when filter names match. "
    "Resampled or coadded reference noise may be correlated; errors are approximate. "
    "Candidates require independent confirmation."
)


@dataclass(frozen=True)
class TemplateRequest:
    """Validated science footprint and explicitly selected survey band."""

    science_fits: Path
    survey: str
    template_filter: str
    science_filter: str
    native_science_filter: str
    ra_deg: float
    dec_deg: float
    size_deg: float
    science_mjd: float | None

    def to_dict(self) -> dict:
        values = asdict(self)
        values["science_fits"] = str(self.science_fits)
        return values


def normalize_filter(value: str) -> str:
    """Recognize explicit band aliases, without claiming identical bandpasses."""
    label = str(value).strip().lower()
    for prefix in ("sloan-", "sdss-", "ps1-", "skymapper-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    if not label or label == "unknown":
        raise ValueError("Science FILTER is missing; provide science_filter explicitly")
    return label


def prepare_template_request(
    science_fits: str | Path,
    template_filter: str,
    *,
    survey: str = "auto",
    science_filter: str | None = None,
) -> TemplateRequest:
    """Read only science input; derive a padded WCS footprint without network or writes."""
    path = Path(science_fits).expanduser().resolve()
    with fits.open(path, memmap=False) as hdus:
        image = _science_hdu(hdus)
        header, shape = hdus[0].header.copy(), image.data.shape
        header.update(image.header)
    wcs = WCS(header)
    if wcs.pixel_n_dim != 2 or not wcs.has_celestial:
        raise ValueError("Survey templates require valid two-dimensional celestial WCS")
    height, width = shape
    center = wcs.celestial.pixel_to_world((width - 1) / 2, (height - 1) / 2).icrs
    positions = np.linspace(0, 1, 17)
    columns = np.concatenate(
        (positions * (width - 1), positions * (width - 1), np.zeros(17), np.full(17, width - 1))
    )
    rows = np.concatenate(
        (np.zeros(17), np.full(17, height - 1), positions * (height - 1), positions * (height - 1))
    )
    boundary = wcs.celestial.pixel_to_world(columns, rows).icrs
    radius = float(np.max(center.separation(boundary).deg))
    if not np.isfinite(radius) or not 0 < radius < 1:
        raise ValueError("Science WCS footprint is invalid or too large for survey cutouts")
    band = str(template_filter).strip().lower()
    if len(band) != 1:
        raise ValueError("Specify one supported template filter")
    available = []
    if np.min(boundary.dec.deg) >= -30 and band in SURVEY_FILTERS["panstarrs"]:
        available.append("panstarrs")
    if np.max(boundary.dec.deg) <= 0 and band in SURVEY_FILTERS["skymapper"]:
        available.append("skymapper")
    selected = available[0] if survey == "auto" and available else survey
    if selected not in SURVEY_FILTERS or selected not in available:
        raise ValueError("Requested survey/filter does not cover the complete science footprint")
    size = max(2 * radius * 1.1, 32 / 3600)
    limit = 9.9 / 60 if selected == "skymapper" else 6000 * 0.25 / 3600
    if size >= limit:
        raise ValueError(
            f"Padded science footprint requires a {size * 60:.2f} arcmin square; "
            f"{selected} requires a cutout smaller than {limit * 60:.2f} arcmin. "
            "Use a separate WCS-preserving science crop, process tiles explicitly, "
            "or provide a larger local reference with --ref/--new --reproject. "
            "Automatic tiling is not implemented."
        )
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
        size,
        None if mjd is None else float(mjd),
    )


@dataclass(frozen=True)
class SurveyTemplate:
    """Locally cached native survey FITS and auditable acquisition metadata."""

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
    request = Request(url, headers={"User-Agent": "ProperTransientFinder/0.1"})
    with build_opener(_SurveyRedirect()).open(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError("Survey response exceeds the download size limit")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Survey response exceeds the download size limit")
    return payload


def _discover_template(request: TemplateRequest, timeout: float) -> dict:
    if request.survey == "panstarrs":
        query = urlencode(
            {
                "ra": request.ra_deg,
                "dec": request.dec_deg,
                "filters": request.template_filter,
                "type": "stack,stack.wt,stack.mask",
            }
        )
        url = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?" + query
        table = Table.read(
            _download_url(url, timeout, 4_000_000).decode("utf-8"), format="ascii.basic"
        )
        rows = [row for row in table if str(row["filter"]) == request.template_filter]
        images = [row for row in rows if str(row["type"]) == "stack" and not int(row["badflag"])]
        if not images:
            raise ValueError("No usable Pan-STARRS template covers the requested field/filter")
        selected = sorted(images, key=lambda row: str(row["filename"]))[0]
        base = str(selected["filename"])

        def cutout(filename: str) -> str:
            parameters = {
                "ra": request.ra_deg,
                "dec": request.dec_deg,
                "size": int(np.ceil(request.size_deg * 3600 / 0.25)),
                "format": "fits",
                "red": filename,
            }
            return "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?" + urlencode(parameters)

        products = {str(row["type"]): str(row["filename"]) for row in rows}
        weight = base.removesuffix(".fits") + ".wt.fits"
        mask = base.removesuffix(".fits") + ".mask.fits"
        return {
            "query_url": url,
            "image_url": cutout(base),
            "image_id": base,
            "variance_url": cutout(weight) if weight in products.values() else None,
            "mask_url": cutout(mask) if mask in products.values() else None,
            "product": "stack",
            "mjd": None,
        }
    parameters = {
        "POS": f"{request.ra_deg},{request.dec_deg}",
        "SIZE": request.size_deg,
        "BAND": request.template_filter,
        "FORMAT": "image/fits",
        "INTERSECT": "covers",
        "VERB": 3,
    }
    if request.science_mjd is not None:
        parameters["MJD_END"] = request.science_mjd
    url = "https://api.skymapper.nci.org.au/public/siap/dr4/query?" + urlencode(parameters)
    table = Table.read(BytesIO(_download_url(url, timeout, 4_000_000)), format="votable")
    rows = [
        row
        for row in table
        if str(row["band"]).strip() == request.template_filter
        and np.isfinite(float(row["mjd_obs"]))
        and (request.science_mjd is None or float(row["mjd_obs"]) < request.science_mjd)
    ]
    if not rows:
        raise ValueError("No usable older SkyMapper image covers the requested field/filter")
    selected = sorted(
        rows,
        key=lambda row: (
            -float(row["exptime"]),
            float(row["mean_fwhm"]),
            str(row["unique_image_id"]),
        ),
    )[0]
    return {
        "query_url": url,
        "image_url": str(selected["get_fits"]),
        "mask_url": str(selected["get_mask"]),
        "variance_url": None,
        "image_id": str(selected["unique_image_id"]),
        "product": str(selected["image_type"]),
        "mjd": float(selected["mjd_obs"]),
    }


def _fits_image(payload: bytes) -> tuple[np.ndarray, fits.Header]:
    with fits.open(BytesIO(payload), memmap=False) as hdus:
        image = _science_hdu(hdus)
        header = hdus[0].header.copy()
        header.update(image.header)
        return np.array(image.data, copy=True), header


def fetch_survey_template(
    request: TemplateRequest,
    cache_dir: str | Path,
    *,
    timeout: float = 30,
    max_download_mb: float = 64,
) -> SurveyTemplate:
    """Download bounded HTTPS FITS products, or verify and reuse a cached template."""
    from .astro import atomic_json

    if not np.isfinite(timeout) or timeout <= 0 or not 0 < max_download_mb <= 512:
        raise ValueError("Use finite positive timeout and download limit (at most 512 MB)")
    cache = Path(cache_dir).expanduser().resolve()
    identity = request.to_dict()
    identity.pop("science_fits")
    identity.pop("native_science_filter")
    identity.pop("science_filter")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path, manifest = cache / f"{key}.fits", cache / f"{key}.json"
    if path.is_file() and manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            raise ValueError("Cached survey FITS checksum mismatch; use a fresh cache directory")
        return SurveyTemplate(path, metadata)
    selected = _discover_template(request, timeout)
    limit = int(max_download_mb * 1_000_000)
    data, header = _fits_image(_download_url(selected["image_url"], timeout, limit))
    wcs = WCS(header)
    if not wcs.has_celestial or not np.isfinite(data).any():
        raise ValueError("Downloaded template has no usable celestial WCS or finite pixels")
    column, row = wcs.celestial.world_to_pixel_values(request.ra_deg, request.dec_deg)
    if not (0 <= column < data.shape[1] and 0 <= row < data.shape[0]):
        raise ValueError("Downloaded template does not contain the requested science center")
    native_filter = str(header.get("FILTER", header.get("FPA.FILTER", ""))).strip()
    if native_filter and normalize_filter(native_filter.split(".")[0]) != request.template_filter:
        raise ValueError("Downloaded template filter does not match the selected service band")
    header["FILTER"] = request.template_filter
    hdus = [fits.PrimaryHDU(data, header)]
    for kind, extension in (("variance_url", "VAR"), ("mask_url", "MASK")):
        if selected.get(kind):
            auxiliary, auxiliary_header = _fits_image(_download_url(selected[kind], timeout, limit))
            if auxiliary.shape != data.shape:
                raise ValueError("Survey mask/variance shape does not match its image")
            reference_pixels = np.array([[0, 0], [data.shape[1] - 1, data.shape[0] - 1]])
            if not np.allclose(
                WCS(auxiliary_header).celestial.all_pix2world(reference_pixels, 0),
                wcs.celestial.all_pix2world(reference_pixels, 0),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Survey mask/variance WCS does not match its image")
            hdus.append(fits.ImageHDU(auxiliary, name=extension))
    mjd = header.get("MJD-OBS", selected.get("mjd"))
    if request.science_mjd is not None and mjd is not None and float(mjd) >= request.science_mjd:
        raise ValueError("Reference template epoch is not older than the science exposure")
    warnings = [SURVEY_WARNING]
    if request.science_mjd is None or mjd is None or selected["product"] == "stack":
        warnings.append("Template chronology is not certified; coadd epochs may span an interval.")
    if not selected.get("variance_url"):
        warnings.append("No survey variance plane; a validated count-space gain model is required.")
    metadata = {
        **identity,
        **selected,
        "native_reference_filter": native_filter,
        "native_bunit": str(header.get("BUNIT", "")),
        "reference_mjd": mjd,
        "warnings": warnings,
        "variance_source": "Pan-STARRS stack.wt pixel variance"
        if selected.get("variance_url")
        else "none",
    }
    cache.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(manifest, metadata)
    return SurveyTemplate(path, metadata)


def _count_unit(value: str) -> str:
    label = str(value).strip().lower()
    if label in {"adu", "dn", "count", "counts"}:
        return "adu"
    if label in {"electron", "electrons", "e-"}:
        return "electron"
    raise ValueError(
        "Survey workflow requires declared detector-count units; ambiguous template BUNIT "
        "requires template_units (adu or electron). Calibrate other units first."
    )


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
    max_download_mb=64,
    threshold_sigma=5,
):
    from .api import _config_values
    from .astro import PipelineConfig

    options = PipelineConfig(**_config_values(config))
    if options.method != "kernel" or options.align or options.shift:
        raise ValueError("Survey reference workflow requires kernel WCS reprojection, not align")
    if template_gain is not None and (not np.isfinite(template_gain) or template_gain <= 0):
        raise ValueError("Template gain must be finite and positive")
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("Detection threshold must be finite and positive")
    if not np.isfinite(timeout) or timeout <= 0 or not 0 < max_download_mb <= 512:
        raise ValueError("Use finite positive timeout and download limit (at most 512 MB)")
    if template_units is not None:
        _count_unit(template_units)
    return replace(options, reproject=True, reproject_units="flux-per-pixel")


def compare_survey_images(
    science_fits: str | Path,
    output_dir: str | Path,
    *,
    template_filter: str,
    survey: str = "auto",
    cache_dir: str | Path = "survey_cache",
    config: PipelineConfig | Mapping[str, Any] | None = None,
    science_filter: str | None = None,
    template_units: str | None = None,
    template_gain: float | None = None,
    timeout: float = 30,
    max_download_mb: float = 64,
    threshold_sigma: float = 5,
    check_only: bool = False,
) -> PipelineResult | dict:
    """Acquire a survey reference and call the unchanged comparison API.

    Check-only validates local WCS/options without downloads or writes; it does
    not certify service availability, calibration, overlap or detectability.
    Ambiguous reference units require explicit template_units. Input FITS files
    are never edited; label normalization occurs only in cache-owned copies.
    """
    from .api import _config_values, compare_images
    from .astro import atomic_json

    request = prepare_template_request(
        science_fits, template_filter, survey=survey, science_filter=science_filter
    )
    options = _survey_options(
        config, template_units, template_gain, timeout, max_download_mb, threshold_sigma
    )
    root = Path(output_dir).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    if root == cache or root in cache.parents or cache in root.parents:
        raise ValueError("Survey cache and results directories must be separate")
    if root == request.science_fits.parent or root in request.science_fits.parents:
        raise ValueError("Survey results must not contain the science input")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Survey results directory must be new or empty")
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
            request, cache, timeout=timeout, max_download_mb=max_download_mb
        )
        values.update(
            template=template.metadata,
            reference_fits=str(template.fits_path),
            warnings=template.metadata["warnings"],
        )
        with fits.open(request.science_fits, memmap=False) as hdus:
            header = hdus[0].header.copy()
            header.update(_science_hdu(hdus).header)
        science_unit = _count_unit(header.get("BUNIT", ""))
        reference_unit = _count_unit(
            template_units if template_units is not None else template.metadata["native_bunit"]
        )
        if science_unit != reference_unit:
            raise ValueError("Science and reference units differ; calibrate compatible inputs")
        options = replace(
            options,
            reproject=True,
            reproject_units="flux-per-pixel",
            cross_filter=cross_filter,
            gain=None,
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
                    "template_gain": template_gain,
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
            request.template_filter,
            reference_unit,
            template_gain,
        )
        with fits.open(prepared_reference, memmap=False) as hdus:
            reference_header = hdus[0].header.copy()
            reference_header.update(_science_hdu(hdus).header)
            if (
                not any(name in hdus for name in ("VAR", "VARIANCE", "ERR"))
                and "GAIN" not in reference_header
            ):
                raise ValueError("Reference has no variance or gain; provide template_gain")
        values.update(
            template=template.metadata,
            normalized_unit=science_unit,
            reference_fits=str(template.fits_path),
            warnings=template.metadata["warnings"],
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
        atomic_json(report_path, {**values, "state": "failed", "error": str(error)})
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
        f"Survey: {details.get('survey', 'unknown')}",
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
    """Sequential survey comparisons with existing lightweight batch result types."""
    from .api import BatchItemResult, BatchResult
    from .astro import atomic_json

    settings = _survey_options(
        **{key: value for key, value in options.items() if key != "science_filter"}
    )
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
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Survey batch results directory must be new or empty")
    items = [
        BatchItemResult(
            f"science-{index:04d}", "pending", root / f"science-{index:04d}", science_fits=path
        )
        for index, path in enumerate(images, start=1)
    ]
    if not check_only:
        root.mkdir(parents=True, exist_ok=True)

    def report(state):
        result = BatchResult(tuple(items), state, root, check_only)
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
                    if settings.plots and not (item.output_dir / "survey_diagnostic.png").exists():
                        if report_path.exists():
                            details = json.loads(report_path.read_text(encoding="utf-8"))
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
        if any(item.state == "failed" for item in items)
        else "checked"
        if check_only
        else "completed"
    )
