# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Prepare tiled survey references without changing the comparison pipeline.

The module is deliberately a pre-processing utility.  It downloads bounded
Pan-STARRS or SkyMapper cutouts through the existing validated acquisition
code, mosaics them onto a native-survey-scale TAN grid, and writes a separate
science copy only when a small centred crop avoids an otherwise needless tile
mosaic.  The normal pipeline still performs the scientific reprojection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from .astro import _science_hdu, atomic_json
from .survey import (
    SURVEY_FILTERS,
    SURVEY_WARNING,
    TemplateRequest,
    fetch_survey_template,
    normalize_filter,
    prepare_template_request,
)

# Stay below the providers' documented request ceilings.  Pan-STARRS rejects
# requests at 6000 pixels (25 arcmin) and SkyMapper documents a 9.9 arcmin
# maximum.  The small margin also protects against service-side rounding.
TILE_LIMIT_DEG = {"panstarrs": 24.9 / 60, "skymapper": 9.8 / 60}


@dataclass(frozen=True)
class MosaicPlan:
    """A local, auditable choice between one cutout, crop, or tile mosaic."""

    science_fits: Path
    survey: str
    template_filter: str
    science_filter: str
    native_science_filter: str
    ra_deg: float
    dec_deg: float
    size_deg: float
    science_mjd: float | None
    tile_limit_deg: float
    mode: Literal["single", "crop", "mosaic"]
    crop_bounds: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["science_fits"] = str(self.science_fits)
        return value


def _science_header_and_shape(path: Path) -> tuple[fits.Header, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdus:
        image = _science_hdu(hdus)
        if image.data is None or image.data.ndim != 2:
            raise ValueError("Science input must contain one two-dimensional image")
        header = hdus[0].header.copy()
        header.update(image.header)
        return header, tuple(int(value) for value in image.data.shape)


def _boundary(wcs: WCS, shape: tuple[int, int]) -> SkyCoord:
    height, width = shape
    positions = np.linspace(0, 1, 33)
    columns = np.concatenate(
        (positions * (width - 1), positions * (width - 1), np.zeros(33), np.full(33, width - 1))
    )
    rows = np.concatenate(
        (np.zeros(33), np.full(33, height - 1), positions * (height - 1), positions * (height - 1))
    )
    return wcs.celestial.pixel_to_world(columns, rows).icrs


def _mjd(header: fits.Header) -> float | None:
    value = header.get("MJD-OBS")
    if value is None and header.get("DATE-OBS"):
        from astropy.time import Time

        value = Time(header["DATE-OBS"]).mjd
    if value is None:
        return None
    if not np.isfinite(float(value)):
        raise ValueError("Science observation epoch must be finite")
    return float(value)


def _select_survey(boundary: SkyCoord, survey: str, band: str) -> str:
    available = []
    if np.min(boundary.dec.deg) >= -30 and band in SURVEY_FILTERS["panstarrs"]:
        available.append("panstarrs")
    if np.max(boundary.dec.deg) <= 0 and band in SURVEY_FILTERS["skymapper"]:
        available.append("skymapper")
    selected = available[0] if survey == "auto" and available else survey
    if selected not in available:
        raise ValueError("Requested survey/filter does not cover the complete science footprint")
    return selected


def plan_survey_mosaic(
    science_fits: str | Path,
    template_filter: str,
    *,
    survey: str = "auto",
    science_filter: str | None = None,
    single_tile_overrun: float = 0.05,
) -> MosaicPlan:
    """Plan a non-mutating survey-reference preparation operation.

    A science field up to five percent larger than one safe survey request is
    cropped centrally by default.  Larger fields use overlapping tiles.  This
    threshold is explicit so callers can set it to zero to always mosaic.
    """
    if survey not in {"auto", *SURVEY_FILTERS}:
        raise ValueError("Survey must be auto, panstarrs, or skymapper")
    if not np.isfinite(single_tile_overrun) or not 0 <= single_tile_overrun <= 0.25:
        raise ValueError("single_tile_overrun must be between zero and 0.25")
    path = Path(science_fits).expanduser().resolve()
    header, shape = _science_header_and_shape(path)
    wcs = WCS(header)
    if wcs.pixel_n_dim != 2 or not wcs.has_celestial:
        raise ValueError("Survey references require valid two-dimensional celestial WCS")
    boundary = _boundary(wcs, shape)
    center = wcs.celestial.pixel_to_world((shape[1] - 1) / 2, (shape[0] - 1) / 2).icrs
    radius = float(np.max(center.separation(boundary).deg))
    if not np.isfinite(radius) or not 0 < radius < 1:
        raise ValueError("Science WCS footprint is invalid or too large for survey cutouts")
    band = str(template_filter).strip().lower()
    if len(band) != 1:
        raise ValueError("Specify one supported template filter")
    selected = _select_survey(boundary, survey, band)
    native = str(header.get("FILTER", "")).strip()
    label = normalize_filter(science_filter if science_filter is not None else native)
    size = max(2 * radius * 1.1, 32 / 3600)
    limit = TILE_LIMIT_DEG[selected]
    mode: Literal["single", "crop", "mosaic"]
    bounds = None
    if size <= limit:
        mode = "single"
    elif size <= limit * (1 + single_tile_overrun):
        mode = "crop"
        # A proportional, centred crop leaves a rounding margin below the
        # request limit.  It is refined before downloading in execute_plan.
        factor = min(1.0, 0.99 * limit / size)
        height, width = shape
        crop_height = max(2, min(height, int(math.floor(height * factor))))
        crop_width = max(2, min(width, int(math.floor(width * factor))))
        row0, column0 = (height - crop_height) // 2, (width - crop_width) // 2
        bounds = (row0, row0 + crop_height, column0, column0 + crop_width)
    else:
        mode = "mosaic"
    return MosaicPlan(
        path,
        selected,
        band,
        label,
        native,
        float(center.ra.deg),
        float(center.dec.deg),
        size,
        _mjd(header),
        limit,
        mode,
        bounds,
    )


def _crop_science(source: Path, destination: Path, bounds: tuple[int, int, int, int]) -> None:
    row0, row1, column0, column1 = bounds
    with fits.open(source, memmap=False) as hdus:
        copied = fits.HDUList([hdu.copy() for hdu in hdus])
    image = _science_hdu(copied)
    shape = image.data.shape
    if not (0 <= row0 < row1 <= shape[0] and 0 <= column0 < column1 <= shape[1]):
        copied.close()
        raise ValueError("Invalid centred crop bounds")
    for hdu in copied:
        data = getattr(hdu, "data", None)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape == shape:
            hdu.data = np.array(data[row0:row1, column0:column1], copy=True)
            if "CRPIX1" in hdu.header:
                hdu.header["CRPIX1"] -= column0
            if "CRPIX2" in hdu.header:
                hdu.header["CRPIX2"] -= row0
    copied[0].header.add_history("Survey mosaic preparer wrote a centred WCS-preserving crop.")
    with NamedTemporaryFile(dir=destination.parent, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        copied.writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(destination)
    finally:
        copied.close()
        temporary.unlink(missing_ok=True)


def _copy_fits(source: Path, destination: Path) -> None:
    with fits.open(source, memmap=False) as hdus:
        copied = fits.HDUList([hdu.copy() for hdu in hdus])
    with NamedTemporaryFile(dir=destination.parent, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        copied.writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(destination)
    finally:
        copied.close()
        temporary.unlink(missing_ok=True)


def _request_from_plan(
    plan: MosaicPlan, science: Path, ra: float, dec: float, size: float
) -> TemplateRequest:
    return TemplateRequest(
        science,
        plan.survey,
        plan.template_filter,
        plan.science_filter,
        plan.native_science_filter,
        ra,
        dec,
        size,
        plan.science_mjd,
    )


def _mosaic_wcs(
    science_wcs: WCS, shape: tuple[int, int], scale_deg: float
) -> tuple[WCS, tuple[int, int]]:
    boundary = _boundary(science_wcs, shape)
    center = science_wcs.celestial.pixel_to_world((shape[1] - 1) / 2, (shape[0] - 1) / 2).icrs
    output = WCS(naxis=2)
    output.wcs.crpix = [1.0, 1.0]
    output.wcs.crval = [center.ra.deg, center.dec.deg]
    output.wcs.cdelt = [-scale_deg, scale_deg]
    output.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    columns, rows = output.all_world2pix(boundary.ra.deg, boundary.dec.deg, 0)
    margin = 2
    min_column, max_column = float(np.min(columns)), float(np.max(columns))
    min_row, max_row = float(np.min(rows)), float(np.max(rows))
    output.wcs.crpix = [1 - min_column + margin, 1 - min_row + margin]
    width = int(math.ceil(max_column - min_column)) + 1 + 2 * margin
    height = int(math.ceil(max_row - min_row)) + 1 + 2 * margin
    return output, (height, width)


def _tile_positions(length: int, span: float) -> np.ndarray:
    if span <= 2:
        raise ValueError("Survey tile pixel scale is invalid")
    if length <= span:
        return np.array([(length - 1) / 2])
    first, last = span / 2, (length - 1) - span / 2
    values = list(np.arange(first, last + 1e-8, max(1.0, span * 0.9)))
    if not values or last - values[-1] > 1e-6:
        values.append(last)
    return np.asarray(values)


def _tile_requests(
    plan: MosaicPlan, output_wcs: WCS, shape: tuple[int, int], scale_deg: float, science: Path
) -> list[TemplateRequest]:
    span = plan.tile_limit_deg / scale_deg
    rows, columns = _tile_positions(shape[0], span), _tile_positions(shape[1], span)
    requests = []
    for row in rows:
        for column in columns:
            sky = output_wcs.pixel_to_world(column, row).icrs
            requests.append(
                _request_from_plan(
                    plan, science, float(sky.ra.deg), float(sky.dec.deg), plan.tile_limit_deg
                )
            )
    return requests


def _tile_image(
    template_path: Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray | None, np.ndarray | None]:
    with fits.open(template_path, memmap=False) as hdus:
        image = _science_hdu(hdus)
        header = hdus[0].header.copy()
        header.update(image.header)
        data = np.asarray(image.data, dtype=float)
        variance = np.asarray(hdus["VAR"].data, dtype=float) if "VAR" in hdus else None
        mask = np.asarray(hdus["MASK"].data) if "MASK" in hdus else None
    return data, header, variance, mask


def _write_mosaic(
    templates: list[Path],
    science_header: fits.Header,
    science_shape: tuple[int, int],
    destination: Path,
) -> dict:
    _, first_header, _, _ = _tile_image(templates[0])
    source_wcs = WCS(first_header).celestial
    scales = proj_plane_pixel_scales(source_wcs)
    scale_deg = float(np.mean(np.abs(scales)))
    if not np.isfinite(scale_deg) or scale_deg <= 0:
        raise ValueError("Downloaded survey tile has an invalid pixel scale")
    output_wcs, shape = _mosaic_wcs(WCS(science_header).celestial, science_shape, scale_deg)
    data_sum = np.zeros(shape, dtype=float)
    weights = np.zeros(shape, dtype=float)
    variance_sum = np.zeros(shape, dtype=float)
    variance_weights = np.zeros(shape, dtype=float)
    all_variance = True
    for path in templates:
        data, header, variance, mask = _tile_image(path)
        if mask is not None:
            data = np.where(mask == 0, data, np.nan)
        projected, footprint = reproject_interp(
            (data, WCS(header).celestial), output_wcs, shape_out=shape, order="bilinear"
        )
        valid = np.isfinite(projected) & (footprint > 0)
        data_sum[valid] += projected[valid] * footprint[valid]
        weights[valid] += footprint[valid]
        if variance is None:
            all_variance = False
        else:
            if mask is not None:
                variance = np.where(mask == 0, variance, np.nan)
            projected_variance, variance_footprint = reproject_interp(
                (variance, WCS(header).celestial), output_wcs, shape_out=shape, order="bilinear"
            )
            variance_valid = np.isfinite(projected_variance) & (variance_footprint > 0)
            variance_sum[variance_valid] += (
                projected_variance[variance_valid] * variance_footprint[variance_valid]
            )
            variance_weights[variance_valid] += variance_footprint[variance_valid]
    mosaic = np.full(shape, np.nan, dtype=float)
    valid = weights > 0
    mosaic[valid] = data_sum[valid] / weights[valid]
    # Require coverage at every science pixel after mapping the science footprint
    # to the mosaic grid.  Empty corners of the rectangular TAN mosaic are fine.
    science_footprint, science_coverage = reproject_interp(
        (np.ones(science_shape), WCS(science_header).celestial),
        output_wcs,
        shape_out=shape,
        order="nearest-neighbor",
    )
    required = (science_coverage > 0) & np.isfinite(science_footprint)
    if not np.all(valid[required]):
        raise ValueError("Downloaded survey tiles do not cover the complete science footprint")
    header = first_header.copy()
    header.update(output_wcs.to_header(relax=True))
    header["NAXIS1"], header["NAXIS2"] = shape[1], shape[0]
    header["MOSAIC"] = (True, "Tiled survey mosaic prepared outside pipeline")
    header["MTILES"] = (len(templates), "Number of downloaded survey tiles")
    header.add_history("Survey mosaic preparer combined overlapping survey cutouts.")
    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = [fits.PrimaryHDU(mosaic, header)]
    hdus.append(fits.ImageHDU((~valid).astype(np.uint8), name="MASK"))
    if all_variance and np.all(variance_weights[required] > 0):
        variance_output = np.full(shape, np.nan, dtype=float)
        variance_valid = variance_weights > 0
        variance_output[variance_valid] = (
            variance_sum[variance_valid] / variance_weights[variance_valid]
        )
        hdus.append(fits.ImageHDU(variance_output, name="VAR"))
    with NamedTemporaryFile(dir=destination.parent, suffix=".fits", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "mosaic_shape": [shape[0], shape[1]],
        "mosaic_pixel_scale_arcsec": scale_deg * 3600,
        "tiles": len(templates),
        "variance_plane": all_variance,
    }


def prepare_survey_mosaic(
    science_fits: str | Path,
    output_dir: str | Path,
    *,
    template_filter: str,
    survey: str = "auto",
    cache_dir: str | Path = "survey_cache",
    science_filter: str | None = None,
    timeout: float = 30,
    max_download_mb: float = 64,
    single_tile_overrun: float = 0.05,
    check_only: bool = False,
) -> dict:
    """Write local reference/science FITS inputs for the unchanged pipeline.

    The output directory must be new or empty.  It contains
    ``science_for_reference.fits``, ``survey_reference.fits``, and an auditable
    ``survey_mosaic.json`` plan.  ``check_only`` only derives the local plan.
    """
    plan = plan_survey_mosaic(
        science_fits,
        template_filter,
        survey=survey,
        science_filter=science_filter,
        single_tile_overrun=single_tile_overrun,
    )
    result = {"plan": plan.to_dict(), "output_written": False, "network_used": False}
    if check_only:
        return result
    root = Path(output_dir).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    if root == cache or root in cache.parents or cache in root.parents:
        raise ValueError("Survey cache and mosaic output directories must be separate")
    if root == plan.science_fits.parent or root in plan.science_fits.parents:
        raise ValueError("Mosaic output must not contain the science input")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Mosaic output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    science_output = root / "science_for_reference.fits"
    reference_output = root / "survey_reference.fits"
    try:
        if plan.mode == "crop":
            assert plan.crop_bounds is not None
            _crop_science(plan.science_fits, science_output, plan.crop_bounds)
            # Re-evaluate the exact cropped WCS and shrink once more if FITS
            # integer rounding left it at the provider's limit.
            request = prepare_template_request(
                science_output,
                plan.template_filter,
                survey=plan.survey,
                science_filter=plan.science_filter,
            )
            if request.size_deg > plan.tile_limit_deg:
                with fits.open(science_output, memmap=False) as hdus:
                    shape = _science_hdu(hdus).data.shape
                factor = 0.99 * plan.tile_limit_deg / request.size_deg
                height, width = shape
                bounds = (
                    (height - int(height * factor)) // 2,
                    (height + int(height * factor)) // 2,
                    (width - int(width * factor)) // 2,
                    (width + int(width * factor)) // 2,
                )
                _crop_science(science_output, science_output, bounds)
                request = prepare_template_request(
                    science_output,
                    plan.template_filter,
                    survey=plan.survey,
                    science_filter=plan.science_filter,
                )
        else:
            _copy_fits(plan.science_fits, science_output)
            request = None
        if plan.mode in {"single", "crop"}:
            if request is None:
                request = prepare_template_request(
                    science_output,
                    plan.template_filter,
                    survey=plan.survey,
                    science_filter=plan.science_filter,
                )
            template = fetch_survey_template(
                request, cache, timeout=timeout, max_download_mb=max_download_mb
            )
            _copy_fits(template.fits_path, reference_output)
            acquisition = {"tiles": 1, "template": template.metadata}
        else:
            header, shape = _science_header_and_shape(science_output)
            # Fetch the central tile first.  Its native survey scale defines a
            # stable TAN mosaic grid; the same tile is harmlessly reused from cache
            # if it appears in the later grid.
            central = fetch_survey_template(
                _request_from_plan(
                    plan, science_output, plan.ra_deg, plan.dec_deg, plan.tile_limit_deg
                ),
                cache,
                timeout=timeout,
                max_download_mb=max_download_mb,
            )
            _, central_header, _, _ = _tile_image(central.fits_path)
            scale = float(np.mean(np.abs(proj_plane_pixel_scales(WCS(central_header).celestial))))
            mosaic_wcs, mosaic_shape = _mosaic_wcs(WCS(header).celestial, shape, scale)
            requests = _tile_requests(plan, mosaic_wcs, mosaic_shape, scale, science_output)
            templates = [
                fetch_survey_template(item, cache, timeout=timeout, max_download_mb=max_download_mb)
                for item in requests
            ]
            acquisition = _write_mosaic(
                [item.fits_path for item in templates], header, shape, reference_output
            )
            acquisition["tile_metadata"] = [item.metadata for item in templates]
        result.update(
            output_written=True,
            network_used=True,
            science_fits=str(science_output),
            reference_fits=str(reference_output),
            acquisition=acquisition,
            warnings=[
                SURVEY_WARNING,
                "Run the normal pipeline with --ref/--new --reproject; tiled mosaics are "
                "resampled survey data.",
            ],
        )
        atomic_json(root / "survey_mosaic.json", result)
        return result
    except Exception as error:
        atomic_json(root / "survey_mosaic.json", {**result, "state": "failed", "error": str(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the standalone mosaic preparer from a source checkout or package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", required=True, help="Science FITS with celestial WCS")
    parser.add_argument("--output", required=True, help="New/empty preparation output directory")
    parser.add_argument("--template-filter", required=True, help="One survey reference band")
    parser.add_argument("--survey", choices=("auto", "panstarrs", "skymapper"), default="auto")
    parser.add_argument("--cache", default="survey_cache", help="Separate survey download cache")
    parser.add_argument("--science-filter", help="Override missing/ambiguous science FILTER")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-download-mb", type=float, default=64)
    parser.add_argument(
        "--single-tile-overrun",
        type=float,
        default=0.05,
        help="Centred-crop limit above one tile (default: 0.05; zero always mosaics)",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Plan locally without downloads/writes"
    )
    args = parser.parse_args(argv)
    report = prepare_survey_mosaic(
        args.science,
        args.output,
        template_filter=args.template_filter,
        survey=args.survey,
        cache_dir=args.cache,
        science_filter=args.science_filter,
        timeout=args.timeout,
        max_download_mb=args.max_download_mb,
        single_tile_overrun=args.single_tile_overrun,
        check_only=args.check_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
