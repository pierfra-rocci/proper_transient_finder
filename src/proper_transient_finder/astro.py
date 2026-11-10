# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""PSF-matched astronomical difference imaging and candidate vetting."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import sep
from astropy import units
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from scipy import ndimage, signal
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

from ._version import __version__

LOGGER = logging.getLogger(__name__)
CROSS_FILTER_WARNING = (
    "Experimental cross-filter residual; color differences can mimic variability. "
    "Flux errors exclude color mismatch; scores are not calibrated variability significance."
)
CANDIDATE_COLUMNS = [
    "id",
    "x_centroid",
    "y_centroid",
    "sign",
    "flux",
    "flux_error",
    "peak_value",
    "snr",
    "npixels",
    "fwhm",
    "fwhm_ratio",
    "elongation",
    "psf_correlation",
    "dipole_ratio",
    "reduced_chi2",
    "ra_deg",
    "dec_deg",
    "flags",
]
BACKGROUND_COLUMNS = [
    "id",
    "valid_annulus_pixels",
    "local_background_snr",
    "local_gradient_snr",
    "local_structure_rms",
    "local_background_warning",
]


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a complete JSON document without exposing a partially written status."""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class ImageFrame:
    """Image samples and noise in the same detector count units."""

    data: np.ndarray
    header: fits.Header
    mask: np.ndarray
    variance: np.ndarray
    background: np.ndarray
    gain: float
    variance_bound: np.ndarray | None = None

    @property
    def noise_variance(self) -> np.ndarray:
        """Variance or a covariance row-sum bound for subsequent linear filtering."""
        return self.variance if self.variance_bound is None else self.variance_bound


def pixel_solid_angles(wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    """Pixel areas in steradians from spherical quadrilaterals, including distortion."""
    row, column = np.indices((shape[0] + 1, shape[1] + 1), dtype=float)
    corners = wcs.pixel_to_world(column - 0.5, row - 0.5)
    vectors = np.moveaxis(corners.cartesian.xyz.value, 0, -1)
    first, second = vectors[:-1, :-1], vectors[:-1, 1:]
    third, fourth = vectors[1:, 1:], vectors[1:, :-1]
    area = np.zeros(shape)
    for middle, last in ((second, third), (third, fourth)):
        determinant = np.einsum("...i,...i->...", first, np.cross(middle - first, last - first))
        denominator = (
            1
            + np.einsum("...i,...i->...", first, middle)
            + np.einsum("...i,...i->...", middle, last)
            + np.einsum("...i,...i->...", last, first)
        )
        area += 2 * np.arctan2(np.abs(determinant), denominator)
    return area


def reproject_frame(
    reference: ImageFrame, science: ImageFrame, *, unit_mode: str = "flux-per-pixel"
) -> tuple[ImageFrame, np.ndarray]:
    """Bilinearly reproject a reference and its noise onto a science celestial WCS.

    For independent input noise, R is the nonnegative interpolation operator:
    diagonal covariance is R**2 @ VAR; C @ 1 = R @ (VAR * (R.T @ 1)) bounds
    subsequent filtering variance via the covariance row sums, without storing C.
    """
    from reproject import reproject_interp

    if unit_mode not in {"flux-per-pixel", "surface-brightness"}:
        raise ValueError("Reprojection units must be flux-per-pixel or surface-brightness")
    if reference.variance_bound is not None:
        raise ValueError("Refusing a second reprojection of already correlated samples")
    input_wcs, output_wcs = WCS(reference.header), WCS(science.header)
    if not all(wcs.pixel_n_dim == 2 and wcs.has_celestial for wcs in (input_wcs, output_wcs)):
        raise ValueError(
            "WCS reprojection requires valid two-dimensional celestial WCS in both inputs"
        )
    input_wcs, output_wcs = input_wcs.celestial, output_wcs.celestial
    input_area = pixel_solid_angles(input_wcs, reference.data.shape)
    output_area = pixel_solid_angles(output_wcs, science.data.shape)
    input_valid = ~reference.mask & np.isfinite(input_area) & (input_area > 0)
    surface = (
        np.divide(reference.data, input_area, out=np.zeros_like(reference.data), where=input_valid)
        if unit_mode == "flux-per-pixel"
        else reference.data
    )
    projected, library_footprint = reproject_interp(
        (np.where(input_valid, surface, 0), input_wcs),
        output_wcs,
        shape_out=science.data.shape,
        order="bilinear",
        roundtrip_coords=True,
    )
    if unit_mode == "flux-per-pixel":
        projected *= output_area
    rows, columns = np.indices(science.data.shape, dtype=float)
    input_x, input_y = pixel_to_pixel(output_wcs, input_wcs, columns, rows)
    finite = np.isfinite(input_x) & np.isfinite(input_y)
    input_x = np.where(finite, input_x, 0)
    input_y = np.where(finite, input_y, 0)
    inside = finite & (input_x >= 0) & (input_y >= 0)
    inside &= (input_x <= reference.data.shape[1] - 1) & (input_y <= reference.data.shape[0] - 1)
    input_x, input_y = np.where(inside, input_x, 0), np.where(inside, input_y, 0)
    left, bottom = np.floor(input_x).astype(int), np.floor(input_y).astype(int)
    fraction_x, fraction_y = input_x - left, input_y - bottom
    valid = inside & (library_footprint > 0) & np.isfinite(projected)
    valid &= np.isfinite(output_area) & (output_area > 0)
    output_indices, input_indices, coefficients = [], [], []
    for delta_y, weight_y in ((0, 1 - fraction_y), (1, fraction_y)):
        for delta_x, weight_x in ((0, 1 - fraction_x), (1, fraction_x)):
            weight = weight_y * weight_x
            native_row, native_column = bottom + delta_y, left + delta_x
            in_bounds = (native_row < reference.data.shape[0]) & (
                native_column < reference.data.shape[1]
            )
            native_row = np.clip(native_row, 0, reference.data.shape[0] - 1)
            native_column = np.clip(native_column, 0, reference.data.shape[1] - 1)
            contributing = weight > 1e-10
            valid &= ~contributing | (in_bounds & input_valid[native_row, native_column])
            factor = weight
            if unit_mode == "flux-per-pixel":
                factor = weight * np.divide(
                    output_area,
                    input_area[native_row, native_column],
                    out=np.zeros_like(weight),
                    where=input_valid[native_row, native_column],
                )
            output_indices.append(np.flatnonzero(contributing))
            input_indices.append(
                np.ravel_multi_index(
                    (native_row[contributing], native_column[contributing]), reference.data.shape
                )
            )
            coefficients.append(factor[contributing])
    if np.count_nonzero(valid) < 100:
        raise ValueError("Insufficient valid WCS overlap for image subtraction")
    coefficient_rows = np.concatenate(output_indices)
    coefficient_columns = np.concatenate(input_indices)
    coefficient_values = np.concatenate(coefficients)
    keep = valid.ravel()[coefficient_rows] & np.isfinite(coefficient_values)
    operator = csr_array(
        (coefficient_values[keep], (coefficient_rows[keep], coefficient_columns[keep])),
        shape=(science.data.size, reference.data.size),
    )
    expected = (operator @ reference.data.ravel()).reshape(science.data.shape)
    if not np.allclose(projected[valid], expected[valid], rtol=1e-6, atol=1e-7):
        raise ValueError("Reprojection image and noise interpolation weights disagree")
    variance = (operator.power(2) @ reference.variance.ravel()).reshape(science.data.shape)
    column_sum = operator.T @ np.ones(science.data.size)
    bound = (operator @ (reference.variance.ravel() * column_sum)).reshape(science.data.shape)
    background = (operator @ reference.background.ravel()).reshape(science.data.shape)
    header = science.header.copy()
    header["BUNIT"] = reference.header.get("BUNIT", "adu")
    return ImageFrame(
        np.where(valid, projected, 0),
        header,
        ~valid,
        variance,
        background,
        reference.gain,
        bound,
    ), valid.astype(float)


def reprojection_unit_mode(header: fits.Header, requested: str) -> str:
    """Infer only recognized flux/brightness units; require a choice for other units."""
    if requested != "auto":
        return requested
    label = str(header.get("BUNIT", "")).strip()
    if label.casefold().replace(" ", "") in {
        "adu",
        "dn",
        "count",
        "counts",
        "electron",
        "electrons",
        "e-",
        "adu/s",
        "dn/s",
        "count/s",
        "counts/s",
        "electron/s",
        "electrons/s",
        "e-/s",
    }:
        return "flux-per-pixel"
    try:
        parsed = units.Unit(label) if label else None
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.is_equivalent(units.Jy / units.sr):
            return "surface-brightness"
        if parsed.is_equivalent(units.Jy):
            return "flux-per-pixel"
    raise ValueError(
        "Ambiguous BUNIT for reprojection; set --reproject-units explicitly to "
        "flux-per-pixel or surface-brightness"
    )


def convolve_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear, zero-padded convolution, with no interpolation of invalid samples."""
    return convolve_fft(
        image,
        kernel,
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        normalize_kernel=False,
        fft_pad=True,
        psf_pad=True,
    )


def normalize_psf(psf: np.ndarray, radius: int = 12) -> np.ndarray:
    """Center and normalize a compact, nonnegative intensity PSF."""
    image = real_image(psf, "PSF")
    if not np.isfinite(image).all() or image.sum() <= 0:
        raise ValueError("PSF must have finite samples and positive integrated flux")
    if image.min() < -0.01 * image.max():
        raise ValueError("PSF has substantial negative lobes; inspect the PSF stars")
    image = np.maximum(image, 0.0)
    center = ndimage.center_of_mass(image)
    padding = radius + max(image.shape)
    padded = np.pad(image, padding)
    target = np.array(padded.shape) // 2
    centered = ndimage.shift(
        padded, target - (np.array(center) + padding), order=3, mode="constant"
    )
    stamp = centered[
        target[0] - radius : target[0] + radius + 1,
        target[1] - radius : target[1] + radius + 1,
    ]
    stamp = np.maximum(stamp, 0.0)
    if stamp.sum() < 0.995 * image.sum():
        raise ValueError("PSF extends beyond the kernel; increase --psf-radius")
    return stamp / stamp.sum()


def psf_fwhm(psf: np.ndarray) -> float:
    """Circularized Gaussian-equivalent FWHM from intensity second moments."""
    row, column = np.indices(psf.shape)
    center_row, center_column = ndimage.center_of_mass(psf)
    moment = np.sum(psf * ((row - center_row) ** 2 + (column - center_column) ** 2))
    return float(2.35482 * np.sqrt(moment / (2 * psf.sum())))


def expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Exclude the full square support of a convolution, including image borders."""
    return ndimage.maximum_filter(
        np.asarray(mask, dtype=np.uint8), size=2 * radius + 1, mode="constant", cval=1
    ).astype(bool)


def _science_hdu(hdus: fits.HDUList):
    image_hdus = [
        hdu
        for hdu in hdus
        if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
        and hdu.data is not None
        and hdu.data.ndim == 2
        and hdu.name.upper() not in {"DQ", "MASK", "BPM", "VAR", "VARIANCE", "ERR"}
    ]
    science_hdus = [hdu for hdu in image_hdus if hdu.name == "SCI"]
    if len(science_hdus) > 1:
        raise ValueError("Multiple SCI extensions are ambiguous; split detectors first")
    if science_hdus:
        return science_hdus[0]
    if len(image_hdus) != 1:
        raise ValueError("FITS must contain one science plane or a single SCI extension")
    return image_hdus[0]


def load_frame(
    path: Path,
    *,
    mask_path: Path | None = None,
    gain: float | None = None,
    saturation: float | None = None,
    cosmic_rays: bool = True,
) -> ImageFrame:
    """Read a SCI/primary image, DQ mask, and optional VAR or ERR extension."""
    if not path.is_file():
        raise ValueError(f"Input FITS file does not exist: {path}")
    with fits.open(path, memmap=False) as hdus:
        science = _science_hdu(hdus)
        data = real_image(science.data, "Science image")
        header = hdus[0].header.copy()
        header.update(science.header)
        mask = ~np.isfinite(data)
        variance = None
        for extension in hdus:
            name = extension.name.upper()
            if name not in {"DQ", "MASK", "BPM", "VAR", "VARIANCE", "ERR"}:
                continue
            if extension.data is None or extension.data.shape != data.shape:
                raise ValueError(f"{name} extension does not match the science image")
            if name in {"DQ", "MASK", "BPM"}:
                mask |= (extension.data != 0) | ~np.isfinite(extension.data)
            elif variance is None:
                variance = real_image(extension.data, name)
                if name == "ERR":
                    mask |= variance <= 0
                    variance = variance**2
    if mask_path is not None:
        external = fits.getdata(mask_path, memmap=False)
        if external.shape != data.shape:
            raise ValueError("External mask does not match the science image")
        mask |= (external != 0) | ~np.isfinite(external)
    gain_value = gain if gain is not None else header.get("GAIN", 1.0)
    gain_value = float(gain_value)
    if not np.isfinite(gain_value) or gain_value <= 0:
        raise ValueError("GAIN must be finite and positive (electrons per image count)")
    if gain is None and "GAIN" not in header and variance is None:
        LOGGER.warning("No GAIN or variance plane: assuming 1 electron/count for %s", path.name)
    saturation_value = saturation if saturation is not None else header.get("SATURATE")
    if saturation_value is not None:
        saturation_value = float(saturation_value)
        if not np.isfinite(saturation_value) or saturation_value <= 0:
            raise ValueError("Saturation must be finite and positive, in image counts")
        mask |= data >= saturation_value
    else:
        LOGGER.warning("No saturation level for %s; supply --saturation", path.name)
    if variance is not None:
        mask |= ~np.isfinite(variance) | (variance <= 0)
    elif str(header.get("BUNIT", "")).strip().casefold() not in {
        "",
        "adu",
        "dn",
        "count",
        "counts",
        "electron",
        "electrons",
        "e-",
    }:
        raise ValueError("Non-count image units require a total VAR/ERR plane")
    if min(data.shape) < 64 or np.count_nonzero(~mask) < 0.5 * data.size:
        raise ValueError("Images must be at least 64x64 with at least half the pixels valid")
    filled = np.where(mask, np.median(data[~mask]), data)
    background = sep.Background(np.ascontiguousarray(filled), mask=mask, bw=32, bh=32)
    sky = background.back()
    if variance is None:
        variance = background.rms() ** 2 + np.maximum(filled - sky, 0) / gain_value
    if cosmic_rays:
        import astroscrappy

        cosmic_mask, _ = astroscrappy.detect_cosmics(
            np.asarray(filled, dtype=np.float32),
            inmask=mask,
            invar=np.asarray(variance, dtype=np.float32),
            gain=gain_value,
            satlevel=saturation_value if saturation_value is not None else np.inf,
            sigclip=5.0,
            objlim=10.0,
            niter=4,
        )
        mask |= cosmic_mask
        LOGGER.info("%s: masked %d cosmic-ray pixels", path.name, cosmic_mask.sum())
    mask = expand_mask(mask, 1)
    clean = np.where(mask, 0.0, data - sky)
    variance = np.where(mask, 0.0, variance)
    return ImageFrame(clean, header, mask, variance, sky, gain_value)


def kernel_subtract(
    reference: ImageFrame,
    new: ImageFrame,
    ref_psf: np.ndarray,
    new_psf: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """Cross-convolve PSFs and propagate covariance through a matched filter."""
    if reference.data.shape != new.data.shape:
        raise ValueError("Images must share a pixel grid before subtraction")
    matched_ref = convolve_image(reference.data, new_psf)
    matched_new = convolve_image(new.data, ref_psf)
    difference = matched_new - scale * matched_ref
    variance = convolve_image(new.noise_variance, ref_psf**2) + scale**2 * convolve_image(
        reference.noise_variance, new_psf**2
    )
    effective_psf = signal.fftconvolve(ref_psf, new_psf, mode="full")
    effective_psf /= effective_psf.sum()
    matched_filter = effective_psf[::-1, ::-1]
    new_weight = signal.fftconvolve(ref_psf, matched_filter, mode="full")
    ref_weight = signal.fftconvolve(new_psf, matched_filter, mode="full")
    numerator = convolve_image(difference, matched_filter)
    score_variance = convolve_image(new.noise_variance, new_weight**2) + scale**2 * convolve_image(
        reference.noise_variance, ref_weight**2
    )
    radius = max(new_weight.shape + ref_weight.shape) // 2
    mask = expand_mask(reference.mask | new.mask, radius)
    mask |= ~np.isfinite(score_variance) | (score_variance <= 0)
    score = np.divide(
        numerator,
        np.sqrt(np.maximum(score_variance, 0)),
        out=np.zeros_like(numerator),
        where=~mask,
    )
    normalization = np.sum(effective_psf**2)
    return {
        "difference": difference,
        "variance": variance,
        "psf": effective_psf,
        "score": score,
        "mask": mask,
        "matched_reference": scale * matched_ref,
        "matched_new": matched_new,
        "flux": numerator / normalization,
        "flux_error": np.sqrt(np.maximum(score_variance, 0)) / normalization,
    }


def real_image(data: np.ndarray, name: str) -> np.ndarray:
    """Accept real images or inverse FFTs with negligible imaginary leakage."""
    array = np.asarray(data)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty two-dimensional image")
    if np.iscomplexobj(array):
        finite = np.isfinite(array)
        if not finite.any():
            raise ValueError(f"{name} has no finite pixels")
        scale = max(float(np.max(np.abs(array.real[finite]))), 1.0)
        if np.max(np.abs(array.imag[finite])) > scale * 1e-6:
            raise ValueError(f"{name} has significant imaginary components")
        array = array.real
    return np.array(array, dtype=np.float64, order="C", copy=True)


def calibrate_score(
    score: np.ndarray, mask: np.ndarray, *, allow_amplification: bool = False
) -> tuple[np.ndarray, float, float]:
    """Empirically standardize the unmasked background, without clipping sources."""
    image = real_image(score, "Detection score")
    invalid = np.asarray(mask, dtype=bool) | ~np.isfinite(image)
    if invalid.shape != image.shape or np.count_nonzero(~invalid) < 100:
        raise ValueError("Too few valid pixels to calibrate the detection score")
    _, center, scale = sigma_clipped_stats(
        image, mask=invalid, sigma=3.0, maxiters=10, stdfunc="mad_std"
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Detection score has zero or invalid background noise")
    scale = float(scale) if allow_amplification else max(float(scale), 1.0)
    calibrated = (image - center) / scale
    calibrated[invalid] = np.nan
    return calibrated, float(center), float(scale)


@dataclass
class PipelineConfig:
    """Explicit assumptions and conservative candidate-selection defaults."""

    method: str = "kernel"
    smooth_psf: bool = False
    fitted_psf: bool = True
    align: bool = False
    reproject: bool = False
    reproject_units: str = "auto"
    iterative: bool = False
    beta: bool = True
    shift: bool = False
    inf_loss: float = 0.15
    gain: float | None = None
    saturation: float | None = None
    cosmic_rays: bool = True
    psf_radius: int = 8
    min_stars: int = 6
    alignment_tolerance: float = 0.35
    flux_scale: float | None = None
    max_scale_scatter: float = 0.15
    ref_mask: str | None = None
    new_mask: str | None = None
    ref_psf: str | None = None
    new_psf: str | None = None
    min_pixels: int = 3
    min_fwhm_ratio: float = 0.6
    max_fwhm_ratio: float = 1.8
    max_elongation: float = 2.0
    max_dipole_ratio: float = 0.5
    min_psf_correlation: float = 0.65
    max_candidates: int = 1000
    plots: bool = True
    overwrite: bool = False
    cross_filter: bool = False
    color_association_fwhm: float = 1.0
    color_reference_snr: float = 5.0
    color_quantile: float = 0.95
    color_error_margin: float = 3.0
    science_filter: str | None = None

    def __post_init__(self) -> None:
        if self.science_filter is not None:
            if (
                not isinstance(self.science_filter, str)
                or not self.science_filter.strip()
                or self.science_filter.strip().lower() == "unknown"
                or not self.science_filter.isascii()
                or not self.science_filter.isprintable()
            ):
                raise ValueError("science_filter must be a nonempty printable ASCII filter label")
            self.science_filter = self.science_filter.strip()
        if not self.fitted_psf:
            raise ValueError(
                "Empirical PSF mode is disabled; use Gaussian PSFs or provide "
                "--ref-psf and --new-psf with --method kernel"
            )
        if self.method not in {"kernel", "proper"}:
            raise ValueError("Subtraction method must be 'kernel' or 'proper'")
        if self.cross_filter and self.method != "kernel":
            raise ValueError("Experimental cross-filter comparison requires --method kernel")
        if self.reproject_units not in {"auto", "flux-per-pixel", "surface-brightness"}:
            raise ValueError("Unknown reprojection unit mode")
        if self.reproject:
            if self.align or self.shift:
                raise ValueError(
                    "Use --reproject or --align/--shift, not both; avoid double resampling"
                )
            if self.method != "kernel":
                raise ValueError("WCS reprojection requires --method kernel for noise propagation")
            if self.ref_psf or self.new_psf:
                raise ValueError(
                    "WCS reprojection requires automatic PSF estimation on the output grid"
                )
        for name in (
            "alignment_tolerance",
            "max_scale_scatter",
            "min_fwhm_ratio",
            "max_fwhm_ratio",
            "max_elongation",
            "max_dipole_ratio",
            "color_association_fwhm",
            "color_reference_snr",
            "color_error_margin",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (
            ("psf_radius", 3),
            ("min_stars", 6),
            ("min_pixels", 1),
            ("max_candidates", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not 0 < self.inf_loss < 1:
            raise ValueError("Information loss must be between 0 and 1, exclusively")
        if not 0 < self.min_psf_correlation <= 1:
            raise ValueError("PSF correlation must be in (0, 1]")
        if not 0 < self.color_quantile <= 1:
            raise ValueError("color_quantile must be in (0, 1]")
        if self.min_fwhm_ratio >= self.max_fwhm_ratio:
            raise ValueError("Minimum FWHM ratio must be less than the maximum")
        for name in ("gain", "saturation", "flux_scale"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (self.ref_psf is None) != (self.new_psf is None):
            raise ValueError("Provide both --ref-psf and --new-psf, or neither")
        if self.method == "proper" and (self.ref_psf or self.flux_scale is not None):
            raise ValueError("External PSFs and manual flux scale require --method kernel")


def star_catalog(frame: ImageFrame) -> np.ndarray:
    """Select isolated, unmasked SEP detections for PSF and registration checks."""
    errors = np.sqrt(np.maximum(frame.variance, np.finfo(float).tiny))
    objects = sep.extract(
        frame.data,
        5.0,
        err=errors,
        mask=frame.mask,
        minarea=5,
        deblend_cont=0.005,
    )
    good = (
        (objects["flag"] == 0)
        & (objects["b"] > 0)
        & (objects["a"] < 1.6 * objects["b"])
        & (objects["flux"] > 0)
    )
    objects = objects[good]
    if len(objects):
        objects = objects[np.argsort(objects["flux"])[::-1]][:300]
        forbidden = expand_mask(frame.mask, 5)
        good = ~forbidden[np.rint(objects["y"]).astype(int), np.rint(objects["x"]).astype(int)]
        objects = objects[good]
    return objects


def matched_stars(
    reference: np.ndarray, new: np.ndarray, radius: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Find unique, mutual nearest neighbors, not many-to-one catalog matches."""
    if not len(reference) or not len(new):
        raise ValueError("No suitable stars detected for registration")
    ref_xy = np.column_stack((reference["x"], reference["y"]))
    new_xy = np.column_stack((new["x"], new["y"]))
    distances, new_indices = cKDTree(new_xy).query(ref_xy)
    _, reverse = cKDTree(ref_xy).query(new_xy)
    selected = (distances < radius) & (reverse[new_indices] == np.arange(len(reference)))
    return reference[selected], new[new_indices[selected]]


class ProperTransientFinder:
    """Validated PSF matching, signed transient detection, and artifact vetting."""

    def __init__(
        self,
        ref_path: str | PathLike[str],
        new_path: str | PathLike[str],
        output_dir: str | PathLike[str],
        config: PipelineConfig | Mapping[str, object] | None = None,
    ) -> None:
        self.ref_path = Path(ref_path).expanduser().resolve()
        self.new_path = Path(new_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        try:
            settings = asdict(config) if isinstance(config, PipelineConfig) else config
            self.options = PipelineConfig(**(settings if settings is not None else {}))
        except TypeError as error:
            raise ValueError("Unknown pipeline configuration option") from error
        self.config = asdict(self.options)
        if self.ref_path == self.new_path:
            raise ValueError("Reference and new image must be different files")
        self.reference = None
        self.new = None
        self.ref_data = None
        self.new_data = None
        self.diff_image = None
        self.psf_image = None
        self.scorr_image = None
        self.mask = None
        self.ref_psf = None
        self.new_psf = None
        self.reprojection_footprint = None
        self.astrometric_residuals = pd.DataFrame()
        self.products = {}
        self.rejected_sources = pd.DataFrame()
        self.local_background_diagnostics = pd.DataFrame(columns=BACKGROUND_COLUMNS)
        self._status_path = None
        self.metadata = {
            "config": self.config,
            "pixel_origin": 0,
            "reference_file": self.ref_path.name,
            "new_file": self.new_path.name,
        }

    def load_fits_image(self, fits_path):
        frame = load_frame(Path(fits_path), cosmic_rays=False)
        return frame.data + frame.background, frame.header

    def _load_inputs(self, *, cosmic_rays: bool) -> None:
        options = self.options
        frames = []
        for path, mask in ((self.ref_path, options.ref_mask), (self.new_path, options.new_mask)):
            frames.append(
                load_frame(
                    path,
                    mask_path=Path(mask) if mask else None,
                    gain=options.gain,
                    saturation=options.saturation,
                    cosmic_rays=cosmic_rays,
                )
            )
        self.reference, self.new = frames
        if options.science_filter is not None:
            original_filter = self.new.header.get("FILTER")
            self.metadata["science_filter_override"] = {
                "original": str(original_filter) if original_filter is not None else None,
                "declared": options.science_filter,
            }
            self.new.header["FILTER"] = options.science_filter
            LOGGER.warning(
                "Using user-declared science FILTER %r instead of header value %r; "
                "verify the actual bandpass",
                options.science_filter,
                original_filter,
            )
        self.new_header = self.new.header.copy()
        self.metadata["input_observation_dates"] = {
            label: {
                keyword: str(frame.header[keyword])
                for keyword in ("DATE-OBS", "MJD-OBS")
                if keyword in frame.header
            }
            for label, frame in zip(("reference", "science"), frames, strict=True)
        }
        for keyword in ("FILTER", "BUNIT"):
            ref_value = self.reference.header.get(keyword)
            new_value = self.new.header.get(keyword)
            if keyword == "FILTER" and options.cross_filter:
                if not all(
                    value is not None and str(value).strip() for value in (ref_value, new_value)
                ):
                    raise ValueError(
                        "Experimental cross-filter comparison requires FILTER in both inputs"
                    )
                self.metadata.update(
                    comparison_mode="experimental cross-filter residual",
                    reference_filter=str(ref_value).strip(),
                    science_filter=str(new_value).strip(),
                    cross_filter_warning=CROSS_FILTER_WARNING,
                )
                LOGGER.warning(
                    "%s Reference %s; science %s", CROSS_FILTER_WARNING, ref_value, new_value
                )
                continue
            if ref_value and new_value and str(ref_value).strip() != str(new_value).strip():
                raise ValueError(f"Input {keyword} values differ; use calibrated compatible images")

    def _validate_output_directory(self) -> None:
        if self.output_dir in {self.ref_path.parent, self.new_path.parent}:
            raise ValueError("Use an output directory separate from the input FITS files")
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise ValueError("Output path must be a directory, not a file")
            if any(self.output_dir.iterdir()) and not self.options.overwrite:
                raise FileExistsError(
                    "Output directory is not empty; choose another or use --overwrite"
                )

    def preflight(self, threshold_sigma=5.0) -> dict:
        """Read and validate inputs without writing outputs, registering, or estimating PSFs."""
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError("Detection threshold must be finite and positive")
        self._validate_output_directory()
        self._load_inputs(cosmic_rays=False)
        if self.options.reproject:
            for frame in (self.reference, self.new):
                wcs = WCS(frame.header)
                if wcs.pixel_n_dim != 2 or not wcs.has_celestial:
                    raise ValueError(
                        "WCS reprojection requires valid two-dimensional celestial WCS"
                    )
            reprojection_unit_mode(self.reference.header, self.options.reproject_units)
        elif not (self.options.align or self.options.shift):
            if self.reference.data.shape != self.new.data.shape:
                raise ValueError("Image grids differ; use --align or --reproject")
        for supplied in (self.options.ref_psf, self.options.new_psf):
            if supplied:
                normalize_psf(fits.getdata(supplied, memmap=False), self.options.psf_radius)
        report = {
            "status": "input_checks_passed",
            "output_written": False,
            "inputs": {
                label: {
                    "file": path.name,
                    "shape": list(frame.data.shape),
                    "filter": str(frame.header.get("FILTER", "unknown")),
                    "unit": str(frame.header.get("BUNIT", "unknown")),
                    "valid_fraction": float((~frame.mask).mean()),
                }
                for label, path, frame in (
                    ("reference", self.ref_path, self.reference),
                    ("science", self.new_path, self.new),
                )
            },
            "not_checked": [
                "Celestial overlap and matched-star registration",
                "PSF seeing/support",
                "Stellar photometric consistency",
                "Cosmic-ray rejection",
                "Disk space and write permissions",
                "Transient detectability",
            ],
        }
        if "science_filter_override" in self.metadata:
            report["science_filter_override"] = self.metadata["science_filter_override"]
        return report

    def _write_run_status(self, state: str, stage: str, error: str | None = None) -> None:
        if self._status_path is not None:
            atomic_json(
                self._status_path,
                {
                    "state": state,
                    "stage": stage,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "started_utc": self.metadata.get("started_utc"),
                    "elapsed_seconds": self.metadata.get("elapsed_seconds"),
                    "timings_seconds": self.metadata.get("stage_timings_seconds", {}),
                    "error": error,
                },
            )

    @contextmanager
    def _stage(self, name: str):
        self._write_run_status("running", name)
        started = time.perf_counter()
        LOGGER.info("Starting stage: %s", name)
        try:
            yield
        except BaseException:
            self.metadata.setdefault("failed_stage", name)
            raise
        finally:
            duration = time.perf_counter() - started
            self.metadata.setdefault("stage_timings_seconds", {})[name] = duration
            LOGGER.info("Stage ended: %s (%.3f s)", name, duration)

    def _fingerprint_inputs(self) -> dict:
        paths = {"reference": self.ref_path, "science": self.new_path}
        for label in ("ref_mask", "new_mask", "ref_psf", "new_psf"):
            supplied = getattr(self.options, label)
            if supplied:
                paths[label] = Path(supplied).expanduser().resolve()
        records = {}
        for label, path in paths.items():
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Input changed while computing fingerprint: {path.name}")
            records[label] = {
                "file": path.name,
                "bytes": after.st_size,
                "mtime_utc": datetime.fromtimestamp(after.st_mtime, UTC).isoformat(),
                "sha256": digest.hexdigest(),
            }
        return records

    def _record_provenance(self) -> None:
        self.metadata["inputs"] = self._fingerprint_inputs()
        self.metadata["started_utc"] = datetime.now(UTC).isoformat()
        self.metadata["runtime"] = {
            "python": platform.python_version(),
            "pipeline_version": __version__,
        }

    def _prepare(self) -> tuple[np.ndarray, np.ndarray]:
        options = self.options
        with self._stage("input_loading"):
            self._load_inputs(cosmic_rays=options.cosmic_rays)
        if options.reproject:
            unit_mode = reprojection_unit_mode(self.reference.header, options.reproject_units)
            native_shape = list(self.reference.data.shape)
            self.reference, self.reprojection_footprint = reproject_frame(
                self.reference, self.new, unit_mode=unit_mode
            )
            self.metadata.update(
                output_grid="science celestial WCS",
                reprojection_method="reproject_interp bilinear",
                reprojection_units=unit_mode,
                reference_native_shape=native_shape,
                output_shape=list(self.new.data.shape),
                reprojection_valid_fraction=float(self.reprojection_footprint.mean()),
                noise_propagation=(
                    "Independent native pixels assumed in both images; exact resampled diagonal "
                    "variance; conservative covariance row-sum bounds for subsequent filtering"
                ),
            )
            LOGGER.warning(
                "Reprojected reference onto science WCS; filter errors use conservative "
                "covariance bounds; pre-existing input correlations are not modeled"
            )
        ref_catalog = star_catalog(self.reference)
        new_catalog = star_catalog(self.new)
        if min(len(ref_catalog), len(new_catalog)) < options.min_stars:
            raise ValueError(
                f"Too few isolated stars: reference={len(ref_catalog)}, "
                f"science={len(new_catalog)}; "
                f"require >= {options.min_stars} in each. Check masks, seeing, and field overlap."
            )
        if options.align or options.shift:
            with self._stage("registration"):
                self._register(ref_catalog, new_catalog)
            new_catalog = star_catalog(self.new)
        elif self.reference.data.shape != self.new.data.shape:
            raise ValueError("Image grids differ; use --align to register new onto reference")
        reference, new = matched_stars(ref_catalog, new_catalog)
        if len(reference) < options.min_stars:
            advice = (
                "check input WCS and overlap"
                if options.reproject
                else "use --align or check field overlap"
            )
            raise ValueError(
                f"Too few matched stars: {len(reference)}; require >= {options.min_stars}; {advice}"
            )
        offsets = np.hypot(reference["x"] - new["x"], reference["y"] - new["y"])
        rms = float(np.sqrt(np.mean(offsets**2)))
        if rms > options.alignment_tolerance:
            advice = "check input astrometric WCS" if options.reproject else "use --align"
            raise ValueError(
                f"Registration RMS {rms:.3f} pixels exceeds tolerance "
                f"{options.alignment_tolerance:.3f}; {advice}"
            )
        self.metadata.update(registration_rms_pixels=rms, matched_stars=len(reference))
        self.astrometric_residuals = pd.DataFrame(
            {
                "x_output": new["x"],
                "y_output": new["y"],
                "dx_pixels": reference["x"] - new["x"],
                "dy_pixels": reference["y"] - new["y"],
                "distance_pixels": offsets,
            }
        )
        self.metadata["registration_p95_pixels"] = float(np.percentile(offsets, 95))
        self.ref_data, self.new_data = self.reference.data, self.new.data
        return reference, new

    def _register(self, reference: np.ndarray, new: np.ndarray) -> None:
        import astroalign

        ref_xy = np.column_stack((reference["x"], reference["y"]))
        new_xy = np.column_stack((new["x"], new["y"]))
        transform, (source, target) = astroalign.find_transform(new_xy, ref_xy)
        residual = np.sqrt(np.mean(np.sum((transform(source) - target) ** 2, axis=1)))
        if len(source) < self.options.min_stars or residual > self.options.alignment_tolerance:
            raise ValueError(
                f"Astrometric transform failed: {len(source)} stars (minimum "
                f"{self.options.min_stars}); RMS {residual:.3f} pixels "
                f"(maximum {self.options.alignment_tolerance:.3f}). Check field overlap."
            )
        source_data = np.ma.array(self.new.data, mask=self.new.mask)
        registered, footprint = astroalign.apply_transform(
            transform, source_data, self.reference.data, fill_value=0, propagate_mask=True
        )
        variance, variance_footprint = astroalign.apply_transform(
            transform, self.new.variance, self.reference.data, fill_value=0
        )
        background, _ = astroalign.apply_transform(
            transform, self.new.background, self.reference.data, fill_value=0
        )
        area = float(transform.scale**2)
        mask = expand_mask(footprint | variance_footprint | (variance <= 0), 2)
        mask |= ~np.isfinite(registered) | ~np.isfinite(variance)
        self.new = ImageFrame(
            np.where(mask, 0, registered / area),
            self.reference.header.copy(),
            mask,
            np.where(mask, 0, variance / area**2),
            background / area,
            self.new.gain,
        )
        self.metadata["registration_matrix"] = transform.params.tolist()
        LOGGER.warning("Registered-image noise is correlated; score calibration is empirical")

    def _estimate_psf(self, frame: ImageFrame, supplied: str | None) -> np.ndarray:
        if supplied:
            return normalize_psf(fits.getdata(supplied, memmap=False), self.options.psf_radius)
        from properimage.single_image import SingleImageGaussPSF

        catalog = star_catalog(frame)
        if len(catalog) < self.options.min_stars:
            raise ValueError(
                f"Too few isolated stars for PSF estimation: {len(catalog)}; "
                f"require >= {self.options.min_stars}. Check masks, saturation, and seeing."
            )
        with SingleImageGaussPSF(
            frame.data,
            mask=frame.mask,
            guessmask=False,
            gain=frame.gain,
            min_sources=self.options.min_stars,
            strict_star_pick=False,
            stamp_shape=(2 * self.options.psf_radius + 1,) * 2,
            smooth_psf=self.options.smooth_psf,
        ) as image:
            # ProperImage's own strict selection silently keeps every detected
            # source when fewer than 15 pass its cuts, so galaxy light, blends
            # and noise widen the Gaussian fit. Seed its source cache with our
            # isolated SEP stars instead, which are already round, unmasked and
            # isolated; this keeps ProperImage's stamping and fitting machinery.
            image._best_sources = catalog[:50]
            _, basis = image.get_variable_psf(inf_loss=self.options.inf_loss)
            if image.n_sources < self.options.min_stars:
                raise ValueError(
                    f"Too few PSF stars selected by ProperImage: {image.n_sources}; "
                    f"require >= {self.options.min_stars}"
                )
            model = basis[1]
            radius = self.options.psf_radius
            row, column = np.mgrid[-radius : radius + 1, -radius : radius + 1]
            psf = model(column, row)
            if 4 * max(model.x_stddev.value, model.y_stddev.value) > radius:
                required = int(np.ceil(4 * max(model.x_stddev.value, model.y_stddev.value)))
                raise ValueError(
                    f"Gaussian PSF too wide for radius {radius}; "
                    f"increase --psf-radius to at least {required}"
                )
        psf = normalize_psf(psf, self.options.psf_radius)
        if psf_fwhm(psf) < 1.5:
            raise ValueError("PSF is undersampled (<1.5 pixels FWHM); inspect the PSF stars")
        return psf

    def _flux_scale(self, reference: np.ndarray, new: np.ndarray) -> float:
        if self.options.flux_scale is not None:
            return self.options.flux_scale
        if not self.options.beta:
            return 1.0
        aperture = 2.5 * max(psf_fwhm(self.ref_psf), psf_fwhm(self.new_psf))
        fluxes = []
        for frame, stars in ((self.reference, reference), (self.new, new)):
            flux, error, flag = sep.sum_circle(
                frame.data,
                stars["x"],
                stars["y"],
                aperture,
                err=np.sqrt(frame.noise_variance),
                mask=frame.mask,
                subpix=5,
            )
            fluxes.append((flux, error, flag))
        ref_flux, ref_error, ref_flag = fluxes[0]
        new_flux, new_error, new_flag = fluxes[1]
        good = (
            (ref_flag == 0)
            & (new_flag == 0)
            & (ref_flux > 20 * ref_error)
            & (new_flux > 20 * new_error)
        )
        ratios = new_flux[good] / ref_flux[good]
        all_ratios = ratios
        ratios = sigma_clip(
            ratios, sigma=3, maxiters=10 if self.options.iterative else 5, stdfunc="mad_std"
        ).compressed()
        if len(ratios) < self.options.min_stars:
            raise ValueError(
                f"Too few stable high-SNR stars for photometric scaling: {len(ratios)}; "
                f"require >= {self.options.min_stars}. Check variance, saturation, and filters."
            )
        scale = float(np.median(ratios))
        scatter = float(np.std(ratios) / scale)
        if not np.isfinite(scale) or scale <= 0 or scatter > self.options.max_scale_scatter:
            raise ValueError(
                f"Inconsistent star flux ratios: scale={scale:.5g}, scatter={scatter:.3f}, "
                f"maximum scatter={self.options.max_scale_scatter:.3f}; "
                "check PSF, filter, or transparency"
            )
        self.metadata.update(flux_scale_scatter=scatter, flux_scale_stars=len(ratios))
        if self.options.cross_filter:
            self.metadata.update(
                flux_scale_unclipped_scatter=float(np.std(all_ratios) / scale),
                flux_scale_stars_rejected=len(all_ratios) - len(ratios),
                flux_scale_assumption="One global stellar scale; no color correction",
                color_fraction_limit=float(
                    np.quantile(np.abs(all_ratios / scale - 1), self.options.color_quantile)
                ),
                color_vetting_aperture_pixels=float(aperture),
            )
            LOGGER.warning(
                "Cross-filter scale %.5g; clipped stellar scatter %.1f%%; unclipped %.1f%%. "
                "Color systematics are not included in flux errors.",
                scale,
                scatter * 100,
                self.metadata["flux_scale_unclipped_scatter"] * 100,
            )
        return scale

    def _proper_subtract(self) -> dict[str, np.ndarray]:
        from properimage.operations import subtract
        from properimage.single_image import SingleImageGaussPSF

        with ExitStack() as stack:
            images = [
                stack.enter_context(
                    SingleImageGaussPSF(
                        frame.data + frame.background,
                        mask=frame.mask,
                        guessmask=False,
                        gain=frame.gain,
                        min_sources=self.options.min_stars,
                    )
                )
                for frame in (self.reference, self.new)
            ]
            difference, psf, score, bad = subtract(
                ref=images[0],
                new=images[1],
                align=False,
                shift=False,
                beta=self.options.beta,
                iterative=self.options.iterative,
                fitted_psf=self.options.fitted_psf,
                smooth_psf=self.options.smooth_psf,
                inf_loss=self.options.inf_loss,
            )
            bad = (
                np.asarray(bad, bool)
                | np.ma.getmaskarray(images[0].data)
                | np.ma.getmaskarray(images[1].data)
            )
        difference = real_image(difference, "ProperImage D")
        psf = real_image(psf, "ProperImage difference PSF")
        peak = np.array(np.unravel_index(np.argmax(psf), psf.shape))
        center = np.array(psf.shape) // 2
        psf = np.roll(psf, center - peak, axis=(0, 1))
        radius = self.options.psf_radius * 2
        psf = psf[
            center[0] - radius : center[0] + radius + 1, center[1] - radius : center[1] + radius + 1
        ]
        if not np.isfinite(psf).all() or psf.sum() <= 0:
            raise ValueError("ProperImage returned an invalid difference PSF")
        psf /= psf.sum()
        mask = expand_mask(self.reference.mask | self.new.mask | np.asarray(bad, bool), radius * 2)
        score = real_image(score, "ProperImage Scorr")
        mask |= ~np.isfinite(difference) | ~np.isfinite(score)
        self.metadata["flux_unit"] = "ProperImage normalized D units (not calibrated counts)"
        self.metadata["flux_scale"] = "ProperImage internal transparency/optimization"
        LOGGER.warning("ProperImage noise uses its internal model, not input VAR/ERR planes")
        return {
            "difference": difference,
            "psf": psf,
            "score": score,
            "mask": mask,
            "flux": convolve_image(difference, psf[::-1, ::-1]) / np.sum(psf**2),
            "flux_error": np.full(difference.shape, np.nan),
        }

    def perform_image_subtraction(self):
        reference, new = self._prepare()
        with self._stage("psf_estimation"):
            self.ref_psf = self._estimate_psf(self.reference, self.options.ref_psf)
            self.new_psf = self._estimate_psf(self.new, self.options.new_psf)
        self.metadata.update(
            ref_fwhm_pixels=psf_fwhm(self.ref_psf),
            new_fwhm_pixels=psf_fwhm(self.new_psf),
            flux_unit="new-image counts",
            method=self.options.method,
        )
        with self._stage("photometric_scaling"):
            measured_scale = self._flux_scale(reference, new)
        self.metadata["photometric_check_scale"] = measured_scale
        if self.options.method == "kernel":
            scale = measured_scale
            self.metadata["flux_scale"] = scale
            with self._stage("subtraction"):
                self.products = kernel_subtract(
                    self.reference, self.new, self.ref_psf, self.new_psf, scale
                )
        else:
            with self._stage("subtraction"):
                self.products = self._proper_subtract()
        self.diff_image = self.products["difference"]
        self.psf_image = self.products["psf"]
        self.mask = self.products["mask"]
        self.mask |= ~np.isfinite(self.diff_image) | ~np.isfinite(self.products["score"])
        score, center, scale = calibrate_score(
            self.products["score"],
            self.mask,
            allow_amplification=self.options.method == "proper",
        )
        self.scorr_image = score
        self.metadata.update(
            score_background_center=center,
            score_background_scale=scale,
            score_type="empirically calibrated matched-filter statistic",
        )
        LOGGER.info(
            "Registration RMS %.3f pixels; %d matched stars",
            self.metadata["registration_rms_pixels"],
            len(reference),
        )
        return self.diff_image, self.psf_image, self.scorr_image, self.mask

    def _candidate_properties(self, source, sign: int, identifier: int) -> dict:
        column, row = int(source["xpeak"]), int(source["ypeak"])
        centroid_x, centroid_y = float(source["x"]), float(source["y"])
        expected_fwhm = psf_fwhm(np.maximum(self.psf_image, 0))
        radius = min(self.psf_image.shape[0] // 2, int(np.ceil(expected_fwhm * 1.5)))
        rows = slice(row - radius, row + radius + 1)
        columns = slice(column - radius, column + radius + 1)
        stamp = sign * self.diff_image[rows, columns]
        bad = self.mask[rows, columns]
        flags = []
        if bad.any():
            flags.append("mask_neighborhood")
        template = ndimage.shift(
            self.psf_image,
            (centroid_y - row, centroid_x - column),
            order=3,
            mode="constant",
        )
        center = np.array(template.shape) // 2
        template = template[
            center[0] - radius : center[0] + radius + 1, center[1] - radius : center[1] + radius + 1
        ]
        if stamp.shape != template.shape:
            raise ValueError("Candidate stamp crosses an unmasked image boundary")
        if "variance" in self.products:
            variance = np.maximum(self.products["variance"][rows, columns], 1e-12)
        else:
            _, _, noise = sigma_clipped_stats(self.diff_image, mask=self.mask)
            variance = np.full(stamp.shape, float(noise) ** 2)
        valid = ~bad & np.isfinite(stamp) & np.isfinite(variance)
        design = np.column_stack((template[valid], np.ones(valid.sum())))
        weights = 1 / np.sqrt(variance[valid])
        fitted, *_ = np.linalg.lstsq(design * weights[:, None], stamp[valid] * weights, rcond=None)
        amplitude, background = fitted
        residual = (stamp[valid] - design @ fitted) * weights
        chi2 = float(np.sum(residual**2) / max(valid.sum() - 2, 1))
        centered_data = stamp[valid] - background
        centered_psf = template[valid]
        correlation = float(
            np.dot(centered_data, centered_psf)
            / max(np.linalg.norm(centered_data) * np.linalg.norm(centered_psf), 1e-12)
        )
        peak = float(np.max(stamp[valid] - background))
        noise = float(np.sqrt(np.median(variance[valid])))
        floor = max(peak * 0.1, noise * 2)
        moments = np.where(valid & (stamp - background > floor), stamp - background, 0)
        expected = np.where(amplitude * template > floor, amplitude * template, 0)
        row_grid, column_grid = np.indices(stamp.shape)
        total = moments.sum()
        if total > 0 and expected.sum() > 0:
            mean_row = np.sum(moments * row_grid) / total
            mean_column = np.sum(moments * column_grid) / total
            offsets = np.stack((row_grid - mean_row, column_grid - mean_column))
            covariance = np.einsum("iyx,jyx,yx->ij", offsets, offsets, moments) / total
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
            fwhm = float(2.35482 * np.sqrt(np.mean(eigenvalues)))
            ratio = fwhm / max(psf_fwhm(expected), 1e-12)
            elongation = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
        else:
            fwhm, ratio, elongation = 0.0, 0.0, np.inf
        distances = np.hypot(row_grid - radius, column_grid - radius)
        annulus = valid & (distances > expected_fwhm * 0.5) & (distances < expected_fwhm * 1.5)
        opposite = max(0.0, -float(np.min(stamp[annulus] - background))) if annulus.any() else 0
        dipole = opposite / max(peak, 1e-12) if opposite > noise * 4 else 0.0
        options = self.options
        if source["npix"] < options.min_pixels:
            flags.append("too_few_pixels")
        if not options.min_fwhm_ratio <= ratio <= options.max_fwhm_ratio:
            flags.append("psf_width")
        if elongation > options.max_elongation:
            flags.append("elongated")
        if correlation < options.min_psf_correlation:
            flags.append("psf_mismatch")
        if dipole > options.max_dipole_ratio:
            flags.append("dipole")
        if amplitude <= 0:
            flags.append("nonpositive_psf_fit")
        if source["flag"] != 0:
            flags.append("blended_or_truncated")
        ra, dec = np.nan, np.nan
        wcs = WCS(self.reference.header)
        if wcs.has_celestial:
            sky = wcs.celestial.pixel_to_world(centroid_x, centroid_y).icrs
            ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        return dict(
            zip(
                CANDIDATE_COLUMNS,
                [
                    identifier,
                    centroid_x,
                    centroid_y,
                    sign,
                    float(self.products["flux"][row, column]),
                    float(self.products["flux_error"][row, column]),
                    float(self.diff_image[row, column]),
                    float(self.scorr_image[row, column]),
                    int(source["npix"]),
                    fwhm,
                    ratio,
                    elongation,
                    correlation,
                    dipole,
                    chi2,
                    ra,
                    dec,
                    ";".join(flags),
                ],
                strict=True,
            )
        )

    def _vet_cross_filter_candidates(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Annotate color ambiguity without changing detection flags or selection."""
        result = catalog.copy()
        result["reference_source_associated"] = pd.Series(
            pd.NA, index=result.index, dtype="boolean"
        )
        for name in (
            "reference_distance_pixels",
            "reference_aperture_flux",
            "reference_aperture_error",
            "residual_aperture_flux",
            "fractional_excess",
        ):
            result[name] = np.nan
        result["within_color_range"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result["vetting_group"] = "color_ambiguous"
        result["vetting_reason"] = "stellar_color_range_unavailable"
        limit = self.metadata.get("color_fraction_limit")
        self.metadata["color_vetting_available"] = limit is not None
        self.metadata["color_vetting_rule"] = (
            f"Reference association within {self.options.color_association_fwhm:g} effective FWHM "
            f"or aperture SNR >= {self.options.color_reference_snr:g}; associated sources remain "
            f"ambiguous. Color range is stellar quantile {self.options.color_quantile:g} plus "
            f"{self.options.color_error_margin:g} errors, not calibrated variability significance."
        )
        if result.empty or limit is None:
            return result
        radius = self.metadata["color_vetting_aperture_pixels"]
        scale = self.metadata["flux_scale"]
        reference_bound = scale**2 * convolve_image(self.reference.noise_variance, self.new_psf)
        residual_bound = reference_bound + convolve_image(self.new.noise_variance, self.ref_psf)
        columns, rows = result["x_centroid"].to_numpy(), result["y_centroid"].to_numpy()
        measurements = []
        for image, bound in (
            (self.products["matched_reference"], reference_bound),
            (self.diff_image, residual_bound),
        ):
            measurements.append(
                sep.sum_circle(
                    np.ascontiguousarray(image),
                    columns,
                    rows,
                    radius,
                    err=np.sqrt(np.maximum(bound, 0)),
                    mask=self.mask,
                    subpix=5,
                )
            )
        reference_flux, reference_error, reference_flags = measurements[0]
        residual_flux, residual_error, residual_flags = measurements[1]
        result["reference_aperture_flux"] = reference_flux
        result["reference_aperture_error"] = reference_error
        result["residual_aperture_flux"] = residual_flux
        valid = (reference_flags == 0) & (residual_flags == 0)
        valid &= np.isfinite(reference_flux) & np.isfinite(reference_error) & (reference_error > 0)
        valid &= np.isfinite(residual_flux) & np.isfinite(residual_error) & (residual_error > 0)
        fraction_valid = valid & (
            reference_flux > self.options.color_reference_snr * reference_error
        )
        result.loc[fraction_valid, "fractional_excess"] = (
            residual_flux[fraction_valid] / reference_flux[fraction_valid]
        )
        result.loc[fraction_valid, "within_color_range"] = np.abs(
            residual_flux[fraction_valid]
        ) <= limit * reference_flux[fraction_valid] + self.options.color_error_margin * (
            residual_error[fraction_valid] + limit * reference_error[fraction_valid]
        )
        objects = sep.extract(
            np.ascontiguousarray(self.products["matched_reference"]),
            self.options.color_reference_snr,
            err=np.sqrt(np.maximum(reference_bound, np.finfo(float).tiny)),
            mask=self.mask,
            minarea=5,
        )
        nearby = np.zeros(len(result), dtype=bool)
        if len(objects):
            distances, _ = cKDTree(np.column_stack((objects["x"], objects["y"]))).query(
                np.column_stack((columns, rows))
            )
            result["reference_distance_pixels"] = distances
            nearby = distances <= self.options.color_association_fwhm * psf_fwhm(self.psf_image)
        associated = nearby | fraction_valid
        result["reference_source_associated"] = associated
        result["vetting_reason"] = "unreliable_reference_photometry"
        result.loc[valid & associated, "vetting_reason"] = "existing_source_excess"
        within = result["within_color_range"].fillna(False).to_numpy(dtype=bool)
        result.loc[valid & associated & within, "vetting_reason"] = "within_stellar_color_range"
        result.loc[valid & ~associated, "vetting_group"] = "less_color_confounded"
        result.loc[valid & ~associated, "vetting_reason"] = "no_significant_reference_source"
        return result

    def candidate_background_diagnostics(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Measure surrounding residual structure; never change candidate selection."""
        records = []
        fwhm = psf_fwhm(np.maximum(self.psf_image, 0))
        radius = int(np.ceil(4 * fwhm))
        for candidate in catalog.itertuples():
            column, row = int(round(candidate.x_centroid)), int(round(candidate.y_centroid))
            rows = slice(max(0, row - radius), min(self.diff_image.shape[0], row + radius + 1))
            cols = slice(
                max(0, column - radius), min(self.diff_image.shape[1], column + radius + 1)
            )
            stamp = self.diff_image[rows, cols]
            grid_y, grid_x = np.mgrid[rows, cols]
            dx, dy = grid_x - candidate.x_centroid, grid_y - candidate.y_centroid
            distances = np.hypot(dx, dy)
            valid = (~self.mask[rows, cols]) & np.isfinite(stamp)
            valid &= (distances >= 2 * fwhm) & (distances <= 4 * fwhm)
            count = int(valid.sum())
            background, gradient, structure = np.nan, np.nan, np.nan
            reasons = []
            if count < 30:
                reasons.append("insufficient_annulus")
            else:
                values = stamp[valid]
                design = np.column_stack((np.ones(count), dx[valid] / fwhm, dy[valid] / fwhm))
                coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
                residual = values - design @ coefficients
                noise = max(
                    float(np.median(np.abs(residual - np.median(residual))) * 1.4826),
                    np.finfo(float).eps,
                )
                if "variance" in self.products:
                    expected = np.sqrt(np.median(self.products["variance"][rows, cols][valid]))
                    noise = max(float(expected), np.finfo(float).eps)
                background = float(np.median(values) / noise)
                gradient = float(np.hypot(*coefficients[1:]) / noise)
                structure = float(np.sqrt(np.mean((values - design @ coefficients) ** 2)) / noise)
                if abs(background) > 1:
                    reasons.append("background_offset")
                if gradient > 0.5:
                    reasons.append("background_gradient")
                if structure > 2:
                    reasons.append("structured_residual")
            records.append(
                [candidate.id, count, background, gradient, structure, ";".join(reasons)]
            )
        return pd.DataFrame(records, columns=BACKGROUND_COLUMNS)

    def detect_sources_in_difference(self, threshold_sigma=5.0):
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError("Detection threshold must be finite and positive")
        if self.scorr_image is None:
            raise ValueError("Run image subtraction before source detection")
        candidates = []
        for sign in (1, -1):
            detections = sep.extract(
                np.ascontiguousarray(np.where(self.mask, 0, sign * self.scorr_image)),
                threshold_sigma,
                mask=self.mask,
                minarea=1,
                deblend_cont=0.005,
            )
            if len(candidates) + len(detections) > self.options.max_candidates:
                raise ValueError(
                    f"Too many candidates: at least {len(candidates) + len(detections)} "
                    f"(maximum {self.options.max_candidates}); inspect subtraction quality"
                )
            for source in detections:
                candidates.append(self._candidate_properties(source, sign, len(candidates) + 1))
        catalog = pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)
        if len(catalog):
            catalog = catalog.iloc[np.argsort(-catalog["snr"].abs().to_numpy())].reset_index(
                drop=True
            )
        if self.options.cross_filter:
            catalog["comparison_warning"] = CROSS_FILTER_WARNING
            catalog = self._vet_cross_filter_candidates(catalog)
        self.local_background_diagnostics = self.candidate_background_diagnostics(catalog)
        self.metadata["candidate_background_warnings"] = int(
            self.local_background_diagnostics["local_background_warning"].ne("").sum()
        )
        accepted = catalog.loc[catalog["flags"] == ""].reset_index(drop=True)
        self.rejected_sources = catalog.loc[catalog["flags"] != ""].reset_index(drop=True)
        self.metadata.update(
            threshold_sigma=float(threshold_sigma),
            accepted_candidates=len(accepted),
            rejected_candidates=len(self.rejected_sources),
            masked_fraction=float(self.mask.mean()),
            difference_sign="new minus scaled reference; positive means brightening",
            status="unconfirmed candidates; artifact cuts do not establish astrophysical origin",
        )
        if self.options.cross_filter:
            self.metadata.update(
                status="experimental cross-filter residuals; color mismatch can mimic variability",
                difference_sign="new minus scaled reference; sign can reflect color differences",
                score_type="empirical residual score; not calibrated for color mismatch",
                flux_error_scope="Statistical noise only; color and scale systematics excluded",
                less_color_confounded_candidates=int(
                    accepted["vetting_group"].eq("less_color_confounded").sum()
                ),
                color_ambiguous_candidates=int(
                    accepted["vetting_group"].eq("color_ambiguous").sum()
                ),
            )
        LOGGER.info(
            "Accepted %d candidates; rejected %d", len(accepted), len(self.rejected_sources)
        )
        return accepted

    def _output_path(self, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        if path.exists() and not self.options.overwrite:
            raise FileExistsError(f"Output already exists: {name}; use --overwrite deliberately")
        inputs = {self.ref_path, self.new_path}
        for supplied in (
            self.options.ref_mask,
            self.options.new_mask,
            self.options.ref_psf,
            self.options.new_psf,
        ):
            if supplied:
                inputs.add(Path(supplied).expanduser().resolve())
        if path.resolve() in inputs:
            raise ValueError("Refusing to overwrite an input image, mask, or PSF")
        return path

    def quality_summary(self) -> dict:
        """Summarize measured checks without claiming astrophysical confirmation."""
        warnings = []
        if self.options.cross_filter:
            warnings.append(CROSS_FILTER_WARNING)
        if self.metadata.get("masked_fraction", 0) > 0.5:
            warnings.append("More than half the output pixels are masked.")
        if self.options.align or self.options.shift:
            warnings.append("Registration covariance is not fully modeled.")
        if self.options.reproject:
            warnings.append(
                "Reprojection covariance is bounded; native input correlations are not modeled."
            )
        if self.options.method == "proper":
            warnings.append("ProperImage uses its internal noise model, not input variance planes.")
        if self.options.flux_scale is not None or not self.options.beta:
            warnings.append("Stellar photometric scale/scatter was not independently estimated.")
        if self.metadata.get("score_background_scale", 1) > 1.5:
            warnings.append("Large empirical score rescaling; inspect subtraction residuals.")
        if self.metadata.get("candidate_background_warnings", 0):
            warnings.append(
                "Some candidates have local-background warnings; see candidate_background.csv."
            )
        counts = {}
        if "flags" in self.rejected_sources:
            reasons = self.rejected_sources["flags"].str.split(";").explode()
            counts = {
                str(reason): int(count)
                for reason, count in reasons.value_counts().items()
                if reason
            }
        names = (
            "registration_rms_pixels",
            "registration_p95_pixels",
            "matched_stars",
            "ref_fwhm_pixels",
            "new_fwhm_pixels",
            "flux_scale",
            "flux_scale_scatter",
            "flux_scale_stars",
            "masked_fraction",
            "score_background_scale",
            "accepted_candidates",
            "rejected_candidates",
            "less_color_confounded_candidates",
            "color_ambiguous_candidates",
            "candidate_background_warnings",
        )
        return {
            "status": "warnings" if warnings else "processing_checks_passed",
            "interpretation": (
                "Processing diagnostics only; not transient confirmation or completeness."
            ),
            "metrics": {name: self.metadata.get(name) for name in names},
            "limits": {
                "registration_rms_pixels": self.options.alignment_tolerance,
                "stellar_scale_scatter": self.options.max_scale_scatter,
                "minimum_stars": self.options.min_stars,
            },
            "rejection_reasons": counts,
            "warnings": warnings,
            "timings_seconds": self.metadata.get("stage_timings_seconds", {}).copy(),
        }

    def save_results_to_csv(self, sources_df):
        sources_df.to_csv(self._output_path("detected_sources.csv"), index=False)
        self.rejected_sources.to_csv(self._output_path("rejected_sources.csv"), index=False)
        self.local_background_diagnostics.to_csv(
            self._output_path("candidate_background.csv"), index=False
        )
        if self.options.cross_filter:
            for group in ("less_color_confounded", "color_ambiguous"):
                sources_df.loc[sources_df["vetting_group"] == group].to_csv(
                    self._output_path(f"{group}_candidates.csv"), index=False
                )
        summary = self.quality_summary()
        self.metadata["quality_summary"] = summary
        self._output_path("quality_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        pd.DataFrame(
            list(summary["rejection_reasons"].items()), columns=["reason", "count"]
        ).to_csv(self._output_path("rejection_summary.csv"), index=False)
        scalars = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, (str, int, float))
        }
        pd.DataFrame(scalars.items(), columns=["parameter", "value"]).to_csv(
            self._output_path("statistics.csv"), index=False
        )
        packages = ["properimage", "astropy", "numpy", "scipy", "sep"]
        if self.options.reproject:
            packages.append("reproject")
            self.astrometric_residuals.to_csv(
                self._output_path("astrometric_residuals.csv"), index=False
            )
        self.metadata["versions"] = {name: version(name) for name in packages}
        with self._output_path("provenance.json").open("w", encoding="utf-8") as stream:
            json.dump(self.metadata, stream, indent=2, allow_nan=False)

    def save_fits_products(self):
        grid_header = WCS(self.reference.header).celestial.to_header(relax=True)
        for keyword in ("FILTER", "EXPTIME", "DATE-OBS", "MJD-OBS"):
            if keyword in self.new_header:
                grid_header[keyword] = self.new_header[keyword]
        grid_header["METHOD"] = self.options.method
        grid_header["DIFFSIGN"] = "NEW-REF"
        grid_header["FLXUNIT"] = self.metadata["flux_unit"]
        if self.options.cross_filter:
            grid_header["XFLTR"] = (True, "Experimental cross-filter residual comparison")
            grid_header["REFFILT"] = self.metadata["reference_filter"]
            grid_header["NEWFILT"] = self.metadata["science_filter"]
            grid_header.add_history(CROSS_FILTER_WARNING)
        if self.options.method == "kernel":
            grid_header["REFSCALE"] = self.metadata["flux_scale"]
        grid_header.add_history(
            "PSFs estimated with ProperImage; original inputs are not modified."
        )
        grid_header.add_history("Masks exclude artifacts and full kernel/filter support.")
        unit = str(self.new_header.get("BUNIT", "count"))
        images = {
            "difference_image": (
                self.diff_image,
                unit if self.options.method == "kernel" else "normalized",
            ),
            "significance_image": (self.scorr_image, "1"),
            "raw_significance_image": (self.products["score"], "1"),
            "flux_image": (
                self.products["flux"],
                unit if self.options.method == "kernel" else "normalized",
            ),
        }
        for key in ("variance", "matched_reference", "matched_new", "flux_error"):
            if key in self.products and (key != "flux_error" or self.options.method == "kernel"):
                images[key] = (self.products[key], f"({unit})**2" if key == "variance" else unit)
        if self.options.reproject:
            images.update(
                {
                    "reprojected_reference": (self.reference.data, unit),
                    "reprojected_reference_variance": (self.reference.variance, f"({unit})**2"),
                    "reference_covariance_bound": (self.reference.noise_variance, f"({unit})**2"),
                }
            )
            coverage_header = grid_header.copy()
            coverage_header["BUNIT"] = "1"
            fits.writeto(
                self._output_path("reprojection_footprint.fits"),
                self.reprojection_footprint,
                coverage_header,
                overwrite=self.options.overwrite,
                checksum=True,
            )
        for name, (image, image_unit) in images.items():
            header = grid_header.copy()
            header["BUNIT"] = image_unit
            if self.options.reproject:
                header.add_history(
                    "Reference reprojected onto science WCS; filtering VAR is a covariance bound."
                )
            if "significance" in name:
                header["SCORECAL"] = self.metadata["score_background_scale"]
                header.add_history("Score is an empirical statistic, not a discovery probability.")
            data = np.where(self.mask, np.nan, image)
            fits.writeto(
                self._output_path(f"{name}.fits"),
                data,
                header,
                overwrite=self.options.overwrite,
                checksum=True,
            )
        fits.writeto(
            self._output_path("quality_mask.fits"),
            self.mask.astype(np.uint8),
            grid_header,
            overwrite=self.options.overwrite,
            checksum=True,
        )
        for name, image in (
            ("reference_psf", self.ref_psf),
            ("new_psf", self.new_psf),
            ("difference_psf", self.psf_image),
        ):
            fits.writeto(
                self._output_path(f"{name}.fits"),
                image,
                overwrite=self.options.overwrite,
                checksum=True,
            )
        if self.options.method == "kernel":
            for name, image in (
                ("reference_matching_kernel", self.new_psf),
                ("new_matching_kernel", self.ref_psf),
            ):
                fits.writeto(
                    self._output_path(f"{name}.fits"),
                    image,
                    overwrite=self.options.overwrite,
                    checksum=True,
                )

    def create_diagnostic_plots(self, sources_df):
        from astropy.visualization import ImageNormalize, ZScaleInterval
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import CenteredNorm
        from matplotlib.figure import Figure

        figure = Figure(figsize=(15, 14 if self.options.reproject else 9), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3 if self.options.reproject else 2, 3)
        if self.options.cross_filter:
            figure.suptitle(
                "EXPERIMENTAL CROSS-FILTER RESIDUALS: color differences can mimic variability",
                fontsize=11,
            )
        reference_title = (
            "Reference (science WCS)"
            if self.options.reproject
            else "Reference (background subtracted)"
        )
        panels = [
            (self.ref_data, reference_title, "gray"),
            (self.new_data, "Science (output pixel grid)", "gray"),
            (np.where(self.mask, np.nan, self.diff_image), "Difference: new - reference", "RdBu_r"),
            (self.scorr_image, "Empirical matched-filter score", "RdBu_r"),
            (self.psf_image, "Difference PSF", "viridis"),
            (self.mask, "Excluded pixels", "binary"),
        ]
        if self.options.reproject:
            panels.append(
                (self.reprojection_footprint, "Valid reference interpolation coverage", "binary")
            )
        for axis, (image, title, cmap) in zip(list(axes.flat)[: len(panels)], panels, strict=True):
            norm = None if cmap == "binary" else ImageNormalize(image, interval=ZScaleInterval())
            if self.options.reproject:
                if cmap == "viridis":
                    norm = None
                elif cmap == "RdBu_r":
                    norm = CenteredNorm(vcenter=0, halfrange=float(np.nanmax(np.abs(image))))
            artist = axis.imshow(image, origin="lower", cmap=cmap, norm=norm)
            axis.set(title=title, xlabel="X [pixels, origin 0]", ylabel="Y [pixels, origin 0]")
            figure.colorbar(artist, ax=axis, shrink=0.8)
        if self.options.reproject:
            residuals = self.astrometric_residuals
            axes[2, 1].quiver(
                residuals["x_output"],
                residuals["y_output"],
                residuals["dx_pixels"],
                residuals["dy_pixels"],
                angles="xy",
                scale_units="xy",
                scale=0.01,
            )
            axes[2, 1].set(
                title="Matched-star residuals (arrows x100)",
                xlabel="X [output pixels]",
                ylabel="Y [output pixels]",
                xlim=(0, self.new_data.shape[1]),
                ylim=(0, self.new_data.shape[0]),
                aspect="equal",
            )
            axes[2, 2].hist(residuals["distance_pixels"], bins=15)
            axes[2, 2].axvline(self.options.alignment_tolerance, color="red", linestyle="--")
            axes[2, 2].set(
                title=f"WCS registration RMS {self.metadata['registration_rms_pixels']:.3f} px",
                xlabel="Matched-star offset [pixels]",
                ylabel="Stars",
            )
        for axis in (axes[0, 2], axes[1, 0]):
            groups = (
                [
                    (
                        sources_df["vetting_group"] == "less_color_confounded",
                        "lime",
                        "o",
                        "Less color-confounded",
                    ),
                    (
                        sources_df["vetting_group"] == "color_ambiguous",
                        "gold",
                        "D",
                        "Color-ambiguous",
                    ),
                ]
                if self.options.cross_filter
                else [
                    (sources_df["sign"] == 1, "lime", "o", "Positive candidates"),
                    (sources_df["sign"] == -1, "gold", "o", "Negative candidates"),
                ]
            )
            for selection, color, marker, label in groups:
                selected = sources_df[selection]
                axis.scatter(
                    selected["x_centroid"],
                    selected["y_centroid"],
                    s=75,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    label=label,
                )
            axis.scatter(
                self.rejected_sources["x_centroid"],
                self.rejected_sources["y_centroid"],
                marker="x",
                color="cyan",
                s=25,
                label="Rejected",
            )
            axis.legend(loc="lower left", fontsize=7)
        figure.savefig(self._output_path("diagnostics.png"), dpi=140)
        if not len(sources_df):
            return
        selected = sources_df.head(12)
        figure = Figure(figsize=(12, 3 * len(selected)), layout="constrained")
        FigureCanvasAgg(figure)
        if self.options.cross_filter:
            figure.suptitle(
                "Experimental cross-filter residuals, not confirmed variability", fontsize=11
            )
        axes = figure.subplots(len(selected), 3, squeeze=False)
        for axes_row, (_, candidate) in zip(axes, selected.iterrows(), strict=True):
            column, row = int(round(candidate["x_centroid"])), int(round(candidate["y_centroid"]))
            radius = int(np.ceil(2 * psf_fwhm(np.maximum(self.psf_image, 0))))
            rows, columns = (
                slice(max(0, row - radius), min(self.diff_image.shape[0], row + radius + 1)),
                slice(max(0, column - radius), min(self.diff_image.shape[1], column + radius + 1)),
            )
            reference = self.products.get("matched_reference", self.ref_data)[rows, columns]
            science = self.products.get("matched_new", self.new_data)[rows, columns]
            difference = self.diff_image[rows, columns]
            common_norm = ImageNormalize(
                np.concatenate((reference.ravel(), science.ravel())), interval=ZScaleInterval()
            )
            difference_norm = CenteredNorm(
                vcenter=0, halfrange=max(float(np.nanpercentile(np.abs(difference), 99)), 1e-12)
            )
            names = (
                ("Matched reference", "Matched science", "Difference")
                if "matched_reference" in self.products
                else ("Reference", "Science", "Difference")
            )
            for axis, stamp, name, norm, cmap in zip(
                axes_row,
                (reference, science, difference),
                names,
                (common_norm, common_norm, difference_norm),
                ("gray", "gray", "RdBu_r"),
                strict=True,
            ):
                axis.imshow(
                    np.ma.masked_invalid(stamp),
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                )
                group = str(candidate.get("vetting_group", "unconfirmed")).replace("_", " ")
                axis.set_title(
                    f"{name}\nID {int(candidate['id'])} / score {candidate['snr']:.1f} / {group}",
                    fontsize=9,
                )
        figure.savefig(self._output_path("candidate_stamps.png"), dpi=120)

    def run_full_pipeline(self, threshold_sigma=5.0, estimate_psf_separately=False):
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError("Detection threshold must be finite and positive")
        self._validate_output_directory()
        status_path = self._output_path("run_status.json")
        log_path = self._output_path("pipeline.log")
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        previous_level = LOGGER.level
        LOGGER.setLevel(logging.INFO)
        LOGGER.addHandler(handler)
        self._status_path = status_path
        self.metadata["stage_timings_seconds"] = {}
        self.metadata.pop("failed_stage", None)
        self.metadata.pop("elapsed_seconds", None)
        self.metadata["started_utc"] = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        try:
            self._write_run_status("running", "initializing")
            with self._stage("provenance"):
                self._record_provenance()
            if estimate_psf_separately:
                LOGGER.info("Both input PSFs are always estimated and saved by this pipeline")
            with self._stage("prepare_and_subtract"):
                self.perform_image_subtraction()
            with self._stage("detection"):
                sources = self.detect_sources_in_difference(threshold_sigma)
            with self._stage("fits_output"):
                self.save_fits_products()
            if self.options.plots:
                with self._stage("plots"):
                    self.create_diagnostic_plots(sources)
            with self._stage("catalog_output"):
                self.save_results_to_csv(sources)
            self.metadata["elapsed_seconds"] = time.perf_counter() - started
            self.metadata["quality_summary"] = self.quality_summary()
            atomic_json(self.output_dir / "quality_summary.json", self.metadata["quality_summary"])
            atomic_json(self.output_dir / "provenance.json", self.metadata)
            LOGGER.info("Run completed: %d accepted candidates", len(sources))
            handler.flush()
            self._write_run_status("completed", "completed")
            return sources
        except BaseException as error:
            self.metadata["elapsed_seconds"] = time.perf_counter() - started
            LOGGER.exception("Run failed: %s", error)
            try:
                self._write_run_status(
                    "failed",
                    self.metadata.get("failed_stage", "processing"),
                    f"{type(error).__name__}: {error}",
                )
            except OSError:
                LOGGER.error("Unable to persist failure status")
            raise
        finally:
            self._status_path = None
            LOGGER.removeHandler(handler)
            handler.close()
            LOGGER.setLevel(previous_level)


def compare_cross_filter(
    reference_fits,
    science_fits,
    output_dir,
    *,
    reproject=False,
    max_scale_scatter=0.15,
    threshold_sigma=5.0,
    **config,
) -> pd.DataFrame:
    """Save an experimental PSF-matched comparison and return warning-labeled candidates.

    A global stellar scale does not correct color. Flux errors and scores exclude
    that systematic. Extra keywords are ordinary PipelineConfig options; the
    cross-filter flag and explicit arguments here take precedence.
    """
    config.update(cross_filter=True, reproject=reproject, max_scale_scatter=max_scale_scatter)
    pipeline = ProperTransientFinder(reference_fits, science_fits, output_dir, config)
    return pipeline.run_full_pipeline(threshold_sigma)


def main(argv: list[str] | None = None) -> int:
    """Command-line interface shared by the standalone wrapper and module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", "--reference", help="Reference FITS image")
    parser.add_argument("--new", "--target", help="New FITS image")
    parser.add_argument(
        "--survey",
        help=(
            "Opt in to HiPS survey-reference acquisition: auto, panstarrs, "
            "skymapper, or an explicit HiPS identifier"
        ),
    )
    parser.add_argument("--template-filter", help="Explicit HiPS template band")
    parser.add_argument("--science-list", help="Text paths or JSON array of science paths")
    parser.add_argument(
        "--science-filter", help="Override the science FITS FILTER with a verified band label"
    )
    parser.add_argument("--template-cache", default="survey_cache")
    parser.add_argument(
        "--template-units",
        choices=("adu", "electron"),
        help=("Declare the count-unit label applied to cache copies; does not convert HiPS pixels"),
    )
    parser.add_argument("--template-gain", type=float, help="Template electrons per HiPS count")
    parser.add_argument("--survey-timeout", type=float, default=30)
    parser.add_argument("--max-download-mb", type=float, default=512)
    parser.add_argument(
        "--no-moc-precheck",
        action="store_false",
        dest="check_moc_coverage",
        help="Skip the CDS MOC coverage pre-check for explicit HiPS identifiers",
    )
    parser.add_argument(
        "--batch", help="JSON manifest of named ref/new pairs; incompatible with --ref/--new"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop a batch after its first failed pair"
    )
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--threshold", "-t", type=float, default=5.0)
    parser.add_argument("--method", choices=("kernel", "proper"), default="kernel")
    parser.add_argument("--inf-loss", type=float, default=0.15)
    parser.add_argument("--psf-radius", type=int, default=8)
    parser.add_argument("--gain", type=float, help="Electrons per input image count")
    parser.add_argument("--saturation", type=float, help="Saturation level in image counts")
    parser.add_argument("--flux-scale", type=float, help="Multiply reference counts by this factor")
    parser.add_argument(
        "--cross-filter",
        action="store_true",
        help="Experimental kernel comparison across filters; color residuals can mimic variability",
    )
    parser.add_argument(
        "--max-scale-scatter",
        type=float,
        default=0.15,
        help="Maximum clipped fractional stellar flux-ratio scatter (default 0.15)",
    )
    parser.add_argument("--ref-mask")
    parser.add_argument("--new-mask")
    parser.add_argument("--ref-psf", help="Centered PSF stamp sampled on the reference grid")
    parser.add_argument("--new-psf", help="Centered PSF stamp sampled on the registered new grid")
    parser.add_argument("--alignment-tolerance", type=float, default=0.35)
    parser.add_argument("--min-stars", type=int, default=6)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--min-fwhm-ratio", type=float, default=0.6)
    parser.add_argument("--max-fwhm-ratio", type=float, default=1.8)
    parser.add_argument("--max-elongation", type=float, default=2.0)
    parser.add_argument("--max-dipole-ratio", type=float, default=0.5)
    parser.add_argument("--min-psf-correlation", type=float, default=0.65)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument(
        "--color-association-fwhm",
        type=float,
        default=1.0,
        help="Cross-filter reference association radius in effective FWHM (default 1)",
    )
    parser.add_argument(
        "--color-reference-snr",
        type=float,
        default=5.0,
        help="Cross-filter reference detection/aperture SNR threshold (default 5)",
    )
    parser.add_argument(
        "--color-quantile",
        type=float,
        default=0.95,
        help="Cross-filter absolute calibration-star color-residual quantile (default 0.95)",
    )
    parser.add_argument(
        "--color-error-margin",
        type=float,
        default=3.0,
        help="Cross-filter color-range error margin multiplier (default 3)",
    )
    parser.add_argument("--smooth-psf", action="store_true")
    parser.add_argument("--align", action="store_true", help="Register new onto reference")
    parser.add_argument(
        "--reproject",
        action="store_true",
        help="Reproject reference onto science celestial WCS (kernel method only)",
    )
    parser.add_argument(
        "--reproject-units",
        choices=("auto", "flux-per-pixel", "surface-brightness"),
        default="auto",
        help="Reference BUNIT interpretation for WCS pixel-area scaling",
    )
    parser.add_argument("--shift", action="store_true", help="Alias for validated registration")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--no-beta", action="store_true", help="Do not estimate photometric scale")
    parser.add_argument(
        "--auto-psf", action="store_true", help="Disabled; supply PSF stamps instead"
    )
    parser.add_argument("--estimate-psf", action="store_true", help="Kept for compatibility")
    parser.add_argument("--no-cosmic-rays", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs without writing outputs or subtracting",
    )
    args = vars(parser.parse_args(argv))
    batch_manifest = args.pop("batch")
    fail_fast = args.pop("fail_fast")
    survey = args.pop("survey")
    science_list = args.pop("science_list")
    survey_options = {
        name: args.pop(name)
        for name in (
            "template_filter",
            "science_filter",
            "template_units",
            "template_gain",
            "template_cache",
            "survey_timeout",
            "max_download_mb",
            "check_moc_coverage",
        )
    }
    survey_options["cache_dir"] = survey_options.pop("template_cache")
    survey_options["timeout"] = survey_options.pop("survey_timeout")
    if survey:
        if batch_manifest or args["ref"] is not None:
            parser.error("--survey cannot be combined with --batch or --ref")
        if bool(science_list) == bool(args["new"]):
            parser.error("--survey requires exactly one of --new or --science-list")
        if not survey_options["template_filter"]:
            parser.error("--survey requires an explicit --template-filter")
        if fail_fast and not science_list:
            parser.error("--fail-fast requires a batch or science list")
    elif (
        science_list
        or not survey_options["check_moc_coverage"]
        or any(
            survey_options[name] is not None
            for name in (
                "template_filter",
                "template_units",
                "template_gain",
            )
        )
    ):
        parser.error("Survey-specific inputs require --survey")
    elif batch_manifest:
        if args["ref"] is not None or args["new"] is not None:
            parser.error("--batch cannot be combined with --ref/--new")
    else:
        if args["ref"] is None or args["new"] is None:
            parser.error("--ref and --new are required unless --batch is provided")
        if fail_fast:
            parser.error("--fail-fast requires --batch")
    if not survey:
        args["science_filter"] = survey_options.pop("science_filter")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    ref, new, output = (args.pop(name) for name in ("ref", "new", "output"))
    threshold = args.pop("threshold")
    estimate = args.pop("estimate_psf")
    verbose = args.pop("verbose")
    check_only = args.pop("check_only")
    args["beta"] = not args.pop("no_beta")
    args["fitted_psf"] = not args.pop("auto_psf")
    args["cosmic_rays"] = not args.pop("no_cosmic_rays")
    args["plots"] = not args.pop("no_plots")
    try:
        if survey:
            from .survey import compare_survey_batch, compare_survey_images, load_science_list

            if science_list:
                report = compare_survey_batch(
                    load_science_list(science_list),
                    output,
                    survey=survey,
                    config=args,
                    threshold_sigma=threshold,
                    check_only=check_only,
                    fail_fast=fail_fast,
                    **survey_options,
                )
                if check_only:
                    print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
                else:
                    print(f"Survey batch {report.state}. Results: {report.output_dir}")
                return 0 if report.successful else 1
            result = compare_survey_images(
                new,
                output,
                survey=survey,
                config=args,
                threshold_sigma=threshold,
                check_only=check_only,
                **survey_options,
            )
            if check_only:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                print(
                    f"Accepted {len(result.candidates)} unconfirmed survey residuals; "
                    f"rejected {len(result.rejected_candidates)}. Results: {output}"
                )
            return 0
        if batch_manifest:
            from .api import compare_batch, load_batch_manifest

            report = compare_batch(
                load_batch_manifest(batch_manifest),
                output,
                config=args,
                threshold_sigma=threshold,
                fail_fast=fail_fast,
                retain_results=False,
                check_only=check_only,
            )
            summary = report.to_dict()
            if check_only:
                print(json.dumps(summary, indent=2, allow_nan=False))
            else:
                counts = summary["counts"]
                print(
                    f"Batch {report.state}: {counts['completed']} completed, "
                    f"{counts['failed']} failed, {counts['skipped']} skipped. "
                    f"Results: {report.output_dir}"
                )
            return 0 if report.successful else 1
        pipeline = ProperTransientFinder(ref, new, output, args)
        if check_only:
            print(json.dumps(pipeline.preflight(threshold), indent=2, allow_nan=False))
            return 0
        sources = pipeline.run_full_pipeline(threshold, estimate)
    except Exception as error:
        LOGGER.error("Analysis failed: %s", error, exc_info=verbose)
        return 1
    label = (
        "experimental cross-filter residuals"
        if pipeline.options.cross_filter
        else "unconfirmed candidates"
    )
    print(
        f"Accepted {len(sources)} {label}; "
        f"rejected {len(pipeline.rejected_sources)}. Results: {pipeline.output_dir}"
    )
    summary = pipeline.metadata["quality_summary"]
    metrics = summary["metrics"]
    scatter = metrics["flux_scale_scatter"]
    scatter_text = "not estimated" if scatter is None else f"{scatter:.3f}"
    print(
        f"Run quality: {summary['status']}; RMS {metrics['registration_rms_pixels']:.3f} px, "
        f"{metrics['matched_stars']} matched stars, "
        f"{100 * (1 - metrics['masked_fraction']):.1f}% usable, scale scatter {scatter_text}; "
        f"{len(summary['warnings'])} warnings. Details: quality_summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
