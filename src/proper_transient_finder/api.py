# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Public Python API for FITS image comparison, with optional product persistence."""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .astro import LOGGER, PipelineConfig, ProperTransientFinder, atomic_json


@dataclass(frozen=True)
class PipelineResult:
    """Comparison output; arrays and tables remain editable by the caller.

    Coordinates are zero-based. Positive residuals are science minus scaled
    reference; in cross-filter mode their sign can instead reflect color.
    ``score`` is empirically calibrated, while ``products['score']`` is raw.
    A missing ``output_dir`` means no products were written to disk.
    """

    candidates: pd.DataFrame
    rejected_candidates: pd.DataFrame
    background_diagnostics: pd.DataFrame
    difference: NDArray[np.float64]
    score: NDArray[np.float64]
    psf: NDArray[np.float64]
    mask: NDArray[np.bool_]
    products: dict[str, np.ndarray]
    metadata: dict[str, Any]
    output_dir: Path | None


@dataclass(frozen=True)
class BatchJob:
    """One named pair. Mapping config overrides batch settings; a dataclass replaces them.

    Names must be unique ignoring case and use 1-64 letters, digits, underscores
    or hyphens, beginning with a letter/digit. Windows device names are forbidden.
    A missing threshold inherits the batch threshold.
    """

    name: str
    reference_fits: str | PathLike[str]
    science_fits: str | PathLike[str]
    config: PipelineConfig | Mapping[str, object] | None = None
    threshold_sigma: float | None = None


@dataclass(frozen=True)
class BatchItemResult:
    """Per-job status; full image results are optional to bound batch memory usage."""

    name: str
    state: str
    output_dir: Path | None
    accepted_candidates: int | None = None
    rejected_candidates: int | None = None
    elapsed_seconds: float = 0.0
    error_type: str | None = None
    error: str | None = None
    result: PipelineResult | None = None
    preflight: dict | None = None
    reference_fits: Path | None = None
    science_fits: Path | None = None


@dataclass(frozen=True)
class BatchResult:
    """Ordered job records, including failures/skips; success is not scientific confirmation."""

    items: tuple[BatchItemResult, ...]
    state: str
    output_dir: Path | None
    check_only: bool = False

    @property
    def successful(self) -> bool:
        return bool(self.items) and all(
            item.state in {"completed", "checked"} for item in self.items
        )

    def to_dict(self) -> dict:
        """Return a JSON-safe summary without image arrays or catalog tables."""
        return {
            "state": self.state,
            "successful": self.successful,
            "check_only": self.check_only,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "counts": {
                state: sum(item.state == state for item in self.items)
                for state in (
                    "pending",
                    "running",
                    "completed",
                    "checked",
                    "failed",
                    "skipped",
                    "interrupted",
                )
            },
            "jobs": [
                {
                    "name": item.name,
                    "state": item.state,
                    "output_dir": str(item.output_dir) if item.output_dir is not None else None,
                    "accepted_candidates": item.accepted_candidates,
                    "rejected_candidates": item.rejected_candidates,
                    "elapsed_seconds": item.elapsed_seconds,
                    "error_type": item.error_type,
                    "error": item.error,
                    "preflight": item.preflight,
                    "reference_fits": str(item.reference_fits)
                    if item.reference_fits is not None
                    else None,
                    "science_fits": str(item.science_fits)
                    if item.science_fits is not None
                    else None,
                }
                for item in self.items
            ],
        }


def _config_values(config: PipelineConfig | Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(config, PipelineConfig):
        return asdict(config)
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("Batch configuration must be a PipelineConfig or mapping")
    return dict(config)


def _prepare_batch(
    jobs: Iterable[BatchJob],
    output_dir: str | PathLike[str] | None,
    config: PipelineConfig | Mapping[str, object] | None,
    threshold_sigma: float,
) -> tuple[Path | None, list[tuple[BatchJob, PipelineConfig, float, Path | None]]]:
    root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    try:
        settings = asdict(PipelineConfig(**_config_values(config)))
    except TypeError as error:
        raise ValueError("Unknown batch configuration option") from error
    prepared, names = [], set()
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    for job in jobs:
        if not isinstance(job, BatchJob):
            raise ValueError("Each batch entry must be a BatchJob")
        if (
            not isinstance(job.name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job.name)
            or job.name.upper() in reserved
            or job.name.casefold() in names
        ):
            raise ValueError(f"Invalid or duplicate batch job name: {job.name!r}")
        names.add(job.name.casefold())
        overrides = _config_values(job.config)
        selected = (
            overrides if isinstance(job.config, PipelineConfig) else {**settings, **overrides}
        )
        destination = root / job.name if root is not None else None
        pipeline = ProperTransientFinder(
            job.reference_fits, job.science_fits, destination or Path.cwd(), selected
        )
        threshold = threshold_sigma if job.threshold_sigma is None else job.threshold_sigma
        try:
            if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("Invalid threshold")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Detection threshold for {job.name!r} must be finite and positive"
            ) from error
        paths = [pipeline.ref_path, pipeline.new_path]
        paths += [
            Path(value).expanduser().resolve()
            for name in ("ref_mask", "new_mask", "ref_psf", "new_psf")
            if (value := getattr(pipeline.options, name)) is not None
        ]
        if root is not None and any(path == root or root in path.parents for path in paths):
            raise ValueError(
                "Batch output directory must not contain any job's input image, mask, or PSF"
            )
        normalized = replace(job, reference_fits=pipeline.ref_path, science_fits=pipeline.new_path)
        prepared.append((normalized, pipeline.options, float(threshold), destination))
    if not prepared:
        raise ValueError("A batch must contain at least one job")
    if root is not None and root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(
                "Batch output directory must be new or empty, even with overwrite enabled"
            )
    return root, prepared


def load_batch_manifest(manifest_path: str | PathLike[str]) -> list[BatchJob]:
    """Load a JSON array of jobs, resolving FITS/mask/PSF paths relative to its file.

    Each object requires ref/new and optionally name, config, threshold_sigma.
    Missing names become pair-0001, pair-0002, etc. Unknown keys and malformed
    objects raise ValueError. Path existence and scientific checks occur during
    batch processing/preflight. Loading alone creates no output files.
    """
    path = Path(manifest_path).expanduser().resolve()
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch manifest must be a nonempty JSON array of job objects")
    jobs = []

    def resolve(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest {label} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())

    for index, entry in enumerate(entries, start=1):
        allowed = {"name", "ref", "new", "config", "threshold_sigma"}
        if not isinstance(entry, dict) or set(entry) - allowed or not {"ref", "new"} <= set(entry):
            raise ValueError(
                f"Manifest job {index} requires ref/new and has unsupported or missing fields"
            )
        options = entry.get("config", {})
        if not isinstance(options, dict):
            raise ValueError(f"Manifest job {index} config must be an object")
        options = dict(options)
        for name in ("ref_mask", "new_mask", "ref_psf", "new_psf"):
            if options.get(name) is not None:
                options[name] = resolve(options[name], name)
        jobs.append(
            BatchJob(
                entry.get("name", f"pair-{index:04d}"),
                resolve(entry["ref"], "ref"),
                resolve(entry["new"], "new"),
                options,
                entry.get("threshold_sigma"),
            )
        )
    return jobs


def compare_batch(
    jobs: Iterable[BatchJob],
    output_dir: str | PathLike[str] | None = None,
    *,
    config: PipelineConfig | Mapping[str, object] | None = None,
    threshold_sigma: float = 5.0,
    fail_fast: bool = False,
    retain_results: bool | None = None,
    check_only: bool = False,
) -> BatchResult:
    """Run named FITS pairs sequentially using the unchanged single-pair pipeline.

    Each persisted job writes under output_dir/name; the root must be new/empty.
    batch_summary.json is atomically updated before/after each job. Structural
    name/config/output errors raise before any processing or output creation.
    Image-processing exceptions are recorded per job. By default execution
    continues; fail_fast stops after the first failure and marks remaining jobs
    skipped. Interrupts propagate after recording partial progress when possible.

    Full PipelineResult objects are retained by default only without output_dir;
    override retain_results explicitly to select memory usage. With check_only,
    input preflight is performed without creating any outputs or subtracting.
    Scientific defaults, input checks, and cross-filter caveats remain unchanged.
    """
    root, prepared = _prepare_batch(jobs, output_dir, config, threshold_sigma)
    retain = root is None if retain_results is None else retain_results
    items = [
        BatchItemResult(
            job.name,
            "pending",
            destination,
            reference_fits=Path(job.reference_fits),
            science_fits=Path(job.science_fits),
        )
        for job, _, _, destination in prepared
    ]
    started_utc = datetime.now(UTC).isoformat()

    def report(state: str) -> BatchResult:
        result = BatchResult(tuple(items), state, root, check_only)
        if root is not None and not check_only:
            payload = result.to_dict()
            payload.update(started_utc=started_utc, updated_utc=datetime.now(UTC).isoformat())
            atomic_json(root / "batch_summary.json", payload)
        return result

    if root is not None and not check_only:
        root.mkdir(parents=True, exist_ok=True)
    report("running")
    try:
        for index, (job, options, threshold, destination) in enumerate(prepared):
            items[index] = replace(items[index], state="running")
            started = time.perf_counter()
            report("running")
            try:
                preflight = None
                if check_only:
                    preflight = ProperTransientFinder(
                        job.reference_fits, job.science_fits, destination or Path.cwd(), options
                    ).preflight(threshold)
                    result = None
                else:
                    result = compare_images(
                        job.reference_fits,
                        job.science_fits,
                        destination,
                        config=options,
                        threshold_sigma=threshold,
                    )
                items[index] = replace(
                    items[index],
                    state="checked" if check_only else "completed",
                    accepted_candidates=len(result.candidates) if result is not None else None,
                    rejected_candidates=len(result.rejected_candidates)
                    if result is not None
                    else None,
                    elapsed_seconds=time.perf_counter() - started,
                    result=result if retain else None,
                    preflight=preflight,
                )
                del result
            except Exception as error:
                items[index] = replace(
                    items[index],
                    state="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                LOGGER.warning("Batch job %s failed: %s", job.name, error)
                if fail_fast:
                    for remaining in range(index + 1, len(items)):
                        items[remaining] = replace(
                            items[remaining],
                            state="skipped",
                            error="Not run after an earlier failure",
                        )
                    return report("stopped")
            report("running")
    except BaseException as error:
        for index, item in enumerate(items):
            if item.state in {"pending", "running"}:
                items[index] = replace(
                    item,
                    state="interrupted" if item.state == "running" else "skipped",
                    error_type=type(error).__name__,
                    error="Batch interrupted",
                )
        try:
            report("interrupted")
        except OSError:
            LOGGER.error("Unable to persist interrupted batch status")
        raise
    failed = any(item.state == "failed" for item in items)
    return report("completed_with_errors" if failed else "checked" if check_only else "completed")


def compare_images(
    reference_fits: str | PathLike[str],
    science_fits: str | PathLike[str],
    output_dir: str | PathLike[str] | None = None,
    *,
    config: PipelineConfig | Mapping[str, object] | None = None,
    threshold_sigma: float = 5.0,
) -> PipelineResult:
    """Compare calibrated FITS exposures and return images and candidate catalogs.

    Omit ``output_dir`` for an in-memory run: inputs are read but no files or
    plots are created. Supply it to save the ordinary full-pipeline products,
    logs and status. ``config`` accepts a copied/revalidated PipelineConfig or
    a mapping. Scientific and overwrite defaults are identical to the CLI.

    Raises ValueError for invalid settings or incompatible images, and OSError
    subclasses for missing inputs or protected/unwritable outputs. Exceptions
    propagate to the caller; this function never exits or prints to stdout.
    Cross-filter comparisons require ``config.cross_filter=True`` and remain
    experimental: candidate scores do not establish variability significance.
    """
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("Detection threshold must be finite and positive")
    pipeline = ProperTransientFinder(
        reference_fits, science_fits, Path.cwd() if output_dir is None else output_dir, config
    )
    if output_dir is None:
        started = time.perf_counter()
        with pipeline._stage("provenance"):
            pipeline._record_provenance()
        with pipeline._stage("prepare_and_subtract"):
            pipeline.perform_image_subtraction()
        with pipeline._stage("detection"):
            candidates = pipeline.detect_sources_in_difference(threshold_sigma)
        pipeline.metadata["elapsed_seconds"] = time.perf_counter() - started
        pipeline.metadata["quality_summary"] = pipeline.quality_summary()
    else:
        candidates = pipeline.run_full_pipeline(threshold_sigma)
    return PipelineResult(
        candidates=candidates,
        rejected_candidates=pipeline.rejected_sources,
        background_diagnostics=pipeline.local_background_diagnostics,
        difference=pipeline.diff_image,
        score=pipeline.scorr_image,
        psf=pipeline.psf_image,
        mask=pipeline.mask,
        products=pipeline.products,
        metadata=copy.deepcopy(pipeline.metadata),
        output_dir=None if output_dir is None else pipeline.output_dir,
    )
