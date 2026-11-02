"""Benchmark medium FITS pairs with known transients and detector/satellite artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("clean", "detector", "satellite", "satellite_masked", "mixed_masked")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def image_side(size_mb: float) -> int:
    """Approximate total on-disk SCI(float32)+VAR(float32)+MASK(uint8), decimal MB."""
    if not math.isfinite(size_mb) or not 1 <= size_mb <= 100:
        raise ValueError("FITS size must be between 1 and 100 decimal MB")
    return int(math.sqrt((size_mb * 1_000_000 - 3 * 2880) / 9))


def add_point(image: np.ndarray, column: float, row: float, flux: float) -> None:
    left, bottom = int(column), int(row)
    fraction_x, fraction_y = column - left, row - bottom
    image[bottom : bottom + 2, left : left + 2] += flux * np.outer(
        [1 - fraction_y, fraction_y], [1 - fraction_x, fraction_x]
    )


def make_pair(directory: Path, side: int, scenario: str, seed: int, null: bool):
    """Create paired fixed fields with independent Poisson/read noise and known truth."""
    if side < 256 or scenario not in SCENARIOS:
        raise ValueError("Use side >=256 and a supported scenario")
    directory.mkdir(parents=True, exist_ok=False)
    shape = (side, side)
    field = np.random.default_rng(20261102)
    noise = np.random.default_rng(seed)
    scale, gain, read_noise = 0.8, 2.0, 5.0
    truth = [
        {"name": "isolated_bright", "x": side * 0.31, "y": side * 0.29, "flux": 16000.0},
        {"name": "isolated_faint", "x": side * 0.68, "y": side * 0.27, "flux": 1800.0},
        {"name": "host_transient", "x": side * 0.62 + 6, "y": side * 0.62, "flux": 12000.0},
        {"name": "fading", "x": side * 0.30, "y": side * 0.71, "flux": -12000.0},
        {
            "name": "trail_overlap",
            "x": side * 0.47,
            "y": side * (0.25 + 0.45 * 0.47),
            "flux": 12000.0,
        },
    ]
    stars = np.zeros(shape, dtype=np.float32)
    spacing = max(38, side // 22)
    for row in np.arange(30, side - 30, spacing):
        for column in np.arange(30, side - 30, spacing):
            star_x, star_y = column + field.uniform(-4, 4), row + field.uniform(-4, 4)
            if any(np.hypot(star_x - item["x"], star_y - item["y"]) < 24 for item in truth):
                continue
            add_point(stars, star_x, star_y, float(field.uniform(60000, 250000)))
    host = Sersic2D(
        amplitude=85, r_eff=10, n=1.5, x_0=side * 0.62, y_0=side * 0.62, ellip=0.4, theta=0.4
    )
    lower, upper = int(side * 0.62) - 70, int(side * 0.62) + 71
    rows, columns = np.mgrid[lower:upper, lower:upper]
    stars[lower:upper, lower:upper] += host(columns, rows).astype(np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(side + 1) / 2] * 2
    wcs.wcs.crval = [150, -25]
    wcs.wcs.cdelt = [-0.8 / 3600, 0.8 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    artifacts = []
    paths = []
    for index, sigma in enumerate((1.3, 1.8)):
        scene = stars.copy() * (scale if index else 1)
        if not null:
            for item in truth:
                if (item["flux"] > 0) == bool(index):
                    add_point(
                        scene, item["x"], item["y"], abs(item["flux"]) / (1 if index else scale)
                    )
        expectation = gaussian_filter(scene, sigma).astype(np.float32)
        expectation += 900 if index else 600
        variance = (expectation / gain + (read_noise / gain) ** 2).astype(np.float32)
        image = (
            noise.poisson(expectation * gain) / gain + noise.normal(0, read_noise / gain, shape)
        ).astype(np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        if index and scenario in {"detector", "mixed_masked"}:
            defects = np.random.default_rng(seed + 50000)
            count = max(12, side * side // 25000)
            for row, column in defects.integers(25, side - 25, (count, 2)):
                image[row, column] += float(defects.uniform(6000, 40000))
            image[side // 5 : side // 5 + 2, side // 6 : side // 6 + 4] += 20000
            image[side // 3 : side // 3 + 22, side // 8 : side // 8 + 3] = 90000
            image[:, side // 9] = 0
            mask[:, side // 9] = 1
            image[40:43, 40:43] = np.nan
            mask[40:43, 40:43] = 1
            artifacts += [
                {"type": "cosmic_and_hot_pixels", "count": count},
                {"type": "compact_cosmic_cluster"},
                {"type": "saturation_bleed"},
                {"type": "masked_bad_column"},
                {"type": "nonfinite_patch"},
            ]
        if index and scenario in {"satellite", "satellite_masked", "mixed_masked"}:
            trail_x = np.arange(side, dtype=float)
            trail_y = side * 0.25 + 0.45 * trail_x
            for offset in range(-10, 11):
                pixel_rows = np.rint(trail_y + offset).astype(int)
                pixels = np.arange(side)
                valid = (pixel_rows >= 0) & (pixel_rows < side)
                amplitude = 1600 * np.exp(-0.5 * (offset / 1.6) ** 2)
                image[pixel_rows[valid], pixels[valid]] += (
                    noise.poisson(amplitude * gain, valid.sum()) / gain
                ).astype(np.float32)
                variance[pixel_rows[valid], pixels[valid]] += amplitude / gain
                if scenario in {"satellite_masked", "mixed_masked"}:
                    mask[pixel_rows[valid], pixels[valid]] = 1
            artifacts.append(
                {
                    "type": "satellite_trail",
                    "peak_adu": 1600,
                    "sigma_pixels": 1.6,
                    "masked": scenario in {"satellite_masked", "mixed_masked"},
                }
            )
        header = wcs.to_header()
        header.update(
            FILTER="r",
            BUNIT="adu",
            GAIN=gain,
            SATURATE=60000,
            EXPTIME=60,
            **{"MJD-OBS": 60000 + index},
        )
        path = directory / ("reference.fits" if index == 0 else "science.fits")
        fits.HDUList(
            [
                fits.PrimaryHDU(image, header),
                fits.ImageHDU(variance, name="VAR"),
                fits.ImageHDU(mask, name="MASK"),
            ]
        ).writeto(path, checksum=True)
        paths.append(path)
    manifest = {
        "kind": "synthetic_injections",
        "shape": list(shape),
        "seed": seed,
        "flux_scale": scale,
        "transients": [] if null else truth,
        "artifacts": artifacts,
        "null": null,
        "input_bytes": [path.stat().st_size for path in paths],
    }
    return paths, manifest


def assess(result, truth: dict, radius: float = 2) -> dict:
    """One-to-one, sign-aware associations; unmatched detections are false positives."""
    candidates = result.candidates
    unmatched = set(range(len(candidates)))
    injections = []
    for source in truth["transients"]:
        column, row = int(round(source["x"])), int(round(source["y"]))
        if not (0 <= row < result.mask.shape[0] and 0 <= column < result.mask.shape[1]):
            raise ValueError("Truth coordinate outside the science image")
        half = min(result.psf.shape[0] // 2, 8)
        footprint = result.mask[
            max(0, row - half) : row + half + 1, max(0, column - half) : column + half + 1
        ]
        usable = footprint.shape == (2 * half + 1,) * 2 and not footprint.any()
        sign = source.get("sign", 1 if source.get("flux", 1) > 0 else -1)
        matches = [
            (
                float(
                    np.hypot(candidate.x_centroid - source["x"], candidate.y_centroid - source["y"])
                ),
                position,
            )
            for position, candidate in enumerate(candidates.itertuples())
            if position in unmatched and candidate.sign == sign
        ]
        distance, position = min(matches, default=(float("inf"), -1))
        recovered = distance < radius
        record = {
            **source,
            "usable": bool(usable),
            "recovered": recovered,
            "masked_center": bool(result.mask[row, column]),
            "position_error_pixels": None,
            "relative_flux_error": None,
            "score": None,
            "rejection_flags": "",
        }
        if recovered:
            unmatched.remove(position)
            candidate = candidates.iloc[position]
            record.update(position_error_pixels=distance, score=float(candidate.snr))
            if source.get("flux"):
                record["relative_flux_error"] = float(candidate.flux / source["flux"] - 1)
        else:
            rejected = result.rejected_candidates
            close = (
                np.hypot(rejected["x_centroid"] - source["x"], rejected["y_centroid"] - source["y"])
                < radius
            )
            record["rejection_flags"] = ";".join(sorted(set(rejected.loc[close, "flags"])))
        injections.append(record)
    usable_count = sum(item["usable"] for item in injections)
    return {
        "injections": injections,
        "injected": len(injections),
        "recovered": sum(item["recovered"] for item in injections),
        "usable": usable_count,
        "usable_recovered": sum(item["recovered"] and item["usable"] for item in injections),
        "false_positives": len(unmatched),
        "accepted": len(candidates),
        "rejected": len(result.rejected_candidates),
        "masked_fraction": float(np.mean(result.mask)),
    }


def known_psf_config(directory: Path) -> dict:
    paths = []
    for label, sigma in (("reference", 1.3), ("science", 1.8)):
        path = directory / f"psf_{label}.fits"
        fits.writeto(path, Gaussian2DKernel(sigma, x_size=17, y_size=17).array)
        paths.append(str(path))
    return {"ref_psf": paths[0], "new_psf": paths[1]}


def peak_memory_mb() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1_000_000
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1_000_000 if sys.platform == "darwin" else 1000
    )


def preview(result, paths, truth: dict, destination: Path) -> None:
    from astropy.visualization import ImageNormalize, ZScaleInterval
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    stride = max(1, result.mask.shape[0] // 700)
    panels = [
        (fits.getdata(paths[1], memmap=False), "Science with artifacts", "gray"),
        (result.difference, "Science - scaled reference", "RdBu_r"),
        (result.score, "Empirical score, not confirmation", "RdBu_r"),
        (result.mask, "Excluded pixels", "binary"),
    ]
    for axis, (image, label, cmap) in zip(axes.flat, panels, strict=True):
        sampled = image[::stride, ::stride]
        norm = ImageNormalize(sampled, interval=ZScaleInterval()) if cmap == "gray" else None
        limits = {"vmin": -10, "vmax": 10} if "score" in label.lower() else {}
        axis.imshow(sampled, origin="lower", cmap=cmap, norm=norm, **limits)
        axis.set_title(label)
        for item in truth["transients"]:
            axis.plot(
                item["x"] / stride,
                item["y"] / stride,
                "o",
                ms=9,
                markerfacecolor="none",
                markeredgecolor="lime",
            )
    figure.suptitle("Known synthetic injections (green); not observed astrophysical events")
    figure.savefig(destination, dpi=120)


def run_worker(task: dict) -> dict:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from proper_transient_finder import PipelineConfig, compare_images

    destination = Path(task["directory"])
    destination.mkdir(parents=True, exist_ok=False)
    row = {**task, "status": "pipeline_error", "error": ""}
    with TemporaryDirectory(prefix="ptf-bench-") as temporary:
        temporary = Path(temporary)
        generated = time.perf_counter()
        paths, truth = make_pair(
            temporary / "inputs", task["side"], task["scenario"], task["seed"], task["null"]
        )
        row.update(
            generation_seconds=time.perf_counter() - generated,
            input_bytes=truth["input_bytes"],
            truth_kind=truth["kind"],
            injected=len(truth["transients"]),
            injections=[
                {**source, "recovered": False, "usable": None} for source in truth["transients"]
            ],
        )
        if task.get("keep_products"):
            output = destination / "analysis"
        else:
            output = temporary / "analysis"
        psf_settings = known_psf_config(temporary) if task.get("known_psf") else {}
        config = PipelineConfig(
            plots=task.get("plots", False), reproject=task.get("reproject", False), **psf_settings
        )
        row["config"] = {
            "plots": config.plots,
            "reproject": config.reproject,
            "cosmic_rays": config.cosmic_rays,
            "threshold_sigma": 5,
            "psf_radius": config.psf_radius,
            "psf_mode": "known_injection_model" if psf_settings else "estimated",
        }
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            result = compare_images(*paths, output, config=config)
            row.update(
                analysis_seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu_started,
                stage_timings_seconds=result.metadata.get("stage_timings_seconds", {}),
                status="completed",
                **assess(result, truth),
            )
            row["quality"] = result.metadata["quality_summary"]
            if not task["null"] and task["trial"] == 1:
                preview(result, paths, truth, destination / "overview.png")
            del result
        except Exception as error:
            row.update(
                analysis_seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu_started,
                error=f"{type(error).__name__}: {error}",
            )
        row["peak_process_rss_mb"] = peak_memory_mb()
        if output.is_dir() and not task.get("keep_products"):
            reports = destination / "reports"
            reports.mkdir()
            for path in output.iterdir():
                if path.suffix in {".json", ".csv", ".log"}:
                    shutil.copy2(path, reports / path.name)
        if task.get("keep_inputs"):
            shutil.copytree(temporary / "inputs", destination / "inputs")
        (destination / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return row


def write_table(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def summarize(rows: list[dict]) -> list[dict]:
    groups = sorted({(row["size_mb"], row["scenario"], row["null"]) for row in rows})
    summary = []
    for size, scenario, null in groups:
        selected = [
            row
            for row in rows
            if (row["size_mb"], row["scenario"], row["null"]) == (size, scenario, null)
        ]
        complete = [row for row in selected if row["status"] == "completed"]
        seconds = [row["analysis_seconds"] for row in complete]
        summary.append(
            {
                "size_mb": size,
                "scenario": scenario,
                "null": null,
                "runs": len(selected),
                "completed": len(complete),
                "errors": len(selected) - len(complete),
                "median_seconds": float(np.median(seconds)) if seconds else None,
                "min_seconds": min(seconds) if seconds else None,
                "max_seconds": max(seconds) if seconds else None,
                "max_peak_rss_mb": max(
                    (
                        row["peak_process_rss_mb"]
                        for row in selected
                        if "peak_process_rss_mb" in row
                    ),
                    default=None,
                ),
                "injected": sum(row.get("injected", 0) for row in selected),
                "median_failed_seconds": float(
                    np.median(
                        [
                            row["analysis_seconds"]
                            for row in selected
                            if row["status"] != "completed" and "analysis_seconds" in row
                        ]
                    )
                )
                if any(
                    row["status"] != "completed" and "analysis_seconds" in row for row in selected
                )
                else None,
                **{
                    key: sum(row.get(key, 0) for row in complete)
                    for key in ("recovered", "usable", "usable_recovered", "false_positives")
                },
            }
        )
    return summary


def save_reports(root: Path, rows: list[dict], environment: dict) -> None:
    summary = summarize(rows)
    from proper_transient_finder.astro import atomic_json

    atomic_json(
        root / "benchmark.json", {"environment": environment, "runs": rows, "summary": summary}
    )
    write_table(
        root / "runs.csv",
        [
            {key: value for key, value in row.items() if key not in {"injections", "quality"}}
            for row in rows
        ],
    )
    write_table(root / "summary.csv", summary)
    write_table(
        root / "injections.csv",
        [{"case": row["name"], **source} for row in rows for source in row.get("injections", [])],
    )
    lines = [
        "# Medium FITS Robustness And Timing",
        "",
        "Synthetic injections only; no observed transient certification.",
        "",
        "MB = 1,000,000 bytes per FITS, including SCI + VAR + MASK. A comparison uses two files.",
        "Times include FITS input, cosmic cleaning, subtraction, detection and output I/O;",
        "generation, interpreter startup and benchmark preview are excluded.",
        "Peak RSS is whole-worker peak (imports, generation, analysis and preview), "
        "not incremental RAM.",
        "No dedicated satellite classifier; rejection/masking is measured, not presumed.",
        "",
        "PSF mode is explicit below: supplied injection-model PSFs isolate subtraction/vetting;",
        "they do not validate automatic PSF estimation. "
        "Failed runs never count as completed timing.",
        "",
        "| MB/file | Scenario | Null | Runs/errors | Median s | Range s | Peak MB | "
        "Recovered/all | Usable recovery | False positives |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary:
        median = item["median_seconds"]
        duration = "--" if median is None else f"{median:.2f}"
        extent = "--" if median is None else f"{item['min_seconds']:.2f}-{item['max_seconds']:.2f}"
        peak = item["max_peak_rss_mb"]
        lines.append(
            f"| {item['size_mb']:g} | {item['scenario']} | {item['null']} | "
            f"{item['runs']}/{item['errors']} | {duration} | {extent} | "
            f"{'--' if peak is None else f'{peak:.0f}'} | "
            f"{item['recovered']}/{item['injected']} | "
            f"{item['usable_recovered']}/{item['usable']} | {item['false_positives']} |"
        )
    lines += [
        "",
        "Machine/run configuration:",
        "",
        "```json",
        json.dumps(environment, indent=2),
        "```",
        "",
        "Small trial counts are exploratory, not completeness or latency guarantees.",
        "Failed-case time to failure is retained separately in summary.csv/benchmark.json.",
        "Inspect per-case truth, rejected flags, quality reports and previews.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes-mb", nargs="+", type=float, default=[20, 50])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--keep-inputs", action="store_true")
    parser.add_argument("--keep-products", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Include production PNGs in timing")
    parser.add_argument(
        "--reproject", action="store_true", help="Include reference WCS reprojection"
    )
    parser.add_argument(
        "--known-psf",
        action="store_true",
        help="Synthetic control using supplied injection-model PSFs, not estimation",
    )
    args = parser.parse_args(argv)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from proper_transient_finder.astro import atomic_json

    if args.worker:
        task = json.loads(args.worker.read_text(encoding="utf-8"))
        result = run_worker(task)
        atomic_json(Path(task["directory"]) / "result.json", result)
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.trials < 1 or args.trials > 20 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Use 1-20 trials and a finite positive per-case timeout")
    if args.known_psf and args.reproject:
        parser.error("Supplied PSF controls cannot use WCS reprojection")
    try:
        sides = {size: image_side(size) for size in args.sizes_mb}
    except ValueError as error:
        parser.error(str(error))
    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError("Choose a new benchmark output directory")
    root.mkdir(parents=True)
    environment = {
        "started_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads_per_worker": 1,
        "timing_mode": "full_pipeline_with_io",
        "plots": args.plots,
        "reproject": args.reproject,
        "trials": args.trials,
        "psf_mode": "known_injection_model" if args.known_psf else "estimated",
        "dependencies": {
            name: version(name)
            for name in ("proper-transient-finder", "numpy", "scipy", "astropy", "astroscrappy")
        },
    }
    rows = []
    warmup_environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1")}
    warmup = subprocess.run(
        [sys.executable, "-c", "import proper_transient_finder"],
        cwd=ROOT,
        env=warmup_environment,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    if warmup.returncode:
        raise RuntimeError("Selected interpreter cannot import proper_transient_finder")
    for size, side in sides.items():
        for scenario in args.scenarios:
            for trial in range(1, args.trials + 1):
                for null in (True, False):
                    name = f"mb{size:g}-{scenario}-t{trial}-{'null' if null else 'injected'}"
                    task = {
                        "name": name,
                        "size_mb": size,
                        "side": side,
                        "scenario": scenario,
                        "trial": trial,
                        "seed": 20261102 + trial,
                        "null": null,
                        "directory": str(root / name),
                        "keep_inputs": args.keep_inputs,
                        "keep_products": args.keep_products,
                        "plots": args.plots,
                        "reproject": args.reproject,
                        "known_psf": args.known_psf,
                    }
                    task_path = root / "task.json"
                    atomic_json(task_path, task)
                    started = time.perf_counter()
                    environment_variables = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1")}
                    try:
                        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
                            process = subprocess.run(
                                [
                                    sys.executable,
                                    str(Path(__file__).resolve()),
                                    "--worker",
                                    str(task_path),
                                ],
                                cwd=ROOT,
                                env=environment_variables,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout,
                                check=False,
                            )
                        result_path = root / name / "result.json"
                        if process.returncode or not result_path.is_file():
                            raise RuntimeError(
                                f"Worker exit {process.returncode}; inspect {name}.log"
                            )
                        row = json.loads(result_path.read_text(encoding="utf-8"))
                    except (subprocess.TimeoutExpired, RuntimeError) as error:
                        row = {
                            **task,
                            "status": "timeout"
                            if isinstance(error, subprocess.TimeoutExpired)
                            else "worker_error",
                            "error": str(error),
                        }
                    row["worker_wall_seconds"] = time.perf_counter() - started
                    rows.append(row)
                    save_reports(root, rows, environment)
                    print(
                        f"{name}: {row['status']}; analysis={row.get('analysis_seconds', 0):.2f}s; "
                        f"recovered={row.get('recovered', 0)}/{row.get('injected', 0)}; "
                        f"false_positives={row.get('false_positives', '--')}",
                        flush=True,
                    )
    print(f"Reports: {root / 'REPORT.md'}", flush=True)
    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
