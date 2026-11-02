# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Run a bounded injection/recovery campaign outside pytest's regression suite."""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]


def scene_generators():
    for directory in (ROOT / "src", ROOT / "tests"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from test_astro import realistic_survey_pair, wcs_survey_pair

    return {"align": realistic_survey_pair, "wcs": wcs_survey_pair}


def run_trial(task):
    logging.disable(logging.WARNING)
    generators = scene_generators()
    from proper_transient_finder.astro import ProperTransientFinder, psf_fwhm

    started = time.monotonic()
    row = {
        **task,
        "status": "pipeline_error",
        "recovered": False,
        "usable_footprint": False,
        "masked_center": None,
        "host_surface_brightness": None,
        "measured_flux": None,
        "relative_flux_error": None,
        "position_error_pixels": None,
        "score": None,
        "extra_detections": None,
        "accepted_positive": None,
        "accepted_negative": None,
        "rejection_flags": "",
        "error": "",
        "masked_fraction": None,
        "flux_scale": None,
        "registration_rms_pixels": None,
        "ref_fwhm_pixels": None,
        "new_fwhm_pixels": None,
    }
    injection = (
        []
        if task["kind"] == "null"
        else [
            {
                "name": "injection",
                "x": task["x"],
                "y": task["y"],
                "flux": task["flux"],
                "environment": "galaxy",
                "host": "G1",
            }
        ]
    )
    try:
        with TemporaryDirectory(prefix="astro-completeness-") as temporary:
            directory = Path(temporary)
            paths, truth = generators[task["workflow"]](
                directory / "inputs", transients=injection, noise_seed=task["noise_seed"]
            )
            options = {
                "psf_radius": 10,
                "plots": False,
                "align": task["workflow"] == "align",
                "reproject": task["workflow"] == "wcs",
            }
            pipeline = ProperTransientFinder(*paths, directory / "results", options)
            pipeline.perform_image_subtraction()
            sources = pipeline.detect_sources_in_difference(threshold_sigma=5)
            row.update(
                status="null" if not injection else "not_detected",
                extra_detections=len(sources),
                accepted_positive=int((sources["sign"] == 1).sum()),
                accepted_negative=int((sources["sign"] == -1).sum()),
                **{
                    key: pipeline.metadata[key]
                    for key in (
                        "masked_fraction",
                        "flux_scale",
                        "registration_rms_pixels",
                        "ref_fwhm_pixels",
                        "new_fwhm_pixels",
                    )
                },
            )
            if injection:
                source = truth["transients"][0]
                column, pixel_row = int(round(source["x"])), int(round(source["y"]))
                radius = min(
                    pipeline.psf_image.shape[0] // 2,
                    int(np.ceil(psf_fwhm(pipeline.psf_image) * 1.5)),
                )
                footprint = pipeline.mask[
                    pixel_row - radius : pixel_row + radius + 1,
                    column - radius : column + radius + 1,
                ]
                row.update(
                    masked_center=bool(pipeline.mask[pixel_row, column]),
                    usable_footprint=bool(
                        footprint.shape == (2 * radius + 1, 2 * radius + 1) and not footprint.any()
                    ),
                    host_surface_brightness=source["host_surface_brightness_adu_per_pixel"],
                )
                distances = np.hypot(
                    sources["x_centroid"] - source["x"], sources["y_centroid"] - source["y"]
                )
                nearby = sources.loc[(distances < 1) & (sources["sign"] == 1)]
                if len(nearby) == 1:
                    candidate = nearby.iloc[0]
                    row.update(
                        status="recovered",
                        recovered=True,
                        extra_detections=len(sources) - 1,
                        measured_flux=float(candidate["flux"]),
                        score=float(candidate["snr"]),
                        relative_flux_error=float(candidate["flux"] / source["flux"] - 1),
                        position_error_pixels=float(distances.loc[nearby.index[0]]),
                    )
                else:
                    rejected = pipeline.rejected_sources
                    close = (
                        np.hypot(
                            rejected["x_centroid"] - source["x"],
                            rejected["y_centroid"] - source["y"],
                        )
                        < 2
                    )
                    row["rejection_flags"] = ";".join(sorted(set(rejected.loc[close, "flags"])))
                    if not row["usable_footprint"]:
                        row["status"] = "masked"
                    elif row["rejection_flags"]:
                        row["status"] = "rejected"
    except Exception as error:
        row.update(status="pipeline_error", error=f"{type(error).__name__}: {error}")
    row["elapsed_seconds"] = time.monotonic() - started
    return row


def wilson(successes, count):
    if not count:
        return (None, None)
    squared = 1.959963984540054**2
    fraction = successes / count
    center = (fraction + squared / (2 * count)) / (1 + squared / count)
    half = np.sqrt(fraction * (1 - fraction) / count + squared / (4 * count**2))
    half *= np.sqrt(squared) / (1 + squared / count)
    return float(center - half), float(center + half)


def summarize(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        eligible = group.loc[(group["status"] != "pipeline_error") & group["usable_footprint"]]
        successes = int(group["recovered"].sum())
        usable_successes = int(eligible["recovered"].sum())
        lower, upper = wilson(successes, len(group))
        usable_lower, usable_upper = wilson(usable_successes, len(eligible))
        recovered = group.loc[group["recovered"]]
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "injected": len(group),
                "recovered": successes,
                "completeness": successes / len(group),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "usable": len(eligible),
                "usable_completeness": usable_successes / len(eligible) if len(eligible) else None,
                "usable_ci95_lower": usable_lower,
                "usable_ci95_upper": usable_upper,
                "masked": int((group["status"] == "masked").sum()),
                "pipeline_errors": int((group["status"] == "pipeline_error").sum()),
                "median_relative_flux_error": recovered["relative_flux_error"].median(),
                "median_position_error_pixels": recovered["position_error_pixels"].median(),
                "extra_detections": int(group["extra_detections"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(bins, surface_bins, workflows, directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(16, 5 * len(workflows)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(workflows), 3, squeeze=False)
    figure.suptitle("Injection recovery on G1 | fixed host and field | threshold 5 | PSF radius 10")
    for row, workflow in enumerate(workflows):
        selected = bins.loc[bins["workflow"] == workflow]
        for column, key in enumerate(("completeness", "usable_completeness")):
            matrix = selected.pivot(index="flux", columns="radius_re", values=key).sort_index()
            axis = axes[row, column]
            artist = axis.imshow(
                matrix.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1
            )
            axis.set(
                xticks=range(len(matrix.columns)),
                xticklabels=[f"{value:g}" for value in matrix.columns],
                yticks=range(len(matrix.index)),
                yticklabels=[f"{value:g}" for value in matrix.index],
                xlabel="Elliptical radius / effective radius",
                ylabel="Transient flux [science ADU]",
                title=f"{workflow}: "
                + ("all injections" if column == 0 else "usable footprints only"),
            )
            for flux_index in range(len(matrix.index)):
                for radius_index in range(len(matrix.columns)):
                    value = matrix.iloc[flux_index, radius_index]
                    axis.text(
                        radius_index,
                        flux_index,
                        "--" if pd.isna(value) else f"{value:.0%}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.6 else "black",
                    )
            figure.colorbar(artist, ax=axis, label="Accepted positive recovery within 1 pixel")
        axis = axes[row, 2]
        for flux, group in surface_bins.loc[surface_bins["workflow"] == workflow].groupby("flux"):
            group = group.sort_values("surface_brightness_bin")
            axis.plot(
                group["surface_brightness_bin"], group["completeness"], "o-", label=f"{flux:g}"
            )
        axis.set(
            xlabel="Local host brightness bin [science ADU/pixel]",
            ylabel="All-injection completeness",
            title=f"{workflow}: local host brightness",
            xscale="log",
            ylim=(-0.05, 1.05),
        )
        axis.legend(title="Transient ADU", fontsize=8)
    figure.savefig(directory / "completeness.png", dpi=140)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New results directory; never overwritten."
    )
    parser.add_argument(
        "--fluxes", nargs="+", type=float, default=[250, 500, 1000, 2000, 4000, 8000]
    )
    parser.add_argument("--radii", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2, 3])
    parser.add_argument(
        "--trials", type=int, default=8, help="Independent noise trials per brightness/radius bin."
    )
    parser.add_argument(
        "--null-trials", type=int, default=8, help="Host-only controls per workflow."
    )
    parser.add_argument(
        "--workflows", nargs="+", choices=["align", "wcs"], default=["align", "wcs"]
    )
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    args = parser.parse_args(argv)
    if (
        args.trials < 2
        or args.null_trials < 1
        or args.seed < 0
        or any(not np.isfinite(value) or value <= 0 for value in args.fluxes)
        or any(not np.isfinite(value) or value < 0 or value > 3 for value in args.radii)
        or len(set(args.fluxes)) != len(args.fluxes)
        or len(set(args.radii)) != len(args.radii)
        or len(set(args.workflows)) != len(args.workflows)
    ):
        parser.error(
            "Require unique finite positive fluxes, radii in [0,3], "
            ">=2 trials, >=1 null, and nonnegative seed."
        )
    if args.output.exists():
        parser.error(
            "Output already exists; choose a fresh directory. Existing data are never removed."
        )
    generators = scene_generators()
    with TemporaryDirectory(prefix="astro-host-") as temporary:
        _, truth = generators["align"](Path(temporary) / "inputs", transients=[])
    host = next(galaxy for galaxy in truth["galaxies"] if galaxy["name"] == "G1")
    tasks = []
    angle_rng = np.random.default_rng(args.seed)
    phases = angle_rng.uniform(0, 2 * np.pi, len(args.radii))
    theta = np.deg2rad(host["position_angle_degrees"])
    for workflow in args.workflows:
        sequence = 0
        for flux in args.fluxes:
            for radius_index, radius in enumerate(args.radii):
                for trial in range(args.trials):
                    angle = phases[radius_index] + 2 * np.pi * (trial + 0.5) / args.trials
                    major = radius * host["effective_radius_pixels"] * np.cos(angle)
                    minor = (
                        radius
                        * host["effective_radius_pixels"]
                        * host["axis_ratio"]
                        * np.sin(angle)
                    )
                    tasks.append(
                        {
                            "workflow": workflow,
                            "kind": "injection",
                            "flux": flux,
                            "radius_re": radius,
                            "angle_degrees": float(np.rad2deg(angle) % 360),
                            "trial": trial,
                            "noise_seed": args.seed + 100 + 10 * sequence,
                            "x": float(host["x"] + major * np.cos(theta) - minor * np.sin(theta)),
                            "y": float(host["y"] + major * np.sin(theta) + minor * np.cos(theta)),
                        }
                    )
                    sequence += 1
        for trial in range(args.null_trials):
            tasks.append(
                {
                    "workflow": workflow,
                    "kind": "null",
                    "flux": None,
                    "radius_re": None,
                    "angle_degrees": None,
                    "trial": trial,
                    "noise_seed": args.seed + 100 + 10 * sequence,
                    "x": None,
                    "y": None,
                }
            )
            sequence += 1
    args.output.mkdir(parents=True)
    config = {
        **vars(args),
        "output": str(args.output.resolve()),
        "host": host,
        "threshold": 5,
        "psf_radius": 10,
        "matching_radius_pixels": 1,
        "surface_brightness_edges": [0, 10, 30, 100, 300, 1000, 3000, 10000, 1e9],
        "versions": {
            name: version(name)
            for name in ("properimage", "astropy", "numpy", "scipy", "sep", "reproject")
        },
    }
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Running {len(tasks)} image pairs using {args.workers} workers; one injection per pair.",
        flush=True,
    )
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with (args.output / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = None
            for row in executor.map(run_trial, tasks):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                if len(rows) % 20 == 0 or len(rows) == len(tasks):
                    errors = sum(item["status"] == "pipeline_error" for item in rows)
                    print(
                        f"Finished {len(rows)}/{len(tasks)}; errors={errors}; "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    frame = pd.DataFrame(rows)
    injections = frame.loc[frame["kind"] == "injection"].copy()
    bins = summarize(injections, ["workflow", "flux", "radius_re"])
    bins.to_csv(args.output / "completeness_bins.csv", index=False)
    edges = config["surface_brightness_edges"]
    injections["surface_brightness_bin"] = pd.cut(
        injections["host_surface_brightness"], edges, labels=edges[1:], include_lowest=True
    )
    surface_bins = summarize(injections, ["workflow", "flux", "surface_brightness_bin"])
    surface_bins["surface_brightness_bin"] = surface_bins["surface_brightness_bin"].astype(float)
    surface_bins.to_csv(args.output / "surface_brightness_bins.csv", index=False)
    nulls = frame.loc[frame["kind"] == "null"]
    null_summary = (
        nulls.groupby("workflow")
        .agg(
            attempted=("kind", "size"),
            successful=("status", lambda values: (values == "null").sum()),
            positive_detections=("accepted_positive", "sum"),
            negative_detections=("accepted_negative", "sum"),
            extra_detections=("extra_detections", "sum"),
        )
        .reset_index()
    )
    null_summary["detections_per_successful_image"] = null_summary[
        "extra_detections"
    ] / null_summary["successful"].replace(0, np.nan)
    null_summary.to_csv(args.output / "null_controls.csv", index=False)
    summary = {
        "attempted_pairs": len(frame),
        "injected": len(injections),
        "pipeline_errors": int((frame["status"] == "pipeline_error").sum()),
        "elapsed_seconds": time.monotonic() - started,
        "null_controls": json.loads(null_summary.to_json(orient="records")),
        "bins": json.loads(bins.to_json(orient="records")),
        "limitations": [
            "One fixed synthetic host/field; not survey completeness.",
            "Eight trials per default bin give broad 95% Wilson intervals.",
            "Angles and science noise are paired across workflows, not independent samples.",
            "Usable means a truth-centered PSF-fit footprint contains no masked pixels.",
            "Flux bias is conditional on recovery; misses and pipeline errors "
            "remain in all-injection denominators.",
            "Host brightness is the noise-free native science host plus nucleus, before sky.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_results(bins, surface_bins, args.workflows, args.output)
    print(
        f"Results: {args.output.resolve()}; pipeline errors: {summary['pipeline_errors']}",
        flush=True,
    )
    return int(summary["pipeline_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
