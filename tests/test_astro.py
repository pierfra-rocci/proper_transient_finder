# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Synthetic regression tests for astronomical difference imaging."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("properimage")
fits = pytest.importorskip("astropy.io.fits")
Gaussian2DKernel = pytest.importorskip("astropy.convolution").Gaussian2DKernel
astro = pytest.importorskip("proper_transient_finder.astro")
ProperTransientFinder = astro.ProperTransientFinder
ImageFrame = astro.ImageFrame
PipelineConfig = astro.PipelineConfig
calibrate_score = astro.calibrate_score
kernel_subtract = astro.kernel_subtract
load_frame = astro.load_frame
normalize_psf = astro.normalize_psf
real_image = astro.real_image


@pytest.mark.parametrize(
    "suffix",
    [
        ".fits",
        ".fit",
        ".fts",
        ".fits.gz",
        ".fit.gz",
        ".fts.gz",
        ".fits.fz",
        ".fit.fz",
        ".fts.fz",
    ],
)
def test_fits_filename_and_compression_variants(tmp_path, suffix):
    from proper_transient_finder.survey import _normalized_copy, prepare_template_request

    wcs = astro.WCS(naxis=2)
    wcs.wcs.crpix = [64.5, 64.5]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.8 / 3600, 0.8 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header.update(FILTER="r", BUNIT="adu", GAIN=2, SATURATE=60000)
    data = np.random.default_rng(42).integers(1000, 1100, (128, 128), dtype=np.int16)
    variance = np.full(data.shape, 25, dtype=np.int16)
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[20, 21] = 4
    image_type = fits.CompImageHDU if suffix.endswith(".fz") else fits.ImageHDU
    path = tmp_path / ("science" + suffix)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            image_type(data, name="SCI"),
            image_type(variance, name="VAR"),
            image_type(mask, name="MASK"),
        ]
    ).writeto(path, checksum=True)
    original = path.read_bytes()
    frame = load_frame(path, cosmic_rays=False)
    assert frame.mask[20, 21]
    assert not frame.mask[64, 64]
    np.testing.assert_allclose((frame.data + frame.background)[~frame.mask], data[~frame.mask])
    np.testing.assert_array_equal(frame.variance[~frame.mask], 25)
    request = prepare_template_request(path, "r")
    assert request.survey == "panstarrs"
    assert request.ra_deg == pytest.approx(180)
    normalized = tmp_path / "normalized.fits"
    _normalized_copy(path, normalized, "r", "adu")
    copied = load_frame(normalized, cosmic_rays=False)
    np.testing.assert_array_equal(copied.mask, frame.mask)
    np.testing.assert_allclose(copied.data, frame.data)
    assert path.read_bytes() == original


def robustness_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments/robustness.py"
    specification = importlib.util.spec_from_file_location("robustness_experiment", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size", [20, 50])
def test_robustness_benchmark_sizes_include_auxiliary_planes(size):
    benchmark = robustness_module()
    side = benchmark.image_side(size)
    estimated = side * side * 9 + 3 * 2880
    assert abs(estimated / 1_000_000 - size) < 0.1


@pytest.mark.parametrize(
    "scenario", ["clean", "detector", "satellite", "satellite_masked", "mixed_masked"]
)
def test_robustness_benchmark_small_scene(tmp_path, scenario):
    benchmark = robustness_module()
    paths, truth = benchmark.make_pair(tmp_path / "inputs", 768, scenario, 19, False)
    result = astro.ProperTransientFinder(
        *paths, tmp_path / "analysis", {"plots": False, **benchmark.known_psf_config(tmp_path)}
    )
    result.perform_image_subtraction()
    sources = result.detect_sources_in_difference()
    from types import SimpleNamespace

    assessment = benchmark.assess(
        SimpleNamespace(
            candidates=sources,
            rejected_candidates=result.rejected_sources,
            mask=result.mask,
            psf=result.psf_image,
        ),
        truth,
    )
    assert assessment["injected"] == 5
    assert assessment["recovered"] >= 3
    if scenario in {"satellite_masked", "mixed_masked"}:
        overlap = next(item for item in assessment["injections"] if item["name"] == "trail_overlap")
        assert overlap["masked_center"]
        assert not overlap["recovered"]
    assert all(0 < size < 6_000_000 for size in truth["input_bytes"])


def test_robustness_summary_keeps_failed_injections_in_denominator():
    benchmark = robustness_module()
    rows = [
        {
            "size_mb": 20,
            "scenario": "clean",
            "null": False,
            "status": "completed",
            "analysis_seconds": 10.0,
            "peak_process_rss_mb": 500.0,
            "injected": 5,
            "recovered": 4,
            "usable": 4,
            "usable_recovered": 4,
            "false_positives": 2,
        },
        {
            "size_mb": 20,
            "scenario": "clean",
            "null": False,
            "status": "pipeline_error",
            "analysis_seconds": 3.0,
            "peak_process_rss_mb": 600.0,
            "injected": 5,
        },
    ]
    summary = benchmark.summarize(rows)[0]
    assert summary["injected"] == 10
    assert summary["recovered"] == 4
    assert summary["median_seconds"] == 10
    assert summary["median_failed_seconds"] == 3
    assert summary["max_peak_rss_mb"] == 600


@pytest.mark.parametrize(
    ("dec", "band", "provider"),
    [
        (30, "r", "panstarrs"),
        (-45, "g", "skymapper"),
        (-10, "r", "panstarrs"),
        (-10, "u", "skymapper"),
    ],
)
def test_survey_request_uses_wcs_coverage_and_requested_filter(tmp_path, dec, band, provider):
    from astropy.wcs import WCS

    from proper_transient_finder.survey import prepare_template_request

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [16.5, 16.5]
    wcs.wcs.crval = [180, dec]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "Sloan-r"
    path = tmp_path / "science.fits"
    fits.writeto(path, np.zeros((32, 32)), header)
    request = prepare_template_request(path, band)
    assert request.survey == provider
    assert request.science_filter == "r"
    assert request.native_science_filter == "Sloan-r"
    assert request.ra_deg == pytest.approx(180)
    assert request.dec_deg == pytest.approx(dec)
    assert request.size_deg >= 32 / 3600
    assert set(tmp_path.iterdir()) == {path}
    with pytest.raises(ValueError, match="survey/filter"):
        prepare_template_request(path, "x")


@pytest.mark.parametrize(
    ("provider", "declination", "limit"),
    [
        ("panstarrs", 30, "25.00"),
        ("skymapper", -45, "9.90"),
    ],
)
def test_survey_oversized_footprint_explains_limit_and_accepts_explicit_crop(
    tmp_path,
    provider,
    declination,
    limit,
):
    from astropy.nddata import Cutout2D

    from proper_transient_finder.survey import prepare_template_request

    wcs = astro.WCS(naxis=2)
    wcs.wcs.crpix = [256.5, 256.5]
    wcs.wcs.crval = [180, declination]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "r"
    data = np.zeros((512, 512), dtype=np.float32)
    source = tmp_path / "science.fits"
    fits.writeto(source, data, header)
    original = source.read_bytes()
    with pytest.raises(ValueError) as raised:
        prepare_template_request(source, "r", survey=provider)
    message = str(raised.value)
    assert f"{provider} requires a cutout smaller than {limit} arcmin" in message
    assert "WCS-preserving science crop" in message
    assert "Automatic tiling is not implemented" in message
    assert set(tmp_path.iterdir()) == {source}
    crop = Cutout2D(data, (255.5, 255.5), (64, 64), wcs=wcs, mode="strict")
    header.update(crop.wcs.to_header())
    cropped = tmp_path / "crop.fits"
    fits.writeto(cropped, crop.data, header)
    request = prepare_template_request(cropped, "r", survey=provider)
    assert request.size_deg * 60 < float(limit)
    assert request.ra_deg == pytest.approx(180)
    assert request.dec_deg == pytest.approx(declination)
    assert source.read_bytes() == original


def test_survey_request_rejects_missing_wcs_without_writes(tmp_path):
    from proper_transient_finder.survey import prepare_template_request

    path = tmp_path / "science.fits"
    fits.writeto(path, np.zeros((32, 32)))
    with pytest.raises(ValueError, match="celestial WCS"):
        prepare_template_request(path, "r")
    assert set(tmp_path.iterdir()) == {path}


def test_survey_science_list_resolves_paths_and_rejects_pair_manifest(tmp_path):
    from proper_transient_finder.survey import load_science_list

    text = tmp_path / "science.txt"
    text.write_text("# science exposures\nfirst.fits\nsecond.fits\n")
    assert load_science_list(text) == [tmp_path / "first.fits", tmp_path / "second.fits"]
    manifest = tmp_path / "science.json"
    manifest.write_text(json.dumps(["first.fits"]))
    assert load_science_list(manifest) == [tmp_path / "first.fits"]
    manifest.write_text(json.dumps([{"ref": "reference.fits", "new": "science.fits"}]))
    with pytest.raises(ValueError, match="path string"):
        load_science_list(manifest)


def test_survey_uses_sci_header_when_variance_is_first(tmp_path):
    from proper_transient_finder import survey

    wcs = astro.WCS(naxis=2)
    wcs.wcs.crpix = [32.5, 32.5]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    primary = fits.PrimaryHDU(header=fits.Header({"FILTER": "Sloan-r", "BUNIT": "adu"}))
    path = tmp_path / "science.fits"
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(np.ones((64, 64)), name="VAR"),
            fits.ImageHDU(np.ones((64, 64)), wcs.to_header(), name="SCI"),
        ]
    ).writeto(path)
    request = survey.prepare_template_request(path, "g")
    assert request.science_filter == "r"
    prepared = tmp_path / "prepared.fits"
    survey._normalized_copy(path, prepared, "r", "adu", gain=2)
    with fits.open(prepared) as hdus:
        assert hdus["SCI"].header["GAIN"] == 2
        assert hdus["SCI"].header["FILTER"] == "r"
        assert "FILTER" not in hdus["VAR"].header


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": float("nan")},
        {"max_download_mb": -1},
        {"threshold_sigma": 0},
        {"template_gain": -1},
        {"template_units": "mag"},
        {"config": {"align": True}},
    ],
)
def test_survey_batch_rejects_invalid_shared_options_before_writes(tmp_path, options):
    from proper_transient_finder import compare_survey_batch

    with pytest.raises(ValueError):
        compare_survey_batch(
            [tmp_path / "science.fits"],
            tmp_path / "results",
            template_filter="r",
            cache_dir=tmp_path / "cache",
            **options,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments",
    [
        ["--survey", "auto", "--new", "science.fits"],
        [
            "--survey",
            "auto",
            "--new",
            "science.fits",
            "--template-filter",
            "r",
            "--ref",
            "ref.fits",
        ],
        ["--science-list", "science.txt"],
        ["--survey", "auto", "--new", "science.fits", "--science-list", "science.txt"],
    ],
)
def test_survey_cli_rejects_incompatible_inputs(arguments):
    with pytest.raises(SystemExit) as raised:
        astro.main(arguments)
    assert raised.value.code == 2


def test_survey_cli_routes_science_list_without_local_reference(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from proper_transient_finder import survey

    path = tmp_path / "images.txt"
    path.write_text("science.fits\n")

    def compare(images, output, **options):
        assert images == [tmp_path / "science.fits"]
        assert options["template_filter"] == "g"
        assert options["check_only"]
        assert options["config"]["cosmic_rays"]
        return SimpleNamespace(successful=True, to_dict=lambda: {"state": "checked"})

    monkeypatch.setattr(survey, "compare_survey_batch", compare)
    assert (
        astro.main(
            [
                "--survey",
                "auto",
                "--science-list",
                str(path),
                "--template-filter",
                "g",
                "--check-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == "checked"


@pytest.mark.parametrize("fail_fast", [False, True])
def test_survey_batch_preserves_failures_and_fail_fast(tmp_path, monkeypatch, fail_fast):
    from types import SimpleNamespace

    from proper_transient_finder import survey

    def compare(science, output, **options):
        if science.name == "bad.fits":
            raise TimeoutError("mock service timeout")
        output.mkdir()
        return SimpleNamespace(
            candidates=[],
            rejected_candidates=[],
            metadata={"survey_reference": {"reference_fits": str(tmp_path / "cached.fits")}},
        )

    monkeypatch.setattr(survey, "compare_survey_images", compare)
    report = survey.compare_survey_batch(
        [tmp_path / "bad.fits", tmp_path / "good.fits"],
        tmp_path / "results",
        template_filter="r",
        cache_dir=tmp_path / "cache",
        fail_fast=fail_fast,
        config={"plots": False},
    )
    assert report.items[0].state == "failed"
    assert report.items[1].state == ("skipped" if fail_fast else "completed")
    assert (tmp_path / "results/batch_summary.json").is_file()
    assert report.items[0].error_type == "TimeoutError"


def test_survey_check_only_uses_no_network_or_writes(tmp_path, monkeypatch):
    from proper_transient_finder import survey

    wcs = astro.WCS(naxis=2)
    wcs.wcs.crpix = [32.5, 32.5]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "r"
    image = tmp_path / "science.fits"
    fits.writeto(image, np.ones((64, 64)), header)

    def fail(*args, **kwargs):
        pytest.fail("Check-only must not access survey services")

    monkeypatch.setattr(survey, "_download_url", fail)
    report = survey.compare_survey_batch(
        [image],
        tmp_path / "results",
        template_filter="g",
        cache_dir=tmp_path / "cache",
        check_only=True,
    )
    assert report.successful
    assert report.items[0].preflight["network_used"] is False
    assert report.items[0].preflight["request"]["comparison_mode"] == "experimental cross-filter"
    assert set(tmp_path.iterdir()) == {image}


@pytest.mark.parametrize("template_band", ["r", "g"])
def test_survey_real_subtraction_and_final_plot(tmp_path, monkeypatch, template_band):
    from proper_transient_finder import survey

    original = synthetic_pair(tmp_path)
    wcs = astro.WCS(naxis=2)
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    paths = []
    for source in original:
        path = tmp_path / ("wcs_" + source.name)
        with fits.open(source, memmap=False) as hdus:
            hdus[0].header.update(wcs.to_header())
            hdus[0].header["BUNIT"] = "adu"
            hdus[0].header["FILTER"] = "r"
            hdus.writeto(path)
        paths.append(path)
    metadata = {"native_bunit": "adu", "warnings": [survey.SURVEY_WARNING]}
    monkeypatch.setattr(
        survey,
        "fetch_survey_template",
        lambda *args, **kwargs: survey.SurveyTemplate(paths[0], metadata),
    )
    result = survey.compare_survey_images(
        paths[1],
        tmp_path / "results",
        template_filter=template_band,
        cache_dir=tmp_path / "cache",
        config={"cosmic_rays": False, "plots": True},
    )
    assert not result.candidates.empty
    assert (tmp_path / "results/survey_diagnostic.png").stat().st_size > 10_000
    assert (
        json.loads((tmp_path / "results/survey_reference.json").read_text())["state"] == "completed"
    )


def test_survey_skymapper_discovery_uses_older_same_band_exposure(monkeypatch):
    from io import BytesIO
    from pathlib import Path
    from urllib.parse import parse_qs, urlsplit

    from astropy.table import Table

    from proper_transient_finder import survey

    table = Table(
        {
            "band": ["r", "r", "r", "g"],
            "mjd_obs": [50000.0, 51000.0, 61000.0, 50000.0],
            "exptime": [5.0, 100.0, 200.0, 200.0],
            "mean_fwhm": [1.0, 2.0, 1.0, 1.0],
            "unique_image_id": ["short", "older-long", "future", "other-band"],
            "image_type": ["main"] * 4,
            "get_fits": ["https://api.skymapper.nci.org.au/image"] * 4,
            "get_mask": ["https://api.skymapper.nci.org.au/mask"] * 4,
        }
    )
    stream = BytesIO()
    table.write(stream, format="votable")

    def download(url, *args):
        assert parse_qs(urlsplit(url).query)["MJD_END"] == ["60000"]
        return stream.getvalue()

    monkeypatch.setattr(survey, "_download_url", download)
    request = survey.TemplateRequest(
        Path("science.fits"), "skymapper", "r", "r", "r", 180, -45, 0.01, 60000
    )
    result = survey._discover_template(request, 30)
    assert result["image_id"] == "older-long"
    assert result["variance_url"] is None


@pytest.mark.parametrize("declared_length", ["1000", None])
def test_survey_download_rejects_oversized_response(monkeypatch, declared_length):
    from io import BytesIO
    from types import SimpleNamespace

    from proper_transient_finder import survey

    class Response(BytesIO):
        headers = {"Content-Length": declared_length}

    monkeypatch.setattr(
        survey,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Response(b"x" * 11)),
    )
    with pytest.raises(ValueError, match="size limit"):
        survey._download_url("https://ps1images.stsci.edu/image", 30, 10)


def test_survey_redirect_rejects_external_host():
    from proper_transient_finder.survey import _SurveyRedirect

    with pytest.raises(ValueError, match="approved HTTPS"):
        _SurveyRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.org/file")


def test_survey_batch_failure_plot_preserves_report(tmp_path, monkeypatch):
    from proper_transient_finder import survey

    def compare(science, output, **options):
        output.mkdir()
        (output / "survey_reference.json").write_text(
            json.dumps(
                {
                    "science_fits": str(science),
                    "survey": "panstarrs",
                    "state": "failed",
                    "image_id": "preserve-template-id",
                    "warnings": [survey.SURVEY_WARNING],
                }
            )
        )
        raise ValueError("mock scientific quality failure")

    monkeypatch.setattr(survey, "compare_survey_images", compare)
    report = survey.compare_survey_batch(
        [tmp_path / "science.fits"],
        tmp_path / "results",
        template_filter="r",
        cache_dir=tmp_path / "cache",
    )
    assert not report.successful
    output = report.items[0].output_dir
    assert (output / "survey_diagnostic.png").stat().st_size > 10_000
    assert (
        json.loads((output / "survey_reference.json").read_text())["image_id"]
        == "preserve-template-id"
    )


@pytest.mark.parametrize("persist", [False, True])
def test_survey_https_rejects_non_service_urls(persist):
    from proper_transient_finder.survey import _validate_url

    with pytest.raises(ValueError, match="approved HTTPS"):
        _validate_url("http://ps1images.stsci.edu/" if persist else "https://example.org/data")


def test_survey_download_validates_caches_and_preserves_native_metadata(tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.parse import parse_qs, urlsplit

    from astropy.wcs import WCS

    from proper_transient_finder import survey

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [32.5, 32.5]
    wcs.wcs.crval = [180, 20]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["FILTER"] = "r"
    science = tmp_path / "science.fits"
    fits.writeto(science, np.ones((64, 64)), header)
    request = survey.prepare_template_request(science, "r")
    calls = []

    def download(url, timeout, limit):
        calls.append(url)
        if "ps1filenames" in url:
            return (
                b"filter type filename badflag\nr stack /test.fits 0\n"
                b"r stack.wt /test.wt.fits 0\nr stack.mask /test.mask.fits 0\n"
            )
        image = np.ones((64, 64))
        if ".mask.fits" in parse_qs(urlsplit(url).query)["red"][0]:
            image[0, 0] = 8
        stream = BytesIO()
        fits.writeto(stream, image, header)
        return stream.getvalue()

    monkeypatch.setattr(survey, "_download_url", download)
    template = survey.fetch_survey_template(request, tmp_path / "cache")
    assert len(calls) == 4
    with fits.open(template.fits_path) as hdus:
        assert hdus["MASK"].data[0, 0] == 8
        assert "VAR" in hdus
    assert template.metadata["native_reference_filter"] == "r"
    assert template.metadata["warnings"]
    assert survey.fetch_survey_template(request, tmp_path / "cache").fits_path == template.fits_path
    assert len(calls) == 4
    template.fits_path.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="checksum mismatch"):
        survey.fetch_survey_template(request, tmp_path / "cache")


@pytest.mark.parametrize("persist", [False, True])
def test_batch_runs_real_pairs_and_preserves_single_pair_measurements(tmp_path, persist) -> None:
    from proper_transient_finder import BatchJob, compare_batch, compare_images

    paths = synthetic_pair(tmp_path)
    config = PipelineConfig(plots=False, cosmic_rays=False)
    expected = compare_images(*paths, config=config)
    output = tmp_path / "batch" if persist else None
    jobs = [BatchJob("first", *paths), BatchJob("second", *paths)]
    report = compare_batch(jobs, output, config=config)
    assert report.successful
    assert [item.name for item in report.items] == ["first", "second"]
    assert all(item.accepted_candidates == len(expected.candidates) for item in report.items)
    if persist:
        summary = json.loads((output / "batch_summary.json").read_text())
        assert summary["state"] == "completed"
        assert summary["counts"]["completed"] == 2
        assert all(item.result is None for item in report.items)
        for name in ("first", "second"):
            assert (
                json.loads((output / name / "run_status.json").read_text())["state"] == "completed"
            )
    else:
        assert all(item.result is not None for item in report.items)
        astro.pd.testing.assert_frame_equal(report.items[0].result.candidates, expected.candidates)
        np.testing.assert_array_equal(report.items[1].result.score, expected.score)
        assert set(tmp_path.iterdir()) == set(paths)


@pytest.mark.parametrize("band", ["r", "g"])
def test_survey_wrapper_reuses_comparison_without_editing_science(tmp_path, monkeypatch, band):
    from dataclasses import replace
    from types import SimpleNamespace

    from proper_transient_finder import api, survey

    paths = synthetic_pair(tmp_path)
    original = paths[1].read_bytes()
    header = fits.getheader(paths[1])
    if not astro.WCS(header).has_celestial:
        wcs = astro.WCS(naxis=2)
        wcs.wcs.crpix = [128, 128]
        wcs.wcs.crval = [180, 20]
        wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        with fits.open(paths[1], memmap=False) as hdus:
            hdus[0].header.update(wcs.to_header())
            hdus[0].header["BUNIT"] = "adu"
            hdus.writeto(tmp_path / "wcs_science.fits")
        science = tmp_path / "wcs_science.fits"
    else:
        science = paths[1]
    original_science = science.read_bytes()
    request = survey.prepare_template_request(science, band)
    cache = tmp_path / "cache"
    cache.mkdir()
    reference = cache / "downloaded.fits"
    with fits.open(science, memmap=False) as hdus:
        hdus.writeto(reference)
    metadata = {"native_bunit": "adu", "warnings": [survey.SURVEY_WARNING]}
    monkeypatch.setattr(
        survey,
        "fetch_survey_template",
        lambda *args, **kwargs: survey.SurveyTemplate(reference, metadata),
    )
    monkeypatch.setattr(
        survey,
        "prepare_template_request",
        lambda *args, **kwargs: replace(request, science_filter="r"),
    )
    seen = {}

    def compare(ref, new, output, *, config, threshold_sigma):
        seen.update(config=config, ref=ref, new=new)
        output.mkdir()
        return SimpleNamespace(metadata={}, output_dir=output)

    monkeypatch.setattr(api, "compare_images", compare)
    result = survey.compare_survey_images(
        science, tmp_path / "result", template_filter=band, cache_dir=cache, config={"plots": False}
    )
    assert seen["config"].reproject
    assert seen["config"].cross_filter == (band == "g")
    assert science.read_bytes() == original_science
    assert paths[1].read_bytes() == original
    assert result.metadata["survey_reference"]["warnings"]


@pytest.mark.parametrize("fail_fast", [False, True])
def test_batch_records_failures_and_optional_skips(tmp_path, fail_fast) -> None:
    from proper_transient_finder import BatchJob, compare_batch

    paths = synthetic_pair(tmp_path)
    jobs = [BatchJob("missing", tmp_path / "absent.fits", paths[1]), BatchJob("valid", *paths)]
    report = compare_batch(
        jobs, tmp_path / "batch", config={"plots": False, "cosmic_rays": False}, fail_fast=fail_fast
    )
    assert not report.successful
    assert report.items[0].state == "failed"
    assert report.items[0].error_type == "FileNotFoundError"
    assert report.items[1].state == ("skipped" if fail_fast else "completed")
    assert report.state == ("stopped" if fail_fast else "completed_with_errors")


@pytest.mark.parametrize("names", [("../escape",), ("same", "SAME"), ("CON",), ()])
def test_batch_rejects_bad_names_before_writing(tmp_path, names) -> None:
    from proper_transient_finder import BatchJob, compare_batch

    jobs = [BatchJob(name, tmp_path / "ref", tmp_path / "new") for name in names]
    with pytest.raises(ValueError):
        compare_batch(jobs, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


def test_batch_manifest_cli_resolves_relative_paths_and_preflight_is_read_only(
    tmp_path, monkeypatch, capsys
) -> None:
    from proper_transient_finder import load_batch_manifest

    paths = synthetic_pair(tmp_path)
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps([{"ref": paths[0].name, "new": paths[1].name}]), encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    jobs = load_batch_manifest(manifest)
    assert jobs[0].name == "pair-0001"
    assert jobs[0].reference_fits == str(paths[0].resolve())
    output = tmp_path / "batch"
    arguments = [
        "--batch",
        str(manifest),
        "--output",
        str(output),
        "--no-cosmic-rays",
        "--no-plots",
    ]
    assert astro.main([*arguments, "--check-only"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "checked"
    assert summary["jobs"][0]["preflight"]["not_checked"]
    assert not output.exists()
    assert astro.main(arguments) == 0
    assert "1 completed" in capsys.readouterr().out
    assert (output / "pair-0001/detected_sources.csv").is_file()


@pytest.mark.parametrize(
    "arguments",
    [[], ["--batch", "jobs.json", "--ref", "ref"], ["--ref", "ref", "--new", "new", "--fail-fast"]],
)
def test_batch_cli_rejects_conflicting_or_missing_modes(arguments) -> None:
    with pytest.raises(SystemExit) as error:
        astro.main(arguments)
    assert error.value.code == 2


def test_batch_interrupt_preserves_partial_summary(tmp_path, monkeypatch) -> None:
    from proper_transient_finder import BatchJob, api, compare_batch

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "compare_images", interrupt)
    jobs = [
        BatchJob("first", tmp_path / "ref", tmp_path / "new"),
        BatchJob("later", tmp_path / "ref", tmp_path / "new"),
    ]
    output = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        compare_batch(jobs, output)
    summary = json.loads((output / "batch_summary.json").read_text())
    assert summary["state"] == "interrupted"
    assert [item["state"] for item in summary["jobs"]] == ["interrupted", "skipped"]


def test_batch_prevalidates_all_configs_and_protects_existing_outputs(tmp_path) -> None:
    from proper_transient_finder import BatchJob, compare_batch

    output = tmp_path / "batch"
    jobs = [BatchJob("first", "ref", "new"), BatchJob("invalid", "ref", "new", {"psf_radius": 0})]
    with pytest.raises(ValueError, match="psf_radius"):
        compare_batch(jobs, output)
    assert not output.exists()
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        compare_batch(jobs[:1], output, config={"overwrite": True})
    assert marker.read_text(encoding="utf-8") == "original"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize(
    "input_kind", ["reference", "science", "ref_mask", "new_mask", "ref_psf", "new_psf"]
)
def test_batch_protects_all_jobs_inputs_before_writing(tmp_path, input_kind) -> None:
    from proper_transient_finder import BatchJob, compare_batch

    output = tmp_path / "batch"
    paths = [tmp_path / "ref", tmp_path / "new"]
    options = {}
    protected = output / "first" / "difference.fits"
    if input_kind in {"ref_psf", "new_psf"}:
        options.update(
            ref_psf=str(tmp_path / "ref-psf.fits"), new_psf=str(tmp_path / "new-psf.fits")
        )
    if input_kind in {"reference", "science"}:
        paths[input_kind == "science"] = protected
    else:
        options[input_kind] = str(protected)
    jobs = [BatchJob("first", "ref", "new"), BatchJob("later", *paths, config=options)]
    with pytest.raises(ValueError, match="must not contain"):
        compare_batch(jobs, output)
    assert not output.exists()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "5", True])
def test_batch_rejects_bad_thresholds_before_writing(tmp_path, value) -> None:
    from proper_transient_finder import BatchJob, compare_batch

    with pytest.raises(ValueError, match="threshold"):
        compare_batch([BatchJob("first", "ref", "new", threshold_sigma=value)], tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "[]",
        '[{"ref":"r"}]',
        '[{"ref":"r","new":"n","typo":1}]',
        '[{"ref":0,"new":"n"}]',
        '[{"ref":"r","new":"n","config":[]}]',
        '[{"ref":"r","new":"n","config":{"ref_mask":0}}]',
        "not json",
    ],
)
def test_batch_rejects_malformed_manifests(tmp_path, content) -> None:
    from proper_transient_finder import load_batch_manifest

    path = tmp_path / "jobs.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_batch_manifest(path)
    assert list(tmp_path.iterdir()) == [path]


def test_batch_overrides_and_retention_use_existing_pair_api(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from proper_transient_finder import BatchJob, api, compare_batch, load_batch_manifest

    calls = []
    result = SimpleNamespace(candidates=[1], rejected_candidates=[])

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(api, "compare_images", capture)
    jobs = [
        BatchJob("mapping", "ref", "new", {"psf_radius": 10}, 6),
        BatchJob("complete", "ref", "new", PipelineConfig(plots=False)),
    ]
    report = compare_batch(
        jobs, tmp_path / "batch", config={"cosmic_rays": False}, retain_results=True
    )
    assert report.items[0].result is result
    assert calls[0][1]["config"].psf_radius == 10
    assert not calls[0][1]["config"].cosmic_rays
    assert calls[0][1]["threshold_sigma"] == 6
    assert calls[1][1]["config"].cosmic_rays
    assert calls[1][1]["threshold_sigma"] == 5
    assert report.to_dict()["jobs"][0]["reference_fits"] == str((tmp_path.cwd() / "ref").resolve())
    assert compare_batch(jobs, retain_results=False).items[0].result is None
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        '[{"ref":"ref","new":"new","config":{"ref_mask":"mask.fits","new_psf":"psf.fits"}}]',
        encoding="utf-8",
    )
    loaded = load_batch_manifest(manifest)[0]
    assert loaded.config["ref_mask"] == str(tmp_path / "mask.fits")
    assert loaded.config["new_psf"] == str(tmp_path / "psf.fits")


def test_batch_cli_reports_failed_pairs_and_does_not_claim_success(tmp_path, capsys) -> None:
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        '[{"ref":"missing","new":"also-missing"},{"ref":"r","new":"n"}]', encoding="utf-8"
    )
    output = tmp_path / "batch"
    assert astro.main(["--batch", str(manifest), "--output", str(output), "--fail-fast"]) == 1
    assert "1 failed, 1 skipped" in capsys.readouterr().out
    assert json.loads((output / "batch_summary.json").read_text())["state"] == "stopped"


def test_batch_summary_io_errors_propagate(tmp_path, monkeypatch) -> None:
    from proper_transient_finder import BatchJob, api, compare_batch

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(api, "atomic_json", fail)
    with pytest.raises(OSError, match="full disk"):
        compare_batch([BatchJob("first", "ref", "new")], tmp_path / "batch")


def test_github_docs_generate_without_dependencies_and_detect_staleness(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/generate_docs.py"
    output = tmp_path / "API.md"
    command = [sys.executable, "-I", "-S", str(script), "--output", str(output)]
    generated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Generated GitHub Markdown" in generated.stdout
    content = output.read_text(encoding="utf-8")
    assert "**Author:** Pier-Francesco Rocci" in content
    assert "Developed with assistance from ChatGPT, Claude, and Mistral." in content
    assert "## compare_images" in content
    assert "## compare_batch" in content
    assert "## load_batch_manifest" in content
    assert "## BatchJob" in content
    assert "## BatchItemResult" in content
    assert "## BatchResult" in content
    assert "| `color_quantile` | `float` | `0.95` |" in content
    assert "https://github.com/quatrope/ProperImage" in content
    subprocess.run([*command, "--check"], cwd=tmp_path, capture_output=True, check=True)
    output.write_text("stale", encoding="utf-8")
    stale = subprocess.run([*command, "--check"], cwd=tmp_path, capture_output=True, check=False)
    assert stale.returncode == 1
    assert output.read_text(encoding="utf-8") == "stale"


def test_distribution_metadata_matches_public_api() -> None:
    import subprocess
    import sys
    from importlib.metadata import distribution
    from pathlib import Path

    import proper_transient_finder as package

    installed = distribution("proper-transient-finder")
    assert installed.version == package.__version__
    assert "ProperTransientFinder" in package.__all__
    finder = package.ProperTransientFinder
    assert finder.__module__ == "proper_transient_finder.astro"
    assert {"PipelineConfig", "PipelineResult", "compare_images"} <= set(package.__all__)
    assert {
        "BatchJob",
        "BatchItemResult",
        "BatchResult",
        "compare_batch",
        "load_batch_manifest",
    } <= set(package.__all__)
    assert (Path(package.__file__).parent / "py.typed").is_file()
    entry_points = {entry.name: entry.value for entry in installed.entry_points}
    assert entry_points["proper-transient-finder"] == "proper_transient_finder.astro:main"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import proper_transient_finder as package; assert package.ProperTransientFinder",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )


def test_public_config_object_is_accepted_validated_and_copied(tmp_path) -> None:
    from proper_transient_finder import PipelineConfig

    config = PipelineConfig(plots=False, cosmic_rays=False)
    pipeline = ProperTransientFinder(
        tmp_path / "ref.fits", tmp_path / "new.fits", tmp_path / "out", config
    )
    assert pipeline.options == config
    assert pipeline.options is not config
    config.psf_radius = 12
    assert pipeline.options.psf_radius == 8
    config.psf_radius = 0
    with pytest.raises(ValueError, match="psf_radius"):
        ProperTransientFinder(
            tmp_path / "ref.fits", tmp_path / "new.fits", tmp_path / "out", config
        )


@pytest.mark.parametrize("persist", [False, True])
def test_public_compare_images_returns_structured_results(
    tmp_path, persist, capsys, monkeypatch
) -> None:
    from proper_transient_finder import PipelineConfig, PipelineResult, __version__, compare_images

    paths = synthetic_pair(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    config = PipelineConfig(plots=False, cosmic_rays=False)
    output = tmp_path / "products" if persist else None
    result = compare_images(*paths, output_dir=output, config=config)
    assert isinstance(result, PipelineResult)
    assert result.difference.shape == result.score.shape == result.mask.shape == (256, 256)
    assert result.difference.dtype == result.score.dtype == np.dtype("float64")
    assert result.mask.dtype == np.dtype("bool")
    assert result.psf.sum() == pytest.approx(1.0)
    assert len(result.candidates) >= 2
    assert result.metadata["runtime"]["pipeline_version"] == __version__
    assert len(result.metadata["inputs"]["reference"]["sha256"]) == 64
    assert result.metadata["quality_summary"]["metrics"]["accepted_candidates"] == len(
        result.candidates
    )
    assert list(result.rejected_candidates.columns) == list(result.candidates.columns)
    assert len(result.background_diagnostics) == len(result.candidates) + len(
        result.rejected_candidates
    )
    assert result.products["difference"] is result.difference
    assert not config.plots
    assert capsys.readouterr().out == ""
    if persist:
        assert result.output_dir == output.resolve()
        assert json.loads((output / "run_status.json").read_text())["state"] == "completed"
    else:
        assert result.output_dir is None
        assert set(tmp_path.iterdir()) == before


def test_public_compare_images_propagates_errors_without_output(tmp_path) -> None:
    from proper_transient_finder import compare_images

    reference, science = tmp_path / "missing-ref", tmp_path / "missing-new"
    with pytest.raises(ValueError, match="threshold"):
        compare_images(reference, science, threshold_sigma=0)
    with pytest.raises(FileNotFoundError):
        compare_images(reference, science)
    assert not list(tmp_path.iterdir())


def test_public_compare_images_cross_filter_remains_explicit(tmp_path) -> None:
    from proper_transient_finder import PipelineConfig, compare_images

    paths = synthetic_pair(tmp_path)
    with fits.open(paths[1], mode="update", memmap=False) as hdus:
        hdus[0].header["FILTER"] = "different-filter"
    with pytest.raises(ValueError, match="FILTER values differ"):
        compare_images(*paths, config=PipelineConfig(cosmic_rays=False))
    result = compare_images(*paths, config=PipelineConfig(cross_filter=True, cosmic_rays=False))
    assert "vetting_group" in result.candidates
    assert result.metadata["comparison_mode"] == "experimental cross-filter residual"
    assert result.output_dir is None
    assert set(tmp_path.iterdir()) == set(paths)


def test_score_calibration_keeps_transients_and_masks_bad_pixels() -> None:
    rng = np.random.default_rng(42)
    score = rng.normal(3.0, 2.0, (128, 128))
    score[64, 64] = 100.0
    mask = np.zeros(score.shape, dtype=bool)
    mask[10, 10] = True
    calibrated, center, scale = calibrate_score(score, mask)
    assert center == pytest.approx(3.0, abs=0.1)
    assert scale == pytest.approx(2.0, rel=0.05)
    assert calibrated[64, 64] > 40
    assert np.isnan(calibrated[10, 10])


def test_fft_products_reject_nontrivial_imaginary_signal() -> None:
    image = np.ones((16, 16))
    np.testing.assert_allclose(real_image(image + 1e-12j, "D"), image)
    with pytest.raises(ValueError, match="imaginary"):
        real_image(image + 1j, "D")


def test_cross_convolution_cancels_stars_and_recovers_signed_flux() -> None:
    shape = (192, 192)
    ref_psf = normalize_psf(Gaussian2DKernel(1.3).array, radius=8)
    new_psf = normalize_psf(Gaussian2DKernel(2.0).array, radius=8)
    ref_stars = np.zeros(shape)
    ref_stars[70, 65] = 10000
    ref_stars[120, 110] = 10000
    new_stars = ref_stars * 1.2
    new_stars[100, 65] += 3000
    new_stars[120, 110] -= 2000
    from proper_transient_finder.astro import convolve_image

    reference = ImageFrame(
        convolve_image(ref_stars, ref_psf),
        fits.Header(),
        np.zeros(shape, bool),
        np.ones(shape) * 4,
        np.zeros(shape),
        1.0,
    )
    new = ImageFrame(
        convolve_image(new_stars, new_psf),
        fits.Header(),
        np.zeros(shape, bool),
        np.ones(shape) * 4,
        np.zeros(shape),
        1.0,
    )
    products = kernel_subtract(reference, new, ref_psf, new_psf, 1.2)
    assert abs(products["difference"][70, 65]) < 1e-8
    assert products["flux"][100, 65] == pytest.approx(3000, rel=0.001)
    assert products["flux"][120, 110] == pytest.approx(-2000, rel=0.001)
    assert products["score"][100, 65] > 20
    assert products["score"][120, 110] < -20
    assert products["mask"][0, :].all()


def test_fits_masks_nan_saturation_and_reads_variance(tmp_path) -> None:
    rng = np.random.default_rng(16)
    image = rng.normal(100, 2, (128, 128))
    image[30, 30] = np.nan
    image[60, 60] = 60000
    quality = np.zeros(image.shape, np.uint8)
    quality[90, 90] = 4
    path = tmp_path / "input.fits"
    header = fits.Header({"SATURATE": 50000, "GAIN": 2.0})
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, header, name="SCI"),
            fits.ImageHDU(quality, name="DQ"),
            fits.ImageHDU(np.full(image.shape, 4.0), name="VAR"),
        ]
    ).writeto(path)
    frame = load_frame(path, cosmic_rays=False)
    assert frame.mask[30, 30]
    assert frame.mask[60, 60]
    assert frame.mask[90, 90]
    assert np.isfinite(frame.data).all()
    assert frame.variance[40, 40] == 4.0
    assert frame.gain == 2.0


def synthetic_pair(tmp_path, *, displaced=False):
    from proper_transient_finder.astro import convolve_image

    rng = np.random.default_rng(2026)
    shape = (256, 256)
    stars = np.zeros(shape)
    for row in (40, 75, 110, 145, 180, 215):
        for column in (40, 75, 110, 145, 180, 215):
            if (row, column) == (180, 180):
                star_row, star_column = row, column
            else:
                star_row = row + rng.integers(-5, 6)
                star_column = column + rng.integers(-5, 6)
            stars[star_row, star_column] = rng.uniform(10000, 20000)
    new_stars = stars * 1.2
    new_stars[128, 128] = 6000
    new_stars[180, 180] -= 6000
    reference = convolve_image(stars, Gaussian2DKernel(1.3).array)
    new = convolve_image(new_stars, Gaussian2DKernel(1.8).array)
    if displaced:
        from scipy.ndimage import shift

        new = shift(new, (1.3, -0.8), order=3)
    header = fits.Header({"GAIN": 2.0, "SATURATE": 50000, "FILTER": "r", "BUNIT": "adu"})
    paths = []
    for name, image in (("reference", reference), ("new", new)):
        path = tmp_path / f"{name}.fits"
        image = image + 100 + rng.normal(0, 2, shape)
        fits.writeto(path, image, header)
        paths.append(path)
    return paths


def test_real_properimage_psfs_and_photometric_scaling(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    pipeline = ProperTransientFinder(*paths, tmp_path / "out", {"cosmic_rays": False})
    difference, psf, score, mask = pipeline.perform_image_subtraction()
    assert difference.shape == (256, 256)
    assert psf.sum() == pytest.approx(1.0)
    assert pipeline.metadata["flux_scale"] == pytest.approx(1.2, rel=0.04)
    assert score[128, 128] > 10
    assert score[180, 180] < -10
    assert mask.dtype == bool


def test_cross_filter_requires_opt_in_and_compatible_units(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[1], header=True, memmap=False)
    header["FILTER"] = "BP-like"
    fits.writeto(paths[1], data, header, overwrite=True)
    with pytest.raises(ValueError, match="FILTER values differ"):
        ProperTransientFinder(*paths, tmp_path / "default").perform_image_subtraction()
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "experimental", {"cross_filter": True, "cosmic_rays": False}
    )
    pipeline.perform_image_subtraction()
    assert pipeline.metadata["reference_filter"] == "r"
    assert pipeline.metadata["science_filter"] == "BP-like"
    assert pipeline.metadata["flux_scale"] == pytest.approx(1.2, rel=0.04)
    header["BUNIT"] = "electron"
    fits.writeto(paths[1], data, header, overwrite=True)
    with pytest.raises(ValueError, match="BUNIT values differ"):
        ProperTransientFinder(
            *paths, tmp_path / "bad_units", {"cross_filter": True}
        ).perform_image_subtraction()


@pytest.mark.parametrize("missing_index", [0, 1])
def test_cross_filter_requires_both_filter_labels(tmp_path, missing_index) -> None:
    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[missing_index], header=True, memmap=False)
    del header["FILTER"]
    fits.writeto(paths[missing_index], data, header, overwrite=True)
    with pytest.raises(ValueError, match="requires FILTER in both inputs"):
        astro.compare_cross_filter(*paths, tmp_path / "results", plots=False)


def test_cross_filter_cli_is_explicit_and_preserves_input_headers(tmp_path, capsys) -> None:
    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[1], header=True, memmap=False)
    header["FILTER"] = "RP-like"
    fits.writeto(paths[1], data, header, overwrite=True)
    arguments = [
        "--ref",
        str(paths[0]),
        "--new",
        str(paths[1]),
        "--output",
        str(tmp_path / "results"),
        "--no-cosmic-rays",
        "--no-plots",
    ]
    assert astro.main(arguments) == 1
    assert json.loads((tmp_path / "results/run_status.json").read_text())["state"] == "failed"
    assert astro.main(arguments) == 1
    assert (
        astro.main([*arguments, "--cross-filter", "--max-scale-scatter", "0.2", "--overwrite"]) == 0
    )
    assert json.loads((tmp_path / "results/run_status.json").read_text())["state"] == "completed"
    assert "experimental cross-filter residuals" in capsys.readouterr().out
    assert fits.getheader(paths[0])["FILTER"] == "r"
    assert fits.getheader(paths[1])["FILTER"] == "RP-like"


@pytest.mark.parametrize(
    "config",
    [
        {"gain": 0},
        {"inf_loss": 1},
        {"alignment_tolerance": float("nan")},
        {"psf_radius": 1},
        {"ref_psf": "only-one.fits"},
        {"cross_filter": True, "method": "proper"},
        {"cross_filter": True, "max_scale_scatter": float("nan")},
    ],
)
def test_invalid_config_is_rejected(config) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**config)


def test_pipeline_finds_brightening_and_fading_and_writes_products(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    pipeline = ProperTransientFinder(*paths, tmp_path / "out", {"cosmic_rays": False})
    sources = pipeline.run_full_pipeline()
    assert list(sources.columns) == astro.CANDIDATE_COLUMNS
    positive = sources[sources["sign"] == 1]
    negative = sources[sources["sign"] == -1]
    assert np.hypot(positive["x_centroid"] - 128, positive["y_centroid"] - 128).min() < 1
    assert np.hypot(negative["x_centroid"] - 180, negative["y_centroid"] - 180).min() < 1
    assert (tmp_path / "out" / "diagnostics.png").is_file()
    assert (tmp_path / "out" / "candidate_stamps.png").is_file()
    assert (tmp_path / "out" / "rejected_sources.csv").is_file()
    difference = fits.getdata(tmp_path / "out" / "difference_image.fits")
    assert np.isnan(difference[0, 0])
    assert fits.getheader(tmp_path / "out" / "difference_image.fits")["DIFFSIGN"] == "NEW-REF"
    summary = json.loads((tmp_path / "out/quality_summary.json").read_text())
    assert summary["metrics"]["accepted_candidates"] == len(sources)
    assert (
        summary["metrics"]["registration_rms_pixels"] < summary["limits"]["registration_rms_pixels"]
    )
    assert list(astro.pd.read_csv(tmp_path / "out/rejection_summary.csv").columns) == [
        "reason",
        "count",
    ]


def test_quality_summary_counts_each_rejection_reason(tmp_path) -> None:
    pipeline = ProperTransientFinder(tmp_path / "reference", tmp_path / "science", tmp_path / "out")
    pipeline.metadata.update(masked_fraction=0.6, registration_rms_pixels=0.1, matched_stars=12)
    pipeline.rejected_sources = astro.pd.DataFrame({"flags": ["dipole;elongated", "dipole"]})
    summary = pipeline.quality_summary()
    assert summary["rejection_reasons"] == {"dipole": 2, "elongated": 1}
    assert summary["status"] == "warnings"
    assert summary["metrics"]["matched_stars"] == 12


def test_preflight_is_read_only_and_cli_lists_unchecked_steps(tmp_path, capsys) -> None:
    paths = synthetic_pair(tmp_path)
    output = tmp_path / "preflight"
    pipeline = ProperTransientFinder(*paths, output)
    report = pipeline.preflight()
    assert report["status"] == "input_checks_passed"
    assert not output.exists()
    assert pipeline.diff_image is None
    assert report["not_checked"]
    assert (
        astro.main(
            [
                "--ref",
                str(paths[0]),
                "--new",
                str(paths[1]),
                "--output",
                str(output),
                "--check-only",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "input_checks_passed" in capsys.readouterr().out


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_preflight_rejects_invalid_threshold_without_output(tmp_path, threshold) -> None:
    pipeline = ProperTransientFinder(
        tmp_path / "absent-ref", tmp_path / "absent-new", tmp_path / "out"
    )
    with pytest.raises(ValueError, match="threshold"):
        pipeline.preflight(threshold)
    assert not pipeline.output_dir.exists()


def test_run_records_complete_status_fingerprints_and_timings(tmp_path) -> None:
    import hashlib

    paths = synthetic_pair(tmp_path)
    for index, path in enumerate(paths):
        with fits.open(path, mode="update", memmap=False) as hdus:
            hdus[0].header["DATE-OBS"] = f"2026-10-{20 + index}T00:00:00"
            hdus[0].header["MJD-OBS"] = 61333.0 + index
    output = tmp_path / "run"
    pipeline = ProperTransientFinder(*paths, output, {"plots": False, "cosmic_rays": False})
    sources = pipeline.run_full_pipeline()
    status = json.loads((output / "run_status.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert status["state"] == "completed"
    assert status["timings_seconds"]["detection"] > 0
    assert provenance["stage_timings_seconds"]["catalog_output"] > 0
    assert (
        provenance["inputs"]["reference"]["sha256"]
        == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    )
    assert provenance["runtime"]["pipeline_version"]
    assert provenance["input_observation_dates"]["reference"]["DATE-OBS"] == "2026-10-20T00:00:00"
    assert status["started_utc"]
    assert status["elapsed_seconds"] > 0
    assert "Run completed" in (output / "pipeline.log").read_text()
    assert len(sources) > 0


def test_failed_run_records_failure_and_keeps_inputs_unchanged(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[1], header=True, memmap=False)
    header["FILTER"] = "different"
    fits.writeto(paths[1], data, header, overwrite=True)
    before = [path.read_bytes() for path in paths]
    handlers_before, level_before = list(astro.LOGGER.handlers), astro.LOGGER.level
    output = tmp_path / "failed"
    pipeline = ProperTransientFinder(*paths, output, {"plots": False, "cosmic_rays": False})
    with pytest.raises(ValueError, match="FILTER values differ"):
        pipeline.run_full_pipeline()
    status = json.loads((output / "run_status.json").read_text())
    assert status["state"] == "failed"
    assert status["stage"] == "input_loading"
    assert "FILTER values differ" in status["error"]
    assert "Run failed" in (output / "pipeline.log").read_text()
    assert [path.read_bytes() for path in paths] == before
    assert astro.LOGGER.handlers == handlers_before
    assert astro.LOGGER.level == level_before
    with (output / "pipeline.log").open("a", encoding="utf-8") as stream:
        stream.write("Handler was closed after failure.\n")


@pytest.mark.parametrize(
    "config",
    [
        {"color_association_fwhm": 0},
        {"color_reference_snr": -1},
        {"color_quantile": float("nan")},
        {"color_quantile": 1.1},
        {"color_error_margin": float("inf")},
    ],
)
def test_invalid_color_vetting_thresholds(config) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**config)


def test_color_vetting_settings_preserve_detection_measurements(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    catalogs = []
    for settings in (
        {},
        {
            "color_association_fwhm": 0.01,
            "color_reference_snr": 1e6,
            "color_quantile": 0.9,
            "color_error_margin": 4,
        },
    ):
        pipeline = ProperTransientFinder(
            *paths, tmp_path / "unused", {"cross_filter": True, "cosmic_rays": False, **settings}
        )
        pipeline.perform_image_subtraction()
        catalogs.append(pipeline.detect_sources_in_difference())
    astro.pd.testing.assert_frame_equal(
        catalogs[0][astro.CANDIDATE_COLUMNS], catalogs[1][astro.CANDIDATE_COLUMNS]
    )
    assert "1e+06" in pipeline.metadata["color_vetting_rule"]


def test_background_diagnostics_warn_without_changing_flags(tmp_path) -> None:
    pipeline = ProperTransientFinder(tmp_path / "ref", tmp_path / "new", tmp_path / "out")
    pipeline.psf_image = normalize_psf(Gaussian2DKernel(1.3).array, 8)
    grid_y, grid_x = np.indices((128, 128))
    pipeline.diff_image = 10 + 3 * (grid_x - 64) + np.random.default_rng(42).normal(size=(128, 128))
    pipeline.mask = np.zeros((128, 128), bool)
    pipeline.products = {"variance": np.ones((128, 128))}
    catalog = astro.pd.DataFrame(
        {"id": [1], "x_centroid": [64.0], "y_centroid": [64.0], "flags": [""]}
    )
    before = catalog.copy()
    report = pipeline.candidate_background_diagnostics(catalog)
    assert "background_gradient" in report.loc[0, "local_background_warning"]
    astro.pd.testing.assert_frame_equal(catalog, before)
    pipeline.mask[:] = True
    assert (
        pipeline.candidate_background_diagnostics(catalog).loc[0, "local_background_warning"]
        == "insufficient_annulus"
    )


def test_diagnostic_plots_share_stamp_scale_and_identify_color_groups(
    tmp_path, monkeypatch
) -> None:
    from matplotlib.figure import Figure

    paths = synthetic_pair(tmp_path)
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "plots", {"cross_filter": True, "cosmic_rays": False}
    )
    pipeline.perform_image_subtraction()
    sources = pipeline.detect_sources_in_difference()
    figures = {}
    monkeypatch.setattr(
        Figure, "savefig", lambda figure, path, **kwargs: figures.update({path.name: figure})
    )
    pipeline.create_diagnostic_plots(sources)
    diagnostics = figures["diagnostics.png"]
    labels = diagnostics.axes[2].get_legend_handles_labels()[1]
    assert labels == ["Less color-confounded", "Color-ambiguous", "Rejected"]
    stamps = figures["candidate_stamps.png"]
    assert stamps.axes[0].images[0].norm is stamps.axes[1].images[0].norm
    assert stamps.axes[2].images[0].norm.vcenter == 0
    assert "Matched reference" in stamps.axes[0].get_title()


def colored_pair(directory, *, transients):
    """Render different object colors, not just different FITS filter labels."""
    from scipy.signal import fftconvolve

    directory.mkdir()
    rng = np.random.default_rng(4110)
    shape = (256, 256)
    reference, science = np.zeros(shape), np.zeros(shape)
    for row_index, row in enumerate((40, 75, 110, 145, 180, 215)):
        for column_index, column in enumerate((40, 75, 110, 145, 180, 215)):
            flux = rng.uniform(10000, 20000)
            reference[row, column] = flux
            science[row, column] = flux * (0.8, 1.2, 1.6)[(row_index + column_index) % 3]
    galaxies = [
        {
            "x": 163,
            "y": 160,
            "flux": 50000.0,
            "effective_radius_pixels": 6.0,
            "sersic_index": 1.0,
            "axis_ratio": 0.55,
            "position_angle_degrees": 30.0,
        }
    ]
    host = elliptical_galaxy_scene({"galaxies": galaxies}, shape)
    host[160, 163] += 15000
    reference += host
    science += host * 1.5
    if transients:
        science[128, 128] += 9000
        science[160, 163] += 7000
    paths = []
    for name, scene, sigma, filter_name in (
        ("reference", reference, 1.3, "PS1-r"),
        ("science", science, 1.8, "BP-like"),
    ):
        expected = (
            np.maximum(fftconvolve(scene, Gaussian2DKernel(sigma).array, mode="same"), 0) + 100
        )
        variance = expected / 2 + 4
        image = rng.poisson(expected * 2) / 2 + rng.normal(0, 2, shape)
        header = fits.Header({"FILTER": filter_name, "BUNIT": "adu", "GAIN": 2.0})
        path = directory / f"{name}.fits"
        fits.HDUList([fits.PrimaryHDU(image, header), fits.ImageHDU(variance, name="VAR")]).writeto(
            path
        )
        paths.append(path)
    return paths


@pytest.mark.parametrize("transients", [False, True])
def test_cross_filter_colored_scene_and_output_warnings(tmp_path, transients, caplog) -> None:
    from proper_transient_finder import compare_cross_filter

    paths = colored_pair(tmp_path / "inputs", transients=transients)
    with pytest.raises(ValueError, match="Inconsistent star flux ratios"):
        compare_cross_filter(*paths, tmp_path / "strict", cosmic_rays=False, psf_radius=10)
    sources = compare_cross_filter(
        *paths,
        tmp_path / "results",
        max_scale_scatter=0.5,
        cosmic_rays=False,
        psf_radius=10,
    )
    assert not sources.empty
    assert sources["comparison_warning"].str.contains("color differences", case=False).all()
    assert sources["flags"].eq("").all()
    assert "color" in caplog.text.lower()
    metadata = json.loads((tmp_path / "results/provenance.json").read_text())
    assert metadata["reference_filter"] == "PS1-r"
    assert metadata["science_filter"] == "BP-like"
    assert metadata["flux_scale"] == pytest.approx(1.2, rel=0.12)
    assert metadata["flux_scale_scatter"] > 0.15
    assert "color" in metadata["flux_error_scope"]
    header = fits.getheader(tmp_path / "results/difference_image.fits")
    assert header["XFLTR"]
    assert header["REFFILT"] == "PS1-r"
    assert header["NEWFILT"] == "BP-like"
    assert "comparison_warning" in (tmp_path / "results/rejected_sources.csv").read_text()
    groups = [
        astro.pd.read_csv(tmp_path / "results" / f"{name}_candidates.csv")
        for name in ("less_color_confounded", "color_ambiguous")
    ]
    assert len(groups[0]) == int(transients)
    assert len(groups[0]) + len(groups[1]) == len(sources)
    assert set(groups[0]["id"]).isdisjoint(groups[1]["id"])
    assert (tmp_path / "results/diagnostics.png").is_file()
    assert (tmp_path / "results/candidate_stamps.png").is_file()
    measurements = []
    for column, row, flux in (
        ((128, 128, 9000), (163, 160, 7000)) if transients else ((163, 160, 0),)
    ):
        distances = np.hypot(sources["x_centroid"] - column, sources["y_centroid"] - row)
        nearby = sources.loc[(distances < 1) & (sources["sign"] == 1)]
        assert len(nearby) == 1
        candidate = nearby.iloc[0]
        expected_group = "less_color_confounded" if column == 128 else "color_ambiguous"
        assert candidate["vetting_group"] == expected_group
        assert bool(candidate["reference_source_associated"]) == (column != 128)
        assert metadata["less_color_confounded_candidates"] == int(transients)
        assert metadata["color_ambiguous_candidates"] == len(sources) - int(transients)
        measurements.append(
            {
                "x": column,
                "y": row,
                "injected_flux": flux,
                "measured_flux": float(candidate["flux"]),
                "score": float(candidate["snr"]),
                "relative_flux_error": float(candidate["flux"] / flux - 1) if flux else None,
                "position_error_pixels": float(distances.loc[nearby.index[0]]),
                "vetting_group": candidate["vetting_group"],
                "vetting_reason": candidate["vetting_reason"],
                "fractional_excess": float(candidate["fractional_excess"])
                if np.isfinite(candidate["fractional_excess"])
                else None,
            }
        )
    report = {
        "transients_injected": transients,
        "measurements": measurements,
        "accepted_residuals": len(sources),
        "unmatched_accepted_residuals": len(sources) - (len(measurements) if transients else 0),
        "less_color_confounded_residuals": len(groups[0]),
        "color_ambiguous_residuals": len(groups[1]),
        "warning": astro.CROSS_FILTER_WARNING,
        "pipeline_metadata": metadata,
    }
    (tmp_path / "results/cross_filter_validation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


@pytest.mark.parametrize("config", [{"flux_scale": 1.2}, {"beta": False}])
def test_cross_filter_without_stellar_calibration_stays_ambiguous(tmp_path, config) -> None:
    paths = synthetic_pair(tmp_path)
    sources = astro.compare_cross_filter(
        *paths, tmp_path / "results", cosmic_rays=False, plots=False, **config
    )
    assert not sources.empty
    assert sources["vetting_group"].eq("color_ambiguous").all()
    assert sources["vetting_reason"].eq("stellar_color_range_unavailable").all()
    assert sources["reference_source_associated"].isna().all()
    assert sources["within_color_range"].isna().all()
    metadata = json.loads((tmp_path / "results/provenance.json").read_text())
    assert not metadata["color_vetting_available"]
    assert metadata["less_color_confounded_candidates"] == 0


@pytest.mark.parametrize("fraction", [-0.3, 0.3])
def test_cross_filter_aperture_fraction_uses_scaled_matched_reference(tmp_path, fraction) -> None:
    pipeline = ProperTransientFinder(
        tmp_path / "reference.fits",
        tmp_path / "science.fits",
        tmp_path / "results",
        {"cross_filter": True},
    )
    scene = np.zeros((192, 192))
    scene[96, 96] = 10000
    pipeline.ref_psf = normalize_psf(Gaussian2DKernel(1.3).array, 8)
    pipeline.new_psf = normalize_psf(Gaussian2DKernel(1.8).array, 8)
    scale = 1.5
    frames = []
    for data, psf in (
        (scene, pipeline.ref_psf),
        (scene * scale * (1 + fraction), pipeline.new_psf),
    ):
        frames.append(
            ImageFrame(
                astro.convolve_image(data, psf),
                fits.Header(),
                np.zeros(scene.shape, bool),
                np.ones(scene.shape),
                np.zeros(scene.shape),
                1.0,
            )
        )
    pipeline.reference, pipeline.new = frames
    pipeline.products = kernel_subtract(*frames, pipeline.ref_psf, pipeline.new_psf, scale)
    pipeline.diff_image = pipeline.products["difference"]
    pipeline.psf_image = pipeline.products["psf"]
    pipeline.mask = pipeline.products["mask"]
    pipeline.metadata.update(
        flux_scale=scale, color_fraction_limit=0.4, color_vetting_aperture_pixels=8.0
    )
    catalog = astro.pd.DataFrame({"x_centroid": [96.0, 50.0], "y_centroid": [96.0, 50.0]})
    result = pipeline._vet_cross_filter_candidates(catalog)
    assert result.loc[0, "fractional_excess"] == pytest.approx(fraction, abs=1e-7)
    assert result.loc[0, "within_color_range"]
    assert result.loc[0, "reference_source_associated"]
    assert result.loc[0, "vetting_group"] == "color_ambiguous"
    assert result.loc[1, "vetting_group"] == "less_color_confounded"
    assert np.isnan(result.loc[1, "fractional_excess"])
    offset_catalog = astro.pd.DataFrame({"x_centroid": [97.0], "y_centroid": [96.0]})
    assert (
        pipeline._vet_cross_filter_candidates(offset_catalog).loc[0, "vetting_group"]
        == "color_ambiguous"
    )
    original_options = pipeline.options
    pipeline.options = PipelineConfig(
        cross_filter=True, color_association_fwhm=0.01, color_reference_snr=1e6
    )
    assert (
        pipeline._vet_cross_filter_candidates(offset_catalog).loc[0, "vetting_group"]
        == "less_color_confounded"
    )
    pipeline.options = original_options
    pipeline.mask[96, 96] = True
    masked = pipeline._vet_cross_filter_candidates(catalog)
    assert masked.loc[0, "vetting_group"] == "color_ambiguous"
    assert masked.loc[0, "vetting_reason"] == "unreliable_reference_photometry"


def test_registration_rejects_shifted_images_and_align_repairs_them(tmp_path) -> None:
    paths = synthetic_pair(tmp_path, displaced=True)
    pipeline = ProperTransientFinder(*paths, tmp_path / "out", {"cosmic_rays": False})
    with pytest.raises(ValueError, match="Registration RMS|matched stars"):
        pipeline.perform_image_subtraction()
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "aligned", {"cosmic_rays": False, "align": True}
    )
    pipeline.perform_image_subtraction()
    assert pipeline.metadata["registration_rms_pixels"] < 0.35
    transform = np.array(pipeline.metadata["registration_matrix"])
    np.testing.assert_allclose(transform[:2, :2], np.eye(2), atol=0.005)
    np.testing.assert_allclose(transform[:2, 2], [0.8, -1.3], atol=0.05)
    assert pipeline.scorr_image[128, 128] > 10


def test_cosmic_saturation_and_nan_do_not_become_candidates(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[1], header=True, memmap=False)
    data[70, 185] = 30000
    data[180, 75] = 60000
    data[60, 120] = np.nan
    fits.writeto(paths[1], data, header, overwrite=True)
    pipeline = ProperTransientFinder(*paths, tmp_path / "out", {"plots": False})
    pipeline.perform_image_subtraction()
    sources = pipeline.detect_sources_in_difference()
    assert pipeline.new.mask[70, 185]
    assert pipeline.mask[180, 75]
    assert pipeline.mask[60, 120]
    for column, row in ((185, 70), (75, 180), (120, 60)):
        distances = np.hypot(sources["x_centroid"] - column, sources["y_centroid"] - row)
        assert not (distances < 8).any()
    assert (np.hypot(sources["x_centroid"] - 128, sources["y_centroid"] - 128) < 1).any()


def test_opposite_sign_dipole_is_rejected(tmp_path) -> None:
    from proper_transient_finder.astro import convolve_image

    paths = synthetic_pair(tmp_path)
    data, header = fits.getdata(paths[1], header=True, memmap=False)
    artifact = np.zeros(data.shape)
    artifact[165, 126] = 6000
    artifact[165, 120] = -6000
    data += convolve_image(artifact, Gaussian2DKernel(1.8).array)
    fits.writeto(paths[1], data, header, overwrite=True)
    pipeline = ProperTransientFinder(*paths, tmp_path / "out", {"cosmic_rays": False})
    pipeline.perform_image_subtraction()
    accepted = pipeline.detect_sources_in_difference()
    rejected = pipeline.rejected_sources
    near_rejected = rejected[
        np.hypot(rejected["x_centroid"] - 123, rejected["y_centroid"] - 165) < 10
    ]
    assert len(near_rejected) >= 1
    assert near_rejected["flags"].str.contains("dipole").all()
    assert not (np.hypot(accepted["x_centroid"] - 123, accepted["y_centroid"] - 165) < 10).any()


def test_empty_catalog_has_headers_and_existing_outputs_are_protected(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "out", {"cosmic_rays": False, "plots": False}
    )
    sources = pipeline.run_full_pipeline(threshold_sigma=1e8)
    assert sources.empty
    catalog = (tmp_path / "out" / "detected_sources.csv").read_text()
    assert "x_centroid" in catalog
    assert "flags" in catalog
    with pytest.raises(FileExistsError):
        pipeline.run_full_pipeline()


def test_matched_filter_variance_accounts_for_kernel_covariance() -> None:
    rng = np.random.default_rng(42)
    shape = (384, 384)
    frames = [
        ImageFrame(
            rng.normal(0, 1, shape),
            fits.Header(),
            np.zeros(shape, bool),
            np.ones(shape),
            np.zeros(shape),
            1.0,
        )
        for _ in range(2)
    ]
    reference_psf = normalize_psf(Gaussian2DKernel(1.3).array, 8)
    new_psf = normalize_psf(Gaussian2DKernel(1.8).array, 8)
    products = kernel_subtract(*frames, reference_psf, new_psf, 1.0)
    assert np.std(products["score"][~products["mask"]]) == pytest.approx(1.0, rel=0.15)


def test_optional_properimage_subtraction_backend(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "out", {"method": "proper", "cosmic_rays": False, "plots": False}
    )
    sources = pipeline.run_full_pipeline()
    assert pipeline.scorr_image[128, 128] > 5
    assert pipeline.scorr_image[180, 180] < -5
    assert np.isfinite(pipeline.scorr_image[~pipeline.mask]).all()
    assert (np.hypot(sources["x_centroid"] - 128, sources["y_centroid"] - 128) < 1).any()
    assert (np.hypot(sources["x_centroid"] - 180, sources["y_centroid"] - 180) < 1).any()
    assert sources["flux_error"].isna().all()


def test_empirical_psf_mode_is_explicitly_disabled() -> None:
    with pytest.raises(ValueError, match="Empirical PSF mode is disabled"):
        PipelineConfig(fitted_psf=False)


def test_catalog_wcs_and_saved_reference_grid(tmp_path) -> None:
    from astropy.wcs import WCS

    paths = synthetic_pair(tmp_path)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.cdelt = [-0.00027, 0.00027]
    wcs.wcs.crval = [120, -30]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    for path in paths:
        data, header = fits.getdata(path, header=True, memmap=False)
        header.update(wcs.to_header())
        fits.writeto(path, data, header, overwrite=True)
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "out", {"cosmic_rays": False, "plots": False}
    )
    sources = pipeline.run_full_pipeline()
    candidate = sources.loc[(sources["x_centroid"] - 128).abs().idxmin()]
    expected = wcs.pixel_to_world(candidate["x_centroid"], candidate["y_centroid"])
    assert candidate["ra_deg"] == pytest.approx(expected.ra.deg)
    assert candidate["dec_deg"] == pytest.approx(expected.dec.deg)
    saved_wcs = WCS(fits.getheader(tmp_path / "out" / "difference_image.fits"))
    assert saved_wcs.has_celestial


def test_supplied_psfs_are_used_and_saved_as_matching_kernels(tmp_path) -> None:
    paths = synthetic_pair(tmp_path)
    psf_paths = [tmp_path / "ref_psf.fits", tmp_path / "new_psf.fits"]
    for path, sigma in zip(psf_paths, (1.3, 1.8), strict=True):
        fits.writeto(path, Gaussian2DKernel(sigma).array)
    pipeline = ProperTransientFinder(
        *paths,
        tmp_path / "out",
        {
            "ref_psf": str(psf_paths[0]),
            "new_psf": str(psf_paths[1]),
            "cosmic_rays": False,
            "plots": False,
        },
    )
    sources = pipeline.run_full_pipeline()
    assert (np.hypot(sources["x_centroid"] - 128, sources["y_centroid"] - 128) < 1).any()
    saved_kernel = fits.getdata(tmp_path / "out" / "reference_matching_kernel.fits")
    np.testing.assert_allclose(saved_kernel, pipeline.new_psf)


def test_count_rate_data_without_variance_is_rejected(tmp_path) -> None:
    path = tmp_path / "rate.fits"
    fits.writeto(path, np.ones((64, 64)), fits.Header({"BUNIT": "adu/s", "GAIN": 1}))
    with pytest.raises(ValueError, match="VAR/ERR"):
        load_frame(path, cosmic_rays=False)


def celestial_frame(*, shift=0.0, pixel_scale=1.0, mask_pixel=False):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [32 + shift, 32 + shift]
    wcs.wcs.crval = [150, -25]
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale]) / 3600
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    mask = np.zeros((64, 64), bool)
    if mask_pixel:
        mask[35, 35] = True
    return ImageFrame(
        np.full((64, 64), 10.0),
        wcs.to_header(),
        mask,
        np.full((64, 64), 4.0),
        np.full((64, 64), 100.0),
        2.0,
    )


def test_wcs_reprojection_propagates_squared_weights_masks_and_covariance() -> None:
    reference = celestial_frame(mask_pixel=True)
    projected, footprint = astro.reproject_frame(reference, celestial_frame(shift=0.5))
    assert projected.data[20, 20] == pytest.approx(10.0, rel=1e-6)
    assert projected.background[20, 20] == pytest.approx(100.0, rel=1e-6)
    assert projected.variance[20, 20] == pytest.approx(1.0, rel=1e-6)
    assert projected.noise_variance[20, 20] == pytest.approx(4.0, rel=1e-6)
    assert projected.mask[35:37, 35:37].all()
    assert (footprint[projected.mask] == 0).all()
    assert projected.mask[0].all()


@pytest.mark.parametrize(
    "unit_mode, expected", [("flux-per-pixel", 40), ("surface-brightness", 10)]
)
def test_wcs_reprojection_respects_pixel_area_and_surface_brightness(unit_mode, expected) -> None:
    projected, _ = astro.reproject_frame(
        celestial_frame(), celestial_frame(pixel_scale=2.0), unit_mode=unit_mode
    )
    assert projected.data[32, 32] == pytest.approx(expected, rel=1e-5)


def test_wcs_reprojection_rejects_missing_wcs_and_empty_overlap() -> None:
    reference = celestial_frame()
    reference.header = fits.Header()
    with pytest.raises(ValueError, match="celestial WCS"):
        astro.reproject_frame(reference, celestial_frame())
    reference = celestial_frame()
    reference.header["CRVAL1"] = 180.0
    with pytest.raises(ValueError, match="WCS overlap"):
        astro.reproject_frame(reference, celestial_frame())


@pytest.mark.parametrize(
    "config",
    [
        {"align": True},
        {"shift": True},
        {"method": "proper"},
        {"ref_psf": "ref.fits", "new_psf": "new.fits"},
    ],
)
def test_reprojection_config_rejects_unsafe_combinations(config) -> None:
    with pytest.raises(ValueError, match="reprojection|reproject"):
        PipelineConfig(reproject=True, **config)


@pytest.mark.parametrize(
    "label, mode",
    [
        ("adu", "flux-per-pixel"),
        ("adu/s", "flux-per-pixel"),
        ("Jy", "flux-per-pixel"),
        ("MJy/sr", "surface-brightness"),
    ],
)
def test_wcs_reprojection_unit_inference(label, mode) -> None:
    assert astro.reprojection_unit_mode(fits.Header({"BUNIT": label}), "auto") == mode


def test_wcs_reprojection_ambiguous_units_require_explicit_choice() -> None:
    with pytest.raises(ValueError, match="Ambiguous BUNIT"):
        astro.reprojection_unit_mode(fits.Header(), "auto")


@pytest.mark.parametrize("projection", ["SIN", "CAR"])
def test_wcs_reprojection_maps_other_celestial_projections(projection) -> None:
    reference = celestial_frame()
    reference.header["CTYPE1"], reference.header["CTYPE2"] = (
        f"RA---{projection}",
        f"DEC--{projection}",
    )
    projected, _ = astro.reproject_frame(
        reference, celestial_frame(), unit_mode="surface-brightness"
    )
    np.testing.assert_allclose(projected.data[20:40, 20:40], 10.0, atol=1e-8)


def test_wcs_reprojection_excludes_invalid_area_from_noise_operator(monkeypatch) -> None:
    original = astro.pixel_solid_angles

    def invalid_corner(wcs, shape):
        area = original(wcs, shape)
        area[:2, :2] = np.nan
        return area

    monkeypatch.setattr(astro, "pixel_solid_angles", invalid_corner)
    projected, _ = astro.reproject_frame(celestial_frame(), celestial_frame(shift=0.5))
    assert projected.mask[:2, :2].all()
    assert np.isfinite(projected.variance).all()
    assert np.isfinite(projected.noise_variance).all()


def test_wcs_reprojection_pipeline_preserves_science_grid(tmp_path) -> None:
    from astropy.wcs import WCS, Sip

    paths = synthetic_pair(tmp_path, displaced=True)
    for path, offset in zip(paths, (0.0, 1.0), strict=True):
        data, header = fits.getdata(path, header=True, memmap=False)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [128 - 0.8 * offset, 128 + 1.3 * offset]
        wcs.wcs.crval = [150, -25]
        wcs.wcs.cdelt = [-0.8 / 3600, 0.8 / 3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        if offset:
            wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
            wcs.sip = Sip(np.zeros((3, 3)), np.zeros((3, 3)), None, None, wcs.wcs.crpix)
        header.update(wcs.to_header(relax=True))
        fits.writeto(path, data, header, overwrite=True)
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "out", {"reproject": True, "cosmic_rays": False}
    )
    sources = pipeline.run_full_pipeline()
    assert (np.hypot(sources["x_centroid"] - 127.2, sources["y_centroid"] - 129.3) < 1).any()
    assert pipeline.metadata["registration_rms_pixels"] < 0.2
    np.testing.assert_allclose(
        WCS(fits.getheader(tmp_path / "out" / "difference_image.fits")).wcs.crpix,
        [127.2, 129.3],
    )
    assert (tmp_path / "out" / "reprojection_footprint.fits").is_file()
    assert (tmp_path / "out" / "astrometric_residuals.csv").is_file()
    assert WCS(fits.getheader(tmp_path / "out" / "difference_image.fits")).sip is not None


def test_wcs_reprojection_noise_bound_covers_correlated_filtering() -> None:
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(411)
    noise = rng.normal(0, 2, (192, 64, 64))
    reference = celestial_frame()
    reference.data = noise[0]
    projected, _ = astro.reproject_frame(reference, celestial_frame(shift=0.5))
    simulated = (noise[:, :-1, :-1] + noise[:, 1:, :-1] + noise[:, :-1, 1:] + noise[:, 1:, 1:]) / 4
    np.testing.assert_allclose(projected.data[2:-2, 2:-2], simulated[0, 1:-2, 1:-2], atol=1e-6)
    kernel = Gaussian2DKernel(1.2).array
    filtered = fftconvolve(simulated, kernel[None], mode="same", axes=(-2, -1))
    measured = float(np.var(filtered[:, 12:-12, 12:-12]))
    upper = float(astro.convolve_image(projected.noise_variance, kernel**2)[20:40, 20:40].mean())
    diagonal_only = float(astro.convolve_image(projected.variance, kernel**2)[20:40, 20:40].mean())
    assert measured <= upper
    assert measured > 2.5 * diagonal_only


def elliptical_galaxy_scene(truth, shape, *, detector_wcs=None, scene_wcs=None):
    """Pixel-integrate the same sky surface-brightness models on either detector."""
    from astropy.modeling.models import Sersic2D
    from astropy.wcs.utils import pixel_to_pixel
    from scipy.special import gamma, gammaincinv

    models = []
    for galaxy in truth["galaxies"]:
        index = galaxy["sersic_index"]
        coefficient = gammaincinv(2 * index, 0.5)
        normalization = (
            2
            * np.pi
            * galaxy["axis_ratio"]
            * galaxy["effective_radius_pixels"] ** 2
            * index
            * np.exp(coefficient)
            * gamma(2 * index)
            / coefficient ** (2 * index)
        )
        models.append(
            Sersic2D(
                amplitude=galaxy["flux"] / normalization,
                r_eff=galaxy["effective_radius_pixels"],
                n=index,
                x_0=galaxy["x"],
                y_0=galaxy["y"],
                ellip=1 - galaxy["axis_ratio"],
                theta=np.deg2rad(galaxy["position_angle_degrees"]),
            )
        )
    rows, columns = np.indices(shape, dtype=float)
    image = np.zeros(shape)
    offsets = (np.arange(4) + 0.5) / 4 - 0.5
    for offset_y in offsets:
        for offset_x in offsets:
            sample_x, sample_y = columns + offset_x, rows + offset_y
            if detector_wcs is not None:
                sample_x, sample_y = pixel_to_pixel(detector_wcs, scene_wcs, sample_x, sample_y)
            for model in models:
                image += model(sample_x, sample_y) / offsets.size**2
    if detector_wcs is not None:
        scene_area = abs(np.linalg.det(scene_wcs.pixel_scale_matrix)) * (np.pi / 180) ** 2
        image *= astro.pixel_solid_angles(detector_wcs, shape) / scene_area
    return image


def realistic_survey_pair(directory, *, transients=None, noise_seed=20261029):
    """Simulate a deeper template and a shallower same-filter science exposure."""
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates, shift
    from scipy.signal import fftconvolve

    directory.mkdir()
    rng = np.random.default_rng(20261029)
    shape = (384, 384)
    row_grid, column_grid = np.indices(shape)
    default_transients = [
        {"name": "T1", "x": 163, "y": 159, "flux": 9000.0, "environment": "isolated"},
        {
            "name": "T2",
            "x": 258,
            "y": 213,
            "flux": 7000.0,
            "environment": "galaxy_core",
            "host": "G1",
        },
    ]
    truth = {
        "seed": noise_seed,
        "field_seed": 20261029,
        "shape": list(shape),
        "filter": "r",
        "pixel_scale_arcsec": 0.8,
        "flux_scale": 0.72,
        "shift_xy_pixels": [-0.7, 1.1],
        "psf_sigma_pixels": [1.25, 1.95],
        "template_extra_noise_adu": 6.0,
        "transients": default_transients if transients is None else transients,
        "galaxies": [
            {
                "name": "G1",
                "x": 258,
                "y": 213,
                "flux": 180000.0,
                "effective_radius_pixels": 8.0,
                "sersic_index": 1.5,
                "axis_ratio": 0.55,
                "position_angle_degrees": 30.0,
                "nucleus_flux": 18000.0,
            },
            {
                "name": "G2",
                "x": 205,
                "y": 240,
                "flux": 100000.0,
                "effective_radius_pixels": 10.0,
                "sersic_index": 1.0,
                "axis_ratio": 0.65,
                "position_angle_degrees": -25.0,
                "nucleus_flux": 0.0,
            },
        ],
        "artifacts": [],
        "stars": [],
        "exposures": [],
    }
    scene = np.zeros(shape)
    for row in range(45, 340, 47):
        for column in range(45, 340, 47):
            star_x, star_y = column + rng.uniform(-5, 5), row + rng.uniform(-5, 5)
            if any(
                np.hypot(star_x - source["x"], star_y - source["y"]) < 24
                for source in default_transients
            ):
                continue
            flux = float(np.exp(rng.uniform(np.log(8000), np.log(50000))))
            column_index, row_index = int(star_x), int(star_y)
            fraction_x, fraction_y = star_x - column_index, star_y - row_index
            scene[row_index : row_index + 2, column_index : column_index + 2] += flux * np.outer(
                [1 - fraction_y, fraction_y], [1 - fraction_x, fraction_x]
            )
            truth["stars"].append({"x": star_x, "y": star_y, "flux": flux})
    galaxy_scene = elliptical_galaxy_scene(truth, shape)
    for galaxy in truth["galaxies"]:
        galaxy_scene[galaxy["y"], galaxy["x"]] += galaxy["nucleus_flux"]
    new_scene = scene * truth["flux_scale"]
    for source in truth["transients"]:
        column, row = int(np.floor(source["x"])), int(np.floor(source["y"]))
        fraction_x, fraction_y = source["x"] - column, source["y"] - row
        new_scene[row : row + 2, column : column + 2] += source["flux"] * np.outer(
            [1 - fraction_y, fraction_y], [1 - fraction_x, fraction_x]
        )
    if noise_seed != 20261029:
        rng = np.random.default_rng(noise_seed)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [192, 192]
    wcs.wcs.crval = [150.0, -25.0]
    wcs.wcs.cdelt = np.array([-0.8, 0.8]) / 3600
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    artifact_positions = (
        (
            (95, 263),
            (305, 104),
            (155, 301),
            (211, 81),
            (99, 263),
            (151, 299),
            (307, 108),
            (91, 268),
            (159, 304),
            (211, 76),
        ),
        (
            (302, 267),
            (87, 179),
            (204, 307),
            (137, 236),
            (298, 271),
            (90, 175),
            (208, 305),
            (306, 264),
            (84, 183),
            (134, 239),
        ),
    )
    paths = []
    for index, (name, image_scene, sigma, gain, read_noise, sky_level) in enumerate(
        (
            ("reference", scene, 1.25, 2.0, 3.0, 80.0),
            ("science", new_scene, 1.95, 1.4, 6.0, 260.0),
        )
    ):
        expected = fftconvolve(image_scene, Gaussian2DKernel(sigma).array, mode="same")
        host_scale = truth["flux_scale"] if index else 1.0
        host_expected = fftconvolve(
            galaxy_scene * host_scale, Gaussian2DKernel(sigma).array, mode="same"
        )
        offset_x, offset_y = truth["shift_xy_pixels"] if index else (0.0, 0.0)
        if index:
            expected = shift(expected, (offset_y, offset_x), order=3, mode="constant")
            host_expected = shift(host_expected, (offset_y, offset_x), order=3, mode="constant")
        host_expected = np.maximum(host_expected, 0)
        if index:
            for source in truth["transients"]:
                source["host_surface_brightness_adu_per_pixel"] = float(
                    map_coordinates(
                        host_expected,
                        [[source["y"] + offset_y], [source["x"] + offset_x]],
                        order=1,
                    )[0]
                )
        sky = sky_level + (0.03 + index * 0.04) * column_grid + 0.02 * row_grid
        expected = np.maximum(expected, 0) + sky
        variance = (expected + host_expected) / gain + (read_noise / gain) ** 2
        extra_noise = truth["template_extra_noise_adu"] if name == "reference" else 0.0
        variance += extra_noise**2
        truth["exposures"].append(
            {
                "name": name,
                "gain_electrons_per_adu": gain,
                "read_noise_electrons": read_noise,
                "sky_origin_adu": sky_level,
                "sky_gradient_xy_adu_per_pixel": [0.03 + index * 0.04, 0.02],
                "extra_noise_rms_adu": extra_noise,
                "host_photon_seed": noise_seed + 2 + index,
                "noise_model": "Poisson(source + sky) + read noise + residual noise; total VAR",
            }
        )
        image = rng.poisson(expected * gain) / gain
        image += rng.normal(0, read_noise / gain, shape)
        if extra_noise:
            image += np.random.default_rng(noise_seed + 1).normal(0, extra_noise, shape)
        image += np.random.default_rng(noise_seed + 2 + index).poisson(host_expected * gain) / gain
        quality = np.zeros(shape, dtype=np.uint8)
        for kind, (column, row) in zip(
            (
                "cosmic_ray",
                "saturation",
                "bad_column",
                "nonfinite",
                "hot_pixel",
                "hot_pixel",
                "hot_pixel",
                "dead_pixel",
                "dead_pixel",
                "dead_pixel",
            ),
            artifact_positions[index],
            strict=True,
        ):
            if kind == "cosmic_ray":
                image[row, column] += 30000
            elif kind == "saturation":
                image[row - 1 : row + 2, column - 1 : column + 2] = 65000
            elif kind == "bad_column":
                image[row - 8 : row + 9, column] += 1500
                quality[row - 8 : row + 9, column] = 1
            elif kind == "hot_pixel":
                image[row, column] += 1500
                quality[row, column] = 2
            elif kind == "dead_pixel":
                image[row, column] = 0.0
                quality[row, column] = 4
            else:
                image[row, column] = np.nan
            truth["artifacts"].append(
                {
                    "image": name,
                    "kind": kind,
                    "native_x": column,
                    "native_y": row,
                    "x": column - offset_x,
                    "y": row - offset_y,
                    "dq_flag": int(quality[row, column]),
                    "pixel_value_adu": float(image[row, column])
                    if np.isfinite(image[row, column])
                    else None,
                }
            )
        header = wcs.to_header()
        header["CRPIX1"] += offset_x
        header["CRPIX2"] += offset_y
        header.update(GAIN=gain, RDNOISE=read_noise, SATURATE=50000, FILTER="r", BUNIT="adu")
        path = directory / f"{name}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(image.astype(np.float32), header, name="SCI"),
                fits.ImageHDU(variance.astype(np.float32), name="VAR"),
                fits.ImageHDU(quality, name="DQ"),
            ]
        ).writeto(path, checksum=True)
        paths.append(path)
    (directory / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return paths, truth


def wcs_survey_pair(directory, *, transients=None, noise_seed=20261029):
    """Render template stars in a separate rotated, finer, SIP-distorted detector."""
    from astropy.wcs import WCS, Sip

    paths, truth = realistic_survey_pair(directory, transients=transients, noise_seed=noise_seed)
    science_wcs = WCS(fits.getheader(paths[1], "SCI"))
    offset_x, offset_y = truth["shift_xy_pixels"]
    ideal_wcs = science_wcs.deepcopy()
    ideal_wcs.wcs.crpix -= [offset_x, offset_y]
    reference_wcs = WCS(naxis=2)
    reference_wcs.wcs.crpix = [269, 238]
    reference_wcs.wcs.crval = [150.004, -25.002]
    angle = np.deg2rad(12.0)
    reference_wcs.wcs.cd = (
        0.65
        / 3600
        * np.array(
            [
                [-np.cos(angle), np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    )
    reference_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    distortion_x, distortion_y = np.zeros((3, 3)), np.zeros((3, 3))
    distortion_x[2, 0], distortion_x[0, 2] = 1.5e-5, -0.5e-5
    distortion_y[0, 2], distortion_y[1, 1] = -1.2e-5, 0.4e-5
    reference_wcs.sip = Sip(distortion_x, distortion_y, None, None, reference_wcs.wcs.crpix)
    shape = (480, 500)
    row_grid, column_grid = np.indices(shape)
    signal = np.zeros(shape)
    host_signal = np.zeros(shape)
    sigma = 1.0 / 0.65
    nuclei = [
        {"x": galaxy["x"], "y": galaxy["y"], "flux": galaxy["nucleus_flux"]}
        for galaxy in truth["galaxies"]
        if galaxy["nucleus_flux"] > 0
    ]
    for objects, component in ((truth["stars"], signal), (nuclei, host_signal)):
        for star in objects:
            sky = ideal_wcs.pixel_to_world(star["x"], star["y"])
            column, row = reference_wcs.world_to_pixel(sky)
            component += (
                star["flux"]
                / (2 * np.pi * sigma**2)
                * np.exp(-((column_grid - column) ** 2 + (row_grid - row) ** 2) / (2 * sigma**2))
            )
    from scipy.signal import fftconvolve

    galaxy_image = elliptical_galaxy_scene(
        truth, shape, detector_wcs=reference_wcs, scene_wcs=ideal_wcs
    )
    host_signal += fftconvolve(galaxy_image, Gaussian2DKernel(sigma).array, mode="same")
    host_signal = np.maximum(host_signal, 0)
    reference_seed = 7711 if noise_seed == 20261029 else noise_seed + 7919
    rng = np.random.default_rng(reference_seed)
    expected = signal + 80 + 0.03 * column_grid + 0.02 * row_grid
    extra_noise = truth["template_extra_noise_adu"]
    variance = (expected + host_signal) / 2 + (3 / 2) ** 2 + extra_noise**2
    image = rng.poisson(expected * 2) / 2 + rng.normal(0, 1.5, shape)
    image += rng.normal(0, extra_noise, shape)
    image += np.random.default_rng(reference_seed + 1).poisson(host_signal * 2) / 2
    quality = np.zeros(shape, np.uint8)
    for artifact in truth["artifacts"]:
        if artifact["image"] == "science":
            artifact["x"], artifact["y"] = artifact["native_x"], artifact["native_y"]
            continue
        sky = ideal_wcs.pixel_to_world(artifact["x"], artifact["y"])
        column, row = (int(round(float(value))) for value in reference_wcs.world_to_pixel(sky))
        artifact["native_x"], artifact["native_y"] = column, row
        sky = reference_wcs.pixel_to_world(column, row)
        artifact["x"], artifact["y"] = [float(value) for value in science_wcs.world_to_pixel(sky)]
        if artifact["kind"] == "cosmic_ray":
            image[row, column] += 30000
        elif artifact["kind"] == "saturation":
            image[row - 1 : row + 2, column - 1 : column + 2] = 65000
        elif artifact["kind"] == "bad_column":
            image[row - 8 : row + 9, column] += 1500
            quality[row - 8 : row + 9, column] = 1
        elif artifact["kind"] == "hot_pixel":
            image[row, column] += 1500
            quality[row, column] = 2
        elif artifact["kind"] == "dead_pixel":
            image[row, column] = 0.0
            quality[row, column] = 4
        else:
            image[row, column] = np.nan
        artifact["dq_flag"] = int(quality[row, column])
        artifact["pixel_value_adu"] = (
            float(image[row, column]) if np.isfinite(image[row, column]) else None
        )
    header = reference_wcs.to_header(relax=True)
    header.update(GAIN=2.0, RDNOISE=3.0, SATURATE=50000, FILTER="r", BUNIT="adu")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image.astype(np.float32), header, name="SCI"),
            fits.ImageHDU(variance.astype(np.float32), name="VAR"),
            fits.ImageHDU(quality, name="DQ"),
        ]
    ).writeto(paths[0], overwrite=True, checksum=True)
    for source in truth["stars"] + truth["transients"] + truth["galaxies"]:
        source["x"], source["y"] = source["x"] + offset_x, source["y"] + offset_y
    truth["shift_xy_pixels"] = [0.0, 0.0]
    truth["wcs_reprojection"] = {
        "reference_shape": list(shape),
        "science_shape": [384, 384],
        "reference_pixel_scale_arcsec": 0.65,
        "science_pixel_scale_arcsec": 0.8,
        "reference_rotation_degrees": 12.0,
        "reference_noise_seed": reference_seed,
        "reference_host_photon_seed": reference_seed + 1,
        "reference_wcs": reference_wcs.to_header(relax=True).tostring(sep="\n"),
        "science_wcs": science_wcs.to_header(relax=True).tostring(sep="\n"),
    }
    (directory / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return paths, truth


def test_wcs_reprojection_distorted_survey_end_to_end(tmp_path) -> None:
    from astropy.wcs import WCS

    paths, truth = wcs_survey_pair(tmp_path / "inputs")
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "results", {"reproject": True, "psf_radius": 10}
    )
    sources = pipeline.run_full_pipeline()
    science_wcs = WCS(fits.getheader(paths[1], "SCI"))
    recovery = []
    for source in truth["transients"]:
        distances = np.hypot(
            sources["x_centroid"] - source["x"], sources["y_centroid"] - source["y"]
        )
        nearby = sources.loc[(distances < 1) & (sources["sign"] == 1)]
        result = {**source, "recovered": len(nearby) == 1}
        if result["recovered"]:
            candidate = nearby.iloc[0]
            expected_sky = science_wcs.pixel_to_world(source["x"], source["y"])
            result.update(
                measured_flux=float(candidate["flux"]),
                flux_error=float(candidate["flux_error"]),
                relative_flux_error=float(candidate["flux"] / source["flux"] - 1),
                score=float(candidate["snr"]),
                position_error_pixels=float(distances.loc[nearby.index[0]]),
                ra_error_arcsec=float(
                    (candidate["ra_deg"] - expected_sky.ra.deg)
                    * 3600
                    * np.cos(expected_sky.dec.rad)
                ),
                dec_error_arcsec=float((candidate["dec_deg"] - expected_sky.dec.deg) * 3600),
            )
        recovery.append(result)
    masked_artifacts = sum(
        bool(pipeline.mask[int(round(artifact["y"])), int(round(artifact["x"]))])
        for artifact in truth["artifacts"]
    )
    report = {
        "recovery": recovery,
        "galaxies": truth["galaxies"],
        "masked_artifacts": masked_artifacts,
        "artifact_count": len(truth["artifacts"]),
        "template_extra_noise_adu": truth["template_extra_noise_adu"],
        "pipeline_metadata": pipeline.metadata,
    }
    report["checks"] = {
        "both_transients_recovered": all(source["recovered"] for source in recovery),
        "galaxy_core_transient_recovered": any(
            source["environment"] == "galaxy_core" and source["recovered"] for source in recovery
        ),
        "no_false_positives": len(sources) == 2,
        "fluxes_within_twenty_percent": all(
            abs(source.get("relative_flux_error", np.inf)) < 0.2 for source in recovery
        ),
        "sky_positions_within_one_arcsec": all(
            np.hypot(source.get("ra_error_arcsec", np.inf), source.get("dec_error_arcsec", np.inf))
            < 1
            for source in recovery
        ),
        "artifacts_masked": masked_artifacts == len(truth["artifacts"]),
        "scale_within_five_percent": abs(pipeline.metadata["flux_scale"] / truth["flux_scale"] - 1)
        < 0.05,
        "registration_below_point_two_pixels": pipeline.metadata["registration_rms_pixels"] < 0.2,
        "partial_coverage_masked": bool((pipeline.reprojection_footprint == 0).any())
        and bool(pipeline.mask[pipeline.reprojection_footprint == 0].all()),
        "science_grid_shape": pipeline.diff_image.shape == (384, 384),
    }
    report["checks"] = {key: bool(value) for key, value in report["checks"].items()}
    (pipeline.output_dir / "wcs_validation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    assert all(report["checks"].values()), f"WCS validation checks failed: {report['checks']}"
    saved_wcs = WCS(fits.getheader(pipeline.output_dir / "difference_image.fits"))
    np.testing.assert_allclose(
        saved_wcs.pixel_to_world_values([100, 200], [150, 250]),
        science_wcs.pixel_to_world_values([100, 200], [150, 250]),
        atol=1e-9,
    )
    assert (pipeline.output_dir / "diagnostics.png").is_file()
    assert np.isfinite(pipeline.scorr_image[~pipeline.mask]).all()


def test_cross_filter_reprojection_labels_empty_catalogs_and_native_filters(tmp_path) -> None:
    paths, _ = wcs_survey_pair(tmp_path / "inputs")
    with fits.open(paths[1], mode="update", memmap=False) as image:
        image["SCI"].header["FILTER"] = "BP-like"
    sources = astro.compare_cross_filter(
        *paths,
        tmp_path / "results",
        reproject=True,
        psf_radius=10,
        threshold_sigma=1e8,
        plots=False,
    )
    assert sources.empty
    assert "comparison_warning" in sources.columns
    metadata = json.loads((tmp_path / "results/provenance.json").read_text())
    assert metadata["reference_filter"] == "r"
    assert metadata["science_filter"] == "BP-like"
    assert metadata["config"]["reproject"]
    assert metadata["registration_rms_pixels"] < 0.2
    assert "comparison_warning" in (tmp_path / "results/detected_sources.csv").read_text()
    assert "vetting_group" in sources.columns
    for name in ("less_color_confounded", "color_ambiguous"):
        assert astro.pd.read_csv(tmp_path / "results" / f"{name}_candidates.csv").empty
    header = fits.getheader(tmp_path / "results/difference_image.fits")
    assert header["FILTER"] == "BP-like"
    assert header["REFFILT"] == "r"
    assert header["NEWFILT"] == "BP-like"
    assert header["XFLTR"]


def plot_survey_validation(pipeline, sources, truth, metrics, blank_sky) -> None:
    """Save a headless, truth-aware overview without changing production plots."""
    from astropy.visualization import ImageNormalize, ZScaleInterval
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(21, 19), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 4)
    figure.suptitle(
        "Synthetic survey comparison | T1 isolated; T2 on a persistent elliptical galaxy nucleus\n"
        "Green circles: injected transients; orange crosses: artifacts; cyan squares: accepted",
        fontsize=16,
    )
    valid = ~pipeline.mask
    difference_limit = float(np.percentile(np.abs(pipeline.diff_image[valid]), 99.9))
    score_limit = max(8, float(np.nanmax(np.abs(pipeline.scorr_image))))
    truth_image = np.zeros(pipeline.ref_data.shape)
    for source in truth["transients"]:
        truth_image[source["y"], source["x"]] = source["flux"]
    panels = (
        (axes[0, 0], fits.getdata(pipeline.ref_path, "SCI"), "Template: sharper, noisy", "gray"),
        (axes[0, 1], fits.getdata(pipeline.new_path, "SCI"), "Science: broader, shifted", "gray"),
        (axes[0, 2], truth_image, "Injected new-only flux [science ADU]", "viridis"),
        (
            axes[1, 0],
            pipeline.products["matched_reference"],
            "Scaled, PSF-matched template",
            "gray",
        ),
        (axes[1, 1], pipeline.products["matched_new"], "Aligned, PSF-matched science", "gray"),
        (
            axes[1, 2],
            np.where(valid, pipeline.diff_image, np.nan),
            "Science - scaled template",
            "RdBu_r",
        ),
        (axes[1, 3], pipeline.scorr_image, "Calibrated score (not a probability)", "RdBu_r"),
        (axes[2, 0], pipeline.mask, "Excluded pixels / propagated artifacts", "binary"),
    )
    for axis, image, title, cmap in panels:
        limits = {}
        if axis is axes[1, 2]:
            limits = {"vmin": -difference_limit, "vmax": difference_limit}
        elif axis is axes[1, 3]:
            limits = {"vmin": -score_limit, "vmax": score_limit}
        elif cmap == "gray":
            limits = {"norm": ImageNormalize(image, interval=ZScaleInterval())}
        artist = axis.imshow(image, origin="lower", cmap=cmap, **limits)
        axis.set(title=title, xlabel="X [pixels]", ylabel="Y [pixels]")
        figure.colorbar(artist, ax=axis, shrink=0.75)
        native_science = axis is axes[0, 1]
        offset_x, offset_y = truth["shift_xy_pixels"] if native_science else (0, 0)
        for source in truth["transients"]:
            column, row = source["x"] + offset_x, source["y"] + offset_y
            axis.scatter(column, row, s=90, facecolors="none", edgecolors="lime")
            label = source["name"] + (" (G1 core)" if source.get("host") else "")
            axis.annotate(label, (column + 6, row + 6), color="lime", fontsize=9)
        for galaxy in truth["galaxies"]:
            if not galaxy["nucleus_flux"]:
                axis.annotate(
                    galaxy["name"],
                    (galaxy["x"] + offset_x + 6, galaxy["y"] + offset_y + 6),
                    color="magenta",
                    fontsize=9,
                )
        for artifact in truth["artifacts"]:
            if axis is axes[0, 0] and artifact["image"] != "reference":
                continue
            if native_science and artifact["image"] != "science":
                continue
            axis.scatter(artifact["x"] + offset_x, artifact["y"] + offset_y, c="orange", marker="x")
        if not native_science:
            axis.scatter(
                sources["x_centroid"],
                sources["y_centroid"],
                s=130,
                facecolors="none",
                edgecolors="cyan",
                marker="s",
            )
    axis = axes[0, 3]
    for psf, sigma, label in zip(
        (pipeline.ref_psf, pipeline.new_psf),
        truth["psf_sigma_pixels"],
        ("Template", "Science"),
        strict=True,
    ):
        center = psf.shape[0] // 2
        radius = np.arange(psf.shape[1]) - center
        line = axis.plot(radius, psf[center] / psf[center].max(), label=f"{label}: estimated")[0]
        axis.plot(
            radius,
            np.exp(-0.5 * (radius / sigma) ** 2),
            "--",
            color=line.get_color(),
            label=f"{label}: injected",
        )
    axis.set(title="PSF central profiles (peak normalized)", xlabel="Offset [pixels]", ylabel="PSF")
    axis.legend(fontsize=8)
    axis = axes[2, 1]
    for image, label in ((pipeline.products["score"], "Raw"), (pipeline.scorr_image, "Calibrated")):
        axis.hist(
            image[blank_sky],
            bins=np.linspace(-6, 6, 75),
            density=True,
            histtype="step",
            label=label,
        )
    normal_grid = np.linspace(-6, 6, 200)
    axis.plot(
        normal_grid, np.exp(-(normal_grid**2) / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)"
    )
    axis.set(title="Blank-sky score distribution", xlabel="Score", ylabel="Density", yscale="log")
    axis.set_ylim(1e-4, 1)
    axis.legend(fontsize=8)
    axis = axes[2, 2]
    axis.axis("off")
    table_rows = [
        [
            source["name"],
            f"{source['flux']:.0f}",
            f"{source['measured_flux']:.0f}" if source["recovered"] else "missed",
            f"{source['score']:.1f}" if source["recovered"] else "-",
            f"{source['position_error_pixels']:.2f}" if source["recovered"] else "-",
        ]
        for source in metrics["recovery"]
    ]
    table = axis.table(
        cellText=table_rows,
        colLabels=["ID", "Truth", "Flux", "Score", "Offset px"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    axis.set_title("Recovery / PSF flux [science ADU]")
    axis = axes[2, 3]
    axis.axis("off")
    axis.set_title("Measured validation metrics")
    axis.text(
        0,
        0.95,
        "\n".join(
            [
                f"Recovered: {metrics['recovered_transients']}/2; "
                f"false positives: {metrics['false_positives']}",
                f"Accepted / rejected: {len(sources)} / {len(pipeline.rejected_sources)}",
                f"Flux scale: {pipeline.metadata['flux_scale']:.4f} (truth 0.72)",
                f"Registration RMS: {pipeline.metadata['registration_rms_pixels']:.3f} px",
                f"Matched stars: {pipeline.metadata['matched_stars']}",
                f"FWHM ref / new: {pipeline.metadata['ref_fwhm_pixels']:.2f} / "
                f"{pipeline.metadata['new_fwhm_pixels']:.2f} px",
                f"Blank score median / std: {metrics['blank_score_median']:.3f} / "
                f"{metrics['blank_score_std']:.3f}",
                f"Masked: {pipeline.mask.mean():.1%}; "
                f"artifacts excluded: {metrics['masked_artifacts']}/{metrics['artifact_count']}",
                f"Template extra noise: {truth['template_extra_noise_adu']:.1f} ADU RMS",
                f"Detection threshold: {pipeline.metadata['threshold_sigma']:.1f}",
                f"Checks passed: {sum(metrics['checks'].values())}/{len(metrics['checks'])}",
                "Synthetic regression, not survey certification.",
            ]
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.8,
    )
    source = next(source for source in truth["transients"] if source.get("host"))
    column, row = source["x"], source["y"]
    slices = (slice(row - 18, row + 19), slice(column - 18, column + 19))
    host_norm = ImageNormalize(
        np.stack([pipeline.products[key][slices] for key in ("matched_reference", "matched_new")]),
    )
    for axis, image, label, cmap in zip(
        axes[3],
        (
            pipeline.products["matched_reference"],
            pipeline.products["matched_new"],
            np.where(valid, pipeline.diff_image, np.nan),
            pipeline.scorr_image,
        ),
        ("persistent host + nucleus", "host + nucleus + transient", "difference", "score"),
        ("gray", "gray", "RdBu_r", "RdBu_r"),
        strict=True,
    ):
        stamp = image[slices]
        limit = float(np.nanmax(np.abs(stamp)))
        limits = {"norm": host_norm} if cmap == "gray" else {"vmin": -limit, "vmax": limit}
        artist = axis.imshow(
            stamp,
            origin="lower",
            cmap=cmap,
            **limits,
            extent=(column - 18.5, column + 18.5, row - 18.5, row + 18.5),
        )
        axis.scatter(column, row, facecolors="none", edgecolors="lime", s=90)
        axis.set(title=f"T2 / G1 core: {label}", xlabel="X [pixels]", ylabel="Y [pixels]")
        figure.colorbar(artist, ax=axis, shrink=0.75)
    figure.savefig(pipeline.output_dir / "validation_overview.png", dpi=140)


def test_realistic_survey_comparison_end_to_end(tmp_path) -> None:
    """Validate complete same-filter FITS subtraction against known injected truth."""
    paths, truth = realistic_survey_pair(tmp_path / "inputs")
    pipeline = ProperTransientFinder(
        *paths, tmp_path / "results", {"align": True, "psf_radius": 9}
    )
    sources = pipeline.run_full_pipeline(threshold_sigma=5.0)
    row_grid, column_grid = np.indices(pipeline.mask.shape)
    blank_sky = ~pipeline.mask.copy()
    for source in truth["stars"] + truth["transients"]:
        blank_sky &= np.hypot(column_grid - source["x"], row_grid - source["y"]) > 18
    for galaxy in truth["galaxies"]:
        blank_sky &= (
            np.hypot(column_grid - galaxy["x"], row_grid - galaxy["y"])
            > 4 * galaxy["effective_radius_pixels"]
        )
    recovery = []
    for source in truth["transients"]:
        distances = np.hypot(
            sources["x_centroid"] - source["x"], sources["y_centroid"] - source["y"]
        )
        nearby = sources.loc[(distances < 2) & (sources["sign"] == 1)]
        result = {**source, "recovered": len(nearby) == 1}
        if result["recovered"]:
            candidate = nearby.iloc[0]
            result.update(
                measured_flux=float(candidate["flux"]),
                flux_error=float(candidate["flux_error"]),
                relative_flux_error=float(candidate["flux"] / source["flux"] - 1),
                score=float(candidate["snr"]),
                position_error_pixels=float(distances.loc[nearby.index[0]]),
                psf_correlation=float(candidate["psf_correlation"]),
                reduced_chi2=float(candidate["reduced_chi2"]),
                ra_deg=float(candidate["ra_deg"]),
                dec_deg=float(candidate["dec_deg"]),
            )
        recovery.append(result)
    masked_artifacts = 0
    artifact_leaks = 0
    for artifact in truth["artifacts"]:
        column, row = int(round(artifact["x"])), int(round(artifact["y"]))
        masked_artifacts += int(pipeline.mask[row, column])
        distances = np.hypot(
            sources["x_centroid"] - artifact["x"], sources["y_centroid"] - artifact["y"]
        )
        artifact_leaks += int((distances < 12).sum())
    recovered = sum(source["recovered"] for source in recovery)
    metrics = {
        "recovery": recovery,
        "galaxies": truth["galaxies"],
        "recovered_transients": recovered,
        "completeness": recovered / len(recovery),
        "purity": recovered / len(sources) if len(sources) else 0.0,
        "false_positives": len(sources) - recovered,
        "masked_artifacts": masked_artifacts,
        "artifact_count": len(truth["artifacts"]),
        "template_extra_noise_adu": truth["template_extra_noise_adu"],
        "artifact_leaks": artifact_leaks,
        "blank_sky_pixels": int(blank_sky.sum()),
        "blank_score_median": float(np.median(pipeline.scorr_image[blank_sky])),
        "blank_score_std": float(np.std(pipeline.scorr_image[blank_sky])),
        "pipeline_metadata": pipeline.metadata,
    }
    metrics["checks"] = {
        "both_transients_recovered": recovered == 2,
        "galaxy_core_transient_recovered": any(
            source["environment"] == "galaxy_core" and source["recovered"] for source in recovery
        ),
        "no_false_positives": len(sources) == 2 and recovered == 2,
        "artifacts_masked": masked_artifacts == len(truth["artifacts"]) and artifact_leaks == 0,
        "centroids_within_one_pixel": all(
            source.get("position_error_pixels", np.inf) < 1 for source in recovery
        ),
        "fluxes_within_twenty_percent": all(
            abs(source.get("relative_flux_error", np.inf)) < 0.2 for source in recovery
        ),
        "scores_above_threshold": all(source.get("score", 0) > 5 for source in recovery),
        "scale_within_five_percent": abs(pipeline.metadata["flux_scale"] / truth["flux_scale"] - 1)
        < 0.05,
        "registration_below_point_two_pixels": pipeline.metadata["registration_rms_pixels"] < 0.2,
        "psfs_within_fifteen_percent": all(
            abs(pipeline.metadata[key] / (2.35482 * sigma) - 1) < 0.15
            for key, sigma in zip(
                ("ref_fwhm_pixels", "new_fwhm_pixels"), truth["psf_sigma_pixels"], strict=True
            )
        ),
        "blank_sky_centered": abs(metrics["blank_score_median"]) < 0.25,
        "blank_sky_noise_reasonable": 0.6 < metrics["blank_score_std"] < 1.4,
        "majority_pixels_usable": pipeline.mask.mean() < 0.5,
    }
    metrics["checks"] = {key: bool(value) for key, value in metrics["checks"].items()}
    report_path = pipeline.output_dir / "validation_summary.json"
    report_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    plot_survey_validation(pipeline, sources, truth, metrics, blank_sky)
    assert all(metrics["checks"].values()), (
        f"Failed validation gates: {metrics['checks']}; {report_path}"
    )
