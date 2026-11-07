# Proper Transient Finder

**Compare two images of the sky to spot what changed.**

Python library for PSF-matched FITS difference imaging and unconfirmed
transient-candidate analysis. Python 3.11 or newer is required.

The distribution is named `proper-transient-finder`; import it as
`proper_transient_finder`.

Current release: `0.2.0`.

Inputs can use `.fits`, `.fit`, `.fts`, gzip compression (`.fits.gz`, `.fit.gz`,
`.fts.gz`) or FITS tiled compression (`.fits.fz`, `.fit.fz`, `.fts.fz`). No manual
decompression is required; files must retain their original binary contents.

## Installation

Install a built wheel into your chosen Python environment:

```powershell
python -m pip install ./dist/proper_transient_finder-0.2.0-py3-none-any.whl
python -m pip check
```

From a source distribution or checkout, `python -m pip install .` also
installs the package and its declared scientific dependencies. Build tools
and test tools are optional: `python -m pip install ".[dev]"`.

ProperImage 0.7.2 depends on legacy `tinynpydb`. If no compatible cached wheel
is available and its source build fails, install the following prerequisites
using your approved package source before installing this library:

```powershell
python -m pip install "setuptools>=77,<81" wheel
python -m pip install --no-build-isolation tinynpydb==0.1.2
```

No package-index, certificate, or execution-policy bypass is required.
See [USER_GUIDE.md](USER_GUIDE.md) for FITS requirements, installation details,
offline installation, and scientific limitations.

## Python API

```python
from proper_transient_finder import PipelineConfig, compare_images

config = PipelineConfig(plots=False)
result = compare_images("reference.fits", "science.fits", config=config)

print(result.candidates)
print(result.rejected_candidates)
print(result.metadata["quality_summary"])
difference, score, mask = result.difference, result.score, result.mask
```

No products are written unless you provide `output_dir`:

```python
result = compare_images(
    "reference.fits", "science.fits", output_dir="results/run_001", config=config
)
print(result.output_dir)
```

`PipelineResult` includes accepted/rejected catalogs, background diagnostics,
the signed difference, calibrated score, effective PSF, exclusion mask, backend
products, and provenance. Arrays and tables remain editable. `PipelineConfig`
is copied and revalidated; dictionaries remain supported. Library calls raise
ordinary Python exceptions rather than exiting the interpreter.

For input-only checks or manual stage control, use `ProperTransientFinder`:

```python
from proper_transient_finder import ProperTransientFinder

pipeline = ProperTransientFinder("reference.fits", "science.fits", "results/run_002", config)
report = pipeline.preflight()
```

Preflight does not certify registration, PSF seeing, photometric consistency,
or detectability. Cross-filter comparison remains explicitly experimental:
set `PipelineConfig(cross_filter=True)` or use the existing
`compare_cross_filter()` helper, which still returns a DataFrame and saves
products. A global stellar scale does not correct color differences.

## Command Line

```powershell
proper-transient-finder --ref reference.fits --new science.fits --output results/run_003
python -m proper_transient_finder --help
```

Both entry points use the same validated pipeline as the Python API.
Scores and artifact cuts do not establish astrophysical origin. Review
candidates and independent observations before claiming variability.

## Batch Processing

Run multiple pairs sequentially, including a shared reference:

```python
from proper_transient_finder import BatchJob, PipelineConfig, compare_batch

report = compare_batch(
    [
        BatchJob("epoch_1", "reference.fits", "science_1.fits"),
        BatchJob("epoch_2", "reference.fits", "science_2.fits"),
    ],
    output_dir="results/batch_001",
    config=PipelineConfig(plots=False),
)
for item in report.items:
    print(item.name, item.state, item.accepted_candidates, item.error)
print(report.successful)
```

Each pair gets a separate output folder and normal logs/products. An atomic
`batch_summary.json` tracks progress and failures. Failed pairs do not stop
later pairs unless `fail_fast=True`. The batch root must be new or empty,
even with overwrite enabled. Persisted batches retain lightweight reports;
without `output_dir`, they write nothing and retain full results in
`item.result`. Use `retain_results` to override that memory policy.

For the CLI, a JSON manifest such as `pairs.json` contains:

```json
[
  {"name": "epoch_1", "ref": "reference.fits", "new": "science_1.fits"},
  {"name": "epoch_2", "ref": "reference.fits", "new": "science_2.fits"}
]
```

```powershell
proper-transient-finder --batch pairs.json --output results/batch_check --check-only
proper-transient-finder --batch pairs.json --output results/batch_002 --no-plots
```

Manifest paths are relative to the manifest, not the current directory.
Shared CLI settings can be overridden by each job's `config` object and
`threshold_sigma`. Add `--fail-fast` to stop on the first failure. The CLI
returns 1 if any job fails; successful preflight does not certify subtraction
or transient detectability. Execution is sequential, with no automatic resume
or parallel workers. See [the full batch guide](USER_GUIDE.md#batch-processing)
for schema, naming, configuration, and retry details.

## Survey References

Opt in to downloading a Pan-STARRS or SkyMapper reference for a science FITS
image with valid celestial WCS. Choose the template filter explicitly:

```powershell
proper-transient-finder --survey auto --new science.fits --template-filter r --check-only
proper-transient-finder --survey auto --new science.fits --template-filter r --template-units adu --output results/survey_001
proper-transient-finder --survey auto --science-list science.txt --template-filter r --template-units adu --output results/survey_batch
```

The `adu` declaration above is appropriate **only after verifying the native
template count units**. It labels cache-owned copies; it does not calibrate or
convert pixels. Typical survey FITS headers have missing or ambiguous BUNIT.
Science input needs declared count units and FILTER, or an explicit science
filter override. No input FITS is edited.

```python
from proper_transient_finder import compare_survey_images

result = compare_survey_images(
    "science.fits", "results/survey_002", template_filter="r",
    survey="auto", template_units="adu", cache_dir="survey_cache",
)
```

Matching band labels select ordinary subtraction; different bands select the
existing **experimental cross-filter** mode. Labels do not guarantee identical
instrument bandpasses. Noise in survey stacks and resampled images is correlated;
scores and photometric errors are approximate, not confirmed variability.

`science.txt` is one path per line, with optional `#` comment lines; a JSON array
of path strings is also supported. Paths are relative to the list file. This is
separate from the existing `--batch` ref/new manifest. Processing is sequential,
continues after failed images unless `--fail-fast` is set, and retains lightweight
reports. Results and cache directories must be separate; results must be new or empty.

Each processed image gets `survey_reference.json`, `survey_diagnostic.png`, and
the existing products under `analysis/`. Failed processing is explicitly labeled;
`--no-plots` disables plots. `--check-only` performs no downloads or writes and
does not certify remote availability, calibration, registration or detectability.
Only coordinates, cutout size, band and optional epoch are sent to the survey;
science FITS pixels are never uploaded. See
[the survey workflow guide](USER_GUIDE.md#automatic-survey-reference-workflow).

### Large Survey Footprints (Standalone Preparer)

The automatic `--survey` mode deliberately remains a one-cutout workflow. For
a science footprint larger than a provider's request limit, prepare local inputs
from a source checkout, without changing the original science FITS or the
pipeline:

```powershell
python scripts\prepare_survey_mosaic.py --science science.fits --survey panstarrs --template-filter r --cache survey_cache --output prepared_reference --check-only
python scripts\prepare_survey_mosaic.py --science science.fits --survey panstarrs --template-filter r --cache survey_cache --output prepared_reference --max-download-mb 512 --timeout 300
proper-transient-finder --ref prepared_reference\survey_reference.fits --new prepared_reference\science_for_reference.fits --reproject --output results\mosaic_001
```

The preparer uses 24.9-arcminute Pan-STARRS or 9.8-arcminute SkyMapper tiles,
with overlap. Full valid coverage is required by default. For tiled mosaics,
`--max-missing-fraction 0.1` explicitly permits up to 10% missing science-grid
coverage. The maximum setting is `0.5` (50%); missing pixels remain masked and
sources there cannot be detected, while downstream masks may exclude more area.
`--crop-border-percent` instead removes 5% from each edge before downloading;
that retains about 81% of the image area. Both options are opt-in and may be
combined. If a padded footprint is at most 5% over one safe tile, the default
automatic centred crop can avoid a mosaic; use `--single-tile-overrun 0` to
disable that automatic crop. Inspect `survey_mosaic.json` for the actual plan,
coverage and warnings.

The defaults are 64 MB per FITS response and a 30-second network timeout; the
example raises these limits for large tiles. Downloads can require substantial
memory and disk space. `--overwrite` explicitly stages replacement preparer
products while preserving unrelated files; without it, the output directory
must be new or empty. Completed tiles remain reusable in the separate cache.
Pan-STARRS masks are taken from the native bitmask, and older cached mask
cutouts are upgraded in separate cache files.

Use the prepared science copy in the local pipeline, especially if it was
cropped. Verify both inputs' filter labels, units and noise calibration first:
`--science-filter` informs the survey plan but does not insert a `FILTER` header
into the prepared science FITS. Different bands require explicit experimental
cross-filter handling; missing or ambiguous BUNIT requires a verified
`--reproject-units` choice. No option converts the pixels or makes different
bandpasses equivalent. See [the full preparer guide](USER_GUIDE.md#standalone-large-footprint-mosaic-preparer).

## Medium FITS Robustness And Timing

Run the standalone experiment from this project directory:

```powershell
python experiments/robustness.py --output artifacts/medium-auto --sizes-mb 20 50 --trials 2
python experiments/robustness.py --output artifacts/medium-known --sizes-mb 20 50 --trials 2 --known-psf
```

Each FITS is approximately 20 or 50 decimal MB **including science, variance and
mask planes**. Five scenes cover clean images, cosmic/hot pixels, saturation/bad
columns, a satellite trail without a supplied mask, the same trail masked, and
combined detector/trail contamination. Each has a null control and five signed
point-source injections, including a host-galaxy source. Reports retain per-stage
times, peak process memory, recoveries, mask losses, false positives and failures.

**Measured 2026-11-02:** 40 known-PSF control analyses completed, with overall median
times of 12.0 s/pair at 20 MB/file and 28.3 s/pair at 50 MB/file. Peak process memory
was approximately 0.78/1.67 GB. All usable injections were recovered; masked
injections were lost. The 50 MB cases had a recurrent false positive for one noise
seed. These are exploratory timings, not guaranteed performance.

**Automatic PSF (fixed 2026-11-03):** the earlier failure of the four clean
20/50 MB pilot cases was traced to ProperImage's strict star selection silently
falling back to *every* detected source when fewer than 15 stars pass its cuts;
galaxy light, blends and noise peaks then widened the Gaussian fit until the
support guard rejected it. The estimator now seeds ProperImage's source cache
with the pipeline's own isolated SEP stars. Re-checked clean pairs with
automatic PSFs completed in 25.5 s at 20 MB and 65.5 s at 50 MB, with FWHM
3.19/4.30 and 3.19/4.34 pixels against injected 3.06/4.24, recovering all five
50 MB injections (the 1,800 ADU source was missed at 20 MB). These are
single-pair checks, not a rerun of the 40-case campaign, and other scenes and
observed transients remain unvalidated. The known-PSF control above used the
injection-model PSFs and did **not** exercise automatic estimation.
See [the full benchmark guide](USER_GUIDE.md#medium-fits-robustness-and-timing).

## Documentation

- [User Guide](USER_GUIDE.md): setup, scientific assumptions, outputs, and validation.
- [API Reference](docs/API.md): generated signatures, configuration defaults, and result fields.
- [Publish Documentation](docs/publishing.md): MkDocs setup, preview, and hosting.

### MkDocs Website

Build a searchable Material for MkDocs site from this README, the user guide,
and the API reference. Run from the project directory:

```powershell
python -m pip install ".[docs]"
.\scripts\docs.ps1 -Build
.\scripts\docs.ps1 -Serve
```

On Linux, macOS, or Git Bash, use `PYTHON` instead of `-PythonPath`:

```bash
python -m pip install ".[docs]"
bash scripts/tasks.sh docs --build
bash scripts/tasks.sh docs --serve
```

The publishable static site is written to `site/`. The local preview is at
<http://127.0.0.1:8001/>. Pass `-PythonPath` (PowerShell) or `PYTHON=<path>`
(`scripts/tasks.sh`) when using another prepared interpreter. See the
publication guide for GitHub Pages and other static hosts. No publication is
performed automatically.

### API Markdown

Regenerate the API reference from the project directory:

```powershell
.\scripts\docs.ps1 -PythonPath .\.venv\Scripts\python.exe
.\scripts\docs.ps1 -PythonPath .\.venv\Scripts\python.exe -Check
```

```bash
PYTHON=.venv/bin/python bash scripts/tasks.sh docs
PYTHON=.venv/bin/python bash scripts/tasks.sh docs --check
```

Use the path to your existing Python 3.11+ interpreter when the project environment
does not exist. Generation uses only the standard library, makes no network
requests, and produces Markdown that GitHub renders without a site build.

## GitHub Readiness

The repository includes a GitHub Actions workflow that tests Python 3.11, 3.12,
and 3.13 on Windows, Linux, and macOS for lint, tests, documentation freshness,
and wheel/source builds. Windows uses the PowerShell scripts; Linux and macOS
use `scripts/tasks.sh`. Documentation is ordinary checked-in Markdown; GitHub
Pages configuration is not required.
Build outputs, virtual environments, and generated observation/test artifacts
are ignored and should not be committed.

**License:** this project is released under the BSD 3-Clause License; see
[LICENSE](LICENSE). Third-party components keep their own licenses (ProperImage,
Astropy, and most of the scientific stack are BSD-3-Clause; SEP is LGPLv3+), and
their citation and license obligations still apply.

Before publishing, confirm permission to publish the repository and review
tracked files for private data/secrets. No GitHub publication or PyPI upload is
performed by the supplied scripts.

The checked-in workflow runs from the repository root (`working-directory: .`)
on every push and pull request. It deliberately has no `paths` filter, so a
change to `pyproject.toml`, the tests, or the documentation still runs the
quality job. If the project files ever move, update that setting and the script
paths in the same commit. Package metadata points at
<https://github.com/pierfra-rocci/proper_transient_finder> for source, issues,
and documentation.

**Author:** Pier-Francesco Rocci
Developed with assistance from ChatGPT, Claude, and Mistral.

## License

Released under the BSD 3-Clause License; see [LICENSE](LICENSE). Third-party
components keep their own licenses, and their citation and license obligations
still apply.

## Acknowledgments

This library builds on the work of the astronomy community, particularly:

- [ProperImage](https://github.com/quatrope/ProperImage), for astronomical image
    processing and PSF/subtraction tools.
- [Astropy](https://www.astropy.org/), for FITS, celestial coordinates, WCS, and
    astronomy utilities.

Please also follow these projects' own citation and license requirements.
