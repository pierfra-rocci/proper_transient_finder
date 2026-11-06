# Proper Transient Finder User Guide

**Author:** Pier-Francesco Rocci

Developed with assistance from ChatGPT, Claude, and Mistral.

This standalone manual covers the astronomical subtraction pipeline, its
launcher, its regression tests, and the commands used to run and validate it.

## 1. Scope And Project Layout

The pipeline compares a reference FITS exposure with a newer exposure to find
possible brightening and fading point sources. It estimates or accepts PSFs,
checks registration, matches image resolution, subtracts the images, and flags
common detector and subtraction artifacts. Opt-in celestial WCS reprojection
maps the reference onto the science grid before matching and subtraction.

The pipeline is a standalone, independently installable Python library.

Current files, relative to this guide's directory:

| File | Role |
| --- | --- |
| `pyproject.toml` | Standalone package metadata, runtime/dev dependencies, pytest, and Ruff settings. |
| `src/proper_transient_finder/astro.py` | FITS loading, PSF matching, subtraction, detection, and CLI implementation. |
| `src/proper_transient_finder/__init__.py` | Public API exports and `__version__`. |
| `src/proper_transient_finder/api.py` | Single-pair and batch APIs, result types, and manifest loading. |
| `src/proper_transient_finder/survey.py` | Opt-in survey planning, HTTPS acquisition, cache, science lists, and final diagnostics. |
| `src/proper_transient_finder/_version.py` | Single version source for imports and built distributions. |
| `src/proper_transient_finder/py.typed` | PEP 561 inline-typing marker, included in installed wheels. |
| `src/proper_transient_finder/__main__.py` | `python -m proper_transient_finder` entry point. |
| `run_proper_transient_finder.py` | Source-checkout command-line launcher. |
| `README.md` | Library installation and API quick start; distribution readme. |
| `USER_GUIDE.md` | This complete standalone manual. |
| `MANIFEST.in` | Source archive includes the guide, tests, scripts, and experiment. |
| `tests/test_astro.py` | Astronomy-only synthetic regression suite. |
| `scripts/bootstrap.ps1` | Install the project and its development tools. |
| `scripts/test.ps1` | Run this project's pytest suite. |
| `scripts/lint.ps1` | Lint and format-check this project's Python files. |
| `scripts/build.ps1` | Build wheel/source distributions and optionally a complete dependency wheelhouse. |
| `scripts/verify_install.ps1` | Install into a fresh environment and test the installed API/CLIs outside the checkout. |
| `scripts/tasks.sh` | POSIX shell equivalents of the PowerShell tasks for Linux, macOS, and Git Bash. |

## 2. Requirements And Setup

Use Python 3.11 or later with Windows PowerShell or a POSIX shell (`bash`). The
verified runtime for this implementation is Python 3.13.17 with ProperImage
0.7.2; the regression suite is also required to pass on Python 3.11 and 3.12.
This does not certify every later interpreter or dependency release. The
PowerShell examples below have POSIX equivalents in `scripts/tasks.sh`.

From the astronomy project directory, define these variables once in
your PowerShell session:

```powershell
$PipelineRoot = (Get-Location).Path
$Python = Join-Path $PipelineRoot ".venv\Scripts\python.exe"
$PipelineScript = Join-Path $PipelineRoot "run_proper_transient_finder.py"
$AstroSource = Join-Path $PipelineRoot "src\proper_transient_finder\astro.py"
$AstroTests = Join-Path $PipelineRoot "tests\test_astro.py"
$Configuration = Join-Path $PipelineRoot "pyproject.toml"
```

Use this explicit interpreter for installation, processing, and tests; shell
activation is optional. This avoids installing into one Python environment
and executing with another.

If the virtual environment does not exist, create it and install the project
from its own directory using an approved local Python 3.11+ installation:

```powershell
Push-Location $PipelineRoot
py -3.13 -m venv .venv
.\scripts\bootstrap.ps1
Pop-Location
```

On Linux, macOS, or Git Bash the equivalent is:

```bash
cd /path/to/project
python3 -m venv .venv
PYTHON=.venv/bin/python bash scripts/tasks.sh bootstrap
```

Install using the package indexes configured for your environment:

```powershell
& $Python -m pip install "setuptools>=77,<81" wheel
& $Python -m pip install --no-build-isolation tinynpydb==0.1.2
& $Python -m pip install --editable $PipelineRoot
& $Python -m pip check
```

The scientific dependencies are ProperImage, Astropy, NumPy, SciPy, pandas,
Matplotlib, SEP, Astroscrappy, Astroalign, and Reproject. The project's
`pyproject.toml` records runtime and development dependency ranges.

### Legacy Build Workaround

ProperImage depends on tinynpydb. Its old build helper needs `pkg_resources`,
which is absent from setuptools 83. Without compatible build tooling, it tries
to download setuptools 18 directly from public PyPI and can fail with HTTP 403
on restricted networks. Installing compatible setuptools and building tinynpydb
without isolation avoids that obsolete download. It does not disable network
or certificate protections. Do not use `--break-system-packages`, disable TLS
verification, or bypass approved package indexes.

### Verify Installation

```powershell
& $Python $PipelineScript --help
& $Python -m proper_transient_finder --help
& $Python -c "from properimage.single_image import SingleImageGaussPSF; from properimage.operations import subtract; print('ProperImage imports OK')"
```

To install development tools for the targeted checks below:

```powershell
& $Python -m pip install --editable "${PipelineRoot}[dev]"
```

### Explicit Interpreter In Project Scripts

The six project scripts (`bootstrap.ps1`, `lint.ps1`, `test.ps1`, `build.ps1`,
`docs.ps1`, `verify_install.ps1`) accept `-PythonPath`; the POSIX equivalents
in `scripts/tasks.sh` use `PYTHON`. Without either they still require this
project's `.venv`; there is no automatic fallback. Every wrapper checks for
Python 3.11 or newer. Bootstrap installs into the selected interpreter's
environment.

To use a different prepared Python 3.11+ environment, set `$Python` to its
interpreter path. That environment must have this library and its development
dependencies installed. Then run the project wrappers:

```powershell
& (Join-Path $PipelineRoot "scripts\lint.ps1") -PythonPath $Python
& (Join-Path $PipelineRoot "scripts\test.ps1") -PythonPath $Python -PytestArgs @('-q', '-k', 'preflight')
```

These commands work from any working directory. The test wrapper explicitly
selects the project's test directory; lint includes `experiments/` as well as
the package, tests, and launcher.

On Linux, macOS, and Git Bash, `scripts/tasks.sh` provides the same tasks and
selects the interpreter with the `PYTHON` environment variable instead of
`-PythonPath`:

```bash
export PYTHON=.venv/bin/python
bash scripts/tasks.sh bootstrap
bash scripts/tasks.sh lint
bash scripts/tasks.sh test -q -k preflight
bash scripts/tasks.sh docs --check --build
bash scripts/tasks.sh build --no-isolation
bash scripts/tasks.sh verify-install --distribution dist/proper_transient_finder-0.1.0-py3-none-any.whl
```

Its options mirror the PowerShell parameters: `--output-directory` for
`-OutputDirectory`, `--address` for `-Address`, `--wheelhouse` for `-Wheelhouse`,
and so on. Both front ends run the same `python -m` commands, and the CI matrix
exercises each on its own platform.

### Documentation Website

The README, this guide, and the generated API reference also form a searchable
MkDocs website. Install the optional documentation tools and build it:

```powershell
& $Python -m pip install "${PipelineRoot}[docs]"
& (Join-Path $PipelineRoot "scripts\docs.ps1") -PythonPath $Python -Build
& (Join-Path $PipelineRoot "scripts\docs.ps1") -PythonPath $Python -Serve
```

The static output is `$PipelineRoot\site`; preview it at
<http://127.0.0.1:8001/>. See the [publication guide](docs/publishing.md) for
strict validation, source maintenance, and hosting. Building does not publish.

### Python Library API

The distribution name is `proper-transient-finder`; the import name is
`proper_transient_finder`. Install this library from its project directory.
Existing `ProperTransientFinder`, `compare_cross_filter`, CLI, and launcher
calls remain supported. Public version metadata is available as `__version__`.

```python
from pathlib import Path
from proper_transient_finder import PipelineConfig, PipelineResult, compare_images

config = PipelineConfig(plots=False, cosmic_rays=True)
result: PipelineResult = compare_images(
  Path("reference.fits"), Path("science.fits"), config=config, threshold_sigma=5.0
)
print(result.candidates)
print(result.metadata["quality_summary"])
```

Without `output_dir`, comparison reads inputs and computes results entirely
in memory: no directory, catalogs, FITS products, log, status, or plots are
written, even when `config.plots=True`. Input fingerprints and observation
metadata are still returned. Supply `output_dir="results/run_001"` to save
the normal full-pipeline outputs, subject to existing overwrite safeguards.
Configuration can be a `PipelineConfig` or mapping and is copied/revalidated.
Input/output paths accept strings and `pathlib.Path` objects.

| Result attribute | Contract |
| --- | --- |
| `candidates`, `rejected_candidates` | pandas DataFrames with the existing accepted/rejected catalog schemas. |
| `background_diagnostics` | Separate per-detection background diagnostic DataFrame, joined by candidate ID. |
| `difference`, `score`, `psf`, `mask` | NumPy arrays: signed residual, calibrated score, effective PSF, and boolean excluded-pixel mask. |
| `products` | Backend products; `products['score']` is raw, unlike calibrated `score`. ProperImage and kernel backends differ. |
| `metadata` | Configuration, measured checks, quality summary, fingerprints, epochs, runtime/pipeline version, and timings. Persisted runs also include output dependency-version metadata. |
| `output_dir` | Resolved `Path` when saved, otherwise `None`. |

The result dataclass does not allow field reassignment; its arrays, tables, and
metadata remain editable. It retains full image products in memory, so release
unneeded results between large pairs. Scientific conventions and limitations
are identical to the CLI. Set `PipelineConfig(cross_filter=True)` explicitly for
experimental cross-filter comparison; its annotations do not confirm variability.
The older `compare_cross_filter()` still returns a DataFrame and saves files.

Library calls return values and propagate exceptions; they do not call
`sys.exit()` or print a CLI summary. Catch `ValueError` for invalid configuration
or incompatible images, `FileNotFoundError` for missing inputs,
`FileExistsError` for protected outputs, and other `OSError` subclasses for I/O
failures. Unexpected backend exceptions are not hidden. Logging uses the
`proper_transient_finder` namespace with a `NullHandler`; applications can attach
their own handlers. Only full persisted runs attach a temporary file handler.

### Build And Verify Installable Distributions

Build tools belong to the optional `[dev]` extra; importing or running the
library does not require pytest, Ruff, or build. This command builds both a
source archive and a wheel **from that source archive**:

```powershell
& (Join-Path $PipelineRoot "scripts\build.ps1") -PythonPath $Python
```

Default PEP 517 build isolation fetches build requirements from the configured
package source. On a prepared/restricted development machine, use
`-NoIsolation` with installed `setuptools>=77,<81`, `wheel`, and `build`.
`-OutputDirectory` overrides the default project `dist/` directory.

To prepare all runtime and test dependencies for offline installation:

```powershell
& (Join-Path $PipelineRoot "scripts\build.ps1") -PythonPath $Python -NoIsolation -IncludeDependencies
$Dist = Join-Path $PipelineRoot "dist"
$Wheelhouse = Join-Path $Dist "wheelhouse"
$Wheel = Join-Path $Dist "proper_transient_finder-0.1.0-py3-none-any.whl"
& $Python -m pip install --no-index --find-links $Wheelhouse $Wheel
```

Preparation accesses the approved package source and builds legacy
`tinynpydb`/ProperImage wheels before resolving the complete wheel set. It
requires the compatible legacy build tools described above. The resulting
wheelhouse includes platform/interpreter-specific scientific binaries: prepare
it on the target platform. A `py3-none-any` library wheel does not make NumPy,
SciPy, or other native dependencies universal. No dependency is silently skipped.

Verify a genuine fresh installation, not an editable checkout:

```powershell
& (Join-Path $PipelineRoot "scripts\verify_install.ps1") -PythonPath $Python -DistributionPath $Wheel -Wheelhouse $Wheelhouse
$SourceArchive = Join-Path $Dist "proper_transient_finder-0.1.0.tar.gz"
& (Join-Path $PipelineRoot "scripts\verify_install.ps1") -PythonPath $Python -DistributionPath $SourceArchive -Wheelhouse $Wheelhouse
```

The verifier creates a new isolated venv, installs runtime dependencies first,
checks `pip check`, installed import location/version/typing marker, and both
CLI entry points. Only then does it install `[dev]` and run the complete suite
outside the checkout with Python isolated mode. `-Wheelhouse` disables indexes
and pip's cache is disabled. Omit it for an online install using the configured
source; legacy source dependencies may need the documented preparation.
`-EnvironmentPath` selects a new directory; existing directories are refused.
Verification environments are retained and printed, not deleted automatically.
Built artifacts are ignored by Git and must be rebuilt after source changes.

These scripts build and verify local distributions; they do not publish to
PyPI or another registry. Publication needs registry credentials and the
license declared in `pyproject.toml`; this repository declares BSD-3-Clause
(`LICENSE`). A verified Windows installation does not certify every operating
system or later Python/dependency release.

## 3. Prepare The Images

Use bias/dark/flat-corrected exposures of the same field and filter. This tool
does not perform detector calibration, absolute photometric calibration,
host-galaxy modeling, moving-object classification, or transient confirmation.

The FITS reader accepts a single 2D image or one named SCI extension. Multiple
SCI planes are rejected rather than silently choosing a detector. Split
multi-detector data into individual science images before processing.

Supported filenames include `.fits`, `.fit` and `.fts`, their gzip forms
(`.fits.gz`, `.fit.gz`, `.fts.gz`) and tiled-compressed forms
(`.fits.fz`, `.fit.fz`, `.fts.fz`). Astropy reads the compression directly;
manual decompression is not required. Regression tests cover science, variance,
mask, inherited primary-header WCS and survey cache-copy creation in all nine
variants. Original inputs are not modified. A filename extension alone does not
make damaged or incorrectly encoded bytes a valid FITS file.

Transfer FITS as binary data. Do not save image bytes through a text editor,
text-mode download or character-encoding conversion. Rice decompression errors,
invalid HDU headers and unexpected file lengths should be investigated as input
integrity failures, not bypassed by filling missing pixels. Re-download the
original binary file and compare the provider's size/checksum when available.

| Input metadata or extension | Meaning |
| --- | --- |
| `FILTER` | When present in both images, the values must agree. |
| `BUNIT` | When present in both images, the labels must agree; unit conversions are not performed. |
| `GAIN` | Electrons per image count; `--gain` overrides both input headers. |
| `SATURATE` | Saturation level in input image counts; `--saturation` overrides both headers. |
| `DQ`, `MASK`, `BPM` | Same-shape quality planes; any nonzero or nonfinite value is invalid. |
| `VAR`, `VARIANCE` | Total per-pixel variance in squared input units. |
| `ERR` | Square root of total per-pixel variance, in input units. |
| Celestial WCS | Used to report ICRS RA/Dec and preserve the reference output grid. |

Supply either a total variance plane or a compatible ERR plane. Do not assume
that separate source/read/background variance components are automatically
combined. Invalid or nonpositive variance samples are masked.

Without a variance plane, local sky and background RMS are estimated with SEP,
and source Poisson variance is added using gain. Missing gain defaults to one
with a warning. Recognized non-count units, such as `adu/s`, require a variance
plane; a raw detector gain must not be used to invent Poisson noise for
flux-calibrated/count-rate images. ProperImage's alternative backend requires
appropriate detector-count input and uses its own internal noise model.

Supply real instrument saturation levels. Without SATURATE or `--saturation`,
the pipeline cannot identify saturated pixels by level and emits a warning.
Gain and saturation overrides apply to both images; retain correct per-image
headers if the two detectors/exposures need different values.

## 4. Run The Pipeline

Paths below are examples; replace the FITS paths with your exposures. Choose a
fresh output directory, separate from the directory containing either input.

### Already Registered Images

```powershell
& $Python $PipelineScript --ref "C:\data\reference.fits" --new "C:\data\target.fits" --output "C:\results\target_001"
```

Even without `--align`, registration is checked using matched stars. Different
image dimensions or a poor astrometric residual are errors, not reasons to
subtract incompatible grids.

### Register Before Subtraction

```powershell
& $Python $PipelineScript --ref "C:\data\reference.fits" --new "C:\data\target.fits" --output "C:\results\target_002" --align --threshold 5
```

Astroalign registers the new image onto the reference grid. At least six
isolated matched stars are required, with a default maximum positional RMS
of 0.35 pixels. A low positional residual alone is not sufficient: robust
stellar flux ratios also check photometric consistency when scale estimation
is enabled.

### Reproject A Survey Template Onto The Science WCS

```powershell
& $Python $PipelineScript --ref "C:\data\survey_reference.fits" --new "C:\data\science.fits" --output "C:\results\wcs_run" --reproject --psf-radius 10
```

`--reproject` uses Reproject's bilinear interpolation to map the **reference onto
the science image's celestial WCS and dimensions**. It supports the celestial
projections and header-based distortions understood by Astropy, including SIP;
the tests exercise TAN/SIP, SIN, and CAR. It is not a plate solver or an
astrometric calibration tool: both inputs need valid, accurate 2D celestial WCS
and sufficient overlapping isolated stars. Matched-star registration RMS must
still pass the configured tolerance. No residual Astroalign resampling is
performed, so inaccurate WCS must be corrected upstream.

Do not combine `--reproject` with `--align`, `--shift`, `--method proper`, or
supplied PSFs. Those combinations fail explicitly. The Gaussian PSFs are
estimated on the output grid after reprojection. Increase `--psf-radius` when
the measured seeing needs larger support; the example and distorted-grid test
use 10 rather than the default 8.

`--reproject-units auto` recognizes common detector counts/count rates and Jy
as **flux per pixel**, and units equivalent to Jy/sr as **surface brightness**.
Flux-per-pixel data are divided by native pixel solid angle, interpolated, and
multiplied by output pixel solid angle. Local areas are estimated from spherical
pixel corners, including WCS distortion. Surface brightness is interpolated
without an area multiplier. Unknown or absent BUNIT requires an explicit
`--reproject-units flux-per-pixel` or `--reproject-units surface-brightness`
choice. This selects the interpretation; it does not convert units. Both
inputs must still have compatible matching FILTER and BUNIT values unless
the experimental cross-filter comparison below is explicitly enabled; BUNIT
checks remain unchanged even in that mode.

Each accepted output pixel must have valid coverage and valid contributing
reference samples. Non-overlap and interpolation supports touching bad pixels
are masked, not filled as usable zero-flux observations. Downstream convolution
and matched-filter masks expand these excluded regions further.

### Automatic Survey Reference Workflow

This workflow accepts a science FITS image or a science-only list, derives its
ICRS center and padded footprint from celestial WCS, downloads a survey image,
and runs the existing WCS/PSF/photometry/detection pipeline. It is opt-in:
ordinary `--ref/--new` and `--batch` never contact a survey service.

| Survey | Template filters | Nominal full-science-footprint coverage | Padded cutout limit |
| --- | --- | --- | --- |
| Pan-STARRS | `g r i z y` | Declination at or above -30 degrees | Below 6000 pixels at 0.25 arcsec/pixel |
| SkyMapper DR4 | `u v g r i z` | Declination at or below 0 degrees | Below 9.9 arcminutes |

These are angular limits, not FITS file-size limits. Pan-STARRS currently permits
a padded request below 25 arcminutes. The planner uses twice the largest sampled
center-to-boundary separation with 10% padding, so its requested square can be
larger than either image side. Even a nominally 20-arcminute square field can
exceed the current limit after its diagonal and padding are included.

The automatic `--survey` workflow intentionally remains a one-cutout operation:
it does not tile or crop inputs. For larger fields, use the separate preparer
below, then pass its generated files through the ordinary local-reference
`--ref/--new --reproject` workflow. Do not remove the service limit or change
pixel-scale headers merely to make a request pass. The original science FITS is
never edited. SkyMapper is not a workaround for large fields: it has a smaller
size limit and different sky coverage.

An Astropy `obsfix` warning about inconsistent observatory coordinates is
separate from the cutout size check. Review the observation metadata upstream;
changing observatory height or hiding the warning does not reduce the footprint.
Check-only still does not validate remote availability, count-unit declarations
or successful subtraction. Missing science BUNIT needs calibration verification
before full survey processing, and an `r` reference against an `ip` science label
selects experimental cross-filter comparison, not a same-band analysis.

`--survey auto` prefers Pan-STARRS in overlapping eligible coverage. It chooses
using nominal footprint coverage and the **requested template band**, not science
filter inference. It does not switch providers after a service failure or silently
substitute filters. Nominal coverage is not a guarantee of usable images; try the
other explicitly selected eligible provider after a failed acquisition.

Science FITS selection matches the ordinary pipeline: one `SCI` extension wins,
otherwise exactly one usable image plane is required; primary and science headers
are merged. Supply accurate 2D celestial WCS, count-space BUNIT and FILTER, with
gain/variance and sufficient usable stars as for any other comparison. Missing
FILTER requires `--science-filter`. Multi-detector/ambiguous images must be split
first. This workflow does not plate-solve or calibrate the science exposure.

```powershell
& $Python $PipelineScript --survey auto --new "C:\data\science.fits" --template-filter r --check-only
& $Python $PipelineScript --survey panstarrs --new "C:\data\science.fits" --template-filter r --template-units adu --template-cache "C:\cache\survey" --output "C:\results\survey_001"
```

#### Standalone Large-Footprint Mosaic Preparer

This utility is deliberately separate from the pipeline script. It derives the
same padded science footprint locally, downloads safe overlapping tiles, and
reprojects them to a north-up TAN mosaic at the native survey pixel scale. The
mosaic covers the complete science footprint; the normal pipeline remains
responsible for its existing reference-to-science reprojection and subtraction.

```powershell
& $Python "scripts\prepare_survey_mosaic.py" --science "C:\data\science.fits" --survey panstarrs --template-filter r --cache "C:\cache\survey" --output "C:\data\prepared_reference"
& $Python $PipelineScript --ref "C:\data\prepared_reference\survey_reference.fits" --new "C:\data\prepared_reference\science_for_reference.fits" --reproject --output "C:\results\mosaic_001"
```

The safe request widths are 24.9 arcminutes for Pan-STARRS and 9.8 arcminutes
for SkyMapper, just below the provider ceilings. Tiles overlap by 10% and the
preparer refuses completion if their combined valid coverage misses any science
footprint pixel. `survey_mosaic.json` records the plan, tile metadata, output
shape and warnings. The preparation directory must be new or empty and must be
separate from the survey cache; it must not be the science input directory or
an ancestor of it.

When a padded field is only a little too large, the default is simpler: if it
is within 5% of one safe tile, the preparer writes a **separate centred crop**
as `science_for_reference.fits` and uses one reference cutout. This discards
only the outer science region from that subsequent comparison; it never alters
the original. Pass `--single-tile-overrun 0` to mosaic every oversized field,
or select a value up to `0.25` for a deliberately larger crop tolerance.
`--check-only` reports the local plan without downloading or writing files.

The preparer preserves native reference headers and does not invent a zero
point, gain, variance, unit conversion, filter transformation, or epoch
certification. Pan-STARRS variance is mosaicked only when every tile supplies
it. Variance contributions follow the same valid science tiles; if a contributing
tile has missing or invalid variance inside the science footprint, preparation
fails rather than writing an incomplete variance plane. SkyMapper still needs a
validated gain/noise model for a full subtraction.
Before running the local pipeline, verify compatible count-space BUNIT/FILTER
values and use the cross-filter precautions described here. Resampling and
overlap make the noise correlated, so all candidates remain unconfirmed.

**Verify native units before declaring `adu`.** Pan-STARRS cutouts can lack BUNIT;
SkyMapper can label it `Data Value`. These ambiguous labels fail unless you provide
`--template-units adu` or `electron` after verification. Declarations only normalize
headers of cache-owned copies. ADU and electron inputs cannot be mixed; count rates,
Jy and other calibrated units must be converted consistently upstream. No arbitrary
zero point, gain or physical calibration is invented. `--gain` applies to science
only here; `--template-gain` independently declares template electrons per count.
Reference variance/ERR or a validated count-space gain model is required. SkyMapper
does not provide variance through this interface; its gain-based noise estimate is
not a measured total variance. Pan-STARRS `stack.wt` is retained as pixel variance,
and native quality masks are preserved. Survey coadd/resampling noise correlations
are not a fully propagated survey covariance model.

Matching normalized labels choose same-band comparison; differing labels automatically
choose the existing experimental cross-filter mode. Common `Sloan-`, `SDSS-`, `PS1-`
and `SkyMapper-` prefixes are recognized. This does not make physical passbands
equivalent. The global stellar scale is not a color transformation; score and flux
errors omit color/scale systematics. All existing registration, PSF, flux-scatter
and artifact guards remain active. Do not loosen them merely to obtain candidates.
The workflow requires kernel reprojection and rejects align/shift, ProperImage
subtraction and supplied PSF combinations, as the local reprojection path does.

#### Science-Only Lists And Python API

Use a text file such as `science.txt`:

```text
# Paths relative to this list file
exposures/night_1.fits
exposures/night_2.fits
```

Or use `science.json` containing an array of strings:

```json
["exposures/night_1.fits", "exposures/night_2.fits"]
```

```powershell
& $Python $PipelineScript --survey auto --science-list science.txt --template-filter g --template-units adu --output "C:\results\survey_batch"
```

`--science-list` is distinct from `--batch` and mutually exclusive with `--new`.
Survey mode forbids an explicit `--ref` or pair manifest. Sequential jobs get portable
names `science-0001`, `science-0002`, etc. Duplicate filenames remain separate jobs.
Each image has independent success/failure status; `--fail-fast` skips remaining
images after the first failure. A partial failure produces CLI exit code 1.
There is no parallel execution, automatic resume or per-image option object in this
list format. Persisted batches discard arrays after each job and return the existing
`BatchResult`/`BatchItemResult` lightweight reports.

```python
from proper_transient_finder import (
    compare_survey_images, compare_survey_batch, load_science_list,
    prepare_template_request, fetch_survey_template,
)

plan = prepare_template_request("science.fits", "r", survey="auto")
print(plan.to_dict())  # Local read only
template = fetch_survey_template(plan, "survey_cache")  # Explicit HTTPS acquisition
result = compare_survey_images(
    "science.fits", "results/survey_002", template_filter="r",
    template_units="adu", cache_dir="survey_cache",
)
report = compare_survey_batch(
    load_science_list("science.txt"), "results/survey_batch_002",
    template_filter="r", template_units="adu", cache_dir="survey_cache",
)
```

`compare_survey_images` returns `PipelineResult`, or a JSON-compatible local planning
report with `check_only=True`. `compare_survey_batch` returns `BatchResult`. Shared
options include `config`, `science_filter`, `template_units`, `template_gain`,
`timeout`, `max_download_mb` and `threshold_sigma`. See the generated
[API reference](docs/API.md) for signatures and request/template fields.

#### Cache, Provenance, Diagnostics And Limits

Only coordinates, footprint size, template band and optional science epoch are sent
over HTTPS to `ps1images.stsci.edu` or `api.skymapper.nci.org.au`. Science pixels and
local paths are not uploaded. Requests and redirects are restricted to these hosts;
TLS verification stays enabled. Default per-response timeout is 30 seconds; FITS
products are limited to 64 MB each, configurable up to 512 MB. Discovery responses
have a separate 4 MB limit. These are transfer limits, not full-run memory bounds.

The cache defaults to `survey_cache/`, relative to the working directory. Hashed FITS
files have JSON manifests, source URLs, image IDs, native labels/units/epochs and
SHA-256 fingerprints. A cache hit verifies the FITS hash; corruption fails rather
than silently reusing data. Label/gain normalization writes separate cached copies
and never alters science inputs. Keep cache separate from results; each results root
must be new or empty even with `--overwrite`. Preserve manifests alongside FITS for
auditing. Concurrent access to the same cache is not supported. Cached products are
not automatically refreshed; use a new cache directory to reacquire.

SkyMapper selection uses the longest exposure then seeing and image ID, requiring
an older observation when science MJD/DATE-OBS is known. Pan-STARRS uses a usable
stack with associated variance/mask products when available. A known non-older
template epoch is rejected. Missing epochs and coadd time spans produce chronology
warnings: a stack timestamp alone does not prove all contributing images predate
the science exposure. Transients present in a reference can be suppressed or reversed.

Single-image results contain:

- `survey_reference.json`: acquisition/processing state, original science path/hash,
  WCS request, chosen survey/native metadata, declared units, warnings and any error.
- `survey_diagnostic.png`: matched reference/science, signed difference, empirical
  score, usable mask, unconfirmed positions, epochs and quality/correlation warnings.
- `analysis/`: all normal catalogs, FITS, logs, quality reports and candidate plots;
  its `provenance.json` additionally includes the survey acquisition record.

Survey batches place these under each generated job directory and add an atomic
`batch_summary.json` at their root. Once processing has started, failures produce an
explicitly labeled failure diagnostic, not a fabricated blank subtraction. Invalid
shared options fail before writes; missing/bad per-image inputs are recorded as batch
failures. `--no-plots` disables PNGs. `--check-only` creates no directories/downloads
and reports remote availability, calibration/noise, overlap/registration, PSF/photometry
and detectability as unchecked. Successful planning is not a successful comparison.

Validation includes mocked acquisition/cache/security, list/CLI/failure regressions
and actual synthetic subtraction with the final plot. Live public example cutouts
from both services were retrieved successfully; this verifies current service
formats, not real-survey subtraction accuracy, photometric calibration or discovery
completeness. Independent confirmation and instrument-specific injection/recovery
tests are still required.

### Experimental Cross-Filter Comparison

This opt-in approximation compares images taken through different filters.
It uses the same Gaussian PSF cross-convolution and one robust global stellar
flux scale, not a spectral transformation. No new dependencies, online
catalogs, or passband models are required. You still need two calibrated FITS
images, not a source-catalog table, with explicit FILTER labels and compatible
BUNIT values. A BP/RP-like camera exposure is not a Gaia catalog image.

```powershell
& $Python $PipelineScript --ref "C:\data\ps1_r.fits" --new "C:\data\bp_like.fits" --output "C:\results\cross_filter" --cross-filter --reproject --psf-radius 10 --max-scale-scatter 0.4
```

Use `--align` instead of `--reproject` for a small image-plane registration,
or neither for already registered grids. The experimental mode requires the
kernel backend. Registration, masks, PSF support, input protection, and
candidate morphology cuts are unchanged. Standard runs still reject different
FILTER values, and their catalog columns and defaults are unchanged.

The public Python entry point returns a DataFrame and saves the standard products:

```python
from proper_transient_finder import compare_cross_filter

candidates = compare_cross_filter(
  "reference_r.fits",
  "science_bp.fits",
  "cross_filter_results",
  reproject=True,
  psf_radius=10,
  max_scale_scatter=0.4,
)
```

The function accepts `threshold_sigma` (default 5), `reproject` (default False),
`max_scale_scatter` (default 0.15), and additional PipelineConfig keywords such
as `align`, `plots`, `cosmic_rays`, and `psf_radius`. Its explicit arguments
and experimental flag take precedence over additional options.

The 0.4 example permits 40% clipped fractional scatter in high-SNR stellar
flux ratios; it is not a recommended universal tolerance or a color correction.
Increase it only after inspecting residuals. The default remains 0.15.
Provenance records the measured clipped and unclipped scatter, retained star
count, rejected star count, and fitted scale. A manual scale or `--no-beta`
bypasses stellar scale estimation and therefore these scatter measurements.

Cross-filter runs now add non-destructive, source-aware vetting after detection.
The original accepted and rejected catalogs retain every residual and their
existing detection flags. Two additional files partition accepted residuals:

- `less_color_confounded_candidates.csv`: valid aperture measurements with no
  nearby reference detection or significant reference light. These are not
  confirmed transients.
- `color_ambiguous_candidates.csv`: candidates on existing sources, including
  galaxy cores, or with unreliable/unavailable vetting measurements. Real
  supernovae can remain in this group; nothing is silently discarded.

Both group files are written even when empty. The returned DataFrame has a
`vetting_group` column with the same classification, plus `vetting_reason`,
`reference_source_associated`, `reference_distance_pixels`,
`reference_aperture_flux`, `reference_aperture_error`,
`residual_aperture_flux`, `fractional_excess`, and `within_color_range`.

Vetting measures the scaled PSF-matched reference and signed difference in
the same aperture (the stellar scaling aperture). A reference association is
a detected source within one effective PSF FWHM or reference aperture SNR
of at least 5. Aperture errors use conservative PSF-convolution covariance
row-sum bounds, subject to the existing native-noise/registration limits.
Fractional excess is residual aperture flux divided by scaled reference
aperture flux; it is unavailable when reference SNR is insufficient.

The empirical color range is the 95th percentile of absolute fractional
residuals among all high-SNR calibration stars, including clipped outliers.
`within_color_range` compares the aperture residual with that range plus a
three-error margin. This is a ranking heuristic, not a 95% confidence bound
or a calibrated variability test. An associated source remains ambiguous
even outside this range: stellar colors do not bound galaxy colors.
An unavailable measurement is left blank, not treated as zero excess.

With a manual scale or `--no-beta`, the stellar color range is unavailable;
all candidates are conservatively retained as ambiguous with an explicit
reason. Provenance records the aperture, color range, heuristic, availability,
and accepted counts per group. No same-filter columns, subtraction, detection
thresholds, or artifact cuts are changed by this post-processing.

The four heuristic thresholds are configurable through CLI options or the
corresponding underscore-named `PipelineConfig`/`compare_cross_filter` keywords:

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--color-association-fwhm` | `1` | Maximum reference-source distance in effective PSF FWHM. |
| `--color-reference-snr` | `5` | Reference detection/aperture significance threshold. |
| `--color-quantile` | `0.95` | Absolute fractional calibration-star residual quantile, in `(0, 1]`. |
| `--color-error-margin` | `3` | Multiplier of the conservative color-range error margin. |

All except the quantile must be finite and positive. Defaults preserve the
original grouping. These options only affect cross-filter annotations, not
which residuals pass the detection/artifact cuts; tuning them cannot turn
color-confounded residuals into confirmed variability.

Accepted and rejected catalogs gain a `comparison_warning` column, including
when empty. Warnings are separate from rejection flags, so candidates remain
available for inspection. Provenance identifies the mode, original filters,
and statistical-only error scope. Image FITS headers include XFLTR=True,
REFFILT, NEWFILT, and warning history; diagnostic figures are labeled.

**An unchanged object can appear as an accepted residual simply because of
its color.** Residual sign does not prove brightening/fading. Flux errors do
not include color or scale uncertainty, and scores are not calibrated
variability significance. Faint sources and galaxy nuclei need particular
caution and independent confirmation. The tests deliberately retain a false
core residual in an unchanged colored galaxy and recover isolated/nuclear
injections without claiming unbiased cross-filter photometry.

### Use Known PSFs And Additional Masks

```powershell
& $Python $PipelineScript --ref "C:\data\reference.fits" --new "C:\data\target.fits" --output "C:\results\target_003" --ref-psf "C:\data\reference_psf.fits" --new-psf "C:\data\registered_target_psf.fits" --ref-mask "C:\data\reference_mask.fits" --new-mask "C:\data\target_mask.fits" --psf-radius 12
```

Provide both PSF files, not just one. They must be finite, nonnegative intensity
PSF stamps sampled on the reference and registered-new grids, respectively.
They are centered and normalized. A PSF losing more than 0.5 percent of its flux
inside the configured support is rejected. External masks must match their
input image shapes before registration.

### Alternative ProperImage Subtraction

```powershell
& $Python $PipelineScript --ref "C:\data\reference.fits" --new "C:\data\target.fits" --output "C:\results\target_004" --align --method proper
```

External PSF files and a manual `--flux-scale` are not supported by this backend.
Use the kernel backend when you need those inputs or supplied variance planes.

### Outputs And Exit Status

For input checks before processing, add `--check-only` to the intended command:

```powershell
& $Python $PipelineScript --ref "C:\data\reference.fits" --new "C:\data\target.fits" --output "C:\results\target_check" --check-only
```

This reads the FITS inputs, variance/mask planes, supplied PSFs, configuration,
grid compatibility, and (in WCS mode) celestial-WCS/unit prerequisites. It
prints a JSON report but creates no output directory or products. The Python
equivalent is `pipeline.preflight(threshold_sigma=5.0)`. Passing preflight
does not validate actual registration, celestial overlap, PSF seeing/support,
stellar photometric consistency, cosmic-ray cleaning, write permissions,
disk space, or transient detectability; these unchecked items are explicit
in the report.

The console reports accepted and rejected candidate counts and the output
directory, followed by registration RMS, matched-star count, usable fraction,
scale scatter, and quality warnings. Exit status 0 means the run completed
(or the requested input checks passed), 1 means analysis failed, and
2 generally indicates invalid/missing command-line arguments. A successful run
can contain zero candidates.

Existing output files are protected. Prefer a new output directory for each
run; use `--overwrite` only deliberately. The pipeline refuses to overwrite
its input images, masks, or PSFs. Failures during writing can leave partial
outputs, so inspect the exit status rather than treating an existing directory
as evidence of a complete run.

Full runs write `run_status.json` atomically: `running`, `completed`, or
`failed`, with UTC timestamps, the processing stage, stage durations, and
an error on failure. Only `completed` indicates all requested outputs finished;
it does not establish scientific validity. Abrupt process termination can
leave `running`. Early configuration/output-path errors can happen before a
run directory exists. A failed run retains diagnostics: retry in a fresh
directory or explicitly use `--overwrite`. Overwrite does not purge stale
products from earlier runs, so a fresh directory remains preferable.

`pipeline.log` retains stage progress, warnings, and failure tracebacks.
`quality_summary.json` combines measured metrics, configured limits, caveats,
candidate-group counts, rejection-reason counts, and timings. Timings for
`prepare_and_subtract` include nested stages; do not sum overlapping durations.
Quality warnings do not change detection thresholds or candidate flags.

### Batch Processing

Batch mode processes an explicit list of pairs sequentially in order, reusing
`compare_images()` for each pair. A reference can be shared across epochs.
Scientific defaults, quality guards, catalog columns, coordinate conventions,
and cross-filter limitations are unchanged. Batch completion is not scientific
confirmation. No new dependency, worker pool, file-discovery heuristic, or
automatic resume/retry mechanism is introduced.

#### Python API

```python
from proper_transient_finder import BatchJob, PipelineConfig, compare_batch

jobs = [
    BatchJob("night_1", "reference.fits", "night_1.fits"),
    BatchJob("night_2", "reference.fits", "night_2.fits",
             config={"plots": False}, threshold_sigma=6.0),
]
report = compare_batch(
    jobs, output_dir="results/batch_001",
    config=PipelineConfig(plots=False), fail_fast=False,
)
for item in report.items:
    print(item.name, item.state, item.accepted_candidates, item.error)
print(report.successful)
summary = report.to_dict()  # JSON-safe; excludes image arrays and tables
```

`BatchJob` accepts `name`, `reference_fits`, `science_fits`, optional `config`,
and optional `threshold_sigma`. API paths are relative to the process working
directory; strings and `Path` objects are supported. Each job's mapping config
merges over shared batch settings. A job-level `PipelineConfig` is a complete
replacement, including all its defaults. Configuration is copied, normalized,
and validated before processing. A missing job threshold inherits the batch
`threshold_sigma` (default 5); thresholds must be finite and positive.

`BatchResult.items` is an ordered tuple of `BatchItemResult` records with
name, resolved input/output paths, state, accepted/rejected counts (or `None`
when unavailable), elapsed seconds, error type/message, optional full `result`,
and optional input-only `preflight` report. `successful` is true only when
every job completed or every requested input check passed. `check_only`
distinguishes preflight reports from actual image processing.

For no-write analysis, omit `output_dir`:

```python
report = compare_batch(jobs, config=PipelineConfig(plots=False))
first = report.items[0]
if first.state == "completed":
    print(first.result.candidates)
    difference = first.result.difference
```

No-write batches retain full `PipelineResult` objects by default; memory then
grows with all successful jobs. Persisted batches discard each full result
after recording its counts by default, keeping image processing to one pair
at a time. Explicit `retain_results=True` retains results even when saving;
`False` discards them even for no-write analysis. Lightweight status records
are always retained for every job.

#### JSON Manifest And CLI

A manifest is a nonempty JSON array. Only `ref` and `new` are required per
object; allowed fields are exactly `name`, `ref`, `new`, `config`, and
`threshold_sigma`:

```json
[
  {"name": "night_1", "ref": "images/reference.fits", "new": "images/night_1.fits"},
  {
    "name": "night_2",
    "ref": "images/reference.fits",
    "new": "images/night_2.fits",
    "threshold_sigma": 6.0,
    "config": {"plots": false, "cosmic_rays": true}
  }
]
```

Paths in `ref`, `new`, and per-job `config` fields `ref_mask`, `new_mask`,
`ref_psf`, `new_psf` resolve relative to the manifest's directory. Absolute
paths also work. Both PSF stamps must be supplied together, as for single
pairs. UTF-8 manifests, including Windows UTF-8 BOM, are accepted. Missing
names become `pair-0001`, `pair-0002`, etc. Names must be unique ignoring case,
1-64 ASCII letters/digits/underscores/hyphens, beginning with a letter or digit;
Windows device names such as `CON`, `NUL`, `COM1`, and `LPT1` are rejected.
Traversal paths and separators are not valid names.

```powershell
& $Python -m proper_transient_finder --batch pairs.json --output results/batch_check --check-only
& $Python -m proper_transient_finder --batch pairs.json --output results/batch_002 --no-plots
& $Python -m proper_transient_finder --batch pairs.json --output results/batch_003 --fail-fast
```

The installed `proper-transient-finder` command and source launcher accept the same
options. `--batch` cannot be combined with `--ref`/`--new`; without it, both
single-pair inputs remain required. Normal CLI science options provide shared
settings, with manifest `config` values taking precedence. Config keys use
Python names (`plots`, `cosmic_rays`), not CLI switches (`--no-plots`, etc.).
Manifest `threshold_sigma` overrides `--threshold`. Global CLI mask/PSF paths
and `--output` remain relative to the shell working directory. Unknown job
fields/config keys, malformed manifests, invalid names/settings, and output
collisions are errors, never silently ignored. `load_batch_manifest(path)`
provides the same path resolution for library callers without executing jobs.

#### Outputs, Failures, And Retries

The batch root must be new or empty, even with `overwrite=True`/`--overwrite`.
It must not contain any job's input image, mask, or PSF, including another
job's input. These structural checks run for the whole batch before any
processing or output creation. Empty batches are rejected. Use one distinct
root per run; concurrent processes must not share an output root.

```text
results/batch_001/
  batch_summary.json
  night_1/             # normal per-pair files, log, status, and provenance
  night_2/
```

The atomic summary includes UTC start/update timestamps, resolved paths,
counts, durations, error details, and ordered job records, but no image arrays
or tables. It is updated before/after each job. Job states are `pending`,
`running`, `completed`, `checked`, `failed`, `skipped`, or `interrupted`.
Batch states are `running`, `completed`, `completed_with_errors`, `checked`,
`stopped` (fail-fast), or `interrupted`. Normal persisted per-job logs and
provenance retain the underlying science configuration, input fingerprints,
quality warnings, and processing diagnostics.

Ordinary processing exceptions become failed records; by default later jobs
continue. `fail_fast=True`/`--fail-fast` stops on the first failed pair and
marks remaining jobs skipped. Structural validation errors and summary-write
errors propagate to Python callers. Interrupts such as `KeyboardInterrupt`
also propagate; a partial summary is attempted, marking the active job
interrupted and pending jobs skipped. Forced process termination can leave
`running`, which must not be interpreted as completion. Review the summary
and per-job files before retrying only failed/skipped jobs into a new root;
there is no automatic continuation of an existing batch directory.

CLI exit code 0 means every requested pair completed (or every preflight
passed), 1 means an analysis/manifest/output error or any failed job, and
2 indicates invalid CLI syntax or mode combinations. The console gives batch
counts; detailed metrics remain in each pair's normal reports. A completed
pair may have zero candidates. Python callers must inspect `successful`
and/or individual states rather than assuming a returned report means success.

`check_only=True`/`--check-only` runs each input preflight, returns/prints its
report with unchecked items, and writes nothing, including no batch summary
or job directories. Fresh-root/input-overlap guards still apply. It does not
validate registration, PSF seeing, photometric agreement, write permissions,
disk capacity, or detectability. Every image pair still requires independent
scientific review; especially across filters, repeated residuals do not
establish astrophysical variability.

## 5. Processing And Scientific Interpretation

### Default Kernel Matching

The default `--method kernel` uses a global Gaussian PSF estimated with
ProperImage, or supplied PSF stamps. The reference is convolved with the new
PSF, and the new image with the reference PSF. Their effective PSFs then match
without unstable deconvolution. This cross-convolution is stable but reduces
resolution; it is not ZOGY's optimal decorrelated subtraction.

Photometric scale is estimated from robustly clipped high-SNR stellar aperture
flux ratios, unless fixed explicitly. The subtraction is **new minus scaled
reference**: positive residuals mean brightening; negative residuals mean fading.

The matched-filter score propagates variance through squared kernels, including
the noise covariance introduced by these convolutions. Independent input noise
is assumed. Background score calibration is empirical and is not allowed to
amplify kernel scores when measured scatter is below predicted scatter.

With `--reproject`, the reference's resampled diagonal variance uses **squared
bilinear weights**, including pixel-area factors. Subsequent convolutions and
matched filtering use a conservative bound from the row sums of the resampling
covariance, not a naive independently resampled variance image. Thus filtered
variance, flux errors, and PSF-fit residual statistics can be conservative.
The covariance matrix itself is not stored. The bound assumes independent
native pixels in both inputs; it does not recover pre-existing coadd or detector
correlations. Empirical score calibration cannot certify a survey noise model.

### ProperImage Backend

`--method proper` calls ProperImage's `subtract()` using mask-bearing Gaussian
SingleImage instances. It supplies sky-inclusive detector counts because the
library uses stored counts in its source-noise calculation. Source detection
uses the returned Scorr, not raw D. Scorr is empirically recentered and rescaled.

ProperImage uses its own transparency/noise calculation, not supplied VAR/ERR
planes. Its D and fitted flux amplitudes have normalized units, not calibrated
detector-count units. Flux uncertainties are left unavailable rather than
invented. Its output P is a difference PSF, not a matched science exposure.

### Artifact Rejection

Invalid pixels, saturation, and L.A.Cosmic detections from Astroscrappy are
masked. Masks and borders are expanded by the full kernel/matched-filter
support to exclude contaminated neighborhoods. Science pixels are not globally
sigma-clipped merely for being bright.

Candidates of both signs are checked for connected area, PSF width, elongation,
PSF correlation, nearby opposite-sign dipoles, and masked neighborhoods.
Rejected candidates are retained with reasons. The reduced PSF-fit chi-squared
is diagnostic; it is not an independently configurable rejection cut.

`rejection_summary.csv` counts each reason separately. A multi-flag candidate
contributes to each applicable reason, so counts can exceed the rejected total.
`candidate_background.csv` reports every detected candidate, accepted or
rejected, joined by `id` without changing the standard catalog columns.
It uses unmasked difference pixels from two to four effective PSF FWHM around
the candidate, fitting a plane to distinguish an offset, gradient, and excess
residual structure. Fewer than 30 valid pixels produces `insufficient_annulus`
and blank measurements, not a clean bill of health.

The diagnostic ratios divide median background, gradient per FWHM, and
detrended RMS by median propagated per-pixel noise (or robust detrended noise
when no variance product is available). Warning levels are absolute offset
above 1, gradient above 0.5, or structure RMS above 2. Despite the `_snr` column
names, these are heuristic ratios, not calibrated source significance or
uncertainty on the background mean. They do not model galaxy light or color
systematics and never reject candidates.

### Important Limits

- **Empirical `--auto-psf` is disabled**, as agreed. Its kernel-support/flux
  preservation regression did not pass with ProperImage 0.7.2. Use Gaussian
  PSFs or supplied PSF files; this option fails explicitly rather than silently
  falling back to another method.
- There is no spatially varying matching solution. Strongly varying/asymmetric
  PSFs, undersampling, crowded fields, extended hosts, and strong distortions
  need specialized/local modeling and real-data validation.
- Existing input coaddition/interpolation covariance and Astroalign resampling
  covariance are not fully modeled. Empirical calibration is an approximation,
  not an instrument-specific noise model. WCS reprojection's own bilinear
  covariance is bounded conservatively, under independent-native-noise assumptions.
- WCS mode is an image-pair/cutout workflow, not an all-sky mosaic builder.
  The in-memory sparse interpolation operator scales with image size. Large
  changes in sampling, severely distorted/undersampled PSFs, and PSFs varying
  across the field still require specialized validation and matching.
- Detection scores are not discovery probabilities or an automatically
  controlled false-discovery rate. A threshold of 5 means five units of this
  calibrated score, not proof of an astrophysical five-sigma discovery.
- Fluxes are approximate PSF-fit amplitudes at peak pixels, without aperture,
  subpixel, or absolute photometric corrections. Conservative morphology cuts
  can reject real faint or blended sources.
- Artifact cuts do not establish astrophysical origin. Compare independent
  epochs and known-source information, and inspect rejected candidates before
  drawing scientific conclusions.

## 6. Command Reference

Run `--help` for the installed interface. Defaults below describe this version.

| Option | Default | Purpose |
| --- | --- | --- |
| `--ref`, `--reference` | Required for local single pairs | Reference FITS image; forbidden in survey mode. |
| `--new`, `--target` | Required for single images | New/target FITS image; survey lists use `--science-list` instead. |
| `--survey` | None (offline/local) | Opt in to `auto`, `panstarrs` or `skymapper` reference acquisition. |
| `--template-filter` | Required with `--survey` | Explicit supported survey template band; never silently substituted. |
| `--science-list` | None | Survey-only text paths or JSON string array; incompatible with `--new`. |
| `--science-filter` | FITS FILTER | Explicit science label for survey mode when missing/ambiguous. |
| `--template-cache` | `survey_cache` | Survey FITS/manifests and normalized copies, separate from results. |
| `--template-units` | Native BUNIT | Verified native `adu` or `electron` declaration for ambiguous template headers; no conversion. |
| `--template-gain` | Native GAIN | Template electrons per count, independently of science gain. |
| `--survey-timeout` | `30` | Positive finite HTTPS timeout in seconds per response. |
| `--max-download-mb` | `64` | Survey FITS transfer limit per product, above zero and at most 512 MB. |
| `--batch` | None | JSON array of ref/new jobs; mutually exclusive with single-pair inputs. |
| `--fail-fast` | Off | Batch only: stop at the first processing failure and mark remaining pairs skipped. |
| `--output`, `-o` | `results` | Output directory, relative to the shell's working directory unless absolute. |
| `--check-only` | Off | Validate inputs/configuration and print a read-only JSON report; no subtraction or output files. |
| `--threshold`, `-t` | `5` | Positive candidate threshold, applied to both score signs. |
| `--method` | `kernel` | `kernel` or `proper`. |
| `--cross-filter` | Off | Experimental kernel-only comparison; permits differing FILTER values, not differing units. Color differences can mimic variability. |
| `--max-scale-scatter` | `0.15` | Maximum clipped fractional high-SNR stellar flux-ratio scatter; does not model color uncertainty. |
| `--color-association-fwhm` | `1` | Cross-filter reference association radius in effective FWHM. |
| `--color-reference-snr` | `5` | Cross-filter reference detection/aperture SNR threshold. |
| `--color-quantile` | `0.95` | Cross-filter absolute calibration-star color-residual quantile. |
| `--color-error-margin` | `3` | Cross-filter color-range error margin multiplier. |
| `--align` | Off | Register new onto reference before subtraction. |
| `--reproject` | Off | Reproject reference onto science celestial WCS; kernel method and automatic PSFs only; incompatible with `--align`/`--shift`. |
| `--reproject-units` | `auto` | Infer recognized BUNIT, or explicitly select `flux-per-pixel` / `surface-brightness`; no unit conversion. |
| `--shift` | Off | Alias for validated registration, not a free untracked subpixel correction. |
| `--alignment-tolerance` | `0.35` | Maximum matched-star registration RMS in pixels. |
| `--min-stars` | `6` | Minimum suitable matched/PSF stars; cannot be below six. |
| `--gain` | FITS GAIN, else `1` | Electrons per input count; both images for local pairs, science only for survey workflow. |
| `--saturation` | FITS SATURATE | Saturation level in input counts; overrides both images. |
| `--flux-scale` | Estimated | Known reference multiplier; kernel backend only. |
| `--no-beta` | Off | Disable photometric scale estimation. Kernel scale is one unless manually supplied. |
| `--iterative` | Off | More ratio-clipping iterations for kernel; iterative estimation for ProperImage. |
| `--psf-radius` | `8` | Input PSF kernel radius in pixels; support is `(2r+1)` squared. |
| `--ref-psf`, `--new-psf` | Estimated Gaussian PSFs | Both supplied PSFs are required together; kernel backend only. |
| `--ref-mask`, `--new-mask` | None | Additional same-shape input quality masks. |
| `--no-cosmic-rays` | Off | Disable the pipeline's preprocessing L.A.Cosmic step. ProperImage can still run its own internal cleaning. |
| `--min-pixels` | `3` | Minimum connected detection area. |
| `--min-fwhm-ratio` | `0.6` | Minimum measured-to-expected PSF-width ratio. |
| `--max-fwhm-ratio` | `1.8` | Maximum measured-to-expected PSF-width ratio. |
| `--max-elongation` | `2` | Maximum major/minor shape ratio. |
| `--min-psf-correlation` | `0.65` | Minimum correlation with the difference PSF. |
| `--max-dipole-ratio` | `0.5` | Maximum significant opposite-sign neighbor/peak ratio. |
| `--max-candidates` | `1000` | Fail rather than silently truncate an excessive candidate list. |
| `--no-plots` | Off | Skip diagnostic PNGs. |
| `--overwrite` | Off | Allow replacing existing single-pair products, never input data; batch roots must still be new/empty. |
| `--verbose` | Off | Include an exception traceback on failure. |
| `--estimate-psf` | Off | Compatibility flag; both PSFs are always estimated or loaded and saved. |
| `--inf-loss` | `0.15` | Retained ProperImage setting; not a tuning control for Gaussian PSF width. |
| `--smooth-psf` | Off | Retained backend setting; does not enable spatial or empirical PSF matching. |
| `--auto-psf` | Disabled | Rejected explicitly; use Gaussian or supplied PSFs. |

Do not relax registration or morphology thresholds merely to suppress an error.
Inspect the images and use injection/recovery tests to select instrument-specific
settings.

## 7. Output Products

| Product | Interpretation |
| --- | --- |
| `batch_summary.json` | Batch root only: atomic ordered job states, counts, errors, paths, and timestamps; each job's normal products live in its named subdirectory. |
| `survey_reference.json` | Survey run/job root: acquisition/processing state, template request, native provenance and warnings. |
| `survey_diagnostic.png` | Survey run/job root: final unconfirmed-candidate review sheet or explicit failure sheet; normal products are under `analysis/`. |
| `detected_sources.csv` | Accepted, unconfirmed candidates sorted by absolute score. |
| `rejected_sources.csv` | Rejected candidates and semicolon-separated rejection flags. |
| `statistics.csv` | Scalar run parameters and quality metrics. |
| `provenance.json` | Input basenames, sizes, modification times, SHA-256 fingerprints (including supplied masks/PSFs), DATE-OBS/MJD-OBS when supplied, configuration, registration/photometric metrics, timings, Python/pipeline/dependency versions. |
| `run_status.json` | Atomic running/completed/failed status with stage, UTC timestamps, elapsed time, timings, and failure details. |
| `pipeline.log` | Persistent per-run progress, stage durations, warnings, and errors; handlers close after each run. |
| `quality_summary.json` | Diagnostic status, measured metrics, configured limits, caveats, candidate groups, rejection counts, and timings; not a scientific certification. |
| `rejection_summary.csv` | Per-reason rejection counts; multi-flag sources count in multiple rows. |
| `candidate_background.csv` | Candidate ID, valid annulus pixel count, local background/gradient/structure ratios, and non-gating warning. |
| `difference_image.fits` | Signed new-minus-reference residual image. |
| `significance_image.fits` | Empirically calibrated score used for detection. |
| `raw_significance_image.fits` | Score before empirical calibration. |
| `flux_image.fits` | Approximate PSF-fit amplitude map. |
| `quality_mask.fits` | `1` means unusable; `0` means eligible for analysis. |
| `reference_psf.fits`, `new_psf.fits` | Normalized PSF stamps for the two inputs. |
| `difference_psf.fits` | Difference-image PSF, not a science exposure. |
| `matched_reference.fits`, `matched_new.fits` | Common-PSF science images; kernel method only. |
| `variance.fits`, `flux_error.fits` | Propagated difference variance and PSF-amplitude uncertainty; kernel method only. |
| `reference_matching_kernel.fits` | New-image PSF applied to reference; kernel method only. |
| `new_matching_kernel.fits` | Reference PSF applied to new image; kernel method only. |
| `diagnostics.png` | Inputs, difference, score, PSF, and excluded pixels. |
| `candidate_stamps.png` | Up to twelve triplets: kernel-matched reference/science share one intensity scale; signed difference has a symmetric scale. ProperImage uses the original reference/science stamps. |
| `reprojection_footprint.fits` | WCS mode: binary valid-reference interpolation support, before downstream mask expansion. |
| `reprojected_reference.fits` | WCS mode: background-subtracted reference on the science grid, before stellar photometric scaling. |
| `reprojected_reference_variance.fits` | WCS mode: exact diagonal variance from squared interpolation weights, assuming independent native reference noise. |
| `reference_covariance_bound.fits` | WCS mode: covariance row-sum bound used for subsequent filtering, not the resampled diagonal variance. |
| `astrometric_residuals.csv` | WCS mode: matched-star output positions, X/Y offsets, and radial registration errors. |

PNG products are omitted with `--no-plots`; candidate triplets are also omitted
when no candidates are accepted. Empty catalogs still contain column headers.
Cross-filter diagnostic legends distinguish less-color-confounded circles,
color-ambiguous diamonds, and rejected crosses; stamp titles name each group.
Masked science pixels are NaN. Image products retain reference-grid celestial
WCS by default, or **science-grid celestial WCS with `--reproject`**, including
SIP, plus available new-exposure metadata. PSF/kernel stamps have no sky WCS.
WCS-mode `diagnostics.png` adds coverage, residual vectors, and a residual
histogram to the usual six panels. In WCS mode, `variance.fits` and
`flux_error.fits` use the conservative filtered covariance bound.

### Catalog Columns

| Columns | Meaning |
| --- | --- |
| `id` | Detection identifier, not the sorted rank. |
| `x_centroid`, `y_centroid` | Zero-based output-grid positions: reference grid normally, science grid with `--reproject`; NumPy indexes data as `[y, x]`. |
| `sign` | `+1` brightening or `-1` fading. |
| `flux`, `flux_error` | Signed approximate amplitude and model uncertainty; ProperImage uncertainty is unavailable. |
| `peak_value`, `snr` | Signed difference peak and signed detection score. The `snr` label is not a discovery guarantee. |
| `npixels` | Connected score-detection area. |
| `fwhm`, `fwhm_ratio`, `elongation` | Morphological shape diagnostics. |
| `psf_correlation`, `dipole_ratio`, `reduced_chi2` | PSF similarity, opposite-sign residual strength, and fit diagnostic. |
| `ra_deg`, `dec_deg` | ICRS sky position in degrees when celestial WCS is available; otherwise empty. |
| `flags` | Empty for accepted candidates; rejection reasons for rejected candidates. |

Possible flags include `mask_neighborhood`, `too_few_pixels`, `psf_width`,
`elongated`, `psf_mismatch`, `dipole`, `nonpositive_psf_fit`, and
`blended_or_truncated`. Review flagged sources rather than discarding that
catalog: real sources can fail conservative cuts.

## 8. Tests And Validation Commands

Use the astronomy-only suite for focused validation. Its seeded synthetic
images and output products are created under pytest's temporary directories;
the tests do not require your observing data or network access.

```powershell
& $Python -m pytest -c $Configuration $AstroTests -q
& (Join-Path $PipelineRoot "scripts\test.ps1") -PythonPath $Python
& (Join-Path $PipelineRoot "scripts\lint.ps1") -PythonPath $Python
```

To see individual test names or investigate one behavior:

```powershell
& $Python -m pytest -c $Configuration $AstroTests -v
& $Python -m pytest -c $Configuration $AstroTests -q -k registration
& $Python -m pytest -c $Configuration $AstroTests -q -k "cosmic or dipole"
& $Python -m pytest -c $Configuration $AstroTests -q -k properimage
```

The current suite expands to 182 test cases, including parameterized invalid
configuration and preflight checks. It covers:

- Score calibration, preservation of bright sources, and invalid-pixel masks.
- Rejection of significant imaginary FFT components.
- Cancellation of unchanged stars despite unequal PSFs and photometric scales.
- Signed flux recovery and positive/negative transient detection.
- Real ProperImage Gaussian PSF estimation and transparency checks.
- Rejection of misregistration and recovery of a known injected translation.
- Cosmic rays, saturation, NaNs, and opposite-sign dipole artifacts.
- Survival of an injected real transient during artifact rejection.
- Kernel-induced noise covariance in the matched-filter statistic.
- The ProperImage backend, its signed candidate catalog, and unavailable flux errors.
- Supplied PSFs and saved matching kernels.
- WCS sky coordinates and reference-grid output headers.
- Empty catalog headers, overwrite protection, and invalid configurations.
- Explicit rejection of empirical mode and count-rate data without variance.
- A complete survey-style comparison with injected truth and a saved validation overview.
- WCS pixel-area scaling, surface brightness, squared-weight variance, and bad-pixel support.
- Monte Carlo verification of correlated filtering noise versus its conservative bound.
- Alternate projections, SIP output preservation, invalid WCS/overlap/area, and unsafe option combinations.
- End-to-end subtraction of a rotated, finer-sampled, SIP-distorted reference with partial coverage.
- Explicit cross-filter opt-in, required filter labels, unit checks, and kernel-only configuration.
- Colored static/transient galaxy scenes, configurable scale scatter, warning-labeled products, and flux-bias reports.
- Cross-filter CLI, WCS reprojection, empty catalogs, and unchanged ordinary catalog columns.
- Source-aware cross-filter grouping, static false-alert reduction, retained isolated/nuclear injections, manual calibration, masks, and signed aperture ratios.
- Configurable color-vetting thresholds with unchanged detection measurements.
- Read-only preflight, invalid detection thresholds, completed/failed status, fingerprints, epochs, timings, and log-handler cleanup.
- Quality/rejection summaries, non-gating local-background warnings, distinct plot groups, and shared stamp intensity scales.
- Public configuration objects, structured in-memory/persisted API results, propagated exceptions, explicit cross-filter opt-in, and installed distribution metadata/typing/entry points.
- Sequential batches with unchanged single-pair measurements, per-pair outputs, no-write analysis/preflight, partial failures, fail-fast skips, interrupts, memory-retention policy, config overrides, manifest-relative paths, unsafe names/inputs/outputs, malformed manifests, CLI exit codes, and summary I/O errors.

The suite skips when required astronomy dependencies are absent.
A skipped suite is **not** scientific validation; install this project's dependencies
and confirm the tests execute and pass.

### Survey-Style End-to-End Validation

`test_realistic_survey_comparison_end_to_end` runs the full pipeline, including
FITS loading, background estimation, cosmic-ray masking, registration, automatic
Gaussian PSF estimation, photometric scaling, subtraction, scoring, candidate
vetting, catalogs, FITS products, and production diagnostic plots. No PSF or
photometric scale is supplied to the pipeline. It uses the default PSF radius
of 8 pixels, with full kernel and matched-filter support masked.

The fixed-seed, 384 x 384 scene contains a sparse field of subpixel-positioned
stars and two positive transients present only in the science exposure.
T1 transient is isolated; T2 coincides with a persistent bright nucleus in an
elliptical Sersic host. A second unchanged elliptical galaxy is also included.
The template is sharper and deeper: injected Gaussian PSF sigmas are 1.25 and 1.95
pixels, with a science/template flux ratio of 0.72 and a science-image offset
of (-0.7, +1.1) pixels in (X, Y). Each exposure has a different sky gradient,
gain, read noise, and independently sampled Poisson noise. The FITS files
include celestial WCS, SCI, total VAR, and DQ planes. Cosmic rays, saturated
patches, flagged bad columns, nonfinite pixels, and DQ-flagged hot/dead pixels
occur at different positions in the two exposures, for 20 injected artifacts.
The template also contains 6 ADU RMS residual noise, included in its VAR plane.

To retain the results in a predictable location, use the variables from section
2 and run:

```powershell
$ValidationRun = Join-Path $PipelineRoot "artifacts\survey-validation"
New-Item -ItemType Directory -Path (Split-Path -Parent $ValidationRun) -Force | Out-Null
& $Python -m pytest -c $Configuration $AstroTests -q -k realistic_survey_comparison --basetemp $ValidationRun
Get-ChildItem -LiteralPath $ValidationRun -Recurse -Filter validation_overview.png
```

**Pytest clears the entire `--basetemp` directory when the run starts.** Reserve
this directory for disposable synthetic results, never observing data. Normal
test runs without `--basetemp` use pytest's temporary directories instead.
Generated artifacts are ignored by Git.

Products below are relative to the test's directory inside `$ValidationRun`:

| Product | Contents |
| --- | --- |
| `inputs/reference.fits`, `inputs/science.fits` | Reproducible mock observations with independent artifacts and noise. |
| `inputs/truth.json` | Seed, star/transient fluxes and coordinates, artifact positions, PSFs, shift, scale, and detector/noise parameters. |
| `results/validation_overview.png` | A 16-panel Matplotlib overview of inputs, truth, PSF profiles, matched images, difference, score, mask, blank-sky histogram, recovery table, metrics, and transient close-ups. |
| `results/validation_summary.json` | All validation gates, completeness/purity for this injection set, false positives, per-transient flux/error/score/position/PSF-fit metrics, and full pipeline metadata. |
| `results/detected_sources.csv`, `results/rejected_sources.csv` | All accepted and rejected candidates, including their scores and quality diagnostics. |
| `results/statistics.csv`, `results/provenance.json` | Processing statistics, settings, registration transform, PSF widths, photometric scaling, score calibration, and dependency versions. |
| Other `results/*.fits` and `results/*.png` | Standard difference, raw/calibrated significance, flux/error, variance, mask, PSFs, matching kernels, matched images, and production plots. |

The summary and overview are saved before the quality assertions, so failed
quality gates remain inspectable if processing itself completed. The 13 gates
require both transients (including the core) and no extra accepted detections, all 20 artifacts
masked with no nearby accepted candidate, centroids within one pixel, fluxes
within 20%, scores above 5, scale within 5%, registration RMS below 0.2 pixels,
PSF FWHMs within 15%, blank-sky score median within 0.25 of zero and scatter
between 0.6 and 1.4, and more than half the image usable after masking.

This is one sparse, Gaussian-PSF synthetic regression, not a measured survey
completeness or false-alarm rate. Its two fixed extended hosts do not establish
performance across galaxy populations, crowded fields, spatially varying or
non-Gaussian PSFs, or general WCS reprojection.
The displayed empirical score is not a discovery probability; flux errors do
not include every calibration or resampling systematic.

For amateur/professional exposures compared with Pan-STARRS or SkyMapper, use a
calibrated **image cutout**, not a source-catalog table. Prepare compatible
filters and flux units, total variance and bad-pixel masks, overlapping sky
coverage, and compatible pixel sampling first. The tested alignment handles a
small image-plane transform; use the opt-in WCS mode below when the grids
differ. Validate correlated input noise, passband/color terms, PSF wings,
registration, and source recovery on real exposures separately. Different filters are not
made scientifically interchangeable by a single photometric scale.

### Distorted-WCS End-to-End Validation

`test_wcs_reprojection_distorted_survey_end_to_end` keeps the mock science
exposure and independently renders the unchanged reference stars at their sky
positions in a 480 x 500 detector. Its 0.65 arcsec pixels, 12-degree rotation,
different pointing, and nonlinear SIP distortion differ from the 384 x 384
science detector's 0.8 arcsec pixels. No transient is rendered in the reference.
The two exposures have independent noise, PSFs, and 20 distinct artifacts.
The test uses `--reproject` semantics, with no supplied PSF or scale.

```powershell
$WcsRun = Join-Path $PipelineRoot "artifacts\wcs-validation"
New-Item -ItemType Directory -Path (Split-Path -Parent $WcsRun) -Force | Out-Null
& $Python -m pytest -c $Configuration $AstroTests -q -k distorted_survey --basetemp $WcsRun
Get-ChildItem -LiteralPath $WcsRun -Recurse -Filter diagnostics.png
Get-ChildItem -LiteralPath $WcsRun -Recurse -Filter wcs_validation_summary.json
```

As above, **pytest clears the entire chosen `--basetemp` directory**. The test
saves input FITS and sky/WCS truth, all standard WCS-mode products and plots,
and `results/wcs_validation_summary.json`. Its assertions check both transient
recoveries without extra accepted candidates, fluxes within 20%, sky positions
within one arcsecond, all 20 artifacts masked, scale within 5%, registration
RMS below 0.2 pixels, excluded incomplete coverage, and correct output WCS and
dimensions. Separate unit tests check TAN/SIN/CAR mappings, SIP serialization,
pixel areas, masks, and invalid input handling. The Monte Carlo test demonstrates
that ignoring interpolation covariance underestimates filtered noise and that
the propagated bound covers it for the simulated independent-noise case.

These are synthetic checks for the stated cases, not certification of every
projection, distortion, real survey coadd, or PSF. Supplied total variance does
not describe all covariance in an already resampled survey reference.

### Galaxy Completeness Campaign

Run the separate experiment with the project environment and dev dependencies
(it reuses the synthetic generators in the tests). It does not add pytest cases
or change production detection thresholds:

```powershell
$Campaign = Join-Path $PipelineRoot "experiments\completeness.py"
$CampaignResults = Join-Path $PipelineRoot "artifacts\galaxy-completeness"
& $Python $Campaign --output $CampaignResults --workers 2
```

The default campaign processes 592 independently generated image pairs:
six transient fluxes (250, 500, 1000, 2000, 4000, 8000 science ADU), six
elliptical radii (0, 0.25, 0.5, 1, 2, 3 effective radii), eight trials per bin,
and eight host-only controls, for each of the alignment and WCS workflows.
Off-center trials sample evenly spaced angles with a seeded angular phase.
The star field and host are fixed; each trial has fresh detector noise and
one science-only transient, including its shot noise in VAR. Workflows share
science noise seeds and injection positions, so they are paired comparisons,
not extra independent trials. Both workflows use PSF radius 10 to accommodate
noise-dependent Gaussian PSF fits; all detection and morphology cuts remain
at their production defaults. FITS pairs are temporary, not archived.

Accepted positive candidates within one output-grid pixel count as recovery;
photometric accuracy does not determine detection completeness. Host-only
controls have no injected transients. The results directory must not exist:
the experiment never deletes or overwrites observations or earlier results.
Run a short pilot with a different results directory first:

```powershell
& $Python $Campaign --output (Join-Path $PipelineRoot "artifacts\completeness-pilot") --fluxes 500 7000 --radii 0 1 --trials 2 --null-trials 2
```

| Product | Contents |
| --- | --- |
| `configuration.json` | Seeds, host properties, grid, matching/settings, package versions. |
| `trials.csv` | Every attempted injection/control, including masks, errors, rejections, flux bias, score, position, and unmatched accepted detections. Written incrementally. |
| `completeness_bins.csv` | All-injection and usable-footprint completeness, 95% Wilson intervals, mask losses, pipeline errors, and conditional flux/position bias by workflow, brightness, and radius. |
| `surface_brightness_bins.csv` | Recovery versus noise-free native science host+nucleus brightness in ADU/pixel, before sky. |
| `null_controls.csv` | Positive/negative accepted detections and detections per successfully processed host-only image. |
| `summary.json` | Aggregate results and interpretation limits. |
| `completeness.png` | Brightness/radius heatmaps and local-host-brightness recovery curves. |

All-injection completeness retains masked trials and pipeline errors in the
denominator. Usable-footprint completeness excludes failed processing and
truth-centered PSF-fit stamps intersecting a mask; its usable count is shown
separately. A nonzero runner exit status indicates pipeline errors, not low
completeness. The CSV retains those errors rather than silently dropping them.

Eight trials per bin give broad intervals: even 8/8 recovery has a Wilson
95% lower bound of about 68%. Zero detections in a few null controls does not
establish a zero false-alarm rate. These are exploratory conditional results
for one host and stellar field, not a population survey selection function.
Increase `--trials` and `--null-trials` for precision before extending the host,
seeing, or registration distributions. Flux bias is measured only among
recoveries and can be selection-biased near the threshold.

### Medium FITS Robustness And Timing

`experiments/robustness.py` is an opt-in performance/robustness experiment, not part
of routine pytest. Its small-scene generator/association/aggregation checks are in
the regression suite. It adds no dependencies and does not change scientific cuts.

From `$PipelineRoot`, run the ordinary automatic-PSF benchmark:

```powershell
& $Python experiments/robustness.py --output artifacts/medium-auto --sizes-mb 20 50 --trials 2
```

For a separately labeled control using the **known injection-model Gaussian PSFs**:

```powershell
& $Python experiments/robustness.py --output artifacts/medium-known --sizes-mb 20 50 --trials 2 --known-psf
```

Do not interpret the known-PSF control as a fix or validation of automatic PSF
estimation. A new results directory is required. Existing results are never erased.
Default runtime settings are threshold 5, kernel method, radius 8, cosmic-ray
cleaning enabled, no WCS reprojection and no production plots. The reference and
science share an accurate TAN grid and `r` band. This is not a cross-filter test.

#### Size, Scenes And Truth

MB means 1,000,000 bytes for the **whole uncompressed FITS**, not just the science
array. Every input contains float32 science, float32 total variance and uint8 mask.
The 20 MB input is 1490 x 1490 pixels (19,992,960 bytes); the 50 MB input is
2356 x 2356 pixels (49,970,880 bytes). One analysis reads a pair of these files.
Thus a 20 MB single-plane image with more pixels is a different workload.

The scene has unsaturated calibration stars, a compact Sersic host, different
Gaussian seeing (sigma 1.3/1.8 pixels), reference/science flux ratio 0.8, independent
Poisson noise, read noise and explicit gain/variance. Subpixel points use bilinear
deposition before convolution. This is a controlled scene, not a fully realistic
optical PSF, detector or astrophysical population model.

| Scenario | Added science contamination |
| --- | --- |
| `clean` | No added detector/trail defects. |
| `detector` | Sparse bright cosmic/hot pixels, compact cosmic cluster, saturated bleed, masked bad column and nonfinite patch. |
| `satellite` | Slanted bright narrow trail; no externally supplied trail mask. Normal cosmic cleaning remains enabled. |
| `satellite_masked` | Same trail with an explicit input mask; no added detector defects. |
| `mixed_masked` | Detector defects plus explicitly masked satellite trail. |

Each scenario has a no-transient control and five truth-labeled injections:
bright/faint isolated positive sources, a positive source on a galaxy, a fading
source (negative signed difference), and a positive source crossing the trail.
Two noise trials, five scenarios, two truth modes and two sizes give **40 analyses**.
Seeds are shared across scenarios, so these are paired comparisons, not 40 fully
independent sky fields. Controls are neither real events nor catalog-confirmed objects.

Accepted candidates are matched one-to-one within two science pixels with the
correct residual sign. Unmatched accepted sources are counted as false positives.
Recovery is reported both for all injections and for usable truth-centered
neighborhoods; the latter excludes output-mask neighborhoods with half-width
`min(PSF array half-width, 8)`. Masked injections stay in the overall denominator.
Failed processing is retained separately and is never counted as a successful
timing or omitted from the injection denominator. Flux bias is conditional on
recovery; mask losses are not successes even when rejection is protective.

#### Timing, Memory And Outputs

Each case runs sequentially in a fresh subprocess with common OpenMP/BLAS thread
limits set to one. The script records wall time for the complete `compare_images`
call, CPU time, normal per-stage timings and a separate whole-worker wall duration.
Analysis time includes input hashes/reading, backgrounds, cosmic cleaning, PSF
handling, matching, subtraction, detection and normal FITS/catalog output writes.
Synthetic generation, interpreter/import startup and the experiment's preview PNG
are excluded from the analysis timer. Production plotting is excluded by default.

Peak process RSS is measured with Windows process counters (or Unix `ru_maxrss`).
It includes imports, generation, analysis and preview allocation; it is **not**
incremental analysis memory, private committed memory or system-wide memory usage.
Temporary FITS inputs/products are removed after the isolated case, but the measured
analysis still writes its normal products to disk. Disk caching and other activity
on a shared workstation can materially change results.

| Option | Meaning |
| --- | --- |
| `--sizes-mb 20 50` | Approximate per-file sizes, bounded to 1-100 MB. |
| `--scenarios clean satellite satellite_masked` | Restrict scenes; all five by default. |
| `--trials 2` | Noise trials per scene/size, from 1 to 20; not a confidence guarantee. |
| `--timeout 600` | Whole-worker per-case timeout, including generation/imports/preview. |
| `--known-psf` | Supply injection-model PSFs; isolates subtraction/vetting rather than estimating PSFs. |
| `--plots` | Include ordinary pipeline PNG generation in the analysis timer. |
| `--reproject` | Include WCS reprojection; incompatible with supplied `--known-psf`. |
| `--keep-inputs`, `--keep-products` | Retain large FITS inputs or normal products; allow several GB of disk space. |

The output contains `benchmark.json`, `runs.csv`, `summary.csv`, `injections.csv`
and `REPORT.md`, updated after each finished case. Each case retains truth, worker
logs, normal JSON/CSV/log reports and a first-trial injection `overview.png`.
No dedicated satellite classifier was added: the experiment measures the effects
of existing masking/vetting, including any loss of a real point source on a trail.
The command returns 1 for a worker/pipeline failure or timeout, not merely because
an injection was missed or a false positive occurred. Incomplete/failed-case
timings are kept separately from successful-case medians.

#### Measured Results: 2026-11-02

Machine: Windows 11, Intel64 Family 6 Model 186 Stepping 2, 20 logical processors;
Python 3.13.14, NumPy 2.5.1, SciPy 1.18.1, Astropy 8.0.1, Astroscrappy 1.3.0.
One worker at a time, one thread limit per native backend, plots/reprojection off.

**Automatic PSF pilot (2026-11-02) and fix (2026-11-03):** the pilot failed before
detection because the estimated PSF exceeded radius 8 (22.4-24.7 s at 20 MB,
61.8-64.3 s at 50 MB). The cause was ProperImage's strict star selection: it logs
`only had N valid sources (< 15). Keeping all M sources` and then fits the
Gaussian to every detected source, so the Sersic host, blends and noise peaks
widened the fit (sigma 3.76 at radius 8, 323 at radius 10). Raising
`--psf-radius` alone made it worse, because fewer stars pass the cuts at larger
stamps.

`ProperTransientFinder._estimate_psf` now seeds ProperImage's source cache with
the pipeline's isolated SEP stars (already round, unmasked and unblended) and
lets ProperImage perform the stamp extraction and Gaussian fit. Re-checked clean
pairs at the default radius 8 with automatic PSFs:

| FITS per input | Analysis | Reference / science FWHM | Injected FWHM | Recoveries |
| --- | --- | --- | --- | --- |
| ~20 MB | 25.5 s | 3.19 / 4.30 px | 3.06 / 4.24 px | 3 positives + 1 fading; 1,800 ADU source missed |
| ~50 MB | 65.5 s | 3.19 / 4.34 px | 3.06 / 4.24 px | all 5 injections |

Both runs used `PipelineConfig(plots=False)` on one clean scene and one noise
seed, so this is not a rerun of the 40-case campaign; the detector, trail and
masked scenes still need their own automatic-PSF pass. The support guard remains
in force: a genuinely broad PSF must raise `--psf-radius` rather than be
cropped, and supplied measured PSFs remain the preferred option for real data.

**Known-PSF control:** all 40 cases completed. Median/range across all 20 cases
at each size (null and injected scenes mixed):

| FITS per input | Median analysis | Observed range | Whole-worker peak RSS |
| --- | --- | --- | --- |
| ~20 MB | 12.0 s/pair | 10.5-17.7 s | ~776 MB |
| ~50 MB | 28.3 s/pair | 14.2-36.7 s | ~1666 MB |

The wide 50 MB range includes substantial within-scenario variation; it is not
evidence that a masked trail makes analysis intrinsically faster. There was no
dedicated quiet-machine, cold-cache or randomized-order timing study. Two trials
are insufficient for p95/p99 latency claims or stable population completeness.

Recovery totals per size (two injection trials, five sources each):

| Scenario | Recovered / all | Recovered / usable | False positives: 20/50 MB |
| --- | --- | --- | --- |
| `clean` | 10/10 | 10/10 | 0 / 1 |
| `detector` | 7/10 | 7/7 | 0 / 1 |
| `satellite` | 8/10 | 8/8 | 0 / 1 |
| `satellite_masked` | 8/10 | 8/8 | 0 / 1 |
| `mixed_masked` | 5/10 | 5/5 | 0 / 1 |

The false-positive column above refers to the two **injected** cases per size/scene.
Null controls had the same counts: none at 20 MB, one recurrent accepted residual
in the first noise seed of each 50 MB scenario. Those repeated detections are not
independent false alarms on five different fields. All lost injections intersected
excluded neighborhoods, including cosmic/hot pixels and the trail; at least one
50 MB host injection was rejected with `mask_neighborhood` even though its center
pixel was not masked. Thus usable-only recovery does not establish total completeness.

Retained results from this run are under `artifacts/medium20-known-20261102/`,
`artifacts/medium50-known-20261102/` and `artifacts/medium-auto-20261102/`.
These generated artifacts are not distributed as source or scientific observations.

**Observed real transients remain untested.** To validate those, provide calibrated
reference/science FITS pairs with permission to use them, event identifiers or a
published truth source, positions, epochs and expected sign. Supply independent
PSFs only when measured/validated for those exposures, not from this simulator.
Apply the real-data acceptance checklist below; injected sources are not a substitute
for confirmed observations, survey completeness or validated astrophysical errors.

### Astronomy-Only Lint And Formatting

```powershell
& (Join-Path $PipelineRoot "scripts\lint.ps1")
& $Python -m ruff check --config $Configuration $AstroSource $AstroTests $PipelineScript
& $Python -m ruff format --check --config $Configuration $AstroSource $AstroTests $PipelineScript
```

The project lint script checks source, tests, experiments, Python tooling, and
the launcher. To deliberately apply formatting, run the second direct command
without `--check`.

### Real-Data Acceptance Checklist

1. Verify detector calibration, filter, units, gain, variance, and saturation.
2. Inspect registration and photometric metrics in the provenance file.
3. Check PSF stamps, matched images, residuals around unchanged stars, and masks.
4. Inspect both accepted and rejected candidate triplets at their sky positions.
5. Inject sources at multiple fluxes and field positions; measure recovery and
   false positives on exposures with no expected new sources.
6. Confirm plausible candidates using independent epochs and appropriate
   catalogs before reporting an astrophysical transient.

Synthetic regression success does not establish survey completeness, a
false-positive rate, or calibrated scientific uncertainties for your instrument.

## 9. Troubleshooting

| Symptom | Action |
| --- | --- |
| tinynpydb build downloads old setuptools and gets HTTP 403 | Apply the compatible-setuptools/no-build-isolation steps in section 2. |
| Astronomy package or dependency is missing | Run `scripts/bootstrap.ps1` (Windows) or `bash scripts/tasks.sh bootstrap` (POSIX) from the project root, then use that environment's interpreter: `.venv\Scripts\python.exe` or `.venv/bin/python`. |
| `.fz` Rice decompression fails or HDU headers are malformed | Verify the original binary download and provider checksum. Renaming extensions or changing detection thresholds cannot repair altered compressed bytes. |
| Too few isolated/matched/PSF stars | Check field overlap, masks, saturation, image quality, and sampling; do not invent a PSF from insufficient stars. |
| Registration RMS exceeds tolerance | Use `--align`, inspect star matches, and verify that both files cover the same field. |
| WCS reprojection has too few matched stars or excessive RMS | Check both astrometric WCS solutions, overlap, and masks; correct the WCS upstream, not by adding `--align` to `--reproject`. |
| Ambiguous BUNIT for reprojection | Choose the correct explicit `--reproject-units` interpretation; supply compatible units and total variance. |
| Invalid celestial WCS or insufficient WCS overlap | Use calibrated 2D image cutouts with accurate celestial WCS and enough common unmasked stars. |
| Inconsistent flux ratios | Check filter, calibration, variable stars, alignment, PSF quality, and spatial transparency changes. |
| PSF extends beyond kernel | Increase `--psf-radius` for a valid broad/supplied PSF and inspect its normalization and wings. |
| Non-count units require VAR/ERR | Supply total variance in the actual image units or convert to calibrated detector counts. |
| Many rejected candidates or candidate limit exceeded | Inspect dipoles, saturated stars, cosmic rays, background structure, and PSF mismatch before changing thresholds. |
| Large masked regions or too few valid score pixels | Inspect bad-pixel/saturation masks and convolution support; large kernels can leave little usable area. |
| Output directory not empty | Choose a new run directory or deliberately use `--overwrite`. |
| `--auto-psf` rejected | Empirical mode is intentionally unavailable; use Gaussian or supplied PSFs. |
| No accepted candidates | Inspect the score, masks, and rejected catalog; success does not require detections. |

Use `--verbose` for a traceback when diagnosing a failure. Review logs,
provenance, file paths, and FITS metadata before sharing them outside approved
channels.

## 10. References

- [ProperImage 0.7.2 documentation](https://properimage.readthedocs.io/en/latest/)
- [ProperImage source and documentation sources](https://github.com/quatrope/ProperImage)
- [Astropy convolution](https://docs.astropy.org/en/stable/convolution/)
- [Astropy robust statistics](https://docs.astropy.org/en/stable/stats/robust.html)
- [Astropy FITS](https://docs.astropy.org/en/stable/io/fits/)
- [Astropy WCS interface](https://docs.astropy.org/en/stable/wcs/wcsapi.html)

During implementation, the hosted documentation returned HTTP 403 in some
dev environments. Official documentation sources on GitHub and installed library
implementations were used to verify API and noise behavior.
