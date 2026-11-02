#!/usr/bin/env bash
# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.
#
# POSIX shell equivalent of the project's PowerShell scripts.
#
# Usage: [PYTHON=python3] bash scripts/tasks.sh <task> [options]
#
#   bootstrap        Install the project and its development tools
#   lint             Ruff check and formatting check
#   test [args...]   Run the astronomy regression suite; args go to pytest
#   docs [--check] [--build | --serve [--address HOST:PORT]]
#                    Generate the API reference and optionally the MkDocs site
#   build [--no-isolation] [--include-dependencies] [--output-directory DIR]
#                    Build the wheel and source archive
#   verify-install --distribution PATH [--wheelhouse DIR] [--environment DIR]
#                    Install a built distribution into a fresh environment
#
# Environment: PYTHON selects the interpreter (default python3); OUTPUT_DIRECTORY
# and ADDRESS mirror the -OutputDirectory and -Address PowerShell parameters.

set -e

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
TARGETS=(src tests experiments scripts run_proper_transient_finder.py)

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

require_python() {
    command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable not found: $PYTHON"
    "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' ||
        die "The selected interpreter must be Python 3.11 or newer."
}

task_bootstrap() {
    require_python
    "$PYTHON" -m pip install "setuptools>=77,<81" wheel
    "$PYTHON" -m pip install --no-build-isolation tinynpydb==0.1.2
    (cd "$PROJECT_ROOT" && "$PYTHON" -m pip install --editable ".[dev]")
}

task_lint() {
    require_python
    cd "$PROJECT_ROOT"
    "$PYTHON" -m ruff check "${TARGETS[@]}"
    "$PYTHON" -m ruff format --check "${TARGETS[@]}"
}

task_test() {
    require_python
    "$PYTHON" -m pytest -c "$PROJECT_ROOT/pyproject.toml" "$PROJECT_ROOT/tests" "$@"
}

task_docs() {
    require_python
    check=0
    mode=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --check) check=1 ;;
            --build | --serve) mode=$1 ;;
            --address)
                shift
                ADDRESS=${1:-}
                ;;
            *) die "Unknown docs option: $1" ;;
        esac
        shift
    done
    cd "$PROJECT_ROOT"
    generator=(-I -S scripts/generate_docs.py)
    if [ "$check" -eq 1 ]; then
        generator+=(--check)
    fi
    "$PYTHON" "${generator[@]}"
    if [ "$mode" = "--build" ]; then
        "$PYTHON" -m mkdocs build --strict --config-file mkdocs.yml
    elif [ "$mode" = "--serve" ]; then
        "$PYTHON" -m mkdocs serve --config-file mkdocs.yml \
            --dev-addr "${ADDRESS:-127.0.0.1:8001}"
    fi
}

task_build() {
    require_python
    destination=${OUTPUT_DIRECTORY:-"$PROJECT_ROOT/dist"}
    isolation=1
    dependencies=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --no-isolation) isolation=0 ;;
            --include-dependencies) dependencies=1 ;;
            --output-directory)
                shift
                destination=${1:-}
                ;;
            *) die "Unknown build option: $1" ;;
        esac
        shift
    done
    [ -n "$destination" ] || die "--output-directory requires a value"
    case "$destination" in
        /*) ;;
        *) destination="$PWD/$destination" ;;
    esac
    arguments=(-m build --outdir "$destination" "$PROJECT_ROOT")
    if [ "$isolation" -eq 0 ]; then
        arguments+=(--no-isolation)
    fi
    "$PYTHON" "${arguments[@]}"
    if [ "$dependencies" -eq 1 ]; then
        wheelhouse="$destination/wheelhouse"
        "$PYTHON" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$wheelhouse" \
            "tinynpydb==0.1.2" "properimage==0.7.2"
        # Run from the project directory so the local path requirement stays
        # relative and portable across MSYS and native POSIX shells.
        (cd "$PROJECT_ROOT" && "$PYTHON" -m pip wheel --only-binary=:all: \
            --find-links "$wheelhouse" --wheel-dir "$wheelhouse" ".[dev]")
    fi
}

task_verify_install() {
    require_python
    distribution=""
    wheelhouse=""
    environment=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --distribution)
                shift
                distribution=${1:-}
                ;;
            --wheelhouse)
                shift
                wheelhouse=${1:-}
                ;;
            --environment)
                shift
                environment=${1:-}
                ;;
            *) die "Unknown verify-install option: $1" ;;
        esac
        shift
    done
    [ -n "$distribution" ] || die "verify-install requires --distribution PATH"
    [ -f "$distribution" ] || die "Distribution not found: $distribution"
    distribution="$(cd "$(dirname -- "$distribution")" && pwd)/$(basename -- "$distribution")"
    case "$distribution" in
        *.whl | *.tar.gz) ;;
        *) die "Choose a wheel (.whl) or source distribution (.tar.gz)." ;;
    esac
    index=()
    if [ -n "$wheelhouse" ]; then
        [ -d "$wheelhouse" ] || die "Wheelhouse directory not found: $wheelhouse"
        wheelhouse="$(cd "$wheelhouse" && pwd)"
        index=(--no-index --find-links "$wheelhouse")
    fi
    if [ -z "$environment" ]; then
        environment="${TMPDIR:-/tmp}/proper-transient-finder-install-$$"
    fi
    [ ! -e "$environment" ] ||
        die "Verification requires a new environment directory; refusing to reuse $environment"
    "$PYTHON" -m venv "$environment"
    if [ -x "$environment/bin/python" ]; then
        isolated="$environment/bin/python"
        entry_point="$environment/bin/proper-transient-finder"
    else
        isolated="$environment/Scripts/python.exe"
        entry_point="$environment/Scripts/proper-transient-finder.exe"
    fi
    printf 'Isolated environment: %s\n' "$environment"
    "$isolated" -I -m pip install --no-cache-dir ${index[@]+"${index[@]}"} "$distribution" ||
        die "Distribution installation failed. See USER_GUIDE.md for legacy dependency preparation."
    "$isolated" -I -m pip check || die "Installed dependency requirements are inconsistent."
    probe=$(cat <<'PROBE'
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
import proper_transient_finder as package

location = Path(package.__file__).resolve()
assert Path(sys.prefix).resolve() in location.parents, location
assert sys.prefix != sys.base_prefix
installed = distribution('proper-transient-finder')
assert installed.version == package.__version__
assert not json.loads(installed.read_text('direct_url.json') or '{}').get('dir_info', {}).get('editable', False)
assert (location.parent / 'py.typed').is_file()
assert package.ProperTransientFinder
assert package.PipelineConfig and package.PipelineResult and package.compare_images
print('Installed API:', package.__version__, location)
PROBE
)
    "$isolated" -I -c "$probe" ||
        die "Installed API import, metadata, typing, or isolation check failed."
    "$entry_point" --help >/dev/null || die "Installed console entry point failed."
    "$isolated" -I -m proper_transient_finder --help >/dev/null ||
        die "Installed module entry point failed."
    printf 'Runtime-only installation, API, and CLI verified before installing development tools.\n'
    dev_requirements=$("$PYTHON" -c '
import sys, tomllib
extra = tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["dev"]
print(" ".join(extra))
' "$PROJECT_ROOT/pyproject.toml") || die "Unable to read [dev] from pyproject.toml"
    # Word splitting is intended: each entry is one requirement specifier.
    "$isolated" -I -m pip install --no-cache-dir ${index[@]+"${index[@]}"} $dev_requirements ||
        die "Installation of optional development tools failed."
    "$isolated" -I -m pip check || die "Development dependency requirements are inconsistent."
    "$isolated" -I -m pytest -c "$PROJECT_ROOT/pyproject.toml" "$PROJECT_ROOT/tests" -q \
        -p no:cacheprovider --basetemp "$environment/regression-tests" ||
        die "Installed-distribution regression tests failed."
    printf 'Installation verified: non-editable package, dependency consistency, API, CLI, and regression suite.\n'
}

usage() {
    cat <<'USAGE'
Usage: [PYTHON=python3] bash scripts/tasks.sh <task> [options]

  bootstrap        Install the project and its development tools
  lint             Ruff check and formatting check
  test [args...]   Run the astronomy regression suite; args go to pytest
  docs [--check] [--build | --serve [--address HOST:PORT]]
                   Generate the API reference and optionally the MkDocs site
  build [--no-isolation] [--include-dependencies] [--output-directory DIR]
                   Build the wheel and source archive
  verify-install --distribution PATH [--wheelhouse DIR] [--environment DIR]
                   Install a built distribution into a fresh environment

Environment: PYTHON selects the interpreter (default python3);
OUTPUT_DIRECTORY and ADDRESS mirror -OutputDirectory and -Address.
USAGE
}

main() {
    [ "$#" -gt 0 ] || {
        usage
        exit 2
    }
    task=$1
    shift
    case "$task" in
        bootstrap) task_bootstrap "$@" ;;
        lint) task_lint "$@" ;;
        test) task_test "$@" ;;
        docs) task_docs "$@" ;;
        build) task_build "$@" ;;
        verify-install) task_verify_install "$@" ;;
        -h | --help | help) usage ;;
        *)
            printf 'Unknown task: %s\n' "$task" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
