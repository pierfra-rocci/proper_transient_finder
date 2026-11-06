# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Documentation drift checks for test counts, Python floors, and version claims."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "proper-transient-finder.yml"
VERSION_CLAIM = re.compile(r"Python 3\.(\d+)(?:\+| or (?:newer|later))")


def read(name: str) -> str:
    """Read a project file as UTF-8 text."""
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def python_floor() -> tuple[int, int]:
    """Return the minimum supported version implied by ``requires-python``."""
    declared = tomllib.loads(read("pyproject.toml"))["project"]["requires-python"]
    match = re.fullmatch(r">=\s*3\.(\d+)", declared.strip())
    assert match is not None, f"Unsupported requires-python form: {declared!r}"
    return 3, int(match.group(1))


def claimed_versions(pattern: str, text: str) -> list[tuple[int, int]]:
    """Extract documented 3.x versions such as ``3.11`` from text."""
    return [(3, int(minor)) for minor in re.findall(pattern, text)]


def test_documented_test_count_matches_collected_suite() -> None:
    """The user guide's stated suite size must match what pytest actually collects."""
    documented = [
        int(count) for count in re.findall(r"expands to (\d+) test cases", read("USER_GUIDE.md"))
    ]
    assert len(documented) == 1, "USER_GUIDE.md must state the suite size exactly once"
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PROJECT_ROOT / "pyproject.toml"),
            str(PROJECT_ROOT / "tests"),
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    summary = re.search(r"(\d+) tests? collected", collected)
    assert summary is not None, f"Could not read a collection summary:\n{collected[-500:]}"
    assert int(summary.group(1)) == documented[0], (
        f"USER_GUIDE.md claims {documented[0]} test cases; pytest collects "
        f"{summary.group(1)}. Update the documented count."
    )


def test_documented_python_floor_matches_package_metadata() -> None:
    """Project guidance must not advertise a floor other than requires-python."""
    floor = python_floor()
    for name in (
        "README.md",
        "USER_GUIDE.md",
        ".github/instructions/proper-transient-finder.instructions.md",
    ):
        claims = claimed_versions(VERSION_CLAIM.pattern, read(name))
        assert claims, f"{name} must state the supported Python version"
        assert set(claims) == {floor}, (
            f"{name} claims Python versions {sorted(set(claims))}; "
            f"pyproject.toml declares 3.{floor[1]} or newer"
        )


def test_project_instructions_match_repository_layout(tmp_path: Path) -> None:
    """Instruction paths and automatic attachment must cover the current project."""
    instructions = read(".github/instructions/proper-transient-finder.instructions.md")
    frontmatter = instructions.split("---", 2)[1]
    apply_to = re.search(r'^applyTo:\s*"([^"]+)"\s*$', frontmatter, re.MULTILINE)
    assert apply_to is not None, "Project instructions must declare matching patterns"
    expected = {
        "run_proper_transient_finder.py",
        "src/proper_transient_finder/astro.py",
        "tests/test_survey_mosaic.py",
        "experiments/robustness.py",
        "scripts/generate_docs.py",
        "scripts/lint.ps1",
        "scripts/tasks.sh",
        "pyproject.toml",
        "MANIFEST.in",
        "USER_GUIDE.md",
    }
    for path in expected:
        assert (PROJECT_ROOT / path).is_file(), f"Expected project file {path} does not exist"
    fixtures = expected | {"scripts/nested/check.ps1", "scripts/nested/check.sh"}
    for path in fixtures:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    matched = {
        path.relative_to(tmp_path).as_posix()
        for pattern in apply_to.group(1).split(",")
        for path in tmp_path.glob(pattern.strip())
        if path.is_file()
    }
    assert fixtures <= matched, f"Instruction patterns miss {sorted(fixtures - matched)}"
    for path in (
        "src/proper_transient_finder",
        "tests",
        "scripts/lint.ps1",
        "scripts/test.ps1",
        "USER_GUIDE.md",
    ):
        assert f"`{path}`" in instructions, f"Project instructions must reference {path}"
        assert (PROJECT_ROOT / path).exists(), f"Instruction path {path} does not exist"


def test_ci_matrix_classifiers_and_wrapper_gates_agree() -> None:
    """Workflow matrix, classifiers, README list, and script gates must agree."""
    floor = python_floor()
    workflow = WORKFLOW.read_text(encoding="utf-8")
    matrices = re.findall(r"python-version:\s*\[([^\]]+)\]", workflow)
    assert len(matrices) == 2, "Windows and POSIX jobs must each declare a matrix"
    tested = [claimed_versions(r"3\.(\d+)", block) for block in matrices]
    assert tested[0] == tested[1], f"CI jobs test different versions: {tested}"
    matrix = set(tested[0])
    assert min(matrix) == floor, (
        f"CI matrix {sorted(matrix)} starts above the declared floor 3.{floor[1]}"
    )
    classifiers = set(
        claimed_versions(r"Programming Language :: Python :: 3\.(\d+)", read("pyproject.toml"))
    )
    assert classifiers == matrix, (
        f"Classifiers {sorted(classifiers)} do not match the tested matrix {sorted(matrix)}"
    )
    documented = re.search(
        r"workflow that tests Python (.{0,80}?) on Windows", read("README.md"), re.DOTALL
    )
    assert documented is not None, "README must describe the tested Python versions"
    assert set(claimed_versions(r"3\.(\d+)", documented.group(1))) == matrix
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.ps1"))
    scripts.append(PROJECT_ROOT / "scripts" / "tasks.sh")
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        gate = f"sys.version_info >= ({floor[0]}, {floor[1]})"
        assert gate in text, f"{script.name} must gate on {gate}"
        assert f"Python {floor[0]}.{floor[1]} or newer." in text, (
            f"{script.name} must state that Python {floor[0]}.{floor[1]} or newer is required"
        )


def test_api_reference_header_matches_public_version() -> None:
    """The generated API page must name the current version and both entry points."""
    import proper_transient_finder as package

    api = read("docs/API.md")
    assert f"Version: `{package.__version__}`" in api, "docs/API.md version is stale"
    header = next(line for line in api.splitlines() if line.startswith("Generated from source"))
    assert "scripts/docs.ps1" in header and "scripts/tasks.sh docs" in header


def test_documented_release_artifacts_match_public_version() -> None:
    """Release labels and distribution examples must use the package version."""
    import proper_transient_finder as package

    documents = (
        ("README.md", "Current release", {"-py3-none-any.whl"}),
        ("USER_GUIDE.md", "Version", {"-py3-none-any.whl", ".tar.gz"}),
    )
    for name, label, expected_suffixes in documents:
        text = read(name)
        assert f"{label}: `{package.__version__}`" in text, f"{name} has a stale release label"
        examples = re.findall(
            r'proper_transient_finder-([^\s/\\`"]+?)(-py3-none-any\.whl|\.tar\.gz)',
            text,
        )
        assert {suffix for _, suffix in examples} >= expected_suffixes, (
            f"{name} must include the expected distribution examples"
        )
        assert {version for version, _ in examples} == {package.__version__}, (
            f"{name} contains stale distribution filenames: {examples}"
        )


def test_posix_task_script_is_checked_out_with_lf() -> None:
    """Keep bash source free of carriage returns on Linux and macOS runners."""
    task_script = PROJECT_ROOT / "scripts" / "tasks.sh"
    assert b"\r" not in task_script.read_bytes(), "tasks.sh must use LF line endings"
    assert "*.sh text eol=lf" in read(".gitattributes")


def test_source_distribution_includes_repository_validation_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "USER_GUIDE.md",
        "LICENSE",
        "mkdocs.yml",
        "run_proper_transient_finder.py",
        ".gitattributes",
    ):
        shutil.copy2(PROJECT_ROOT / name, source / name)
    for name in (
        "src",
        "tests",
        "experiments",
        "scripts",
        "docs",
        ".github/workflows",
        ".github/instructions",
    ):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            PROJECT_ROOT / name,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    dist = tmp_path / "dist"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with tarfile.open(next(dist.glob("*.tar.gz")), "r:gz") as archive:
        names = {name.split("/", 1)[-1] for name in archive.getnames()}
    assert {
        ".gitattributes",
        ".github/workflows/proper-transient-finder.yml",
        ".github/instructions/proper-transient-finder.instructions.md",
    } <= names
