---
description: "Use when modifying the Proper Transient Finder Python project, FITS pipeline, synthetic tests, packaging, scripts, or user guide."
name: "Proper Transient Finder Project"
applyTo: "**/*.py, scripts/**/*.ps1, scripts/**/*.sh, pyproject.toml, MANIFEST.in, USER_GUIDE.md"
---
# Proper Transient Finder Project

- Treat this repository as a separately installable Python 3.11+ project; its package is `src/proper_transient_finder`.
- Keep astronomy production code in that package and focused synthetic regression tests in `tests`.
- Run `scripts/lint.ps1` and `scripts/test.ps1` for project changes.
- Follow the shared Python security requirements: validate FITS/CLI input, avoid unsafe subprocesses, and do not log private image data.
- Keep the user guide at `USER_GUIDE.md`.