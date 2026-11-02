---
description: "Use when modifying the standalone astronomy child project, its FITS pipeline, synthetic tests, packaging, or project scripts."
name: "Proper Transient Finder Project"
applyTo: "proper_transient_finder/**/*.py"
---
# Proper Transient Finder Project

- Treat `proper_transient_finder/` as a separately installable Python 3.13+ project; its package is `src/proper_transient_finder`.
- Keep astronomy production code in that package and focused synthetic regression tests in `proper_transient_finder/tests/`.
- Run `proper_transient_finder/scripts/lint.ps1` and `proper_transient_finder/scripts/test.ps1` for child-project changes.
- Follow the shared Python security requirements: validate FITS/CLI input, avoid unsafe subprocesses, and do not log private image data.
- Keep the standalone manual at `proper_transient_finder/USER_GUIDE.md`; do not add astronomy guidance to IP Lab's diagnostics guide.