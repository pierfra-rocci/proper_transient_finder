# Publish Documentation

**Author:** Pier-Francesco Rocci

Developed with assistance from ChatGPT, Claude, and Mistral.

The documentation site is built with MkDocs 1.x and Material for MkDocs.
The generated `site/` directory contains static HTML, local styles and scripts,
and the search index. It can be hosted on GitHub Pages or another static host.
No Python process is required on the hosting server.

## Install Documentation Tools

Run these commands from the directory containing `mkdocs.yml` with your
chosen Python 3.11+ interpreter:

```powershell
python -m pip install ".[docs]"
```

Documentation tools are optional and are not runtime dependencies of the
astronomy library. The default PowerShell scripts use `.venv\Scripts\python.exe`;
pass `-PythonPath` (or `PYTHON=<path>` for `scripts/tasks.sh`) to select another
prepared interpreter.

## Build And Preview

```powershell
.\scripts\docs.ps1 -Build
.\scripts\docs.ps1 -Serve
```

On Linux, macOS, or Git Bash:

```bash
bash scripts/tasks.sh docs --build
bash scripts/tasks.sh docs --serve
bash scripts/tasks.sh docs --serve --address 127.0.0.1:8002
```

The build regenerates the API reference and validates the site with
`mkdocs build --strict`. The preview is at <http://127.0.0.1:8001/>.
Stop it with Ctrl+C. To use another local port:

```powershell
.\scripts\docs.ps1 -Serve -Address 127.0.0.1:8002
```

The preview watches the README, user guide, documentation directory, and build
hook. Re-run the build after changes to Python API signatures or defaults.

## Source Of Truth

| Website page | Maintained source |
| --- | --- |
| Quick Start | `README.md` |
| User Guide | `USER_GUIDE.md` |
| API Reference | `docs/API.md`, generated from Python source by `scripts/docs.ps1` (or `scripts/tasks.sh docs` on POSIX) |
| Publish Documentation | `docs/publishing.md` |

The MkDocs hook exposes the README and user guide as pages without copying
them. It adapts their internal links for the website while leaving the original
Markdown readable on GitHub. Do not edit generated HTML in `site/`.

## Validate Before Publishing

```powershell
.\scripts\docs.ps1 -Check -Build
```

This command fails if the API Markdown is stale, a page is missing, or the
strict site build reports a broken documentation link or anchor. The repository's
quality workflow also runs this check. Inspect the preview before publishing.

Before a public release, confirm permission to publish and review the site for
private observations, credentials, and internal paths. The project declares the
BSD 3-Clause License in `pyproject.toml` and `LICENSE`; follow ProperImage and
Astropy citation and license requirements in addition.

## Static Hosting

Upload the **contents** of `site/` to the hosting service's web root or chosen
subdirectory. Do not upload virtual environments, FITS observations, experiment
artifacts, or the entire source checkout. The built site uses local theme assets;
it does not require Google Fonts or an external script CDN.

`site_url` in `mkdocs.yml` is already set to this repository's GitHub Pages
address:

```yaml
site_url: https://pierfra-rocci.github.io/proper_transient_finder/
```

It enables correct canonical URLs and sitemap entries. If the site is hosted
either, change this value to that exact address, including any hosting
subdirectory, and rebuild.

## GitHub Pages

In this repository, GitHub Pages is published by the `.github/workflows/docs.yml`
workflow at
<https://pierfra-rocci.github.io/proper_transient_finder/>. The workflow runs on
pushes to `main` that touch `docs/`, `README.md`, `USER_GUIDE.md`, `mkdocs.yml`,
`pyproject.toml`, the package source, the documentation generator or MkDocs hook,
or the workflow itself, and can also be triggered manually from the Actions tab.
It installs the `[docs]` extra declared in `pyproject.toml`, regenerates the API
reference, builds the site with `mkdocs build --strict`, and deploys the result
through GitHub's Pages actions; deployments are serialized through a `pages`
concurrency group. The separate `publish.yml` workflow only publishes the package
to PyPI.

Configure GitHub Pages once: open **Settings → Pages** and set **Source** to
**GitHub Actions**. No `gh-pages` branch is used.

To publish manually from a workstation instead, run:

```powershell
python -m mkdocs gh-deploy --strict --config-file mkdocs.yml
```

That command pushes the built site to the repository's `gh-pages` branch, so use
it only if Pages is switched back to branch-based publishing. Confirm the Git
remote and write permissions first: it replaces the published documentation on
that branch.
