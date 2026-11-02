# Publish Documentation

**Author:** Pier-Francesco Rocci

Developed with assistance from ChatGPT, Claude, and Mistral.

The documentation site is built with MkDocs 1.x and Material for MkDocs.
The generated `site/` directory contains static HTML, local styles and scripts,
and the search index. It can be hosted on GitHub Pages or another static host.
No Python process is required on the hosting server.

## Install Documentation Tools

Run these commands from the directory containing `mkdocs.yml` with your
chosen Python 3.13+ interpreter:

```powershell
python -m pip install ".[docs]"
```

Documentation tools are optional and are not runtime dependencies of the
astronomy library. The default scripts use `.venv\Scripts\python.exe`;
pass `-PythonPath` to select another prepared interpreter.

## Build And Preview

```powershell
.\scripts\docs.ps1 -Build
.\scripts\docs.ps1 -Serve
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
| API Reference | `docs/API.md`, generated from Python source by `scripts/docs.ps1` |
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

Before a public release, choose and add a project license, confirm permission
to publish, and review the site for private observations, credentials, and
internal paths. Follow ProperImage and Astropy citation and license requirements.

## Static Hosting

Upload the **contents** of `site/` to the hosting service's web root or chosen
subdirectory. Do not upload virtual environments, FITS observations, experiment
artifacts, or the entire source checkout. The built site uses local theme assets;
it does not require Google Fonts or an external script CDN.

Once the public address is known, set `site_url` in `mkdocs.yml` to that exact
address, including any hosting subdirectory, then rebuild. For example:

```yaml
site_url: https://example.org/proper-transient-finder/
```

Setting the real address enables correct canonical URLs and sitemap entries.
No placeholder address is configured in the supplied site.

## GitHub Pages

In the intended repository, configure GitHub Pages to publish the `gh-pages`
branch at `/ (root)`. After validation, run:

```powershell
python -m mkdocs gh-deploy --strict --config-file mkdocs.yml
```

This command builds the site and pushes it to the repository's `gh-pages`
branch. Confirm the Git remote and write permissions first: it replaces the
published documentation on that branch. The supplied quality workflow builds
the site but does not deploy it automatically. No publication is performed by
the local build or preview commands.
