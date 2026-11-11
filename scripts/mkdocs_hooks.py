# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Expose project Markdown as MkDocs pages without duplicate sources."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files

SOURCE_PAGES = {"README.md": "index.md", "USER_GUIDE.md": "user-guide.md"}
REPO_FILES = {"LICENSE": "blob/main/LICENSE"}


def on_files(files: Files, config: MkDocsConfig) -> Files:
    project = Path(__file__).resolve().parents[1]
    for source, destination in SOURCE_PAGES.items():
        path = str(project / source)
        files.append(File.generated(config, destination, abs_src_path=path))
    return files


def on_page_markdown(markdown: str, *, config: MkDocsConfig, **kwargs: object) -> str:
    targets = {
        **SOURCE_PAGES,
        "docs/API.md": "API.md",
        "docs/publishing.md": "publishing.md",
    }
    for source, destination in targets.items():
        for prefix in ("", "../"):
            for suffix in (")", "#"):
                original = f"]({prefix}{source}{suffix}"
                replacement = f"]({destination}{suffix}"
                markdown = markdown.replace(original, replacement)
    repository = str(config.get("repo_url") or "").rstrip("/")
    if repository:
        # The README links to absolute GitHub URLs so PyPI renders them; point
        # those links back at the local pages in the built site.
        for source, destination in targets.items():
            markdown = markdown.replace(f"]({repository}/blob/main/{source}", f"]({destination}")
    for name, path in REPO_FILES.items():
        if not repository:
            break
        for prefix in ("", "../"):
            markdown = markdown.replace(f"]({prefix}{name})", f"]({repository}/{path})")
    return markdown
