from collections import defaultdict
from os import walk
from pathlib import Path
import posixpath
import re
import shutil
from urllib.parse import quote

from markdown.extensions.toc import slugify_unicode

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / ".wiki-docs"
PROJECT_DIRS = {"scripts", "site", "web-assets"}
ATTACHMENTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}
WIKI_LINK = re.compile(r"(?<!\\)(!?)\[\[([^\]\n]+)\]\]")
CODE = re.compile(
    r"(^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\2[ \t]*(?:\n|$)|(`+)[^\n]*?\3)",
    re.MULTILINE | re.DOTALL,
)
HOME = Path("总览.md")


def wiki_files():
    for directory, folders, names in walk(ROOT):
        parent = Path(directory)
        folders[:] = [
            name for name in folders
            if not name.startswith(".")
            and not (parent == ROOT and name in PROJECT_DIRS)
            and not (parent / name).is_symlink()
        ]
        for name in names:
            path = parent / name
            if name.startswith(".") or path == ROOT / "README.md" or path.is_symlink():
                continue
            if path.suffix.lower() in ATTACHMENTS | {".md"}:
                yield path.relative_to(ROOT)


def on_config(config):
    def navigation(directory):
        pages = []
        sections = []
        for path in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
            if path.is_dir():
                children = navigation(path)
                if children:
                    sections.append({path.name: children})
            elif path.suffix.lower() == ".md":
                relative = path.relative_to(DOCS)
                if relative == Path("index.md"):
                    pages.insert(0, {HOME.stem: relative.as_posix()})
                else:
                    pages.append({path.stem: relative.as_posix()})
        return pages + sections

    config.nav = navigation(DOCS)
    return config


def export():
    files = sorted(wiki_files())
    by_name = defaultdict(list)
    for path in files:
        key = path.stem if path.suffix.lower() == ".md" else path.name
        by_name[key.casefold()].append(path)

    unresolved = set()
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()
    for asset, folder in (("sidebar-icons.css", "stylesheets"), ("sidebar-icons.js", "javascripts")):
        asset_dir = DOCS / folder
        asset_dir.mkdir()
        shutil.copyfile(ROOT / "web-assets" / asset, asset_dir / asset)
    for source in files:
        destination = Path("index.md") if source == HOME else source
        output = DOCS / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() != ".md":
            shutil.copyfile(ROOT / source, output)
            continue

        def convert(match):
            embed, body = match.groups()
            target, _, alias = body.partition("|")
            name, _, heading = target.partition("#")
            name = name.strip()
            label = alias or heading or name.rsplit("/", 1)[-1]
            if name:
                key = name.removesuffix(".md")
                candidates = by_name[key.rsplit("/", 1)[-1].casefold()]
                if "/" in key:
                    candidates = [
                        path for path in candidates
                        if ("/" + path.as_posix().removesuffix(".md")).casefold()
                        .endswith("/" + key.casefold())
                    ]
                candidates = sorted(
                    (len(Path(posixpath.relpath(path.as_posix(), source.parent.as_posix())).parts), path)
                    for path in candidates
                )
                if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
                    unresolved.add((source.as_posix(), target))
                    return label
                linked = candidates[0][1]
                linked = Path("index.md") if linked == HOME else linked
                url = quote(posixpath.relpath(linked.as_posix(), destination.parent.as_posix()))
            else:
                linked = destination
                url = ""
            if heading:
                url += "#" + slugify_unicode(heading, "-")
            if embed and linked.suffix.lower() in ATTACHMENTS - {".pdf"}:
                width = alias.split("x", 1)[0]
                if width.isdecimal():
                    return f'![{linked.stem}]({url}){{ width="{width}" }}'
                return f"![{label}]({url})"
            label = label.replace("[", "\\[").replace("]", "\\]")
            return f"[{label}]({url})"

        raw = (ROOT / source).read_text(encoding="utf-8-sig")
        chunks = []
        start = 0
        for code in CODE.finditer(raw):
            chunks.append(WIKI_LINK.sub(convert, raw[start:code.start()]))
            chunks.append(code.group())
            start = code.end()
        chunks.append(WIKI_LINK.sub(convert, raw[start:]))
        content = "".join(chunks)
        if source == HOME:
            content = "# 总览\n\n" + content
        output.write_text(content, encoding="utf-8")

    for source, target in sorted(unresolved):
        print(f"Unresolved Obsidian link in {source}: {target}")
    print(f"Exported {len(files)} files. Unresolved links: {len(unresolved)}")


if __name__ == "__main__":
    export()
