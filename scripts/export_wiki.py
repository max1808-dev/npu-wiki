from collections import defaultdict
from pathlib import Path
import posixpath
import re
import shutil
from urllib.parse import quote

from markdown.extensions.toc import slugify_unicode

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / ".wiki-docs"
SECTIONS = ("微架构", "图编译器", "SDK", "推理框架", "产品与市场")
ATTACHMENTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}
WIKI_LINK = re.compile(r"(?<!\\)(!?)\[\[([^\]\n]+)\]\]")
CODE = re.compile(
    r"(^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\2[ \t]*(?:\n|$)|(`+)[^\n]*?\3)",
    re.MULTILINE | re.DOTALL,
)
HOME = Path("总览.md")


def export():
    files = [HOME, Path("Software Stack.md")]
    for section in SECTIONS:
        files.extend(
            path.relative_to(ROOT)
            for path in (ROOT / section).rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
            and path.suffix.lower() in ATTACHMENTS | {".md"}
        )
    by_name = defaultdict(list)
    for path in sorted(files):
        key = path.stem if path.suffix == ".md" else path.name
        by_name[key.casefold()].append(path)

    unresolved = set()
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()
    for source in files:
        destination = Path("index.md") if source == HOME else source
        output = DOCS / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix != ".md":
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
