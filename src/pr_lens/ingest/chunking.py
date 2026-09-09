"""Split a file into retrievable chunks along the structure it already has.

A function, a class, a docs section. Fixed-size windows are the alternative and they cut
through the middle of a function body, which produces a retrieval hit that cannot be
read on its own and a citation that points at half a thing.

MAX_CHARS is a deliberate compromise, not a tuned number. Whole functions are the unit
worth retrieving, so the threshold is set high enough that most of them survive intact.
That is past the 512-token window of the small embedding models, so the first stage will
truncate long chunks. Phase 2 measures exactly that; changing the number afterwards
changes every unit id, so it is fixed here and the cost is measured rather than guessed.
"""

import ast
import re
from dataclasses import dataclass

MAX_CHARS = 4000
MIN_CHARS = 32

_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})
_RST_SUFFIXES = frozenset({".rst"})

_ATX_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_RST_UNDERLINE = re.compile(r"^(?P<char>[=\-~`:'\"^_*+#])(?P=char){2,}\s*$")


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    start_line: int
    end_line: int
    symbol: str | None


def chunk_file(path: str, text: str) -> list[Chunk]:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix in _PYTHON_SUFFIXES:
        return chunk_python(text)
    if suffix in _MARKDOWN_SUFFIXES:
        return chunk_markdown(text)
    if suffix in _RST_SUFFIXES:
        return chunk_rst(text)
    return chunk_lines(text)


def chunk_python(source: str) -> list[Chunk]:
    """One chunk per top-level definition, plus the module-level code between them.

    Falls back to line windows on a syntax error. The mining set spans years of history
    and includes files written for older Pythons, so a parse failure is a normal event
    rather than a bug, and losing the file entirely would be the worse outcome.
    """
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError):
        return chunk_lines(source)

    lines = source.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    cursor = 1
    for node in module.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        start = _leading_comment_start(lines, _declaration_line(node), floor=cursor)
        end = node.end_lineno or node.lineno
        if start > cursor:
            chunks.extend(_sized(lines, cursor, start - 1, None))
        if isinstance(node, ast.ClassDef) and _span_length(lines, start, end) > MAX_CHARS:
            chunks.extend(_split_class(lines, node, start, end))
        else:
            chunks.extend(_sized(lines, start, end, node.name))
        cursor = end + 1
    if cursor <= len(lines):
        chunks.extend(_sized(lines, cursor, len(lines), None))
    return [c for c in chunks if c.text.strip()]


def _split_class(lines: list[str], node: ast.ClassDef, start: int, end: int) -> list[Chunk]:
    """A class too big to keep whole becomes its header plus one chunk per method."""
    chunks: list[Chunk] = []
    cursor = start
    for member in node.body:
        if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        member_start = _leading_comment_start(lines, _declaration_line(member), floor=cursor)
        member_end = member.end_lineno or member.lineno
        if member_start > cursor:
            chunks.extend(_sized(lines, cursor, member_start - 1, node.name))
        chunks.extend(_sized(lines, member_start, member_end, f"{node.name}.{member.name}"))
        cursor = member_end + 1
    if cursor <= end:
        chunks.extend(_sized(lines, cursor, end, node.name))
    return _merge_runts(chunks) or _sized(lines, start, end, node.name)


def _sized(lines: list[str], start: int, end: int, symbol: str | None) -> list[Chunk]:
    """One chunk if it fits, line windows carrying the same symbol if it does not."""
    whole = _chunk(lines, start, end, symbol)
    if not whole.text.strip():
        return []
    if len(whole.text) <= MAX_CHARS:
        return [whole]
    return _relocate(chunk_lines(whole.text), whole)


def _span_length(lines: list[str], start: int, end: int) -> int:
    return len("\n".join(lines[start - 1 : end]))


def chunk_markdown(text: str) -> list[Chunk]:
    """Split at headings, naming each chunk by its breadcrumb.

    Fenced blocks are tracked because a shell comment inside one looks exactly like a
    level-one heading, and splitting there tears the example in half.
    """
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    trail: list[str] = []
    fenced = False
    for index, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = _ATX_HEADING.match(line)
        if heading is None:
            continue
        level = len(heading["hashes"])
        del trail[level - 1 :]
        trail.append(heading["title"])
        starts.append((index, " > ".join(trail)))
    return _sections(lines, starts)


def chunk_rst(text: str) -> list[Chunk]:
    """Same idea as markdown, but reStructuredText marks a heading by underlining it."""
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for index in range(1, len(lines)):
        title = lines[index - 1].strip()
        if not title or not _RST_UNDERLINE.match(lines[index]):
            continue
        if len(lines[index].strip()) < len(title):
            continue
        starts.append((index, title))
    return _sections(lines, starts)


def chunk_lines(text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """The fallback: whole lines, packed up to the limit, never split mid-line."""
    lines = text.splitlines()
    chunks: list[Chunk] = []
    start = 1
    size = 0
    for index, line in enumerate(lines, start=1):
        if size and size + len(line) + 1 > max_chars:
            chunks.append(_chunk(lines, start, index - 1, None))
            start, size = index, 0
        size += len(line) + 1
    if start <= len(lines):
        chunks.append(_chunk(lines, start, len(lines), None))
    return [c for c in chunks if c.text.strip()]


def _sections(lines: list[str], starts: list[tuple[int, str]]) -> list[Chunk]:
    if not lines:
        return []
    if not starts or starts[0][0] > 1:
        starts = [(1, ""), *starts]

    chunks: list[Chunk] = []
    for position, (start, symbol) in enumerate(starts):
        end = starts[position + 1][0] - 1 if position + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start - 1 : end])
        if not text.strip():
            continue
        if len(text) <= MAX_CHARS:
            chunks.append(Chunk(text, start, end, symbol or None))
            continue
        for part in chunk_lines(text):
            chunks.append(
                Chunk(
                    part.text,
                    start + part.start_line - 1,
                    start + part.end_line - 1,
                    symbol or None,
                )
            )
    return _merge_runts(chunks)


def _merge_runts(chunks: list[Chunk]) -> list[Chunk]:
    """Fold a chunk with almost nothing in it into the one that follows.

    Docs are full of headings with a line under them, and splitting a class leaves a bare
    `class C:` behind. On their own these retrieve as a title with no answer under it.

    Deliberately not applied to top-level Python definitions. Merging two functions
    because the first one is short destroys exactly the structure this module exists to
    keep, and a four-line function is a legitimate retrieval result.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and len(merged[-1].text) < MIN_CHARS:
            previous = merged.pop()
            joined = f"{previous.text}\n{chunk.text}"
            if len(joined) <= MAX_CHARS:
                merged.append(Chunk(joined, previous.start_line, chunk.end_line, previous.symbol))
                continue
            merged.append(previous)
        merged.append(chunk)
    return merged


def _relocate(parts: list[Chunk], parent: Chunk) -> list[Chunk]:
    return [
        Chunk(
            p.text,
            parent.start_line + p.start_line - 1,
            parent.start_line + p.end_line - 1,
            parent.symbol,
        )
        for p in parts
    ]


def _chunk(lines: list[str], start: int, end: int, symbol: str | None) -> Chunk:
    return Chunk("\n".join(lines[start - 1 : end]), start, end, symbol)


def _declaration_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Decorators sit above the def and belong to it, but ast.lineno points at the def."""
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _leading_comment_start(lines: list[str], declaration_line: int, *, floor: int) -> int:
    """Walk back over the comment block directly above a definition and take it with it.

    A comment above a function is about that function. Leaving it behind puts it at the
    end of the previous chunk, where it reads as a note about something unrelated.
    """
    index = declaration_line - 1
    while index > floor - 1:
        previous = lines[index - 1].strip()
        if previous and not previous.startswith("#"):
            break
        index -= 1
    while index < declaration_line - 1 and not lines[index].strip():
        index += 1
    return index + 1
