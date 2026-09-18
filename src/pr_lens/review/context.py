"""What the drafting call is shown about a pull request, and what it is spared.

A call has 7,000 tokens for prompt and completion together (review/provider.py), so what
goes in is a choice. This module makes that choice explicitly and records it. Every file
left out carries its reason, and every hunk that did not fit is listed as dropped, so a
quiet review can be told apart from a review that never saw the code.

The listing shown to the model numbers each line by its place in the new file, because a
line comment is anchored to that number and the diff itself does not carry it. Lines the
pull request touched or showed as context are commentable; lines of the surrounding
window are not, since GitHub refuses a comment outside the diff, and are marked so.

The import block of every changed file is shown outside the ranking and outside the hunk
budget. The first live review claimed a name was not imported when the same pull request
added the import, because that one added line ranked twelfth of fifteen and the budget
stopped at five. Whether a name is in scope is not a fact the ranking should get a vote
on, so it is not left to the ranking.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

from pr_lens.github.client import GitHubClient
from pr_lens.ingest.diff import Hunk, parse_patch
from pr_lens.review.provider import estimate_tokens

# Files a reviewer does not read line by line. The lockfile names catch the ones without a
# .lock suffix; the suffix catches poetry, uv, Cargo, Gemfile, composer and yarn.
LOCKFILES = frozenset({"package-lock.json", "pnpm-lock.yaml", "npm-shrinkwrap.json", "go.sum"})
GENERATED_SUFFIXES = (".lock", ".min.js", ".min.css", ".map", ".snap")
VENDORED_DIRS = frozenset({"vendor", "node_modules", "third_party", "_vendor", "site-packages"})

# Enough lines either side of a hunk to show the enclosing block, few enough that three or
# four hunks with their windows still fit the budget.
WINDOW_LINES = 10

# Head files fetched for windows. Each is one API call, and only the best-ranked hunks can
# fit the budget anyway.
MAX_WINDOW_FILES = 8

# A new file arrives as one hunk of every line in it. Past this the listing is cut, so one
# large file cannot spend the budget that several ordinary hunks would have shared.
MAX_HUNK_LINES = 80

# A minified line inside an ordinary file would otherwise spend hundreds of tokens alone.
MAX_LINE_CHARS = 200

# The import block of one changed file, and of all of them together. A one line import is
# what decides whether a "this name is not defined" comment is true, and ranking sorts it
# near the bottom every time because it adds one line. So it is shown outside the ranking.
# The caps are what stop a file with a hundred imports from spending the diff's budget.
MAX_IMPORT_LINES_PER_FILE = 25
MAX_IMPORT_LINES_TOTAL = 100

IMPORTS_HEADING = (
    "Names already in scope in the changed files at this commit. These lines are outside "
    "the diff, so they are not commentable, and a name listed here IS imported:"
)

# Test and documentation changes are reviewable, but a hunk of code the tests exercise is
# the likelier home for a comment worth leaving.
SECONDARY_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class PullRequest:
    repo: str
    number: int
    title: str
    body: str
    author: str
    base_sha: str
    head_sha: str
    draft: bool
    head_ref: str = ""


@dataclass(frozen=True, slots=True)
class Skipped:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class Block:
    """One hunk as the prompt could show it, richest rendering first."""

    hunk: Hunk
    variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Fitted:
    shown: list[tuple[Hunk, str]]
    dropped: list[Hunk]
    tokens: int
    # Where each shown block sat in the ranking. fit no longer keeps a prefix, so a caller
    # that has one list per candidate cannot index it by position in `shown`.
    positions: tuple[int, ...] = ()


async def fetch_pull(client: GitHubClient, repo: str, number: int) -> PullRequest:
    pull = await client.get_json(f"/repos/{repo}/pulls/{number}")
    return PullRequest(
        repo=repo,
        number=number,
        title=pull.get("title") or "",
        body=pull.get("body") or "",
        author=(pull.get("user") or {}).get("login", ""),
        base_sha=pull["base"]["sha"],
        head_sha=pull["head"]["sha"],
        draft=bool(pull.get("draft")),
        head_ref=pull["head"].get("ref") or "",
    )


def skip_reason(changed: dict[str, Any]) -> str | None:
    """Why a file from the pulls files endpoint is not reviewed, or None if it is."""
    path = PurePosixPath(changed["filename"])
    status = changed.get("status")
    if status == "removed":
        return "deleted"
    if status == "renamed" and not changed.get("changes"):
        return "renamed without changes"
    if path.name in LOCKFILES or path.name.endswith(GENERATED_SUFFIXES):
        return "lockfile or generated"
    if VENDORED_DIRS.intersection(path.parts[:-1]):
        return "vendored"
    if not changed.get("patch"):
        # GitHub omits the patch for binary files and for diffs too large to show.
        return "no patch: binary or too large"
    return None


def _weight(path: str) -> float:
    parts = PurePosixPath(path).parts
    name = parts[-1]
    is_test = name.startswith("test_") or name.endswith(("_test.py", ".test.ts", ".spec.ts"))
    is_test = is_test or bool({"tests", "test", "__tests__"}.intersection(parts[:-1]))
    is_doc = name.endswith((".md", ".rst", ".txt")) or "docs" in parts[:-1]
    return SECONDARY_WEIGHT if is_test or is_doc else 1.0


def rank(hunks: Sequence[Hunk]) -> list[Hunk]:
    """Most added code first. Stable, so equal hunks keep their order in the diff."""
    return sorted(hunks, key=lambda hunk: -len(hunk.added()) * _weight(hunk.path))


async def fetch_changes(
    client: GitHubClient, repo: str, number: int
) -> tuple[list[Hunk], list[Skipped]]:
    hunks: list[Hunk] = []
    skipped: list[Skipped] = []
    async for changed in client.paginate(f"/repos/{repo}/pulls/{number}/files"):
        reason = skip_reason(changed)
        if reason:
            skipped.append(Skipped(changed["filename"], reason))
            continue
        hunks.extend(parse_patch(changed["patch"], changed["filename"]))
    return rank(hunks), skipped


async def fetch_windows(
    client: GitHubClient, repo: str, sha: str, hunks: Sequence[Hunk]
) -> dict[str, list[str]]:
    """The head version of the files the best-ranked hunks belong to, as lines."""
    paths: list[str] = []
    for hunk in hunks:
        if hunk.path not in paths:
            paths.append(hunk.path)
    windows: dict[str, list[str]] = {}
    for path in paths[:MAX_WINDOW_FILES]:
        raw = await client.get_raw(
            f"/repos/{repo}/contents/{quote(path)}?ref={sha}", accept="application/vnd.github.raw"
        )
        windows[path] = raw.decode("utf-8", errors="replace").splitlines()
    return windows


def _row(number: int | None, marker: str, text: str) -> str:
    if len(text) > MAX_LINE_CHARS:
        text = text[:MAX_LINE_CHARS] + " [cut]"
    return f"{number if number is not None else '':>5} {marker} {text}"


def render(hunk: Hunk, file_lines: Sequence[str] | None = None) -> str:
    """The hunk as numbered new-file lines, with an optional window of the file around it.

    Markers: "+" added, " " unchanged but in the diff, "-" removed (no number, it is not in
    the new file), "." outside the diff and so not commentable.
    """
    rows = [f"{hunk.path}"]
    end = hunk.new_start + hunk.new_lines - 1
    if file_lines is not None:
        above = range(
            max(1, hunk.new_start - WINDOW_LINES), min(hunk.new_start, len(file_lines) + 1)
        )
        rows.extend(_row(number, ".", file_lines[number - 1]) for number in above)

    new_line = hunk.new_start
    shown = 0
    for line in hunk.lines:
        if shown == MAX_HUNK_LINES:
            rows.append(f"      ... {len(hunk.lines) - shown} more lines of this hunk not shown")
            break
        marker = line[:1] or " "
        if marker == "\\":
            continue
        if marker == "-":
            rows.append(_row(None, "-", line[1:]))
        else:
            rows.append(_row(new_line, marker, line[1:]))
            new_line += 1
        shown += 1

    if file_lines is not None and shown < MAX_HUNK_LINES:
        below = range(end + 1, min(len(file_lines), end + WINDOW_LINES) + 1)
        rows.extend(_row(number, ".", file_lines[number - 1]) for number in below)
    return "\n".join(rows)


def fit(blocks: Sequence[Block], budget: int) -> Fitted:
    """The code first, then what enriches it. Two passes, both in rank order.

    Pass one takes every hunk in its cheapest rendering, so no hunk is lost to a higher
    ranked hunk's window or past comments. Pass two spends what is left upgrading hunks to
    richer renderings, best ranked first.

    The earlier single greedy pass dropped a hunk that did not fit AND everything ranked
    below it, which on pull request 16 left 2,500 tokens unspent and ten one line hunks
    unshown, one of them the import that made the posted comment false. A hunk skipped
    here is skipped for its own size, not for its rank, and the budget it could not use is
    offered to the next one. Order in the prompt stays rank order either way.
    """
    chosen: dict[int, str] = {}
    spent = 0
    for position, block in enumerate(blocks):
        cheapest = block.variants[-1]
        cost = estimate_tokens(cheapest)
        if spent + cost <= budget:
            chosen[position] = cheapest
            spent += cost
    for position, block in enumerate(blocks):
        current = chosen.get(position)
        if current is None:
            continue
        paid = estimate_tokens(current)
        for text in block.variants:
            if text == current:
                break
            cost = estimate_tokens(text)
            if spent - paid + cost <= budget:
                chosen[position] = text
                spent += cost - paid
                break
    order = sorted(chosen)
    return Fitted(
        [(blocks[position].hunk, chosen[position]) for position in order],
        [block.hunk for position, block in enumerate(blocks) if position not in chosen],
        spent,
        tuple(order),
    )


def _is_import_start(line: str) -> bool:
    """A top level import, in the two languages the sandbox runs.

    Indented lines are excluded: a local import inside a function says nothing about what
    is in scope at module level, which is the question a NameError claim turns on.
    """
    if line[:1] in (" ", "\t"):
        return False
    stripped = line.strip()
    if stripped.startswith(("import ", "import(", "import{", "import *", "from ")):
        return True
    if stripped.startswith("export ") and " from " in stripped:
        return True
    return stripped.startswith(("const ", "let ", "var ")) and "require(" in stripped


def _unclosed(text: str) -> bool:
    depth = 0
    for character in text:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
    return depth > 0


def imports(file_lines: Sequence[str]) -> list[tuple[int, str]]:
    """Top level import lines with their number in the file, continuations included.

    A parenthesised import spans several lines and the names are on the continuation
    lines, so taking only the first line would list the module and hide every name it
    brings in, which is the opposite of the point.
    """
    found: list[tuple[int, str]] = []
    number = 0
    total = len(file_lines)
    while number < total and len(found) < MAX_IMPORT_LINES_PER_FILE:
        line = file_lines[number]
        number += 1
        if not _is_import_start(line):
            continue
        found.append((number, line))
        buffered = line
        while (_unclosed(buffered) or buffered.rstrip().endswith("\\")) and number < total:
            if len(found) >= MAX_IMPORT_LINES_PER_FILE:
                break
            line = file_lines[number]
            number += 1
            found.append((number, line))
            buffered += line
    return found


def render_imports(windows: Mapping[str, Sequence[str]]) -> str:
    """The import block of every changed file whose head was fetched, one section each.

    This is shown outside the hunk budget because it is the context that decides whether a
    claim about an undefined name is true, and it is small: one line each, already paid
    for by the file fetches the windows needed.
    """
    sections: list[str] = []
    budget = MAX_IMPORT_LINES_TOTAL
    for path in sorted(windows):
        if budget <= 0:
            break
        found = imports(windows[path])[:budget]
        if not found:
            continue
        budget -= len(found)
        rows = [path]
        rows.extend(_row(number, ".", text) for number, text in found)
        sections.append("\n".join(rows))
    if not sections:
        return ""
    return "\n".join([IMPORTS_HEADING, *sections])


def commentable(hunk: Hunk) -> frozenset[int]:
    """New-file line numbers GitHub accepts a right-side comment on: the ones in the diff."""
    return frozenset(number for number, _ in hunk.new_file_lines())
