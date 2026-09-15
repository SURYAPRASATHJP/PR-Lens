"""What the drafting call is shown about a pull request, and what it is spared.

A call has 7,000 tokens for prompt and completion together (review/provider.py), so what
goes in is a choice. This module makes that choice explicitly and records it. Every file
left out carries its reason, and every hunk that did not fit is listed as dropped, so a
quiet review can be told apart from a review that never saw the code.

The listing shown to the model numbers each line by its place in the new file, because a
line comment is anchored to that number and the diff itself does not carry it. Lines the
pull request touched or showed as context are commentable; lines of the surrounding
window are not, since GitHub refuses a comment outside the diff, and are marked so.
"""

from collections.abc import Sequence
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
    """Greedy, in rank order: each hunk gets its richest rendering that still fits.

    A hunk that fits in no rendering is dropped and so is everything ranked below it, since
    showing a lower-ranked hunk in place of a higher one would invert the ranking.
    """
    shown: list[tuple[Hunk, str]] = []
    spent = 0
    for position, block in enumerate(blocks):
        chosen = next(
            (text for text in block.variants if spent + estimate_tokens(text) <= budget), None
        )
        if chosen is None:
            return Fitted(shown, [b.hunk for b in blocks[position:]], spent)
        shown.append((block.hunk, chosen))
        spent += estimate_tokens(chosen)
    return Fitted(shown, [], spent)


def commentable(hunk: Hunk) -> frozenset[int]:
    """New-file line numbers GitHub accepts a right-side comment on: the ones in the diff."""
    return frozenset(number for number, _ in hunk.new_file_lines())
