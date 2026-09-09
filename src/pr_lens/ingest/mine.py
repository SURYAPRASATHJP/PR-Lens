"""One repository into corpus units.

Source comes from the tarball rather than the contents API, which is the difference
between one request per repo and one per file. On a 27-repo mining set against a
5,000/hour budget that is not an optimisation, it is the only version that finishes.

Four kinds of unit come out of the API instead: merged pull requests, the review comments
left on them, and issues. The review comments are the valuable ones. They are a record of
what an experienced reviewer thought was worth saying about a specific hunk, which is the
exact thing this project is trying to learn to do.
"""

import io
import logging
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from pr_lens.github.client import IMMUTABLE, GitHubClient
from pr_lens.ingest.chunking import chunk_file
from pr_lens.ingest.units import CorpusUnit, UnitKind

logger = logging.getLogger(__name__)

# Listings move, so they are revalidated; a resumed run inside the same night still reads
# them from disk. The tarball is pinned to a commit sha and can never change.
LISTING_MAX_AGE = timedelta(hours=6)

SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".cfg", ".toml", ".ini"})
DOC_SUFFIXES = frozenset({".md", ".markdown", ".mdx", ".rst", ".txt"})

# Vendored and generated trees. Chunking them buys nothing and they are often the largest
# thing in the tarball.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".github/workflows/generated",
        ".mypy_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "third_party",
        "vendor",
    }
)

# Past this a file is generated, minified or a fixture. Chunking it produces noise that
# outranks real code because it is long and repetitive.
MAX_FILE_BYTES = 400_000


@dataclass(frozen=True, slots=True)
class MiningLimits:
    """How far back to walk each listing.

    These are windows, not totals. Source is the whole tree because the tarball gives it
    for free; the API-backed kinds are capped because each page is a request and the
    oldest pages are the least useful.
    """

    pull_requests: int = 300
    review_comments: int = 1000
    issues: int = 300


@dataclass(frozen=True, slots=True)
class RepoHead:
    repo: str
    default_branch: str
    sha: str


async def resolve_head(client: GitHubClient, repo: str) -> RepoHead:
    metadata = await client.get_json(f"/repos/{repo}", max_age=LISTING_MAX_AGE)
    branch = str(metadata["default_branch"])
    commit = await client.get_json(f"/repos/{repo}/commits/{branch}", max_age=LISTING_MAX_AGE)
    return RepoHead(repo=repo, default_branch=branch, sha=str(commit["sha"]))


async def source_and_doc_units(client: GitHubClient, head: RepoHead) -> list[CorpusUnit]:
    """The whole default-branch tree in one request, chunked along its own structure."""
    tarball = await client.get_raw(
        f"/repos/{head.repo}/tarball/{head.sha}",
        accept="application/vnd.github+json",
        max_age=IMMUTABLE,
    )
    units: list[CorpusUnit] = []
    for path, text in _walk_tarball(tarball):
        kind = _kind_for(path)
        if kind is None:
            continue
        counters: dict[str | None, int] = {}
        for chunk in chunk_file(path, text):
            ordinal = counters.get(chunk.symbol, 0)
            counters[chunk.symbol] = ordinal + 1
            units.append(
                CorpusUnit(
                    repo=head.repo,
                    kind=kind,
                    identity=f"{path}#{chunk.symbol or ''}#{ordinal}",
                    text=chunk.text,
                    ref=head.sha,
                    path=path,
                    symbol=chunk.symbol,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
            )
    logger.info("%s: %s source and doc units at %s", head.repo, len(units), head.sha[:7])
    return units


async def pull_request_units(client: GitHubClient, repo: str, limit: int) -> list[CorpusUnit]:
    """Merged pull requests only. A closed-unmerged PR is a rejected idea, not context."""
    units = []
    seen = 0
    async for pr in client.paginate(
        f"/repos/{repo}/pulls",
        params={"state": "closed", "sort": "updated", "direction": "desc"},
        max_items=limit,
        max_age=LISTING_MAX_AGE,
    ):
        seen += 1
        if not pr.get("merged_at"):
            continue
        number = int(pr["number"])
        units.append(
            CorpusUnit(
                repo=repo,
                kind="pull_request",
                identity=f"pull/{number}",
                text=_joined(pr.get("title"), pr.get("body")),
                ref=_as_str(pr.get("merge_commit_sha")),
                metadata={
                    "number": number,
                    "title": _as_str(pr.get("title")) or "",
                    "state": "merged",
                    "author": _login(pr.get("user")),
                    "html_url": _as_str(pr.get("html_url")) or "",
                    "merged_at": _as_str(pr.get("merged_at")) or "",
                },
            )
        )
    logger.info("%s: %s merged pull requests from %s closed", repo, len(units), seen)
    return units


async def review_comment_units(client: GitHubClient, repo: str, limit: int) -> list[CorpusUnit]:
    """The point of the whole corpus.

    The text deliberately carries the hunk as well as the remark. A comment on its own
    embeds as an opinion with no subject, and what retrieval needs to match is the pair:
    this shape of code drew this observation.
    """
    units = []
    async for comment in client.paginate(
        f"/repos/{repo}/pulls/comments",
        params={"sort": "created", "direction": "desc"},
        max_items=limit,
        max_age=LISTING_MAX_AGE,
    ):
        body = _as_str(comment.get("body"))
        if not body:
            continue
        comment_id = int(comment["id"])
        path = _as_str(comment.get("path"))
        diff_hunk = _as_str(comment.get("diff_hunk")) or ""
        line = comment.get("line") or comment.get("original_line")
        units.append(
            CorpusUnit(
                repo=repo,
                kind="review_comment",
                identity=f"review_comment/{comment_id}",
                text=_joined(f"{path}:{line}" if path else None, diff_hunk, body),
                ref=_as_str(comment.get("commit_id")),
                path=path,
                start_line=line if isinstance(line, int) else None,
                end_line=line if isinstance(line, int) else None,
                metadata={
                    "comment_id": comment_id,
                    "author": _login(comment.get("user")),
                    "diff_hunk": diff_hunk,
                    "html_url": _as_str(comment.get("html_url")) or "",
                    "side": _as_str(comment.get("side")) or "",
                    "pull_request_number": _pr_number(comment.get("pull_request_url")),
                },
            )
        )
    logger.info("%s: %s review comments", repo, len(units))
    return units


async def issue_units(client: GitHubClient, repo: str, limit: int) -> list[CorpusUnit]:
    """Issues, minus the pull requests. The issues endpoint returns both."""
    units = []
    async for issue in client.paginate(
        f"/repos/{repo}/issues",
        params={"state": "all", "sort": "updated", "direction": "desc"},
        max_items=limit,
        max_age=LISTING_MAX_AGE,
    ):
        if "pull_request" in issue:
            continue
        number = int(issue["number"])
        units.append(
            CorpusUnit(
                repo=repo,
                kind="issue",
                identity=f"issue/{number}",
                text=_joined(issue.get("title"), issue.get("body")),
                metadata={
                    "number": number,
                    "title": _as_str(issue.get("title")) or "",
                    "state": _as_str(issue.get("state")) or "",
                    "author": _login(issue.get("user")),
                    "html_url": _as_str(issue.get("html_url")) or "",
                },
            )
        )
    logger.info("%s: %s issues", repo, len(units))
    return units


def _walk_tarball(tarball: bytes) -> Iterator[tuple[str, str]]:
    """Yield decodable text files, with the archive's own top-level directory stripped.

    Nothing is written to disk. Members are read straight out of the archive, so the
    usual tar path-traversal problem does not arise: the paths are only ever used as
    corpus metadata.
    """
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                continue
            path = member.name.split("/", 1)[-1]
            if not path or _is_skipped(path):
                continue
            if _kind_for(path) is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                yield path, handle.read().decode("utf-8")
            except UnicodeDecodeError:
                # A .txt fixture full of binary, or a file in an encoding nothing else
                # in the repo uses. Neither is worth guessing at.
                logger.debug("skipping undecodable file %s", path)


def _is_skipped(path: str) -> bool:
    parts = path.split("/")
    return any(part in SKIP_DIRECTORIES for part in parts) or any(
        part.endswith(".egg-info") for part in parts
    )


def _kind_for(path: str) -> UnitKind | None:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if suffix in SOURCE_SUFFIXES:
        return "source"
    if suffix in DOC_SUFFIXES:
        return "doc"
    return None


def _joined(*parts: str | Any) -> str:
    return "\n\n".join(str(p).strip() for p in parts if isinstance(p, str) and p.strip())


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _login(user: object) -> str:
    return str(user.get("login", "")) if isinstance(user, dict) else ""


def _pr_number(url: object) -> int | None:
    if not isinstance(url, str):
        return None
    tail = url.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None
