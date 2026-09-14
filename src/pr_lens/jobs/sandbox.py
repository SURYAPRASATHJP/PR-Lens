"""What one sandbox.yml job does: take a repository, run its tests in the sandbox, report.

    python -m pr_lens.jobs.sandbox repos     the matrix, one entry per mined repository
    python -m pr_lens.jobs.sandbox run       one repository, SANDBOX_REPO, to one JSON row
    python -m pr_lens.jobs.sandbox table     every row in SANDBOX_ROWS, to the markdown table

Configuration comes from the environment and nowhere else: SANDBOX_REPO, SANDBOX_ROWS for
the row directory, SANDBOX_TABLE for the table's path, and GH_MINING_TOKEN to read the
tarball. Never the workflow's GITHUB_TOKEN, for the reason pr_lens.github.mining gives.

Every way a run can end is a row, not an exception. NO_TESTS, a failed install, a
timeout, an OOM kill and a sandbox error are all things Phase 4 needs to see counted, so
the job exits 0 having written one. A job that crashed would leave a hole in the table,
and a hole reads as nothing to report.

The source comes from the tarball of the default branch's head commit. A tarball carries
no .git, and so no credential in a remote URL or an extraheader for the code under
review to find.
"""

import asyncio
import io
import json
import logging
import os
import statistics
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pr_lens.eval.split import MINING_SET
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import IMMUTABLE, GitHubClient, GitHubError
from pr_lens.github.mining import mining_token
from pr_lens.logging import configure
from pr_lens.sandbox import runner
from pr_lens.sandbox.detect import Detection, Framework, detect
from pr_lens.sandbox.evidence import EXCERPT_CHARS, parse
from pr_lens.sandbox.spec import FetchSpec, Outcome, SandboxResult, SandboxSpec

logger = logging.getLogger(__name__)

DEFAULT_ROWS = Path("sandbox-rows")
DEFAULT_TABLE = Path("docs/sandbox/phase-3-runs.md")

# A workflow annotation is how a row leaves a runner in a form anyone can read back
# without a token. GitHub cuts a message at 4096 bytes, found when eleven of the first
# twenty-seven rows came back truncated, so a row is sent as numbered chunks under that.
ANNOTATION_CHUNK = 3500
MAX_ANNOTATION_CHUNKS = 9


@dataclass(slots=True)
class Row:
    repo: str
    outcome: str
    sha: str | None = None
    reason: str = ""
    language: str | None = None
    framework: str | None = None
    notes: list[str] = field(default_factory=list)
    source_mb: float | None = None
    pull_seconds: float | None = None
    fetch_seconds: float | None = None
    fetch_outcome: str | None = None
    requirements: int | None = None
    dropped: list[str] = field(default_factory=list)
    setup_exit_code: int | None = None
    run_seconds: float | None = None
    exit_code: int | None = None
    memory_peak_mb: float | None = None
    evidence: dict[str, Any] | None = None
    excerpt: str = ""
    error: str | None = None


def _tail(result: SandboxResult) -> str:
    text = f"{result.stdout}\n{result.stderr}".strip()
    return text[-EXCERPT_CHARS:]


def extract(tarball: bytes, into: Path) -> int:
    """Unpack a GitHub tarball without its top-level directory. Returns members skipped.

    tarfile's data filter refuses absolute paths, parent references, device files and
    links that leave the tree. A real repository occasionally carries a symlink pointing
    outside itself, so a refused member is skipped and counted rather than failing the run.
    """
    skipped = 0

    def keep(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        nonlocal skipped
        name = member.name.split("/", 1)[1] if "/" in member.name else ""
        if not name:
            return None
        try:
            return tarfile.data_filter(member.replace(name=name, deep=False), path)
        except tarfile.FilterError:
            skipped += 1
            return None

    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        archive.extractall(into, filter=keep)  # noqa: S202 -- keep applies data_filter
    return skipped


async def _source(repo: str, into: Path) -> tuple[str, float, int]:
    """Download and unpack the head of the default branch. Returns sha, MB, skipped."""
    token = mining_token()
    if not token:
        raise GitHubError("GH_MINING_TOKEN is not set")
    with tempfile.TemporaryDirectory() as cache_dir:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
            client = GitHubClient(http, token, HttpCache(Path(cache_dir)))
            metadata = await client.get_json(f"/repos/{repo}")
            branch = str(metadata["default_branch"])
            commit = await client.get_json(f"/repos/{repo}/commits/{branch}")
            sha = str(commit["sha"])
            tarball = await client.get_raw(
                f"/repos/{repo}/tarball/{sha}",
                accept="application/vnd.github+json",
                max_age=IMMUTABLE,
            )
    skipped = extract(tarball, into)
    return sha, len(tarball) / 1e6, skipped


def _outcome(detection: Detection, result: SandboxResult) -> Outcome:
    if (
        result.outcome in (Outcome.PASSED, Outcome.FAILED)
        and result.exit_code in detection.no_tests_exit_codes
    ):
        return Outcome.NO_TESTS
    return result.outcome


async def run_detected(detection: Detection, source: Path, row: Row) -> Row:
    """Pull, fetch, run and parse, for a detection that found a suite."""
    if detection.image is None or detection.fetcher is None:
        raise ValueError(f"detection found a suite but named no image: {detection.reason}")
    row.pull_seconds = round(await runner.pull(detection.image), 2)

    async with runner.deps_volume() as volume:
        started = time.monotonic()
        fetched, dropped = await runner.fetch_dropping_unavailable(
            FetchSpec(
                image=detection.image,
                fetcher=detection.fetcher,
                source_dir=source,
                deps_volume=volume,
                requirements=detection.requirements,
            )
        )
        row.fetch_seconds = round(time.monotonic() - started, 2)
        row.fetch_outcome = fetched.outcome.value
        row.dropped = list(dropped)
        if fetched.outcome is not Outcome.PASSED:
            row.outcome = Outcome.INSTALL_FAILED.value
            row.excerpt = _tail(fetched)
            return row

        result = await runner.run(
            SandboxSpec(
                image=detection.image,
                command=detection.command,
                source_dir=source,
                setup=detection.setup,
                deps_volume=volume,
                env=detection.env,
            )
        )

    outcome = _outcome(detection, result)
    evidence = parse(detection.framework or Framework.UNKNOWN, result.stdout, result.stderr)
    row.outcome = outcome.value
    row.run_seconds = round(result.duration_seconds, 2)
    row.exit_code = result.exit_code
    row.setup_exit_code = result.setup_exit_code
    if result.memory_peak_bytes is not None:
        row.memory_peak_mb = round(result.memory_peak_bytes / 2**20, 1)
    row.evidence = evidence.to_dict()
    if outcome in (Outcome.ERROR, Outcome.TIMED_OUT, Outcome.OOM_KILLED) or (
        outcome is Outcome.FAILED and not evidence.failures
    ):
        row.excerpt = _tail(result)
    return row


async def run_repo(repo: str) -> Row:
    row = Row(repo=repo, outcome=Outcome.ERROR.value)
    with tempfile.TemporaryDirectory(prefix="pr-lens-src-") as workspace:
        source = Path(workspace) / "src"
        source.mkdir()
        try:
            row.sha, size, skipped = await _source(repo, source)
        except GitHubError as exc:
            row.error = f"could not fetch the source: {exc}"
            return row
        row.source_mb = round(size, 1)
        if skipped:
            row.notes.append(f"{skipped} tarball members refused by the extraction filter")

        detection = detect(source)
        row.reason = detection.reason
        row.notes += list(detection.notes)
        row.language = detection.language.value if detection.language else None
        row.framework = detection.framework.value if detection.framework else None
        row.requirements = len(detection.requirements)
        if not detection.found:
            row.outcome = Outcome.NO_TESTS.value
            return row
        try:
            return await run_detected(detection, source, row)
        except runner.SandboxError as exc:
            row.error = str(exc)
            return row


def annotate(title: str, message: str) -> None:
    """Notice annotations, readable from the public API, unlike the job log.

    Titled "title i/n" so the chunks can be put back in order. A message too long for
    MAX_ANNOTATION_CHUNKS keeps its tail, where a runner's summary is.
    """
    budget = ANNOTATION_CHUNK * MAX_ANNOTATION_CHUNKS
    if len(message) > budget:
        message = message[-budget:]
    chunks = [message[i : i + ANNOTATION_CHUNK] for i in range(0, len(message), ANNOTATION_CHUNK)]
    for index, chunk in enumerate(chunks or [""], start=1):
        escaped = chunk.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        sys.stdout.write(f"::notice title={title} {index}/{len(chunks)}::{escaped}\n")
    sys.stdout.flush()


def _seconds(values: list[float]) -> str:
    if not values:
        return "none measured"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return (
        f"p50 {statistics.median(ordered):.1f}s, p95 {p95:.1f}s, max {ordered[-1]:.1f}s "
        f"over {len(ordered)}"
    )


def _cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/").replace("\n", " ")


def render(rows: list[Row]) -> str:
    rows = sorted(rows, key=lambda row: row.repo.lower())
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.outcome] = counts.get(row.outcome, 0) + 1

    lines = [
        "# Phase 3 sandbox runs on the mined repositories",
        "",
        f"{len(rows)} repositories, each at the head of its default branch, each on a fresh",
        "GitHub-hosted runner. Generated by `python -m pr_lens.jobs.sandbox table` from the",
        "rows the sandbox.yml real-data jobs wrote. Nothing here is edited by hand.",
        "",
        "## Outcomes",
        "",
        "| outcome | repositories |",
        "|---|---|",
        *(f"| {outcome} | {count} |" for outcome, count in sorted(counts.items())),
        "",
        "## Latency",
        "",
        "Image pull is measured on its own, because on a cold runner it is the latency",
        "nobody budgets for.",
        "",
        f"    image pull   {_seconds([r.pull_seconds for r in rows if r.pull_seconds])}",
        f"    fetch        {_seconds([r.fetch_seconds for r in rows if r.fetch_seconds])}",
        f"    test run     {_seconds([r.run_seconds for r in rows if r.run_seconds])}",
        "",
        "## Per repository",
        "",
        "| repository | outcome | pull s | fetch s | run s | peak MB | passed | failed "
        "| errors | reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        tallies = (row.evidence or {}).get("counts") or {}
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row.repo,
                    row.outcome,
                    row.pull_seconds,
                    row.fetch_seconds,
                    row.run_seconds,
                    row.memory_peak_mb,
                    tallies.get("passed"),
                    tallies.get("failed"),
                    tallies.get("errors"),
                    row.error or row.reason,
                )
            )
            + " |"
        )

    lines += ["", "## What each repository said", ""]
    for row in rows:
        lines.append(f"### {row.repo}")
        lines.append("")
        lines.append(f"    commit    {row.sha}")
        lines.append(f"    outcome   {row.outcome}")
        if row.requirements is not None:
            lines.append(f"    fetched   {row.requirements} requirements, {row.fetch_outcome}")
        for requirement in row.dropped:
            lines.append(f"    dropped   {requirement}, no wheel")
        if row.setup_exit_code is not None:
            lines.append(f"    setup     exit {row.setup_exit_code}")
        for note in row.notes:
            lines.append(f"    note      {note}")
        failures = (row.evidence or {}).get("failures") or []
        for failure in failures[:3]:
            where = f"{failure['file']}:{failure['line']}" if failure.get("line") else ""
            lines.append(f"    failure   {failure['test']} {where}")
            lines.append(f"              {failure['message'][:160]}")
        if row.excerpt:
            tail = [line for line in row.excerpt.splitlines() if line.strip()][-6:]
            lines += [f"    tail      {line[:160]}" for line in tail]
        lines.append("")
    return "\n".join(lines)


def _rows_dir() -> Path:
    return Path(os.environ.get("SANDBOX_ROWS", str(DEFAULT_ROWS)))


def main(argv: list[str]) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    command = argv[1] if len(argv) > 1 else ""

    if command == "repos":
        sys.stdout.write(json.dumps(sorted(MINING_SET, key=str.lower)) + "\n")
        return 0

    if command == "run":
        repo = os.environ.get("SANDBOX_REPO", "")
        if repo not in MINING_SET:
            logger.error("SANDBOX_REPO %r is not a mined repository", repo)
            return 1
        started = time.monotonic()
        row = asyncio.run(run_repo(repo))
        logger.info("%s: %s in %.0fs", repo, row.outcome, time.monotonic() - started)
        directory = _rows_dir()
        directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(row), sort_keys=True)
        (directory / f"{repo.replace('/', '__')}.json").write_text(text, encoding="utf-8")
        annotate(f"sandbox-row {repo}", text)
        return 0

    if command == "table":
        rows = [
            Row(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(_rows_dir().rglob("*.json"))
        ]
        table = render(rows)
        target = Path(os.environ.get("SANDBOX_TABLE", str(DEFAULT_TABLE)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(table + "\n", encoding="utf-8")
        annotate("sandbox-table", table)
        return 0

    logger.error("usage: python -m pr_lens.jobs.sandbox repos|run|table")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
