"""One repository at one commit, fetched, installed and run, as a record of what happened.

Shared by the Phase 3 table job and by Phase 4's base-versus-head comparison, so both run
a suite exactly the same way. jobs/ is the Actions entry-point layer and nothing under
src/ imports from it, which is why this lives beside the runner rather than beside the job
that first needed it.
"""

import io
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import IMMUTABLE, GitHubClient, GitHubError
from pr_lens.github.mining import mining_token
from pr_lens.sandbox import runner
from pr_lens.sandbox.detect import Detection, Framework, detect
from pr_lens.sandbox.evidence import EXCERPT_CHARS, parse
from pr_lens.sandbox.spec import FetchSpec, Outcome, SandboxResult, SandboxSpec

logger = logging.getLogger(__name__)


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


async def source_at(
    repo: str, into: Path, sha: str | None = None, token: str | None = None
) -> tuple[str, float, int]:
    """Download and unpack one commit, the default branch's head unless a sha is given.

    Returns the sha, the tarball's size in MB and the members the extraction filter refused.
    Phase 4 passes a sha: it runs the base commit and the head of a pull request.
    """
    token = token or mining_token()
    if not token:
        raise GitHubError("no token: set GH_MINING_TOKEN, or pass an installation token")
    with tempfile.TemporaryDirectory() as cache_dir:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
            client = GitHubClient(http, token, HttpCache(Path(cache_dir)))
            if sha is None:
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


async def run_repo(repo: str, sha: str | None = None) -> Row:
    """Detect and run one repository's suite, at its default branch head unless given a sha."""
    row = Row(repo=repo, outcome=Outcome.ERROR.value)
    with tempfile.TemporaryDirectory(prefix="pr-lens-src-") as workspace:
        source = Path(workspace) / "src"
        source.mkdir()
        try:
            row.sha, size, skipped = await source_at(repo, source, sha)
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
