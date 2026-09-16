"""Did this pull request break a test that passed before it?

The Phase 3 table is why this is narrow. Of 222 failures recorded across 27 repositories,
6 were about the code; the rest were environment and version differences. So a failing test
is not evidence about a pull request. Only a test that passes at the base commit and fails
at the head is, and that is the only thing this module reports.

Two economies keep it affordable at Phase 3's measured cost, fetch p95 123 s and a run
capped at 300 s. Only the tests nearest the diff are run, not the suite, which is what the
table said a fifth of repositories cannot afford. And the head is run first: with nothing
failing there, no regression is possible and the base is never fetched at all.

Silence rather than a guess, every time the ground is unsure:

    no test file near the diff     nothing to run
    the language is not Python     selecting tests for a pytest run is what this knows
    install failed, timed out,     Phase 3's outcomes that say the sandbox, not the code
    ran out of memory, errored
    a dependency was dropped       the fetch drops a package with no wheel and records it,
                                   so the suite that ran is not the repository's suite
"""

import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pr_lens.github.client import GitHubError
from pr_lens.ingest.diff import Hunk
from pr_lens.sandbox.detect import Language, detect
from pr_lens.sandbox.evidence import Failure
from pr_lens.sandbox.runner import SandboxError
from pr_lens.sandbox.session import Row, run_detected, source_at
from pr_lens.sandbox.spec import Outcome

logger = logging.getLogger(__name__)

# Enough to cover a change that touches a few modules, few enough to stay inside the run
# timeout. A pull request whose diff names more test files than this is not a small one.
MAX_SELECTED = 20

# Shown to the drafting call. More than a few regressions is a broken branch, not a review.
MAX_REPORTED = 5
MAX_MESSAGE_CHARS = 200

RAN_CLEANLY = (Outcome.PASSED, Outcome.FAILED)


@dataclass(frozen=True, slots=True)
class Verification:
    ran: bool
    reason: str = ""
    selected: tuple[str, ...] = ()
    regressions: tuple[Failure, ...] = ()
    head_outcome: str = ""
    base_outcome: str = ""
    seconds: float = 0.0


def _is_test_file(path: Path) -> bool:
    name = path.name
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def select(hunks: Sequence[Hunk], source: Path) -> tuple[str, ...]:
    """The test files nearest the diff: the ones it changed, and the ones named for what it
    changed. `tests/test_cache.py` is near `pkg/cache.py` by the convention pytest projects
    follow, and a project that does not follow it gets nothing selected and no verification,
    which is silence rather than a whole suite nobody can afford."""
    changed = {hunk.path for hunk in hunks}
    selected = {path for path in changed if _is_test_file(Path(path))}
    stems = {Path(path).stem for path in changed if path.endswith(".py")} - {"__init__"}
    wanted = {f"test_{stem}.py" for stem in stems} | {f"{stem}_test.py" for stem in stems}
    for path in sorted(source.rglob("*.py")):
        if path.name in wanted:
            selected.add(path.relative_to(source).as_posix())
    return tuple(sorted(selected)[:MAX_SELECTED])


def regressions(base: Row, head: Row) -> tuple[Failure, ...]:
    """Failures at the head that the base did not have. Both rows must have run cleanly."""
    if base.outcome not in {o.value for o in RAN_CLEANLY}:
        return ()
    healthy = {failure["test"] for failure in (base.evidence or {}).get("failures", [])}
    return tuple(
        Failure(**failure)
        for failure in (head.evidence or {}).get("failures", [])
        if failure["test"] not in healthy
    )


def unusable(row: Row, side: str) -> str | None:
    """Why this side's run says nothing about the pull request, or None when it does."""
    if row.outcome not in {o.value for o in RAN_CLEANLY}:
        return f"the {side} commit's tests came back {row.outcome}"
    if row.dropped:
        return (
            f"the {side} commit's install dropped {len(row.dropped)} dependencies, so the "
            "suite that ran was not the repository's"
        )
    return None


async def verify(
    repo: str, base_sha: str, head_sha: str, hunks: Sequence[Hunk], token: str | None = None
) -> Verification:
    """Run the tests near the diff at the head, and at the base only if the head failed.

    Never raises. A sandbox that cannot run, a commit that cannot be fetched and a daemon
    that is not there are all the same answer as a test that says nothing: no evidence,
    with the reason recorded.
    """
    try:
        return await _verify(repo, base_sha, head_sha, hunks, token)
    except (GitHubError, SandboxError) as exc:
        logger.warning("the sandbox could not run for %s: %s", repo, exc)
        return Verification(False, f"the sandbox could not run: {exc}"[:300])


async def _verify(
    repo: str, base_sha: str, head_sha: str, hunks: Sequence[Hunk], token: str | None
) -> Verification:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pr-lens-verify-") as workspace:
        head_dir = Path(workspace) / "head"
        head_dir.mkdir()
        await source_at(repo, head_dir, head_sha, token)
        detection = detect(head_dir)
        if not detection.found:
            return Verification(False, detection.reason, seconds=_since(started))
        if detection.language is not Language.PYTHON:
            return Verification(
                False, "selecting the tests near a diff is a pytest rule", seconds=_since(started)
            )
        selected = select(hunks, head_dir)
        if not selected:
            return Verification(False, "no test file near the diff", seconds=_since(started))

        chosen = replace(detection, command=(*detection.command, *selected))
        head = await run_detected(chosen, head_dir, Row(repo=repo, outcome=Outcome.ERROR.value))
        blocked = unusable(head, "head")
        if blocked:
            return Verification(
                False, blocked, selected, head_outcome=head.outcome, seconds=_since(started)
            )
        if not (head.evidence or {}).get("failures"):
            return Verification(
                True,
                "nothing near the diff fails at the head",
                selected,
                head_outcome=head.outcome,
                seconds=_since(started),
            )

        base_dir = Path(workspace) / "base"
        base_dir.mkdir()
        await source_at(repo, base_dir, base_sha, token)
        base = await run_detected(chosen, base_dir, Row(repo=repo, outcome=Outcome.ERROR.value))
        blocked = unusable(base, "base")
        if blocked:
            return Verification(
                False,
                blocked,
                selected,
                head_outcome=head.outcome,
                base_outcome=base.outcome,
                seconds=_since(started),
            )
        found = regressions(base, head)
    logger.info("%s: %s regressions from %s selected tests", repo, len(found), len(selected))
    return Verification(
        True,
        "",
        selected,
        found,
        head_outcome=head.outcome,
        base_outcome=base.outcome,
        seconds=_since(started),
    )


def _since(started: float) -> float:
    return round(time.monotonic() - started, 2)


def render(verification: Verification) -> str:
    """The block the drafting call sees, or nothing at all when there is no evidence."""
    if not verification.ran or not verification.regressions:
        return ""
    rows = [
        "The tests nearest this diff were run at the base commit and at this pull request's "
        "head. These pass before it and fail on it:"
    ]
    for failure in verification.regressions[:MAX_REPORTED]:
        where = f" ({failure.file}:{failure.line})" if failure.file else ""
        rows.append(f"- {failure.test}{where}: {failure.message[:MAX_MESSAGE_CHARS]}")
    if len(verification.regressions) > MAX_REPORTED:
        rows.append(f"- and {len(verification.regressions) - MAX_REPORTED} more")
    return "\n".join(rows)
