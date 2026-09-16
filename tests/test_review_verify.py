from pathlib import Path
from typing import Any

import pytest

from pr_lens.github.client import GitHubError
from pr_lens.ingest.diff import parse_patch
from pr_lens.review.verify import (
    MAX_REPORTED,
    Verification,
    regressions,
    render,
    select,
    unusable,
    verify,
)
from pr_lens.sandbox.evidence import Failure
from pr_lens.sandbox.runner import SandboxError
from pr_lens.sandbox.session import Row
from pr_lens.sandbox.spec import Outcome

PATCH = "@@ -1,2 +1,2 @@\n def evict(keys):\n-    keys.pop()\n+    keys.pop(0)\n"


def hunks(*paths: str) -> list[Any]:
    return [parse_patch(PATCH, path)[0] for path in paths]


def tree(root: Path, *paths: str) -> Path:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("", encoding="utf-8")
    return root


def failure(test: str, message: str = "AssertionError") -> dict[str, Any]:
    return {
        "test": test,
        "kind": "failure",
        "file": "tests/test_cache.py",
        "line": 7,
        "message": message,
    }


def row(outcome: Outcome, *failures: dict[str, Any], dropped: list[str] | None = None) -> Row:
    return Row(
        repo="o/r",
        outcome=outcome.value,
        dropped=dropped or [],
        evidence={"failures": list(failures)},
    )


def test_the_tests_near_a_diff_are_the_ones_it_changed_and_the_ones_named_for_it(
    tmp_path: Path,
) -> None:
    source = tree(
        tmp_path,
        "pkg/cache.py",
        "tests/test_cache.py",
        "tests/test_unrelated.py",
        "pkg/store_test.py",
    )
    assert select(hunks("pkg/cache.py", "tests/test_unrelated.py", "pkg/store.py"), source) == (
        "pkg/store_test.py",
        "tests/test_cache.py",
        "tests/test_unrelated.py",
    )


def test_a_project_that_does_not_name_its_tests_that_way_gets_no_verification(
    tmp_path: Path,
) -> None:
    """Silence, rather than a whole suite a fifth of repositories cannot afford."""
    source = tree(tmp_path, "pkg/cache.py", "tests/check_everything.py")
    assert select(hunks("pkg/cache.py"), source) == ()
    assert select(hunks("README.md"), source) == ()


def test_an_init_file_does_not_select_every_test_named_init(tmp_path: Path) -> None:
    source = tree(tmp_path, "pkg/__init__.py", "tests/test___init__.py")
    assert select(hunks("pkg/__init__.py"), source) == ()


def test_only_a_test_that_passed_before_and_fails_now_is_a_regression() -> None:
    base = row(Outcome.FAILED, failure("test_old_flake"))
    head = row(Outcome.FAILED, failure("test_old_flake"), failure("test_evict_oldest"))
    assert [f.test for f in regressions(base, head)] == ["test_evict_oldest"]


def test_a_base_that_never_ran_proves_nothing() -> None:
    head = row(Outcome.FAILED, failure("test_evict_oldest"))
    assert regressions(row(Outcome.INSTALL_FAILED), head) == ()
    assert regressions(row(Outcome.TIMED_OUT), head) == ()


@pytest.mark.parametrize(
    ("side_row", "says"),
    [
        (row(Outcome.INSTALL_FAILED), "install_failed"),
        (row(Outcome.TIMED_OUT), "timed_out"),
        (row(Outcome.OOM_KILLED), "oom_killed"),
        (row(Outcome.NO_TESTS), "no_tests"),
        (row(Outcome.PASSED, dropped=["pyspark"]), "dropped 1 dependencies"),
    ],
    ids=["install", "timeout", "oom", "no tests", "dropped"],
)
def test_a_run_the_sandbox_spoiled_says_nothing_about_the_pull_request(
    side_row: Row, says: str
) -> None:
    """Phase 3's outcomes, and the dependency review's finding: a suite missing a dropped
    package is not the repository's suite."""
    reason = unusable(side_row, "head")
    assert reason is not None
    assert says in reason


def test_a_clean_run_is_usable_whether_it_passed_or_failed() -> None:
    assert unusable(row(Outcome.PASSED), "head") is None
    assert unusable(row(Outcome.FAILED, failure("test_x")), "head") is None


@pytest.mark.parametrize(
    "failing",
    [GitHubError("the commit is gone"), SandboxError("docker is not running")],
    ids=["commit gone", "no docker"],
)
async def test_a_sandbox_that_cannot_run_is_silence_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, failing: Exception
) -> None:
    """A review is not lost because the sandbox was unavailable; it goes on without it."""

    async def raises(*args: Any, **kwargs: Any) -> tuple[str, float, int]:
        raise failing

    monkeypatch.setattr("pr_lens.review.verify.source_at", raises)
    result = await verify("o/r", "b" * 40, "h" * 40, hunks("pkg/cache.py"))
    assert not result.ran
    assert str(failing) in result.reason
    assert result.regressions == ()


def test_nothing_renders_unless_there_is_a_regression() -> None:
    assert render(Verification(False, "no test file near the diff")) == ""
    assert render(Verification(True, "", ("tests/test_cache.py",))) == ""


def test_the_block_names_the_tests_and_stops_at_a_few() -> None:
    found = tuple(
        Failure(f"test_{n}", "failure", "tests/test_cache.py", n, "AssertionError: x")
        for n in range(MAX_REPORTED + 2)
    )
    block = render(Verification(True, "", ("tests/test_cache.py",), found))
    assert "pass before it and fail on it" in block
    assert "- test_0 (tests/test_cache.py:0): AssertionError: x" in block
    assert f"and {len(found) - MAX_REPORTED} more" in block
    assert f"test_{MAX_REPORTED}" not in block.replace(f"and {len(found) - MAX_REPORTED} more", "")
