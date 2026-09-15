from pathlib import Path

import httpx
import pytest
import respx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.jobs.record_delivery import repo_outputs
from pr_lens.jobs.review import commented_lines, skip_reason
from pr_lens.review.seeded import branches, seed_source


def test_a_seeded_testbed_branch_names_where_it_was_copied_from() -> None:
    assert seed_source("SURYAPRASATHJP/pr-lens-testbed", "seed/encode__httpx/3312/head") == (
        "encode/httpx",
        3312,
    )


def test_the_branch_seed_pushes_is_the_branch_review_reads() -> None:
    base, head = branches("pydantic/pydantic-settings", 949)
    assert seed_source("suryaprasathjp/pr-lens-testbed", head) == (
        "pydantic/pydantic-settings",
        949,
    )
    assert seed_source("suryaprasathjp/pr-lens-testbed", base) is None


@pytest.mark.parametrize(
    ("repo", "head_ref"),
    [
        ("someone/else", "seed/encode__httpx/3312/head"),
        ("SURYAPRASATHJP/pr-lens-testbed", "feature/seed"),
        ("SURYAPRASATHJP/pr-lens-testbed", "seed/encode__httpx/0/head"),
        ("SURYAPRASATHJP/pr-lens-testbed", "seed/encode__httpx/3312/base"),
        ("SURYAPRASATHJP/pr-lens-testbed", "seed/../x__y/1/head"),
    ],
    ids=["another repo", "not a seed", "pull zero", "base branch", "path tricks"],
)
def test_a_branch_name_means_nothing_anywhere_else(repo: str, head_ref: str) -> None:
    """Anyone opening a pull request chooses its branch name, so outside the testbed a name
    that looks like a seed must not redirect retrieval to another repository."""
    assert seed_source(repo, head_ref) is None


def test_drafts_and_bots_are_not_reviewed() -> None:
    assert skip_reason(True, "dev") is not None
    assert skip_reason(False, "dependabot[bot]") == "opened by a bot"
    assert skip_reason(False, "dev") is None


@pytest.mark.parametrize(
    ("name", "outputs"),
    [
        ("octo/hello.world", [("owner", "octo"), ("repo_name", "hello.world")]),
        ("octo/hello\nowner=evil", []),
        ("no-slash", []),
        (None, []),
    ],
    ids=["ordinary", "newline", "no slash", "none"],
)
def test_the_token_step_gets_owner_and_name_only_for_a_real_repo_name(
    name: str | None, outputs: list[tuple[str, str]]
) -> None:
    assert repo_outputs(name) == outputs


@respx.mock
async def test_lines_already_commented_on_are_collected(tmp_path: Path) -> None:
    respx.get("https://api.github.com/repos/o/r/pulls/7/comments").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"path": "a.py", "line": 3},
                {"path": "a.py", "line": None},
                {"path": "b.py", "line": 9},
            ],
        )
    )
    async with httpx.AsyncClient() as http:
        lines = await commented_lines(GitHubClient(http, "t", HttpCache(tmp_path)), "o/r", 7)
    assert lines == frozenset({("a.py", 3), ("b.py", 9)})
