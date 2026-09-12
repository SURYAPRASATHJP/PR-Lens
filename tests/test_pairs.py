import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pr_lens.eval.pairs import (
    MAX_PAIRS_PER_PR,
    RepoPairs,
    ReviewPair,
    before_state,
    cap_per_pr,
    cap_per_repo,
    mine_repo_pairs,
    nit_reason,
    screen_comment,
)
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import API_ROOT, GitHubClient
from pr_lens.ingest.mine import MiningLimits
from pr_lens.ingest.units import unit_id_for

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "review_comments.json").read_text(encoding="utf-8")
)["comments"]


@pytest.mark.parametrize("case", FIXTURES, ids=lambda c: c["case"])
def test_every_pathological_case_lands_where_expected(case: dict[str, Any]) -> None:
    screened = screen_comment(case["payload"])
    if case["expect"] == "pair":
        assert isinstance(screened, ReviewPair)
    else:
        assert screened == case["expect"]


def test_before_state_keeps_context_and_removals_and_drops_additions() -> None:
    hunk = (
        '@@ -517,7 +517,7 @@ name = "exceptiongroup"\n'
        " dependencies = [\n"
        "-    old_line\n"
        "+    new_line\n"
        "\\ No newline at end of file"
    )
    assert before_state(hunk) == 'name = "exceptiongroup"\ndependencies = [\n    old_line'


def test_before_state_of_an_all_addition_hunk_is_empty() -> None:
    assert before_state("@@ -0,0 +1,2 @@\n+a = 1\n+b = 2") == ""


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("LGTM", "acknowledgement"),
        ("+1", "acknowledgement"),
        ("🚀🚀", "emoji_only"),
        ("Thanks! Fixed in the latest commit, cheers.", "acknowledgement"),
        ("Could you add a test?", "short"),
        ("Nice, but this breaks the fallback path when the cache directory is missing.", None),
        ("This will raise on Windows because the path separator is hardcoded here.", None),
    ],
)
def test_the_nit_filter(body: str, reason: str | None) -> None:
    assert nit_reason(body) == reason


def pair(comment_id: int, pr: int, repo: str = "octocat/hello-world") -> ReviewPair:
    return ReviewPair(
        repo=repo,
        comment_id=comment_id,
        pull_request=pr,
        path="a.py",
        diff_hunk="@@ -1 +1 @@\n-a\n+b",
        body="x" * 50,
        author="reviewer",
        created_at="",
        html_url="",
    )


def test_a_pull_request_contributes_at_most_three_pairs_and_the_earliest_win() -> None:
    result = RepoPairs(repo="octocat/hello-world")
    kept = cap_per_pr([pair(i, pr=1) for i in (50, 10, 40, 30, 20)] + [pair(99, pr=2)], result)
    assert [p.comment_id for p in kept if p.pull_request == 1] == [10, 20, 30]
    assert len([p for p in kept if p.pull_request == 1]) == MAX_PAIRS_PER_PR
    assert result.dropped["per_pr_cap"] == 2


def test_every_repo_is_capped_at_the_median_and_the_sample_is_stable() -> None:
    def results() -> list[RepoPairs]:
        return [
            RepoPairs(repo="a", pairs=[pair(i, pr=i) for i in range(100)]),
            RepoPairs(repo="b", pairs=[pair(i, pr=i) for i in range(1000, 1010)]),
            RepoPairs(repo="c", pairs=[pair(i, pr=i) for i in range(2000, 2004)]),
        ]

    first, second = results(), results()
    assert cap_per_repo(first) == 10
    cap_per_repo(second)
    assert [len(r.pairs) for r in first] == [10, 10, 4]
    assert first[0].dropped["per_repo_cap"] == 90
    assert [p.comment_id for p in first[0].pairs] == [p.comment_id for p in second[0].pairs]
    # A hash sample, not the first ten by id, so the cap does not favour older comments.
    assert [p.comment_id for p in first[0].pairs] != list(range(10))


REPO = "fastapi/typer"


def comment(
    comment_id: int,
    pr: int,
    author: str = "reviewer",
    body: str = "This will raise on Windows because the separator is hardcoded.",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": comment_id,
        "in_reply_to_id": None,
        "path": "typer/main.py",
        "diff_hunk": "@@ -1,3 +1,3 @@ def main():\n     x = 1\n-    y = '/'\n+    y = os.sep",
        "body": body,
        "user": {"login": author, "type": "User"},
        "pull_request_url": f"{API_ROOT}/repos/{REPO}/pulls/{pr}",
        "created_at": "2026-09-01T00:00:00Z",
        "html_url": f"https://github.com/{REPO}/pull/{pr}#discussion_r{comment_id}",
        **extra,
    }


def pull(number: int, author: str, merged: bool) -> dict[str, Any]:
    return {
        "number": number,
        "user": {"login": author},
        "merged_at": "2026-09-02T00:00:00Z" if merged else None,
    }


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[GitHubClient]:
    async with httpx.AsyncClient() as http:
        yield GitHubClient(http, "test-token", HttpCache(tmp_path / "cache"))


@respx.mock
async def test_mining_applies_every_rule_and_fetches_only_the_prs_it_needs(
    client: GitHubClient,
) -> None:
    comments = [
        comment(1, pr=10),  # kept
        comment(2, pr=11, author="author-of-11"),  # self review
        comment(3, pr=12),  # PR not merged
        comment(4, pr=13),  # PR outside the listing window, fetched on its own, kept
        comment(5, pr=14),  # gold not in the corpus snapshot
        comment(6, pr=10, body="LGTM"),  # nit, and never costs a PR lookup
        comment(7, pr=99, in_reply_to_id=1),  # reply
    ]
    respx.get(f"{API_ROOT}/repos/{REPO}/pulls/comments").mock(
        return_value=httpx.Response(200, json=comments)
    )
    respx.get(f"{API_ROOT}/repos/{REPO}/pulls").mock(
        return_value=httpx.Response(
            200,
            json=[pull(10, "someone", True), pull(11, "author-of-11", True), pull(12, "x", False)],
        )
    )
    single = respx.get(f"{API_ROOT}/repos/{REPO}/pulls/13").mock(
        return_value=httpx.Response(200, json=pull(13, "someone", True))
    )
    never = respx.get(f"{API_ROOT}/repos/{REPO}/pulls/99")

    corpus = {
        unit_id_for(REPO, "review_comment", f"review_comment/{i}") for i in (1, 2, 3, 4, 6, 7)
    }
    result = await mine_repo_pairs(client, REPO, corpus, MiningLimits())

    assert [p.comment_id for p in result.pairs] == [1, 4]
    assert all(p.repo == REPO for p in result.pairs)
    assert result.seen == 7
    assert dict(result.dropped) == {
        "self_review": 1,
        "not_merged": 1,
        "gold_not_in_corpus": 1,
        "acknowledgement": 1,
        "reply": 1,
    }
    assert single.call_count == 1
    assert not never.called
