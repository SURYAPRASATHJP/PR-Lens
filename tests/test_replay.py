import json
from pathlib import Path

import httpx
import pytest
import respx

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.eval.corpus import read_corpus
from pr_lens.eval.pairs import ReviewPair
from pr_lens.eval.parts import EMBED_PARTS
from pr_lens.eval.split import HOLDOUT, TUNE, HoldoutViolation
from pr_lens.eval.vectors import corpus_part
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.jobs.embed import embed_input
from pr_lens.jobs.replay import Chosen, choose, render, replay_one, spent_the_day
from pr_lens.retrieval.embed import MODELS
from pr_lens.review.pipeline import NoComment, Review
from pr_lens.review.provider import ChatClient, Inference, Provider
from pr_lens.review.retrieve import load_index
from tests.fakes import BagEncoder

API = "https://api.github.com"
LLM = "https://llm.example/v1/chat/completions"
TUNED = sorted(TUNE)
MODEL = MODELS["gte-modernbert"]


def pair(
    repo: str, pull: int, comment_id: int = 1, body: str = "Check the empty case."
) -> ReviewPair:
    return ReviewPair(
        repo=repo,
        comment_id=comment_id,
        pull_request=pull,
        path="pkg/cache.py",
        diff_hunk="@@ -1 +1 @@\n-a\n+b\n",
        body=body,
        author="maintainer",
        created_at="2025-01-01T00:00:00Z",
        html_url=f"https://github.com/{repo}/pull/{pull}#discussion_r{comment_id}",
    )


PAIRS = [pair(repo, pull, pull) for repo in TUNED[:3] for pull in (10, 20, 30, 40)]


def test_a_batch_spreads_across_repos_and_always_means_the_same_pulls() -> None:
    first = choose(PAIRS, 6, "b1", set())
    assert len(first) == 6
    assert len({c.repo for c in first[:3]}) == 3
    assert [(c.repo, c.number) for c in choose(PAIRS, 6, "b1", set())] == [
        (c.repo, c.number) for c in first
    ]
    assert {(c.repo, c.number) for c in choose(PAIRS, 6, "b2", set())} != {
        (c.repo, c.number) for c in first
    }


def test_pulls_another_batch_drafted_are_not_chosen_again() -> None:
    taken = {(c.repo, c.number) for c in choose(PAIRS, 6, "b1", set())}
    later = choose(PAIRS, 12, "b2", taken)
    assert len(later) == 6
    assert not taken & {(c.repo, c.number) for c in later}


def test_the_pairs_of_one_pull_request_travel_together() -> None:
    both = [pair(TUNED[0], 10, 1, "first"), pair(TUNED[0], 10, 2, "second")]
    [chosen] = choose(both, 5, "b", set())
    assert [r["body"] for r in chosen.reference] == ["first", "second"]


def test_a_holdout_pair_stops_the_batch_before_anything_is_drafted() -> None:
    with pytest.raises(HoldoutViolation):
        choose([*PAIRS, pair(sorted(HOLDOUT)[0], 5)], 3, "b", set())


def test_only_a_limit_before_any_draft_counts_as_the_day_spent() -> None:
    assert spent_the_day(Review(NoComment.RATE_LIMITED))
    assert not spent_the_day(Review(NoComment.MODEL_SILENT))
    assert not spent_the_day(Review(None))


def completion(content: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 100},
        },
    )


@pytest.fixture
def sink(tmp_path: Path) -> LocalSink:
    repo = TUNED[0]
    store = LocalSink(tmp_path / "corpus")
    units = []
    for comment_id, pull, body in [
        (1, 5, "evict keys from the cache oldest first"),
        (2, 30, "evict keys answer key from pull thirty"),
    ]:
        unit = review_comment_unit(
            repo,
            {
                "id": comment_id,
                "path": "pkg/cache.py",
                "line": 2,
                "diff_hunk": "@@ -1 +1 @@\n-a\n+b\n",
                "body": body,
                "user": {"login": "r"},
                "pull_request_url": f"https://api.github.com/repos/{repo}/pulls/{pull}",
            },
        )
        assert unit is not None
        units.append(unit)
    write_units(store, repo, units)
    corpus = read_corpus(store, repo)
    for part in range(EMBED_PARTS[MODEL.key]):
        embed_input(
            store, corpus_part(corpus, MODEL, part, EMBED_PARTS[MODEL.key]), BagEncoder(MODEL)
        )
    return store


@respx.mock
async def test_one_pull_request_replays_through_the_pipeline_without_its_answer_key(
    sink: LocalSink, tmp_path: Path
) -> None:
    repo = TUNED[0]
    respx.get(f"{API}/repos/{repo}/pulls/30").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Evict oldest",
                "body": "",
                "user": {"login": "dev"},
                "base": {"sha": "b" * 40},
                "head": {"sha": "h" * 40},
            },
        )
    )
    respx.get(f"{API}/repos/{repo}/pulls/30/files").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "filename": "pkg/cache.py",
                    "status": "modified",
                    "changes": 2,
                    "patch": (
                        "@@ -1,2 +1,2 @@\n def evict(keys):\n-    keys.pop()\n+    keys.pop(0)\n"
                    ),
                }
            ],
        )
    )
    respx.get(f"{API}/repos/{repo}/contents/pkg/cache.py").mock(
        return_value=httpx.Response(200, content=b"def evict(keys):\n    keys.pop(0)\n")
    )
    llm = respx.post(LLM).mock(
        return_value=completion({"plan": "fine", "drafts": [], "no_comment_reason": "ok"})
    )
    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, "t", HttpCache(tmp_path / "cache"))
        inference = Inference(
            [ChatClient(http, Provider("l", "https://llm.example/v1", "m", "L"), "k")]
        )
        index = load_index(sink, repo, BagEncoder(MODEL))
        result, head_sha, timings = await replay_one(client, Chosen(repo, 30, ()), index, inference)

    assert head_sha == "h" * 40
    assert result.no_comment is NoComment.MODEL_SILENT
    assert set(timings) == {"fetch", "review"}
    prompt = json.loads(llm.calls.last.request.content)["messages"][1]["content"]
    assert "oldest first" in prompt
    assert "answer key" not in prompt


@respx.mock
async def test_a_pull_request_that_cannot_be_read_is_recorded_not_raised(
    sink: LocalSink, tmp_path: Path
) -> None:
    repo = TUNED[0]
    respx.get(f"{API}/repos/{repo}/pulls/30").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, "t", HttpCache(tmp_path / "cache"))
        inference = Inference(
            [ChatClient(http, Provider("l", "https://llm.example/v1", "m", "L"), "k")]
        )
        index = load_index(sink, repo, BagEncoder(MODEL))
        result, head_sha, _ = await replay_one(client, Chosen(repo, 30, ()), index, inference)
    assert result.no_comment is NoComment.FETCH_FAILED
    assert head_sha == ""


def test_the_summary_counts_outcomes() -> None:
    rows = [
        (Chosen("o/r", 1, ()), Review(NoComment.MODEL_SILENT)),
        (Chosen("o/r", 2, ()), Review(None)),
    ]
    table = render("b", rows)
    assert "| o/r#1 | model_silent |" in table
    assert "| o/r#2 | commented |" in table
    assert "2 pull requests, 0 comments kept, 1 silent." in table
