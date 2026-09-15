import json
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from pr_lens.eval.corpus import EvalDocument
from pr_lens.ingest.diff import parse_patch
from pr_lens.retrieval.embed import MODELS
from pr_lens.review import prompts
from pr_lens.review.context import PullRequest, Skipped
from pr_lens.review.pipeline import (
    MAX_COMMENTS_PER_PR,
    DraftAnswer,
    DraftItem,
    Fate,
    FilterAnswer,
    NoComment,
    Verdict,
    review,
)
from pr_lens.review.provider import ChatClient, Inference, Provider
from pr_lens.review.retrieve import CommentIndex
from tests.fakes import BagEncoder

URL = "https://llm.example/v1/chat/completions"
PROVIDER = Provider("llm", "https://llm.example/v1", "m", "LLM_API_KEY")
PULL = PullRequest("o/r", 30, "Evict the oldest entry", "", "dev", "b" * 40, "h" * 40, False)
PATCH = (
    "@@ -1,3 +1,5 @@ def evict(cache):\n"
    "     keys = list(cache)\n"
    "-    cache.pop(keys[-1])\n"
    "+    if not keys:\n"
    "+        return\n"
    "+    cache.pop(keys[0])\n"
    "     return cache\n"
)
HUNK = parse_patch(PATCH, "pkg/cache.py")[0]


def completion(content: Any, finish_reason: str = "stop") -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def draft(
    line: int, body: str = "Popping keys[0] on an empty dict raises KeyError.", **flags: Any
) -> dict[str, Any]:
    return {
        "path": "pkg/cache.py",
        "line": line,
        "body": body,
        "evidence": f"line {line}",
        "critique": "the guard above may cover it",
        "specific": flags.get("specific", True),
        "non_obvious": flags.get("non_obvious", True),
        "grounded": flags.get("grounded", True),
    }


def drafted(*drafts: dict[str, Any]) -> dict[str, Any]:
    return {"plan": "check eviction order", "drafts": list(drafts), "no_comment_reason": ""}


def verdicts(*keeps: bool) -> dict[str, Any]:
    return {"verdicts": [{"index": i, "keep": k, "reason": f"r{i}"} for i, k in enumerate(keeps)]}


async def run(responses: list[httpx.Response], **kwargs: Any) -> tuple[Any, respx.Route]:
    route = respx.post(URL).mock(side_effect=responses)
    async with httpx.AsyncClient() as http:
        inference = Inference([ChatClient(http, PROVIDER, "k")], sleep=_no_sleep)
        result = await review(PULL, kwargs.pop("hunks", [HUNK]), [], {}, inference, **kwargs)
    return result, route


async def _no_sleep(seconds: float) -> None:
    del seconds


def test_the_schemas_and_the_models_name_the_same_fields() -> None:
    """The provider is held to the schema and the answer to the model. If they drift, every
    answer fails validation and the pipeline goes silent without anyone noticing why."""
    item = prompts.DRAFT_SCHEMA["properties"]["drafts"]["items"]
    assert set(item["properties"]) == set(DraftItem.model_fields)
    assert set(prompts.DRAFT_SCHEMA["properties"]) == set(DraftAnswer.model_fields)
    verdict = prompts.FILTER_SCHEMA["properties"]["verdicts"]["items"]
    assert set(verdict["properties"]) == set(Verdict.model_fields)
    assert set(prompts.FILTER_SCHEMA["properties"]) == set(FilterAnswer.model_fields)


@respx.mock
async def test_nothing_reviewable_is_silence_without_a_call() -> None:
    result, route = await run([], hunks=[])
    assert result.no_comment is NoComment.NOTHING_REVIEWABLE
    assert not route.called


@respx.mock
async def test_a_hunk_too_large_for_any_rendering_is_silence_without_a_call() -> None:
    huge = parse_patch(
        "@@ -0,0 +1,80 @@\n" + "".join(f"+{'x' * 190}\n" for _ in range(80)), "big.py"
    )[0]
    result, route = await run([], hunks=[huge])
    assert result.no_comment is NoComment.TOO_LARGE
    assert result.dropped == [huge]
    assert not route.called


@respx.mock
async def test_a_model_with_nothing_to_say_is_silence_with_its_reason() -> None:
    result, route = await run(
        [completion({"plan": "fine", "drafts": [], "no_comment_reason": "the guard is right"})]
    )
    assert result.no_comment is NoComment.MODEL_SILENT
    assert result.detail == "the guard is right"
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        completion("I think this looks good!"),
        completion({"plan": "x", "drafts": [{"path": "a"}], "no_comment_reason": ""}),
        completion(drafted(draft(4)), finish_reason="length"),
    ],
    ids=["prose", "wrong shape", "cut off"],
)
async def test_an_answer_that_is_not_the_schema_is_silence(response: httpx.Response) -> None:
    result, _ = await run([response])
    assert result.no_comment is NoComment.MALFORMED
    assert result.kept == []


@respx.mock
async def test_every_gate_removes_what_it_should_and_the_filter_decides_the_rest() -> None:
    answer = drafted(
        draft(2, "Guard is fine but the empty case returns None, not the cache."),
        draft(5, specific=False),
        draft(9, "Line 9 is outside the diff."),
        draft(2, "Same line again."),
        draft(3, "Human already said this."),
        draft(4, "Evicting keys[0] assumes insertion order is recency order."),
    )
    result, route = await run(
        [completion(answer), completion(verdicts(False, True))],
        existing=frozenset({("pkg/cache.py", 3)}),
    )
    fates = [d.fate for d in result.drafts]
    assert fates == [
        Fate.FILTERED,
        Fate.SELF_CRITIQUE,
        Fate.NOT_COMMENTABLE,
        Fate.DUPLICATE,
        Fate.ALREADY_COMMENTED,
        Fate.KEPT,
    ]
    assert result.no_comment is None
    assert [item.line for item in result.kept] == [4]
    assert [call.step for call in result.calls] == ["draft", "filter"]

    filter_request = json.loads(route.calls[1].request.content)
    shown = filter_request["messages"][1]["content"]
    assert "pkg/cache.py:4: cache.pop(keys[0])" in shown
    assert "Line 9 is outside the diff." not in shown


@respx.mock
async def test_the_cap_holds_however_many_drafts_the_filter_keeps() -> None:
    answer = drafted(*(draft(line, f"Point {line}.") for line in (1, 2, 3, 4, 5)))
    result, _ = await run([completion(answer), completion(verdicts(*[True] * 5))])
    assert len(result.kept) == MAX_COMMENTS_PER_PR
    assert [d.fate for d in result.drafts].count(Fate.OVER_CAP) == 5 - MAX_COMMENTS_PER_PR


@respx.mock
async def test_a_draft_the_filter_forgot_is_dropped() -> None:
    answer = drafted(draft(4, "One."), draft(5, "Two."))
    result, _ = await run(
        [completion(answer), completion({"verdicts": [{"index": 1, "keep": True, "reason": "ok"}]})]
    )
    assert [d.fate for d in result.drafts] == [Fate.FILTERED, Fate.KEPT]
    assert result.drafts[0].filter_reason == "no verdict given"


@respx.mock
async def test_a_rate_limit_on_the_draft_call_never_becomes_a_comment() -> None:
    result, _ = await run([httpx.Response(429, headers={"retry-after": "86400"})])
    assert result.no_comment is NoComment.RATE_LIMITED
    assert result.kept == []


@respx.mock
async def test_a_rate_limit_on_the_filter_call_leaves_the_drafts_unjudged_and_unposted() -> None:
    answer = drafted(draft(4, "One."), draft(5, "Two."))
    result, _ = await run([completion(answer), httpx.Response(429)])
    assert result.no_comment is NoComment.RATE_LIMITED
    assert result.kept == []
    assert {d.fate for d in result.drafts} == {Fate.UNJUDGED}


@respx.mock
async def test_past_comments_reach_the_prompt_and_stop_at_the_cutoff() -> None:
    encoder = BagEncoder(MODELS["gte-modernbert"])
    comments = [
        EvalDocument(
            "u1", "o/r", "review_comment", "pkg/cache.py", "h1", "", "evict pops newest keys", 12
        ),
        EvalDocument(
            "u2", "o/r", "review_comment", "pkg/cache.py", "h2", "", "evict keys answer key", 30
        ),
    ]
    index = CommentIndex("o/r", comments, encoder.encode([c.body or "" for c in comments]), encoder)
    result, route = await run([completion(drafted())], index=index)
    prompt = json.loads(route.calls[0].request.content)["messages"][1]["content"]
    assert "[1.1] pkg/cache.py (#12) evict pops newest keys" in prompt
    assert "answer key" not in prompt
    assert result.past_shown == 1


@respx.mock
async def test_skipped_files_are_carried_into_the_result() -> None:
    route = respx.post(URL).mock(side_effect=[completion(drafted())])
    async with httpx.AsyncClient() as http:
        inference = Inference([ChatClient(http, PROVIDER, "k")])
        result = await review(PULL, [HUNK], [Skipped("uv.lock", "lockfile")], {}, inference)
    assert result.skipped == [Skipped("uv.lock", "lockfile")]
    assert route.called
    assert np.isfinite(result.calls[0].seconds)
