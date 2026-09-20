import json

import httpx
import pytest
import respx

from pr_lens.review.provider import (
    MAX_WAITS_PER_PROVIDER,
    PROVIDERS,
    ChatClient,
    Inference,
    InferenceUnavailable,
    NotConfigured,
    Provider,
    ProviderError,
    configured,
    estimate_tokens,
    from_env,
)

FIRST = Provider("first", "https://first.example/v1", "model-a", "FIRST_API_KEY")
SECOND = Provider("second", "https://second.example/v1", "model-b", "SECOND_API_KEY")
MESSAGES = [{"role": "user", "content": "hello"}]


def ok(content: str = '{"ok": true}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "model-a",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
    )


def limited(retry_after: str | None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, json={"error": "rate limited"})


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def inference(client: httpx.AsyncClient, sleeps: Sleeps) -> Inference:
    return Inference(
        [ChatClient(client, FIRST, "key-1"), ChatClient(client, SECOND, "key-2")], sleep=sleeps
    )


@respx.mock
async def test_a_completion_carries_content_usage_and_the_schema_it_was_asked_for() -> None:
    route = respx.post("https://first.example/v1/chat/completions").mock(return_value=ok())
    async with httpx.AsyncClient() as client:
        completion = await ChatClient(client, FIRST, "key-1").complete(
            MESSAGES, max_tokens=100, schema={"type": "object"}
        )
    assert completion.content == '{"ok": true}'
    assert completion.usage.total == 17
    assert completion.finish_reason == "stop"
    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["temperature"] == 0
    assert route.calls.last.request.headers["authorization"] == "Bearer key-1"


@respx.mock
async def test_a_short_retry_after_is_waited_out_on_the_same_provider() -> None:
    respx.post("https://first.example/v1/chat/completions").mock(side_effect=[limited("2.5"), ok()])
    second = respx.post("https://second.example/v1/chat/completions").mock(return_value=ok())
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        completion = await inference(client, sleeps).complete(MESSAGES, max_tokens=10)
    assert completion.provider == "first"
    assert sleeps.waits == [2.5]
    assert not second.called


@respx.mock
@pytest.mark.parametrize("retry_after", ["3600", None], ids=["daily cap", "no header"])
async def test_a_long_or_missing_retry_after_fails_over(retry_after: str | None) -> None:
    respx.post("https://first.example/v1/chat/completions").mock(return_value=limited(retry_after))
    respx.post("https://second.example/v1/chat/completions").mock(return_value=ok())
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        completion = await inference(client, sleeps).complete(MESSAGES, max_tokens=10)
    assert completion.provider == "second"
    assert sleeps.waits == []


@respx.mock
async def test_waiting_on_one_provider_is_bounded() -> None:
    first = respx.post("https://first.example/v1/chat/completions").mock(return_value=limited("1"))
    respx.post("https://second.example/v1/chat/completions").mock(return_value=ok())
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        completion = await inference(client, sleeps).complete(MESSAGES, max_tokens=10)
    assert completion.provider == "second"
    assert len(sleeps.waits) == MAX_WAITS_PER_PROVIDER
    assert first.call_count == MAX_WAITS_PER_PROVIDER + 1


@respx.mock
async def test_every_provider_rate_limited_is_unavailable_and_says_so() -> None:
    respx.post("https://first.example/v1/chat/completions").mock(return_value=limited(None))
    respx.post("https://second.example/v1/chat/completions").mock(return_value=limited("9999"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(InferenceUnavailable) as raised:
            await inference(client, Sleeps()).complete(MESSAGES, max_tokens=10)
    assert raised.value.rate_limited


@respx.mock
async def test_errors_fail_over_and_are_not_reported_as_rate_limits() -> None:
    respx.post("https://first.example/v1/chat/completions").mock(
        side_effect=httpx.ConnectTimeout("slow")
    )
    respx.post("https://second.example/v1/chat/completions").mock(
        return_value=httpx.Response(413, text="Request too large for tokens per minute")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(InferenceUnavailable) as raised:
            await inference(client, Sleeps()).complete(MESSAGES, max_tokens=10)
    assert not raised.value.rate_limited
    assert "413" in str(raised.value)


@respx.mock
async def test_an_unreadable_body_is_a_provider_error() -> None:
    respx.post("https://first.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="<html>gateway</html>")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await ChatClient(client, FIRST, "k").complete(MESSAGES, max_tokens=10)


async def test_a_provider_pointed_at_github_is_refused() -> None:
    """The inference POST is the one write that bypasses pr_lens.github.writes."""
    github = Provider("x", "https://api.github.com/v1", "m", "X_API_KEY")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="GitHub"):
            ChatClient(client, github, "k")


async def test_only_providers_with_a_key_are_used_in_failover_order() -> None:
    environ = {"NVIDIA_API_KEY": "n", "GROQ_API_KEY": "g", "GROQ_MODEL": "groq/compound"}
    async with httpx.AsyncClient() as client:
        chosen = from_env(client, environ)
    assert chosen.names == ["groq:groq/compound", f"nim:{PROVIDERS[2].model}"]


async def test_no_key_at_all_fails_loudly_rather_than_going_quiet() -> None:
    """Silence is for a provider that said no, not for a job that was never configured."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(NotConfigured, match="GROQ_API_KEY"):
            from_env(client, {})


@respx.mock
async def test_a_key_pasted_with_a_line_break_still_builds_a_header() -> None:
    """The failure this prevents is not a bad request, it is a provider that vanishes.
    httpx raises LocalProtocolError while building the header, before anything is sent, so
    failover never sees a failure to fail over from. Both fallback secrets reached Actions
    this way on 2026-09-21 and read as unconfigured."""
    route = respx.post(f"{PROVIDERS[0].base_url}/chat/completions").mock(return_value=ok())
    async with httpx.AsyncClient() as client:
        clients = configured(client, {PROVIDERS[0].key_env: "secret-key\n"})
        await clients[0].complete(MESSAGES, max_tokens=10)
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret-key"


@respx.mock
async def test_a_key_padded_with_spaces_still_builds_a_header() -> None:
    route = respx.post(f"{PROVIDERS[0].base_url}/chat/completions").mock(return_value=ok())
    async with httpx.AsyncClient() as client:
        clients = configured(client, {PROVIDERS[0].key_env: "  secret-key  "})
        await clients[0].complete(MESSAGES, max_tokens=10)
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret-key"


async def test_a_model_override_pasted_with_a_newline_is_stripped() -> None:
    async with httpx.AsyncClient() as client:
        clients = configured(
            client, {PROVIDERS[0].key_env: "k", PROVIDERS[0].model_env: "groq/compound\n"}
        )
    assert clients[0].model == "groq/compound"


async def test_a_key_that_is_only_whitespace_is_unset_not_an_empty_bearer() -> None:
    """An empty Bearer token is a 401 on every call. A provider that was never configured
    is both the truth and the more useful thing to report."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(NotConfigured):
            configured(client, {PROVIDERS[0].key_env: "   \n"})


def test_both_fallbacks_carry_the_same_model() -> None:
    """A failover already changes provider mid-batch. A second model would change the
    reviewer again on the same run, and then a drop in noise cannot be told from a change
    of reviewer. Measured 2026-09-21: the previous ids were dead on both fallbacks, 404 on
    OpenRouter and 410 on NIM, so the failover had never once worked."""
    openrouter, nim = PROVIDERS[1], PROVIDERS[2]
    assert openrouter.model.removesuffix(":free") == nim.model
    assert "nemotron" in nim.model


def test_the_estimate_rounds_up() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2
