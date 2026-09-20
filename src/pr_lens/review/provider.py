"""Chat completions from the free inference tiers, behind one class.

Groq, OpenRouter and NVIDIA NIM all speak the OpenAI chat completions protocol, so one
client with a base URL covers the three. Four free tiers have already moved under this
project, and the next move should be an edit to PROVIDERS rather than to the pipeline.

The tier, checked 15 Sep 2026 in notes/phase-4-provider-check.md, sets the shape of
everything downstream. Groq's gpt-oss-120b allows 1,000 requests a day but only 8K tokens a
minute, and a request larger than that is refused outright, however it is paced. So a call
has a hard token budget, CALL_TOKEN_BUDGET, and the pipeline is two calls rather than four.

At this budget a 429 is a normal operating state, not an error. Inference waits when the
provider asks for less than a minute, which is the token window rolling over, and fails
over when it asks for longer, which is a daily cap. When every provider is spent it raises
InferenceUnavailable, and the pipeline goes quiet. A rate limit never becomes a comment.
"""

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# Groq's free gpt-oss-120b is 8,000 tokens per minute, counting prompt and completion
# together. A call has to fit inside one minute's window with room left for the tokenizer
# estimate being wrong, so the prompt and the completion share 7,000.
CALL_TOKEN_BUDGET = 7000

# Code tokenizes worse than prose. Three characters a token errs toward overestimating,
# which costs context rather than a refused request. provider_check reports the real ratio.
CHARS_PER_TOKEN = 3.0

# A retry-after inside a minute is the per-minute token window rolling over, worth waiting
# for. Anything longer is a daily cap, and the next provider is the better bet.
MAX_WAIT_SECONDS = 65.0
MAX_WAITS_PER_PROVIDER = 3

# Large enough for a drafting call on a slow free tier, small enough that a hung provider
# does not hold a review job for the whole of its timeout.
REQUEST_TIMEOUT_SECONDS = 120.0


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str

    @property
    def model_env(self) -> str:
        return f"{self.name.upper()}_MODEL"


# In failover order. Groq is the base. OpenRouter's free models are 50 requests a day, and
# its 429s do not correlate with Groq's. NIM's 1,000 credits are a pool that never renews,
# so it is last: a reserve for when both of the others are spent.
#
# Both fallbacks carry nemotron rather than gpt-oss-120b, measured 2026-09-21. The old ids
# were dead and the failover had therefore never worked: OpenRouter serves no free
# gpt-oss-120b at all, 404, and NIM retired it on 2026-09-03, 410 Gone. Both nemotron ids
# honour the strict schema and call read_file with exact arguments in about a second.
#
# The same model sits on both fallbacks on purpose. A failover already changes provider
# mid-batch, and a second model would change the reviewer again on the same run, so a
# drop in noise could not be told from a change of reviewer.
PROVIDERS: tuple[Provider, ...] = (
    Provider("groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "OPENROUTER_API_KEY",
    ),
    Provider(
        "nim",
        "https://integrate.api.nvidia.com/v1",
        "nvidia/nemotron-3-super-120b-a12b",
        "NVIDIA_API_KEY",
    ),
)


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool the model asked for, as the provider named it."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Completion:
    provider: str
    model: str
    content: str
    finish_reason: str | None
    usage: Usage
    seconds: float
    tool_calls: tuple[ToolCall, ...] = ()
    # The assistant message exactly as the provider sent it. A tool calling loop has to
    # send it back unchanged in the next turn, and rebuilding it from the fields above
    # loses whatever the provider added, which is how a loop starts arguing with itself.
    message: Mapping[str, Any] = field(default_factory=dict)


class RateLimited(Exception):
    def __init__(self, provider: str, retry_after: float | None) -> None:
        super().__init__(f"{provider} rate limited, retry-after {retry_after}")
        self.provider = provider
        self.retry_after = retry_after


class ProviderError(Exception):
    """This provider could not answer this request. Another provider might."""


class NotConfigured(RuntimeError):
    """No provider key is set. A job that was never configured fails loudly, it does not go
    quiet: silence is for a provider that said no."""


class InferenceUnavailable(Exception):
    """No provider answered. The pipeline's answer to this is silence."""

    def __init__(self, rate_limited: bool, detail: str) -> None:
        super().__init__(detail)
        self.rate_limited = rate_limited


class ChatClient:
    def __init__(
        self, client: httpx.AsyncClient, provider: Provider, key: str, model: str | None = None
    ) -> None:
        # The inference call is the one POST in this project that does not go through
        # pr_lens.github.writes, so it must never be pointed at GitHub.
        if urlsplit(provider.base_url).netloc.endswith("github.com"):
            raise ValueError(f"{provider.name} points at GitHub; writes go through the chokepoint")
        self._client = client
        self._provider = provider
        self._key = key
        self.model = model or provider.model

    @property
    def name(self) -> str:
        return self._provider.name

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        started = time.monotonic()
        try:
            response = await self._client.post(
                f"{self._provider.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        seconds = time.monotonic() - started

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimited(self.name, _retry_after(response))
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} returned {response.status_code}: {response.text[:300]}"
            )

        try:
            payload = response.json()
            choice = payload["choices"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name} returned an unreadable body") from exc
        usage = payload.get("usage") or {}
        message = choice.get("message") or {}
        return Completion(
            provider=self.name,
            model=str(payload.get("model", self.model)),
            content=message.get("content") or "",
            finish_reason=choice.get("finish_reason"),
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            seconds=seconds,
            tool_calls=_tool_calls(message),
            message=message,
        )


class Inference:
    """The configured providers in failover order, and the policy for when they say no."""

    def __init__(
        self,
        clients: Sequence[ChatClient],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not clients:
            raise NotConfigured("no inference provider is configured")
        self._clients = tuple(clients)
        self._sleep = sleep

    @property
    def names(self) -> list[str]:
        return [f"{client.name}:{client.model}" for client in self._clients]

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Completion:
        failures: list[str] = []
        every_failure_was_a_rate_limit = True
        for client in self._clients:
            waits = 0
            while True:
                try:
                    return await client.complete(
                        messages, max_tokens=max_tokens, schema=schema, tools=tools
                    )
                except RateLimited as limited:
                    wait = limited.retry_after
                    if (
                        wait is not None
                        and wait <= MAX_WAIT_SECONDS
                        and waits < MAX_WAITS_PER_PROVIDER
                    ):
                        logger.info("%s asked for %.1fs, waiting", client.name, wait)
                        await self._sleep(wait)
                        waits += 1
                        continue
                    logger.warning("%s is rate limited past waiting, failing over", client.name)
                    failures.append(str(limited))
                except ProviderError as error:
                    logger.warning("%s", error)
                    failures.append(str(error))
                    every_failure_was_a_rate_limit = False
                break
        raise InferenceUnavailable(every_failure_was_a_rate_limit, "; ".join(failures))


def configured(
    client: httpx.AsyncClient, environ: Mapping[str, str] = os.environ
) -> list[ChatClient]:
    """Every provider whose key is set, in PROVIDERS order. A model override per provider,
    such as GROQ_MODEL=groq/compound, is how the provider check's answer gets applied."""
    clients = [
        ChatClient(client, provider, environ[provider.key_env], environ.get(provider.model_env))
        for provider in PROVIDERS
        if environ.get(provider.key_env)
    ]
    if not clients:
        names = ", ".join(provider.key_env for provider in PROVIDERS)
        raise NotConfigured(f"no inference provider is configured; set one of {names}")
    return clients


def from_env(client: httpx.AsyncClient, environ: Mapping[str, str] = os.environ) -> Inference:
    return Inference(configured(client, environ))


def _tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """The tool calls in an assistant message, skipping anything malformed.

    A provider that sends a call with no name or no id is not answerable, and inventing an
    id to reply to would put the conversation out of step with what the provider thinks it
    asked. Dropping it costs one turn; the loop's cap covers the rest.
    """
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        function = function if isinstance(function, dict) else {}
        name, call_id = function.get("name"), entry.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            continue
        arguments = function.get("arguments")
        calls.append(ToolCall(call_id, name, arguments if isinstance(arguments, str) else "{}"))
    return tuple(calls)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
