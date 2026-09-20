"""Ask each configured model what it can do before the pipeline depends on the answer.

The provider check in notes/phase-4-provider-check.md read the vendors' documentation. What
it could not settle from documentation is settled here, on the real endpoints, with the real
keys: whether a model honours a strict JSON schema, whether it honours `tools`, and whether
a call sized to CALL_TOKEN_BUDGET is accepted under an 8K-a-minute cap. The budget probe
also measures how many characters a token really is for code, which is what CHARS_PER_TOKEN
guesses at.

The schema and tools probes are separate because the pipeline now needs them separately.
The research phase sends tools and no schema, the drafting and filter calls send a schema
and no tools, so a model that can only do one of the two can still hold one of those jobs.
That is not hypothetical: the 15 Sep check verified gpt-oss-120b with a schema and never
asked Compound the same question, and Compound answers 400 to every schema call.

Alternative models are probed beside each provider's default, because choosing between two
models is a question about both of them. Each is asked on its own, not through failover.
The result is a table in the job summary. A model saying no is a finding, not a failure, so
this exits 0 unless no provider is configured at all.
"""

import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pr_lens.logging import configure
from pr_lens.review.provider import (
    CALL_TOKEN_BUDGET,
    CHARS_PER_TOKEN,
    PROVIDERS,
    ChatClient,
    Completion,
    NotConfigured,
    ProviderError,
    RateLimited,
    configured,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "count": {"type": "integer"}},
    "required": ["ok", "count"],
    "additionalProperties": False,
}

# Shaped like review/tools.py's read_file, so the answer is about the tool the pipeline
# actually sends rather than about a simpler one invented for the probe.
TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a numbered region of a file in the pull request's repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path within the repository."},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
        },
    },
}

# Models to probe beside each provider's default. Compound is the high tokens-a-minute
# option the tool loop wants, 70K against gpt-oss-120b's 8K.
ALSO_PROBE: Mapping[str, tuple[str, ...]] = {"groq": ("groq/compound",)}

# Room for the completion inside the budget, reasoning tokens included.
PROBE_COMPLETION_TOKENS = 800


@dataclass(frozen=True, slots=True)
class Probe:
    provider: str
    model: str
    question: str
    verdict: str
    seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chars_per_token: float | None = None


def _code_filler(tokens: int) -> str:
    """Real source rather than repeated words, because words tokenize far better than code
    and would flatter the estimate."""
    source = Path(__file__).read_text(encoding="utf-8")
    wanted = int(tokens * CHARS_PER_TOKEN)
    return (source * (wanted // len(source) + 1))[:wanted]


def _schema_verdict(completion: Completion) -> str:
    content = completion.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return "answered, but not JSON"
    if isinstance(parsed, dict) and parsed.get("ok") is True and parsed.get("count") == 3:
        return "honours the schema"
    return f"JSON, wrong shape: {content[:60]!r}"


def _tool_verdict(completion: Completion) -> str:
    """Whether the model asked for the tool. Answering the question from its own head is a
    no, not an error, and it is the answer that decides whether a tool loop can run here."""
    if completion.tool_calls:
        return "calls the tool: " + ", ".join(call.name for call in completion.tool_calls)
    return f"answered without calling: {completion.content[:60]!r}"


async def _ask(
    client: ChatClient,
    question: str,
    prompt: str,
    *,
    system: str = "Reply with JSON only.",
    schema: Mapping[str, Any] | None = SCHEMA,
    tools: Sequence[Mapping[str, Any]] | None = None,
    verdict: Callable[[Completion], str] = _schema_verdict,
) -> Probe:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        completion = await client.complete(
            messages, max_tokens=PROBE_COMPLETION_TOKENS, schema=schema, tools=tools
        )
    except RateLimited as limited:
        return Probe(client.name, client.model, question, f"429, retry-after {limited.retry_after}")
    except ProviderError as error:
        return Probe(client.name, client.model, question, str(error)[:120])
    usage = completion.usage
    ratio = len(prompt) / usage.prompt_tokens if usage.prompt_tokens else None
    return Probe(
        client.name,
        completion.model,
        question,
        verdict(completion),
        round(completion.seconds, 2),
        usage.prompt_tokens,
        usage.completion_tokens,
        round(ratio, 2) if ratio else None,
    )


async def check(client: ChatClient) -> list[Probe]:
    task = 'Return {"ok": true, "count": 3}.'
    near_budget = CALL_TOKEN_BUDGET - PROBE_COMPLETION_TOKENS - 200
    filler = _code_filler(near_budget)
    return [
        await _ask(client, "strict JSON schema", task),
        await _ask(
            client,
            "tools, no schema",
            "What is on line 12 of src/pr_lens/review/tools.py? "
            "Use the tool to look it up rather than answering from memory.",
            system="Use the tools available to you before answering.",
            schema=None,
            tools=[TOOL],
            verdict=_tool_verdict,
        ),
        await _ask(
            client,
            f"call of ~{estimate_tokens(filler)} estimated prompt tokens",
            f"Ignore this source code.\n\n{filler}\n\n{task}",
        ),
    ]


def clients_to_probe(
    http: httpx.AsyncClient, environ: Mapping[str, str] = os.environ
) -> list[ChatClient]:
    """Every configured provider at its own model, plus the alternatives worth comparing it
    against, so the choice between two models rests on both having been asked."""
    clients = configured(http, environ)
    asked = {(client.name, client.model) for client in clients}
    return clients + [
        ChatClient(http, provider, environ[provider.key_env], model)
        for provider in PROVIDERS
        if environ.get(provider.key_env)
        for model in ALSO_PROBE.get(provider.name, ())
        if (provider.name, model) not in asked
    ]


def render(probes: list[Probe]) -> str:
    lines = [
        "## Inference provider check",
        "",
        f"Budget per call {CALL_TOKEN_BUDGET} tokens, estimate {CHARS_PER_TOKEN} chars/token.",
        "",
        "| provider | model | probe | verdict | s | prompt tok | completion tok | chars/tok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for probe in probes:
        cells = [
            probe.provider,
            probe.model,
            probe.question,
            probe.verdict.replace("|", "/"),
            probe.seconds,
            probe.prompt_tokens,
            probe.completion_tokens,
            probe.chars_per_token,
        ]
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


async def probe_all() -> list[Probe]:
    async with httpx.AsyncClient() as http:
        return [probe for client in clients_to_probe(http) for probe in await check(client)]


def main() -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        probes = asyncio.run(probe_all())
    except NotConfigured:
        logger.exception("nothing to check")
        return 1
    markdown = render(probes)
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
