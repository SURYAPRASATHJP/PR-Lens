import httpx
import respx

from pr_lens.jobs.provider_check import check, render
from pr_lens.review.provider import CALL_TOKEN_BUDGET, ChatClient, Provider

PROVIDER = Provider("groq", "https://groq.example/v1", "openai/gpt-oss-120b", "GROQ_API_KEY")
URL = "https://groq.example/v1/chat/completions"


def answer(content: str, prompt_tokens: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "openai/gpt-oss-120b",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 40},
        },
    )


@respx.mock
async def test_both_probes_are_reported_with_the_measured_ratio() -> None:
    respx.post(URL).mock(
        side_effect=[
            answer('{"ok": true, "count": 3}', 30),
            answer('{"ok": true, "count": 3}', 5000),
        ]
    )
    async with httpx.AsyncClient() as http:
        probes = await check(ChatClient(http, PROVIDER, "k"))
    schema, budget = probes
    assert schema.verdict == "honours the schema"
    assert budget.verdict == "honours the schema"
    assert budget.prompt_tokens == 5000
    assert budget.chars_per_token is not None and budget.chars_per_token > 2


@respx.mock
async def test_a_refusal_is_a_finding_not_a_crash() -> None:
    respx.post(URL).mock(
        side_effect=[
            answer("sure, here you go", 30),
            httpx.Response(413, text="Request too large: tokens per minute limit 8000"),
        ]
    )
    async with httpx.AsyncClient() as http:
        probes = await check(ChatClient(http, PROVIDER, "k"))
    assert probes[0].verdict == "answered, but not JSON"
    assert "413" in probes[1].verdict
    table = render(probes)
    assert f"Budget per call {CALL_TOKEN_BUDGET}" in table
    assert table.count("| groq |") == 2
