import json

import httpx
import respx

from pr_lens.jobs.provider_check import check, clients_to_probe, render
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


def tool_answer() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "a.py", "start_line": 1}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 20},
        },
    )


@respx.mock
async def test_all_three_probes_are_reported_with_the_measured_ratio() -> None:
    respx.post(URL).mock(
        side_effect=[
            answer('{"ok": true, "count": 3}', 30),
            tool_answer(),
            answer('{"ok": true, "count": 3}', 5000),
        ]
    )
    async with httpx.AsyncClient() as http:
        probes = await check(ChatClient(http, PROVIDER, "k"))
    schema, tools, budget = probes
    assert schema.verdict == "honours the schema"
    assert tools.verdict == "calls the tool: read_file"
    assert budget.verdict == "honours the schema"
    assert budget.prompt_tokens == 5000
    assert budget.chars_per_token is not None and budget.chars_per_token > 2


@respx.mock
async def test_the_tools_probe_carries_tools_and_no_schema() -> None:
    """The split the hybrid rests on. A tools probe that also sent a schema would fail on a
    model that honours tools perfectly well, which is the confusion this check exists to end.
    """
    route = respx.post(URL).mock(side_effect=[answer("{}", 30), tool_answer(), answer("{}", 40)])
    async with httpx.AsyncClient() as http:
        await check(ChatClient(http, PROVIDER, "k"))
    schema_body, tools_body, _ = (call.request for call in route.calls)
    assert "tools" not in json.loads(schema_body.content)
    sent = json.loads(tools_body.content)
    assert "response_format" not in sent
    assert sent["tools"][0]["function"]["name"] == "read_file"


@respx.mock
async def test_a_model_answering_from_memory_is_a_no_not_an_error() -> None:
    respx.post(URL).mock(
        side_effect=[answer("{}", 30), answer("line 12 is an import", 60), answer("{}", 40)]
    )
    async with httpx.AsyncClient() as http:
        probes = await check(ChatClient(http, PROVIDER, "k"))
    assert probes[1].verdict.startswith("answered without calling")


@respx.mock
async def test_a_refusal_is_a_finding_not_a_crash() -> None:
    respx.post(URL).mock(
        side_effect=[
            answer("sure, here you go", 30),
            httpx.Response(400, text="This model does not support response format json_schema"),
            httpx.Response(413, text="Request too large: tokens per minute limit 8000"),
        ]
    )
    async with httpx.AsyncClient() as http:
        probes = await check(ChatClient(http, PROVIDER, "k"))
    assert probes[0].verdict == "answered, but not JSON"
    assert "400" in probes[1].verdict
    assert "413" in probes[2].verdict
    table = render(probes)
    assert f"Budget per call {CALL_TOKEN_BUDGET}" in table
    assert table.count("| groq |") == 3


async def test_an_alternative_model_is_probed_beside_the_default() -> None:
    """Choosing between two models is a question about both, so both get asked."""
    async with httpx.AsyncClient() as http:
        clients = clients_to_probe(http, {"GROQ_API_KEY": "k"})
    models = [(client.name, client.model) for client in clients]
    assert ("groq", "openai/gpt-oss-120b") in models
    assert ("groq", "groq/compound") in models


async def test_a_model_already_configured_is_not_probed_twice() -> None:
    async with httpx.AsyncClient() as http:
        clients = clients_to_probe(http, {"GROQ_API_KEY": "k", "GROQ_MODEL": "groq/compound"})
    models = [(client.name, client.model) for client in clients]
    assert models.count(("groq", "groq/compound")) == 1
