import json

import httpx
import pytest
import respx

from pr_lens.github.dispatch import API_ROOT, DispatchError, GitHubDispatcher

REPO = "SURYAPRASATHJP/pr-lens"
URL = f"{API_ROOT}/repos/{REPO}/dispatches"


@respx.mock
async def test_a_204_is_success_and_sends_the_expected_body() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        await GitHubDispatcher(client, REPO, "token", "pr_event")({"pr_number": 7})

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert json.loads(request.content) == {
        "event_type": "pr_event",
        "client_payload": {"pr_number": 7},
    }


@pytest.mark.parametrize("code", [401, 403])
@respx.mock
async def test_auth_failures_name_the_expired_token(code: int) -> None:
    respx.post(URL).mock(return_value=httpx.Response(code))

    async with httpx.AsyncClient() as client:
        dispatcher = GitHubDispatcher(client, REPO, "token", "pr_event")
        with pytest.raises(DispatchError, match="GH_DISPATCH_TOKEN"):
            await dispatcher({})


@respx.mock
async def test_an_unexpected_status_is_an_error_not_a_silent_success() -> None:
    respx.post(URL).mock(return_value=httpx.Response(422, text="no such event type"))

    async with httpx.AsyncClient() as client:
        dispatcher = GitHubDispatcher(client, REPO, "token", "pr_event")
        with pytest.raises(DispatchError, match="422"):
            await dispatcher({})


@respx.mock
async def test_a_transport_error_becomes_a_dispatch_error() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("no route to host"))

    async with httpx.AsyncClient() as client:
        dispatcher = GitHubDispatcher(client, REPO, "token", "pr_event")
        with pytest.raises(DispatchError):
            await dispatcher({})
