"""Backoff, conditional requests, pagination and the identity check.

The identity check is the one that earns its place. A mining run authenticated as the
workflow's GITHUB_TOKEN does not fail, it runs at a fifth of the speed against a limit
that is per repository, and it reads as a slow network rather than a misconfiguration.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import API_ROOT, GitHubClient, GitHubError, NotFound


class Clock:
    """Records what the client asked to wait for instead of actually waiting."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def client(tmp_path: Path, clock: Clock):  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient(follow_redirects=True) as http:
        yield GitHubClient(http, "test-token", HttpCache(tmp_path), sleep=clock)


@respx.mock
async def test_a_token_with_the_workflows_limit_is_refused(client: GitHubClient) -> None:
    _rate_limit(1000)

    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        await client.assert_mining_budget()


@respx.mock
async def test_a_token_with_the_mining_limit_is_accepted(client: GitHubClient) -> None:
    _rate_limit(5000, remaining=4999)

    identity = await client.assert_mining_budget()

    assert identity.limit == 5000
    assert "personal access token" in identity.description


@respx.mock
async def test_the_token_is_sent_as_a_bearer(client: GitHubClient) -> None:
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(httpx.Response(200, json={"id": 1}))

    await client.get_json("/repos/o/r")

    assert route.calls[0].request.headers["authorization"] == "Bearer test-token"


@respx.mock
async def test_a_404_is_its_own_error_because_repos_go_private(client: GitHubClient) -> None:
    respx.get(f"{API_ROOT}/repos/o/gone").mock(httpx.Response(404, json={}))

    with pytest.raises(NotFound):
        await client.get_json("/repos/o/gone")


@respx.mock
async def test_a_secondary_limit_backs_off_and_then_succeeds(
    client: GitHubClient, clock: Clock
) -> None:
    respx.get(f"{API_ROOT}/repos/o/r").mock(
        side_effect=[
            httpx.Response(403, headers={"retry-after": "7"}, text="secondary rate limit"),
            httpx.Response(200, json={"id": 1}),
        ]
    )

    assert await client.get_json("/repos/o/r") == {"id": 1}
    assert clock.slept == [7.0]


@respx.mock
async def test_a_spent_primary_budget_waits_for_the_reset(
    client: GitHubClient, clock: Clock
) -> None:
    reset = (datetime.now(UTC) + timedelta(seconds=30)).timestamp()
    respx.get(f"{API_ROOT}/repos/o/r").mock(
        side_effect=[
            httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset))},
                text="rate limit exceeded",
            ),
            httpx.Response(200, json={"id": 1}),
        ]
    )

    await client.get_json("/repos/o/r")

    assert 25 <= clock.slept[0] <= 35


@respx.mock
async def test_a_403_that_is_a_permissions_problem_is_not_retried(
    client: GitHubClient, clock: Clock
) -> None:
    # Retrying a permissions failure spends the budget to learn the same thing again.
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(
        httpx.Response(403, text="Resource not accessible by integration")
    )

    with pytest.raises(GitHubError):
        await client.get_json("/repos/o/r")

    assert route.call_count == 1
    assert clock.slept == []


@respx.mock
async def test_retrying_stops_at_a_ceiling_rather_than_looping(client: GitHubClient) -> None:
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(httpx.Response(500, text="boom"))

    with pytest.raises(GitHubError, match="after 6 attempts"):
        await client.get_json("/repos/o/r")

    assert route.call_count == 6


@respx.mock
async def test_a_wait_past_the_ceiling_stops_the_run_instead_of_sitting_idle(
    client: GitHubClient,
) -> None:
    respx.get(f"{API_ROOT}/repos/o/r").mock(
        httpx.Response(429, headers={"retry-after": "3600"}, text="slow down")
    )

    with pytest.raises(GitHubError, match="ceiling"):
        await client.get_json("/repos/o/r")


@respx.mock
async def test_a_second_fetch_revalidates_with_the_stored_etag(client: GitHubClient) -> None:
    # A 304 does not count against the primary rate limit, which is the whole reason the
    # ETag is stored at all.
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(
        side_effect=[
            httpx.Response(200, json={"id": 1}, headers={"etag": 'W/"v1"'}),
            httpx.Response(304),
        ]
    )

    await client.get_json("/repos/o/r")
    assert await client.get_json("/repos/o/r") == {"id": 1}

    assert route.calls[1].request.headers["if-none-match"] == 'W/"v1"'


@respx.mock
async def test_a_fresh_cache_entry_is_read_without_any_request(client: GitHubClient) -> None:
    # This is what makes a killed run resume from disk rather than from the API.
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(httpx.Response(200, json={"id": 1}))

    await client.get_json("/repos/o/r", max_age=timedelta(hours=1))
    await client.get_json("/repos/o/r", max_age=timedelta(hours=1))

    assert route.call_count == 1


@respx.mock
async def test_pagination_follows_the_link_header(client: GitHubClient) -> None:
    route = _two_pages()

    numbers = [item["number"] async for item in client.paginate("/repos/o/r/issues")]

    assert numbers == [1, 2]
    assert "page=2" in str(route.calls[1].request.url)


@respx.mock
async def test_pagination_stops_when_a_link_points_back_at_a_page_already_walked(
    client: GitHubClient,
) -> None:
    respx.get(f"{API_ROOT}/repos/o/r/issues").mock(
        httpx.Response(
            200,
            json=[{"number": 1}],
            headers={"link": f'<{API_ROOT}/repos/o/r/issues?per_page=100>; rel="next"'},
        )
    )

    items = [item async for item in client.paginate("/repos/o/r/issues")]

    assert items == [{"number": 1}]


@respx.mock
async def test_pagination_stops_at_max_items(client: GitHubClient) -> None:
    respx.get(f"{API_ROOT}/repos/o/r/issues").mock(
        httpx.Response(
            200,
            json=[{"number": n} for n in range(10)],
            headers={"link": f'<{API_ROOT}/repos/o/r/issues?page=2>; rel="next"'},
        )
    )

    items = [item async for item in client.paginate("/repos/o/r/issues", max_items=3)]

    assert len(items) == 3


@respx.mock
async def test_a_replayed_pagination_walk_needs_no_requests_at_all(
    client: GitHubClient,
) -> None:
    # Without the next link stored beside the body, page one comes from disk and the
    # client still has to ask the API where page two is.
    route = _two_pages()
    fresh = timedelta(hours=1)
    assert [i async for i in client.paginate("/repos/o/r/issues", max_age=fresh)] == [
        {"number": 1},
        {"number": 2},
    ]
    assert route.call_count == 2

    replayed = [i async for i in client.paginate("/repos/o/r/issues", max_age=fresh)]

    assert replayed == [{"number": 1}, {"number": 2}]
    assert route.call_count == 2


@respx.mock
async def test_raw_bytes_come_back_untouched(client: GitHubClient) -> None:
    payload = bytes(range(256))
    respx.get(f"{API_ROOT}/repos/o/r/tarball/abc").mock(httpx.Response(200, content=payload))

    assert await client.get_raw("/repos/o/r/tarball/abc", accept="application/x-gzip") == payload


def _two_pages() -> respx.Route:
    """Page one links to page two, and page two ends the walk."""
    return respx.get(f"{API_ROOT}/repos/o/r/issues").mock(
        side_effect=[
            httpx.Response(
                200,
                json=[{"number": 1}],
                headers={"link": f'<{API_ROOT}/repos/o/r/issues?page=2>; rel="next"'},
            ),
            httpx.Response(200, json=[{"number": 2}]),
        ]
    )


def _rate_limit(limit: int, remaining: int | None = None) -> None:
    respx.get(f"{API_ROOT}/rate_limit").mock(
        httpx.Response(
            200,
            content=json.dumps(
                {"resources": {"core": {"limit": limit, "remaining": remaining or limit}}}
            ),
        )
    )


@respx.mock
async def test_an_empty_token_sends_no_authorization_header(tmp_path: Path) -> None:
    # "Bearer " with nothing after it is a malformed header, and httpx rejects it with a
    # protocol error that says nothing about credentials. Anonymous reads are only for
    # inspecting a public repo by hand; assert_mining_budget refuses them for a run.
    route = respx.get(f"{API_ROOT}/repos/o/r").mock(httpx.Response(200, json={"id": 1}))

    async with httpx.AsyncClient() as http:
        await GitHubClient(http, "", HttpCache(tmp_path)).get_json("/repos/o/r")

    assert "authorization" not in route.calls[0].request.headers
