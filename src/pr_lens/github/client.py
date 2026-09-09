"""The read side of the GitHub API: backoff, conditional requests and an identity check.

Two things from the 2026-09-09 rate-limit re-check shape this file.

The first is that the workflow's built-in GITHUB_TOKEN is 1,000 requests per hour per
repository, while a PAT or an App installation token is 5,000 per hour. Mining as the
workflow does not fail, it runs at a fifth of the speed and reads as a slow network. So
the identity is asserted at startup and logged, rather than assumed.

The second is that the secondary limits bite first and cannot be queried: 100 concurrent
requests, 900 points per minute per endpoint, 90 seconds of CPU per 60 seconds of real
time, and no endpoint that reports how much of any of them is left. There is therefore no
number to stay under. What works instead is to keep concurrency low, back off on 403 and
429, honour retry-after, and raise at a ceiling rather than loop.
"""

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from pr_lens.github.cache import HttpCache

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"

# A PAT or an App installation token gets 5,000. Anything under it is the workflow token.
MINING_RATE_LIMIT = 5000

# Long enough for a spent primary budget to refill, short enough that a run does not sit
# silently for most of an hour before anyone sees it.
MAX_SLEEP_SECONDS = 900.0

# Well under the 100 concurrent requests GitHub allows. The budget that runs out first is
# points per minute, not connections, so opening more of them buys nothing.
DEFAULT_CONCURRENCY = 6

IMMUTABLE = timedelta(days=3650)

_NEXT_LINK = re.compile(r'<(?P<url>[^>]+)>;\s*rel="next"')


class GitHubError(RuntimeError):
    """A request failed in a way the caller cannot paper over."""


class NotFound(GitHubError):
    """404. Repos get renamed, deleted and made private between one run and the next."""


@dataclass(frozen=True, slots=True)
class Identity:
    """What GitHub says our budget is. There is no endpoint that names the credential.

    /user is not it: it returns 403 for an App installation token. The limit is the one
    signal that separates the three cases we care about, and it is enough.
    """

    limit: int
    remaining: int

    @property
    def is_mining_grade(self) -> bool:
        return self.limit >= MINING_RATE_LIMIT

    @property
    def description(self) -> str:
        if self.limit >= MINING_RATE_LIMIT:
            return "a personal access token or an App installation token"
        if self.limit > 0:
            return "the workflow's built-in GITHUB_TOKEN, capped per repository"
        return "an unauthenticated client"


class GitHubClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        token: str,
        cache: HttpCache,
        *,
        max_attempts: int = 6,
        concurrency: int = DEFAULT_CONCURRENCY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._token = token
        self._cache = cache
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._gate = asyncio.Semaphore(concurrency)

    async def identity(self) -> Identity:
        """Who we are and what budget we have, straight from GitHub.

        /rate_limit is the only honest answer. The token prefix does not distinguish an
        App installation token from the workflow's own, and both are ghs_.
        """
        response = await self._send("GET", f"{API_ROOT}/rate_limit", headers={}, attempt=0)
        core = json.loads(response.content)["resources"]["core"]
        return Identity(limit=int(core["limit"]), remaining=int(core["remaining"]))

    async def assert_mining_budget(self) -> Identity:
        identity = await self.identity()
        if not identity.is_mining_grade:
            raise GitHubError(
                f"this token has a core rate limit of {identity.limit}/hour, and mining "
                f"needs {MINING_RATE_LIMIT}. A limit of 1,000 means the workflow's "
                "built-in GITHUB_TOKEN, which is capped per repository. Pass the "
                "fine-grained PAT or an App installation token instead."
            )
        logger.info(
            "mining as %s: %s requests/hour, %s remaining",
            identity.description,
            identity.limit,
            identity.remaining,
        )
        return identity

    async def get_json(
        self, path: str, *, params: dict[str, Any] | None = None, max_age: timedelta | None = None
    ) -> Any:
        response = await self._get(path, params=params, max_age=max_age, accept=None)
        return json.loads(response) if response else None

    async def get_raw(self, path: str, *, accept: str, max_age: timedelta | None = None) -> bytes:
        """A non-JSON representation: a diff, a patch, a tarball."""
        return await self._get(path, params=None, max_age=max_age, accept=accept)

    async def paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        max_items: int | None = None,
        max_age: timedelta | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Walk rel="next" rather than incrementing a page number.

        GitHub caps some listings well before the page arithmetic says it should, and the
        Link header is the only thing that knows where a listing actually ends.
        """
        # Resolved once, so that the page walked and the page recorded as walked are the
        # same string. Every later url comes from a Link header and is already absolute.
        request = self._client.build_request(
            "GET", _url(path), params={"per_page": 100, **(params or {})}
        )
        url = str(request.url)
        yielded = 0
        # A Link header pointing back at a page already walked would otherwise spin here
        # forever. It should not happen against the real API, but an unbounded loop over
        # a paginated endpoint is not a failure mode worth leaving open.
        visited: set[str] = set()
        while url and url not in visited:
            visited.add(url)
            body, next_url = await self._get_page(url, params=None, max_age=max_age)
            page = json.loads(body)
            if not isinstance(page, list):
                raise GitHubError(f"expected a list from {url}, got {type(page).__name__}")
            for item in page:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            url = next_url or ""

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None,
        max_age: timedelta | None,
        accept: str | None,
    ) -> bytes:
        body, _ = await self._get_page(_url(path), params=params, max_age=max_age, accept=accept)
        return body

    async def _get_page(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        max_age: timedelta | None,
        accept: str | None = None,
    ) -> tuple[bytes, str | None]:
        request = self._client.build_request("GET", url, params=params)
        key = f"GET {request.url}|{accept or 'json'}"
        cached = self._cache.get(key)

        if cached is not None and max_age is not None and cached.age() < max_age.total_seconds():
            return cached.body, cached.next_url

        headers = {"Accept": accept} if accept else {}
        if cached is not None and cached.etag:
            headers["If-None-Match"] = cached.etag

        response = await self._send("GET", str(request.url), headers=headers, attempt=0)

        next_url = _next_link(response.headers.get("link"))

        if response.status_code == httpx.codes.NOT_MODIFIED and cached is not None:
            self._cache.touch(key)
            return cached.body, next_url or cached.next_url

        self._cache.put(
            key,
            status=response.status_code,
            etag=response.headers.get("etag"),
            body=response.content,
            next_url=next_url,
        )
        return response.content, next_url

    async def _send(
        self, method: str, url: str, *, headers: dict[str, str], attempt: int
    ) -> httpx.Response:
        while True:
            async with self._gate:
                try:
                    response = await self._client.request(
                        method, url, headers={**self._auth_headers(), **headers}
                    )
                except httpx.HTTPError as exc:
                    if attempt >= self._max_attempts - 1:
                        raise GitHubError(f"{method} {url} failed: {exc}") from exc
                    await self._sleep(_backoff(attempt))
                    attempt += 1
                    continue

            if response.status_code == httpx.codes.NOT_FOUND:
                raise NotFound(f"{method} {url} returned 404")
            if response.status_code < 400 or response.status_code == httpx.codes.NOT_MODIFIED:
                return response

            if attempt >= self._max_attempts - 1:
                raise GitHubError(
                    f"{method} {url} returned {response.status_code} after "
                    f"{self._max_attempts} attempts: {response.text[:200]}"
                )

            wait = _retry_delay(response, attempt)
            if wait is None:
                raise GitHubError(
                    f"{method} {url} returned {response.status_code}: {response.text[:200]}"
                )
            if wait > MAX_SLEEP_SECONDS:
                raise GitHubError(
                    f"{method} {url} returned {response.status_code} and asked us to wait "
                    f"{wait:.0f}s, past the {MAX_SLEEP_SECONDS:.0f}s ceiling. Stopping so "
                    "the run can be resumed from the cache rather than sitting idle."
                )
            logger.warning(
                "%s on %s, waiting %.1fs before attempt %s",
                response.status_code,
                url,
                wait,
                attempt + 2,
            )
            await self._sleep(wait)
            attempt += 1

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        # No token means no header, rather than "Bearer " with nothing after it, which
        # httpx rejects as a malformed header before the request leaves the process. The
        # anonymous client is only useful for reading a public repo by hand; a mining run
        # gets 60 requests an hour that way and assert_mining_budget refuses it.
        return headers


def _retry_delay(response: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait, or None if this status is not worth retrying.

    403 covers both a spent primary budget and a secondary limit, and the two are told
    apart by x-ratelimit-remaining rather than by the status code. A 403 that is neither
    is a permissions problem and retrying it just burns the budget.
    """
    status = response.status_code

    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    if status in (httpx.codes.FORBIDDEN, httpx.codes.TOO_MANY_REQUESTS):
        if response.headers.get("x-ratelimit-remaining") == "0":
            reset = response.headers.get("x-ratelimit-reset")
            if reset:
                try:
                    seconds = float(reset) - datetime.now(UTC).timestamp()
                except ValueError:
                    seconds = 0.0
                return max(1.0, seconds) + 1.0
        if status == httpx.codes.TOO_MANY_REQUESTS or "secondary rate" in response.text.lower():
            return _backoff(attempt)
        return None

    if status >= 500:
        return _backoff(attempt)
    return None


def _backoff(attempt: int) -> float:
    """Exponential with jitter. The jitter matters because repos are mined in parallel."""
    return min(60.0, 2.0**attempt) + random.random()  # noqa: S311 -- backoff, not crypto


def _url(path: str) -> str:
    return path if path.startswith("http") else f"{API_ROOT}{path}"


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = _NEXT_LINK.search(link_header)
    return match["url"] if match else None
