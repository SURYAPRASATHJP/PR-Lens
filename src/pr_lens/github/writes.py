"""Every write this project makes to the GitHub API goes through send(), and nothing else.

PR-Lens promises it never opens a pull request, pushes, applies a suggestion or approves.
The App's permissions do not keep that promise. Contents read-only stops a push, but Pull
requests read and write, which posting a review comment needs, also permits opening a
pull request and approving one. So the promise is kept here: ALLOWED is an explicit list
of (method, path) routes, and a write matching none of them raises before it leaves the
process.

What enforces it, in tests/test_github_writes.py: the allowlist holds no route that opens
a pull request or approves one, and no module outside this one makes an HTTP call that
could write. The read client's own send is typed to GET alone.

Adding a route is a deliberate edit to ALLOWED with its reason, and to the test that
lists what ALLOWED holds. Phase 4 will add the review comment route here. If it ever
posts through the reviews endpoint, which accepts event=APPROVE in the same body as a
comment, the route needs a body check as well as a path, and this module is where it goes.
"""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

API_HOST = "api.github.com"

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# One path segment of an owner or repository name, as GitHub allows them.
_NAME = r"[A-Za-z0-9._-]+"


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    pattern: re.Pattern[str]
    why: str


ALLOWED: tuple[Route, ...] = (
    Route(
        "POST",
        re.compile(rf"^/repos/{_NAME}/{_NAME}/dispatches$"),
        "start a review run on our own repository, from the webhook receiver",
    ),
)


class ForbiddenWrite(RuntimeError):
    """A write that is not on the allowlist. A bug in the caller, never a network error."""


def route_for(method: str, url: str) -> Route:
    """The allowed route this write matches, or ForbiddenWrite saying why it has none."""
    method = method.upper()
    if method not in WRITE_METHODS:
        raise ForbiddenWrite(f"{method} is not a write; reads go through GitHubClient")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != API_HOST:
        raise ForbiddenWrite(f"{method} {url} is not a request to https://{API_HOST}")
    if parts.query or parts.fragment:
        raise ForbiddenWrite(f"{method} {url} carries a query; no allowed write takes one")
    for route in ALLOWED:
        if route.method == method and route.pattern.match(parts.path):
            return route
    raise ForbiddenWrite(
        f"{method} {parts.path} is not on the write allowlist in pr_lens.github.writes"
    )


async def send(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    route_for(method, url)
    return await client.request(method, url, json=json, headers=headers)
