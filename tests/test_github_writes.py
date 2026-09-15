"""The write chokepoint. These tests are the whole proof of "never opens a pull request and
never approves one", because the App's Pull requests read and write permission allows both."""

import ast
import typing
from pathlib import Path

import httpx
import pytest
import respx

from pr_lens.github import writes
from pr_lens.github.client import GitHubClient
from pr_lens.github.writes import ALLOWED, ForbiddenWrite, route_for

API = "https://api.github.com"
SOURCE = Path(__file__).resolve().parents[1] / "src" / "pr_lens"

# Every route that would break a published promise, with the promise it breaks.
FORBIDDEN = [
    ("POST", "/repos/o/r/pulls", "open a pull request"),
    ("POST", "/repos/o/r/pulls/7/reviews", "create a review, whose body can say APPROVE"),
    ("POST", "/repos/o/r/pulls/7/reviews/9/events", "submit a review as APPROVE"),
    ("PUT", "/repos/o/r/pulls/7/merge", "merge a pull request"),
    ("PATCH", "/repos/o/r/pulls/7", "edit or close a pull request"),
    ("POST", "/repos/o/r/pulls/7/update-branch", "push to the pull request's branch"),
    ("PUT", "/repos/o/r/pulls/7/reviews/9/dismissals", "dismiss a human's review"),
    ("PUT", "/repos/o/r/contents/README.md", "push a file, which is how a suggestion lands"),
    ("POST", "/repos/o/r/git/refs", "create a branch"),
    ("PATCH", "/repos/o/r/git/refs/heads/main", "move a branch"),
    ("POST", "/repos/o/r/git/commits", "write a commit"),
    ("POST", "/repos/o/r/merges", "merge branches"),
    ("POST", "/repos/o/r/forks", "fork the repository"),
    ("POST", "/graphql", "anything a GraphQL mutation can do, approval included"),
]


def test_the_allowlist_holds_exactly_these_routes() -> None:
    """A new route fails here until this list is edited on purpose, beside its reason."""
    assert [(route.method, route.pattern.pattern) for route in ALLOWED] == [
        ("POST", r"^/repos/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/dispatches$"),
    ]
    assert all(route.why for route in ALLOWED)


@pytest.mark.parametrize(("method", "path", "breaks"), FORBIDDEN, ids=[f[2] for f in FORBIDDEN])
def test_no_route_opens_a_pull_request_approves_one_or_pushes(
    method: str, path: str, breaks: str
) -> None:
    with pytest.raises(ForbiddenWrite):
        route_for(method, API + path)
    assert not any(route.method == method and route.pattern.match(path) for route in ALLOWED), (
        f"the allowlist permits a route that would {breaks}"
    )


def test_the_dispatch_route_is_allowed_on_the_api_host_and_nowhere_else() -> None:
    assert route_for("POST", f"{API}/repos/SURYAPRASATHJP/pr-lens/dispatches") is ALLOWED[0]
    for url in (
        "https://evil.example/repos/o/r/dispatches",
        "http://api.github.com/repos/o/r/dispatches",
        f"{API}/repos/o/r/dispatches?x=1",
        f"{API}/repos/o/r/../pulls/dispatches",
        f"{API}/repos/o/r/pulls/dispatches",
    ):
        with pytest.raises(ForbiddenWrite):
            route_for("POST", url)


def test_a_read_is_not_a_write() -> None:
    with pytest.raises(ForbiddenWrite, match="reads go through GitHubClient"):
        route_for("GET", f"{API}/repos/o/r/dispatches")


@respx.mock
async def test_a_forbidden_write_never_reaches_the_network() -> None:
    route = respx.post(f"{API}/repos/o/r/pulls").mock(return_value=httpx.Response(201))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ForbiddenWrite):
            await writes.send(client, "POST", f"{API}/repos/o/r/pulls", json={}, headers={})
    assert not route.called


def test_the_read_client_is_typed_to_get_alone() -> None:
    hints = typing.get_type_hints(GitHubClient._send)
    assert hints["method"] == typing.Literal["GET"]


# Calls that could send an HTTP request, and every place in src/ that makes one. A new one
# fails here: route it through pr_lens.github.writes, or add it below with the reason it
# is not a write to GitHub.
HTTP_VERBS = frozenset({"post", "put", "patch", "delete", "request", "send", "stream"})
KNOWN_CALLS = {
    # FastAPI registering the webhook route. Receiving, not sending.
    ("api/main.py", "app.post"),
    # HttpCache.put, a file on disk.
    ("github/client.py", "self._cache.put"),
    # The read client, whose _send is typed Literal["GET"].
    ("github/client.py", "self._client.request"),
    # The chokepoint itself.
    ("github/writes.py", "client.request"),
    # The inference call to Groq, OpenRouter or NIM. ChatClient refuses a GitHub host.
    ("review/provider.py", "self._client.post"),
}


def test_nothing_outside_the_chokepoint_can_send_a_write() -> None:
    found = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in HTTP_VERBS
                # Going through the chokepoint is the one way to write, from anywhere.
                and ast.unparse(node.func) != "writes.send"
            ):
                found.add((path.relative_to(SOURCE).as_posix(), ast.unparse(node.func)))
    assert found == KNOWN_CALLS


def test_dispatch_writes_through_the_chokepoint() -> None:
    """The only writer today. If it stopped calling writes.send the allowlist would guard
    nothing, and the scan above would catch the direct call that replaced it."""
    text = (SOURCE / "github" / "dispatch.py").read_text(encoding="utf-8")
    assert "writes.send(" in text
