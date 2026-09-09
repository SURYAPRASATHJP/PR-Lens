import hashlib
import hmac
import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pr_lens.api.main import WEBHOOK_PATH, app
from pr_lens.settings import Settings

WEBHOOK_SECRET = "test-webhook-secret"
DELIVERY_ID = "11111111-2222-3333-4444-555555555555"


class FakeDispatcher:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def __call__(self, client_payload: dict[str, Any]) -> None:
        self.calls.append(client_payload)
        if self.error is not None:
            raise self.error


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gh_webhook_secret=WEBHOOK_SECRET,
        gh_dispatch_token="test-token",
        dispatch_repo="SURYAPRASATHJP/pr-lens",
    )


@pytest.fixture
def dispatcher() -> FakeDispatcher:
    return FakeDispatcher()


@pytest.fixture
def client(settings: Settings, dispatcher: FakeDispatcher) -> Iterator[TestClient]:
    # The receiver reads both of these off app.state when they are present, which is how a
    # test avoids reaching the environment or the network.
    app.state.settings = settings
    app.state.dispatcher = dispatcher
    with TestClient(app) as test_client:
        yield test_client
    del app.state.settings
    del app.state.dispatcher


def sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def pull_request_body(
    action: str = "opened",
    number: int = 7,
    repo: str = "octocat/hello-world",
    sha: str = "a" * 40,
    installation_id: int = 12345,
) -> bytes:
    return json.dumps(
        {
            "action": action,
            "number": number,
            "pull_request": {"number": number, "head": {"sha": sha}},
            "repository": {"full_name": repo},
            "installation": {"id": installation_id},
        }
    ).encode()


def post_webhook(
    client: TestClient,
    body: bytes,
    *,
    event: str = "pull_request",
    delivery_id: str = DELIVERY_ID,
    signature: str | None = None,
) -> Any:
    headers = {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery_id,
        "Content-Type": "application/json",
    }
    signature = sign(body) if signature is None else signature
    if signature:
        headers["X-Hub-Signature-256"] = signature
    return client.post(WEBHOOK_PATH, content=body, headers=headers)


def database_url() -> str | None:
    """Set in CI by the postgres service container. Absent locally, and that is fine."""
    return os.environ.get("DATABASE_URL")


requires_postgres = pytest.mark.skipif(
    database_url() is None, reason="DATABASE_URL is not set, skipping database tests"
)
