import hashlib
import hmac
import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pr_lens.api.deps import get_dispatcher, get_settings_dep, get_store
from pr_lens.api.main import WEBHOOK_PATH, create_app
from pr_lens.models import Delivery
from pr_lens.settings import Settings

WEBHOOK_SECRET = "test-webhook-secret"


class FakeStore:
    def __init__(self) -> None:
        self.recorded: list[Delivery] = []
        self.marked: list[tuple[str, str]] = []
        self._seen: set[str] = set()

    async def record(self, delivery: Delivery) -> bool:
        if delivery.delivery_id in self._seen:
            return False
        self._seen.add(delivery.delivery_id)
        self.recorded.append(delivery)
        return True

    async def mark_dispatched(self, delivery_id: str, status: str) -> None:
        self.marked.append((delivery_id, status))


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
        database_url="postgresql://unused/unused",
        dispatch_repo="SURYAPRASATHJP/pr-lens",
    )


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def dispatcher() -> FakeDispatcher:
    return FakeDispatcher()


@pytest.fixture
def client(
    settings: Settings, store: FakeStore, dispatcher: FakeDispatcher
) -> Iterator[TestClient]:
    app = create_app(with_lifespan=False)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_dispatcher] = lambda: dispatcher
    app.dependency_overrides[get_settings_dep] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


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
    delivery_id: str = "11111111-2222-3333-4444-555555555555",
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
