from fastapi.testclient import TestClient

from pr_lens.api.main import WEBHOOK_PATH
from pr_lens.github.dispatch import DispatchError

from .conftest import FakeDispatcher, FakeStore, post_webhook, pull_request_body, sign


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_signed_pull_request_is_recorded_and_dispatched(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, pull_request_body())

    assert response.status_code == 202
    assert len(store.recorded) == 1
    assert store.marked == [("11111111-2222-3333-4444-555555555555", "ok")]
    assert dispatcher.calls == [
        {
            "delivery_id": "11111111-2222-3333-4444-555555555555",
            "repo_full_name": "octocat/hello-world",
            "pr_number": 7,
            "head_sha": "a" * 40,
            "installation_id": 12345,
        }
    ]


def test_client_payload_stays_under_the_ten_key_limit(
    client: TestClient, dispatcher: FakeDispatcher
) -> None:
    post_webhook(client, pull_request_body())
    assert len(dispatcher.calls[0]) <= 10


def test_bad_signature_is_rejected_and_nothing_is_recorded(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, pull_request_body(), signature="sha256=" + "0" * 64)

    assert response.status_code == 401
    assert store.recorded == []
    assert dispatcher.calls == []


def test_missing_signature_is_rejected(client: TestClient, store: FakeStore) -> None:
    response = post_webhook(client, pull_request_body(), signature="")

    assert response.status_code == 401
    assert store.recorded == []


def test_body_tampered_after_signing_is_rejected(client: TestClient) -> None:
    body = pull_request_body()
    signature = sign(body)
    tampered = body.replace(b'"number": 7', b'"number": 9')
    response = post_webhook(client, tampered, signature=signature)

    assert response.status_code == 401


def test_a_redelivery_is_recorded_once_and_dispatched_once(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    body = pull_request_body()
    first = post_webhook(client, body)
    second = post_webhook(client, body)

    assert (first.status_code, second.status_code) == (202, 202)
    assert len(store.recorded) == 1
    assert len(dispatcher.calls) == 1


def test_ping_is_recorded_but_not_dispatched(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, b'{"zen":"Keep it logically awesome."}', event="ping")

    assert response.status_code == 202
    assert len(store.recorded) == 1
    assert dispatcher.calls == []


def test_uninteresting_pull_request_action_is_recorded_but_not_dispatched(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, pull_request_body(action="labeled"))

    assert response.status_code == 202
    assert len(store.recorded) == 1
    assert dispatcher.calls == []


def test_missing_delivery_header_is_a_bad_request(client: TestClient) -> None:
    body = pull_request_body()
    response = client.post(
        WEBHOOK_PATH,
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": sign(body)},
    )
    assert response.status_code == 400


def test_non_json_body_with_a_valid_signature_is_a_bad_request(client: TestClient) -> None:
    response = post_webhook(client, b"not json at all")
    assert response.status_code == 400


def test_a_failed_dispatch_still_answers_202_and_records_the_failure(
    client: TestClient, store: FakeStore, dispatcher: FakeDispatcher
) -> None:
    dispatcher.error = DispatchError("token expired")

    response = post_webhook(client, pull_request_body())

    assert response.status_code == 202
    assert store.marked == [("11111111-2222-3333-4444-555555555555", "failed")]
