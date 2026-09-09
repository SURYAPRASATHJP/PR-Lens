from fastapi.testclient import TestClient

from pr_lens.api.main import WEBHOOK_PATH
from pr_lens.github.dispatch import DispatchError

from .conftest import DELIVERY_ID, FakeDispatcher, post_webhook, pull_request_body, sign


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_signed_pull_request_is_dispatched(client: TestClient, dispatcher: FakeDispatcher) -> None:
    response = post_webhook(client, pull_request_body())

    assert response.status_code == 202
    assert dispatcher.calls == [
        {
            "delivery_id": DELIVERY_ID,
            "event": "pull_request",
            "action": "opened",
            "repo_full_name": "octocat/hello-world",
            "pr_number": 7,
            "head_sha": "a" * 40,
            "installation_id": 12345,
        }
    ]


def test_the_payload_carries_everything_the_delivery_row_needs(
    client: TestClient, dispatcher: FakeDispatcher
) -> None:
    # The receiver no longer writes the row, so anything missing here is lost for good.
    post_webhook(client, pull_request_body())
    sent = dispatcher.calls[0]

    assert set(sent) == {
        "delivery_id",
        "event",
        "action",
        "repo_full_name",
        "pr_number",
        "head_sha",
        "installation_id",
    }
    assert len(sent) <= 10, "GitHub caps client_payload at ten top-level keys"


def test_bad_signature_is_rejected_and_nothing_is_dispatched(
    client: TestClient, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, pull_request_body(), signature="sha256=" + "0" * 64)

    assert response.status_code == 401
    assert dispatcher.calls == []


def test_missing_signature_is_rejected(client: TestClient, dispatcher: FakeDispatcher) -> None:
    response = post_webhook(client, pull_request_body(), signature="")

    assert response.status_code == 401
    assert dispatcher.calls == []


def test_body_tampered_after_signing_is_rejected(
    client: TestClient, dispatcher: FakeDispatcher
) -> None:
    body = pull_request_body()
    tampered = body.replace(b'"number": 7', b'"number": 9')
    response = post_webhook(client, tampered, signature=sign(body))

    assert response.status_code == 401
    assert dispatcher.calls == []


def test_ping_is_not_dispatched(client: TestClient, dispatcher: FakeDispatcher) -> None:
    response = post_webhook(client, b'{"zen":"Keep it logically awesome."}', event="ping")

    assert response.status_code == 202
    assert dispatcher.calls == []


def test_uninteresting_pull_request_action_is_not_dispatched(
    client: TestClient, dispatcher: FakeDispatcher
) -> None:
    response = post_webhook(client, pull_request_body(action="labeled"))

    assert response.status_code == 202
    assert dispatcher.calls == []


def test_a_redelivery_is_dispatched_again(client: TestClient, dispatcher: FakeDispatcher) -> None:
    # The receiver holds no state, so it cannot tell. Suppressing the duplicate is the
    # Actions job's responsibility, covered in test_record_delivery.py.
    body = pull_request_body()
    post_webhook(client, body)
    post_webhook(client, body)

    assert len(dispatcher.calls) == 2


def test_missing_delivery_header_is_a_bad_request(client: TestClient) -> None:
    body = pull_request_body()
    response = client.post(
        WEBHOOK_PATH,
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": sign(body)},
    )
    assert response.status_code == 400


def test_non_json_body_with_a_valid_signature_is_a_bad_request(client: TestClient) -> None:
    assert post_webhook(client, b"not json at all").status_code == 400


def test_a_failed_dispatch_is_a_502(client: TestClient, dispatcher: FakeDispatcher) -> None:
    # Nothing is recorded anywhere, so the App's Advanced tab is the only place this can
    # surface, and it only shows the response code. A 202 here would hide the failure.
    dispatcher.error = DispatchError("token expired")

    assert post_webhook(client, pull_request_body()).status_code == 502
