from pr_lens.models import Delivery, delivery_from_webhook


def test_reads_the_fields_we_dispatch_on() -> None:
    delivery = delivery_from_webhook(
        "d-1",
        "pull_request",
        {
            "action": "synchronize",
            "pull_request": {"number": 42, "head": {"sha": "b" * 40}},
            "repository": {"full_name": "octocat/hello-world"},
            "installation": {"id": 999},
        },
    )

    assert delivery.action == "synchronize"
    assert delivery.repo_full_name == "octocat/hello-world"
    assert delivery.pr_number == 42
    assert delivery.head_sha == "b" * 40
    assert delivery.installation_id == 999
    assert delivery.is_actionable


def test_a_payload_with_nothing_we_need_parses_to_nulls() -> None:
    delivery = delivery_from_webhook("d-2", "ping", {"zen": "Keep it logically awesome."})

    assert delivery.action is None
    assert delivery.repo_full_name is None
    assert not delivery.is_actionable


def test_wrong_types_in_the_payload_do_not_raise() -> None:
    # GitHub will not send this, but the endpoint is public and anything that gets past
    # the HMAC should still fail as a 202 no-op rather than a 500.
    delivery = delivery_from_webhook(
        "d-3",
        "pull_request",
        {"action": 5, "pull_request": "not an object", "repository": [], "installation": None},
    )

    assert delivery.action is None
    assert delivery.pr_number is None
    assert not delivery.is_actionable


def test_a_boolean_is_not_accepted_as_a_pull_request_number() -> None:
    delivery = delivery_from_webhook(
        "d-4", "pull_request", {"action": "opened", "pull_request": {"number": True}}
    )
    assert delivery.pr_number is None


def test_only_the_four_actions_worth_reviewing_are_actionable() -> None:
    for action in ("opened", "synchronize", "reopened", "ready_for_review"):
        assert _with_action(action).is_actionable, action
    for action in ("closed", "labeled", "assigned", "edited", "review_requested"):
        assert not _with_action(action).is_actionable, action


def _with_action(action: str) -> Delivery:
    return delivery_from_webhook(
        "d",
        "pull_request",
        {
            "action": action,
            "pull_request": {"number": 1, "head": {"sha": "c" * 40}},
            "repository": {"full_name": "o/r"},
        },
    )
