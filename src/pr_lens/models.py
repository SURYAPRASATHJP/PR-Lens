from dataclasses import dataclass
from typing import Any

# GitHub only sends us work for these. Everything else is recorded and ignored,
# so we can see in Postgres what we chose not to act on.
ACTIONABLE_PULL_REQUEST_ACTIONS = frozenset(
    {"opened", "synchronize", "reopened", "ready_for_review"}
)


@dataclass(frozen=True, slots=True)
class Delivery:
    """One webhook delivery, as recorded before any work is dispatched."""

    delivery_id: str
    event: str
    action: str | None
    repo_full_name: str | None
    pr_number: int | None
    head_sha: str | None
    installation_id: int | None

    @property
    def is_actionable(self) -> bool:
        return (
            self.event == "pull_request"
            and self.action in ACTIONABLE_PULL_REQUEST_ACTIONS
            and self.repo_full_name is not None
            and self.pr_number is not None
        )

    def as_client_payload(self) -> dict[str, Any]:
        """GitHub caps client_payload at 10 top-level keys. This uses seven.

        The receiver no longer writes the delivery row, so everything the row needs has to
        survive the trip. Only identifiers travel: the Actions job re-fetches the pull
        request from the API rather than trusting a body that arrived over the internet.
        """
        return {
            "delivery_id": self.delivery_id,
            "event": self.event,
            "action": self.action,
            "repo_full_name": self.repo_full_name,
            "pr_number": self.pr_number,
            "head_sha": self.head_sha,
            "installation_id": self.installation_id,
        }

    @classmethod
    def from_client_payload(cls, payload: dict[str, Any]) -> "Delivery":
        """Rebuild what the receiver sent. Raises KeyError if the dispatch shape drifts."""
        return cls(
            delivery_id=str(payload["delivery_id"]),
            event=str(payload["event"]),
            action=_as_str(payload.get("action")),
            repo_full_name=_as_str(payload.get("repo_full_name")),
            pr_number=_as_int(payload.get("pr_number")),
            head_sha=_as_str(payload.get("head_sha")),
            installation_id=_as_int(payload.get("installation_id")),
        )


def delivery_from_webhook(delivery_id: str, event: str, payload: dict[str, Any]) -> Delivery:
    pull_request = payload.get("pull_request")
    pull_request = pull_request if isinstance(pull_request, dict) else {}
    repository = payload.get("repository")
    repository = repository if isinstance(repository, dict) else {}
    installation = payload.get("installation")
    installation = installation if isinstance(installation, dict) else {}
    head = pull_request.get("head")
    head = head if isinstance(head, dict) else {}

    return Delivery(
        delivery_id=delivery_id,
        event=event,
        action=_as_str(payload.get("action")),
        repo_full_name=_as_str(repository.get("full_name")),
        pr_number=_as_int(pull_request.get("number")),
        head_sha=_as_str(head.get("sha")),
        installation_id=_as_int(installation.get("id")),
    )


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
