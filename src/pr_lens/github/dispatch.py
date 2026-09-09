import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"


class DispatchError(RuntimeError):
    """The dispatch did not reach GitHub. The caller must not report success."""


class GitHubDispatcher:
    """Fires repository_dispatch at our own repo to start an Actions run.

    This uses a fine-grained PAT scoped to pr-lens alone, not the App's installation
    token. repository_dispatch needs Contents: write on the target, and the App is
    deliberately Contents: read-only so that "never pushes to your repo" is true at the
    token level rather than only in our code. See ADR 0002.
    """

    def __init__(self, client: httpx.AsyncClient, repo: str, token: str, event_type: str) -> None:
        self._client = client
        self._repo = repo
        self._token = token
        self._event_type = event_type

    async def __call__(self, client_payload: dict[str, Any]) -> None:
        try:
            response = await self._client.post(
                f"{API_ROOT}/repos/{self._repo}/dispatches",
                json={"event_type": self._event_type, "client_payload": client_payload},
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        except httpx.HTTPError as exc:
            raise DispatchError(f"dispatch to {self._repo} failed: {exc}") from exc

        if response.status_code in (401, 403):
            # Fine-grained PATs expire, 90 days by default. A silent failure here looks
            # like the reviewer quietly getting worse, three months after anyone touched
            # the code. Say the actual reason out loud.
            raise DispatchError(
                f"GitHub returned {response.status_code} for dispatch to {self._repo}. "
                "GH_DISPATCH_TOKEN is expired, revoked, or missing Contents: write on "
                "that repo. Fine-grained PATs expire; regenerate it and update the Space "
                "secret."
            )
        if response.status_code != httpx.codes.NO_CONTENT:
            raise DispatchError(
                f"GitHub returned {response.status_code} for dispatch to {self._repo}: "
                f"{response.text[:200]}"
            )
        logger.info("dispatched %s to %s", self._event_type, self._repo)
