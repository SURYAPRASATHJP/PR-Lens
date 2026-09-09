from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from fastapi import Request

from pr_lens.github.dispatch import GitHubDispatcher
from pr_lens.settings import Settings, get_settings


class DispatchesWork(Protocol):
    async def __call__(self, client_payload: dict[str, Any]) -> None: ...


def get_settings_dep(request: Request) -> Settings:
    """Prefer an override placed on the app, otherwise read the environment."""
    override: Settings | None = getattr(request.app.state, "settings", None)
    return override if override is not None else get_settings()


async def get_dispatcher(request: Request) -> AsyncIterator[DispatchesWork]:
    """One client per request, closed when the request ends.

    A module-level client would be reused across serverless invocations, which is only
    safe while the event loop survives between them. It does not reliably, and a client
    bound to a closed loop fails in a way that reads like a network problem. A fresh
    client costs a few milliseconds and is always correct.
    """
    override: DispatchesWork | None = getattr(request.app.state, "dispatcher", None)
    if override is not None:
        yield override
        return

    settings = get_settings_dep(request)
    # GitHub gives a webhook ten seconds and does not retry a failure.
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield GitHubDispatcher(
            client,
            settings.dispatch_repo,
            settings.gh_dispatch_token,
            settings.dispatch_event_type,
        )
