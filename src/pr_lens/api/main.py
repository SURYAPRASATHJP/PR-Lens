import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request, Response, status

from pr_lens import __version__
from pr_lens import logging as app_logging
from pr_lens.api.deps import (
    DispatchesWork,
    RecordsDeliveries,
    get_dispatcher,
    get_settings_dep,
    get_store,
)
from pr_lens.api.security import SIGNATURE_HEADER, is_valid_signature
from pr_lens.db.deliveries import DeliveryStore
from pr_lens.db.pool import create_pool
from pr_lens.github.dispatch import DispatchError, GitHubDispatcher
from pr_lens.models import Delivery, delivery_from_webhook
from pr_lens.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# The GitHub App is configured to post here. Changing it means changing the App too.
WEBHOOK_PATH = "/webhooks/github"

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post(WEBHOOK_PATH, status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    store: Annotated[RecordsDeliveries, Depends(get_store)],
    dispatch: Annotated[DispatchesWork, Depends(get_dispatcher)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> Response:
    """Verify, record, dispatch, return. No model call ever happens on this path.

    Anything slow or interesting runs as an Actions job on the public repo, where compute
    is free and unlimited and where taking four minutes costs us nothing.
    """
    body = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER)
    if not is_valid_signature(body, signature, settings.gh_webhook_secret):
        logger.warning("rejected webhook with bad or missing signature")
        return _text(status.HTTP_401_UNAUTHORIZED, "signature mismatch")

    delivery_id = request.headers.get("X-GitHub-Delivery")
    event = request.headers.get("X-GitHub-Event")
    if not delivery_id or not event:
        return _text(status.HTTP_400_BAD_REQUEST, "missing delivery or event header")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return _text(status.HTTP_400_BAD_REQUEST, "body is not json")
    if not isinstance(payload, dict):
        return _text(status.HTTP_400_BAD_REQUEST, "body is not a json object")

    delivery = delivery_from_webhook(delivery_id, event, payload)

    if not await store.record(delivery):
        # Already seen. GitHub redelivers on its own, and a reviewer that comments twice
        # on one pull request is exactly the noise this project measures against.
        logger.info("ignoring redelivery %s", delivery_id)
        return _text(status.HTTP_202_ACCEPTED, "duplicate delivery")

    if not delivery.is_actionable:
        logger.info("recorded %s/%s, nothing to dispatch", event, delivery.action)
        return _text(status.HTTP_202_ACCEPTED, "recorded")

    await _dispatch(dispatch, store, delivery)
    return _text(status.HTTP_202_ACCEPTED, "dispatched")


async def _dispatch(dispatch: DispatchesWork, store: RecordsDeliveries, delivery: Delivery) -> None:
    """Record the outcome either way.

    We still answer 202 on failure because GitHub will not retry usefully and a 5xx only
    turns a visible database row into an invisible one. The failure lives in
    deliveries.dispatch_status, which is queryable.
    """
    try:
        await dispatch(delivery.as_client_payload())
    except DispatchError:
        logger.exception("dispatch failed for delivery %s", delivery.delivery_id)
        await store.mark_dispatched(delivery.delivery_id, "failed")
    else:
        await store.mark_dispatched(delivery.delivery_id, "ok")


def _text(code: int, message: str) -> Response:
    return Response(content=message, status_code=code, media_type="text/plain")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app_logging.configure(settings.log_level)
    pool = await create_pool(settings.database_url)
    # GitHub times webhook deliveries out at 10 seconds and does not retry. Every second
    # spent here is a second the delivery can be lost, so the ceiling is deliberately low.
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=2.0))
    app.state.settings = settings
    app.state.store = DeliveryStore(pool)
    app.state.dispatcher = GitHubDispatcher(
        client, settings.dispatch_repo, settings.gh_dispatch_token, settings.dispatch_event_type
    )
    try:
        yield
    finally:
        await client.aclose()
        await pool.close()


def create_app(*, with_lifespan: bool = True) -> FastAPI:
    """Factory so tests get a fresh app that never opens a pool or a socket."""
    created = FastAPI(
        title="PR-Lens",
        version=__version__,
        lifespan=lifespan if with_lifespan else None,
    )
    created.include_router(router)
    return created


app = create_app()
