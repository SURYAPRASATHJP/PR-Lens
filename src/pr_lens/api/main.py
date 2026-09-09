import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response, status

from pr_lens import __version__
from pr_lens.api.deps import DispatchesWork, get_dispatcher, get_settings_dep
from pr_lens.api.security import SIGNATURE_HEADER, is_valid_signature
from pr_lens.github.dispatch import DispatchError
from pr_lens.models import delivery_from_webhook
from pr_lens.settings import Settings

logger = logging.getLogger(__name__)

# The GitHub App posts here. Changing it means changing the App's webhook URL too.
WEBHOOK_PATH = "/webhooks/github"

app = FastAPI(title="PR-Lens", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post(WEBHOOK_PATH, status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    dispatch: Annotated[DispatchesWork, Depends(get_dispatcher)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> Response:
    """Prove GitHub sent this, hand it to Actions, answer 202.

    That is the whole receiver. It opens no database connection and holds no state, so it
    stays inside GitHub's ten second timeout and can move to another host in an afternoon.
    Three free tiers moved under this project on day one; assume a fourth will.
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
    if not delivery.is_actionable:
        # Deliberately not recorded anywhere. Writing every ping and label change would
        # mean a database on the serverless path to store events we never act on.
        logger.info("ignoring %s/%s", event, delivery.action)
        return _text(status.HTTP_202_ACCEPTED, "ignored")

    try:
        await dispatch(delivery.as_client_payload())
    except DispatchError:
        # 502 rather than 202. The receiver keeps no record, so the App's Advanced tab is
        # the only place this failure can show up, and it only shows non-2xx.
        logger.exception("dispatch failed for delivery %s", delivery_id)
        return _text(status.HTTP_502_BAD_GATEWAY, "dispatch failed")

    return _text(status.HTTP_202_ACCEPTED, "dispatched")


def _text(code: int, message: str) -> Response:
    return Response(content=message, status_code=code, media_type="text/plain")
