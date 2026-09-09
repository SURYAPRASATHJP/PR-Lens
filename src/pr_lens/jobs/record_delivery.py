"""Write the delivery row that the receiver deliberately does not write.

Run as the first step of review.yml. Postgres lives on the Actions side of the line, so
the receiver stays stateless and portable and Neon never sees serverless traffic.

Exits non-zero on a real failure. A delivery we have already seen is not a failure: it
exits 0 having set new=false, and the workflow skips the rest of the run.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pr_lens.db.connection import connect
from pr_lens.db.deliveries import record
from pr_lens.logging import configure
from pr_lens.models import Delivery

logger = logging.getLogger(__name__)


def emit_output(name: str, value: str) -> None:
    """Hand a value back to the workflow so later steps can branch on it."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path is None:
        logger.info("%s=%s", name, value)
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def parse_payload(raw: str) -> Delivery:
    payload: Any = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("client_payload is not a json object")
    return Delivery.from_client_payload(payload)


async def main() -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is not set")
        return 1

    raw = os.environ.get("CLIENT_PAYLOAD")
    if not raw:
        logger.error("CLIENT_PAYLOAD is not set")
        return 1

    try:
        delivery = parse_payload(raw)
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.exception("client_payload did not match the shape the receiver sends")
        return 1

    conn = await connect(dsn)
    try:
        is_new = await record(conn, delivery)
    finally:
        await conn.close()

    emit_output("new", "true" if is_new else "false")
    if is_new:
        logger.info(
            "recorded %s for %s#%s",
            delivery.delivery_id,
            delivery.repo_full_name,
            delivery.pr_number,
        )
    else:
        logger.info("delivery %s already recorded, skipping the run", delivery.delivery_id)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
