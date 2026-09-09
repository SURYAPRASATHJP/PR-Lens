import asyncpg

from pr_lens.models import Delivery

_INSERT = """
insert into deliveries (
    delivery_id, event, action, repo_full_name, pr_number, head_sha, installation_id
) values ($1, $2, $3, $4, $5, $6, $7)
on conflict (delivery_id) do nothing
returning delivery_id
"""

_MARK_DISPATCHED = """
update deliveries set dispatched_at = now(), dispatch_status = $2 where delivery_id = $1
"""


class DeliveryStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, delivery: Delivery) -> bool:
        """Insert the delivery. False means we have seen this delivery id before.

        GitHub redelivers, both automatically and from the App's Advanced tab. This
        return value is the only thing standing between a redelivery and a second round
        of review comments on the same pull request.
        """
        row = await self._pool.fetchrow(
            _INSERT,
            delivery.delivery_id,
            delivery.event,
            delivery.action,
            delivery.repo_full_name,
            delivery.pr_number,
            delivery.head_sha,
            delivery.installation_id,
        )
        return row is not None

    async def mark_dispatched(self, delivery_id: str, status: str) -> None:
        await self._pool.execute(_MARK_DISPATCHED, delivery_id, status)
