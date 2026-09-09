import asyncpg

from pr_lens.models import Delivery

_INSERT = """
insert into deliveries (
    delivery_id, event, action, repo_full_name, pr_number, head_sha, installation_id
) values ($1, $2, $3, $4, $5, $6, $7)
on conflict (delivery_id) do nothing
returning delivery_id
"""


async def record(conn: asyncpg.Connection, delivery: Delivery) -> bool:
    """Insert the delivery. False means this id has been seen before.

    This return value is the whole redelivery guard. The receiver keeps no state, so if
    this said nothing, a redelivered webhook would produce a second set of review comments
    on a pull request that already has them.
    """
    row = await conn.fetchrow(
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
