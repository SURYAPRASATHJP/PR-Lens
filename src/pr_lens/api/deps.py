from typing import Any, Protocol

from fastapi import Request

from pr_lens.models import Delivery
from pr_lens.settings import Settings


class RecordsDeliveries(Protocol):
    async def record(self, delivery: Delivery) -> bool: ...

    async def mark_dispatched(self, delivery_id: str, status: str) -> None: ...


class DispatchesWork(Protocol):
    async def __call__(self, client_payload: dict[str, Any]) -> None: ...


# Resolved from app.state so tests can swap in fakes through dependency_overrides and
# run the whole handler without a database or a network.
def get_store(request: Request) -> RecordsDeliveries:
    store: RecordsDeliveries = request.app.state.store
    return store


def get_dispatcher(request: Request) -> DispatchesWork:
    dispatcher: DispatchesWork = request.app.state.dispatcher
    return dispatcher


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings
