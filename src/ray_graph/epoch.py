# ruff: noqa: D102
from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

import sunray

from .event import Event
from .graph import RayAsyncNode, RayAsyncNodeActor, handle


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from typing import Any, TypeAlias

    from ray_graph.graph import ActorRemoteOptions

Epoch: TypeAlias = int
EPOCH_MANAGER_NAME = "EpochManager"


class NextEpochEvent(Event[None]):
    """Generate next epoch."""


class EpochManagerNode(RayAsyncNode):
    """The epoch manager node."""

    async def remote_init(self) -> Any:
        self.queue: asyncio.Queue[Epoch] = asyncio.Queue(maxsize=1)
        self.current_epoch = 0
        await self.queue.put(self.current_epoch)

    @handle(NextEpochEvent)
    async def handle_next_epoch(self, _event: NextEpochEvent) -> None:
        self.current_epoch += 1
        await self.queue.put(self.current_epoch)

    def actor_options(self) -> ActorRemoteOptions:
        return {"num_cpus": 0.1}


class EpochManagerNodeActor(RayAsyncNodeActor):
    """The epoch manager node actor."""

    ray_node: EpochManagerNode

    @sunray.remote_method
    async def epochs(self) -> AsyncGenerator[Epoch]:
        while True:
            yield await self.ray_node.queue.get()


def epochs() -> Generator[Epoch, None, None]:
    """Get the epoch from EpochManager."""
    actor = sunray.get_actor[EpochManagerNodeActor](EPOCH_MANAGER_NAME)
    for epoch_ref in actor.methods.epochs.remote():
        yield sunray.get(epoch_ref)
