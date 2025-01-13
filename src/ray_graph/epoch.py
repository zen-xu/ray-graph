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

    from ray_graph.graph import ActorRemoteOptions, RayGraph

Epoch: TypeAlias = int
EPOCH_MANAGER_NAME = "EpochManager"


class NextEpochEvent(Event[None]):
    """Generate next epoch."""


class EpochManagerNode(RayAsyncNode):  # pragma: no cover
    """The epoch manager node."""

    queue: asyncio.Queue[Epoch]

    def __init__(self, epoch: Epoch | None = None) -> None:
        """If the epoch is not None, it will trigger all nodes to recover from that epoch."""
        self.current_epoch = epoch

    async def remote_init(self) -> Any:
        self.queue = asyncio.Queue(maxsize=1)
        self.current_epoch = 0
        await self.queue.put(self.current_epoch)

    async def recovery_from_snapshot(self, epoch: int) -> None:
        self.queue = asyncio.Queue(maxsize=1)
        self.current_epoch = epoch + 1
        await self.queue.put(self.current_epoch)

    @handle(NextEpochEvent)
    async def handle_next_epoch(self, _event: NextEpochEvent) -> None:
        assert self.current_epoch is not None
        self.current_epoch += 1
        await self.queue.put(self.current_epoch)

    def actor_options(self) -> ActorRemoteOptions:
        return {"num_cpus": 0.1}


class EpochManagerNodeActor(RayAsyncNodeActor):  # pragma: no cover
    """The epoch manager node actor."""

    ray_node: EpochManagerNode

    @sunray.remote_method
    async def epochs(self) -> AsyncGenerator[Epoch]:
        while True:
            yield await self.ray_node.queue.get()


def epochs(graph: RayGraph) -> Generator[Epoch, None, None]:  # pragma: no cover
    """Get the epoch from EpochManager."""
    actor = sunray.get_actor[EpochManagerNodeActor](EPOCH_MANAGER_NAME)
    for epoch_ref in actor.methods.epochs.remote():
        epoch = sunray.get(epoch_ref)
        if epoch != 0:
            # take previous epoch snapshot
            previous_epoch = epoch - 1
            nodes = graph.filter(lambda node: node.name != EPOCH_MANAGER_NAME)
            sunray.get([node.actor.methods.take_snapshot.remote(previous_epoch) for node in nodes])
        try:
            from opentelemetry.trace import get_tracer

            tracer = get_tracer(__name__)
            with tracer.start_as_current_span("New Epoch", attributes={"ray_graph.epoch": epoch}):
                yield epoch
        except ImportError:
            yield epoch
