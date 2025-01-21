# ruff: noqa: D102
from __future__ import annotations

import asyncio

from dataclasses import field
from typing import TYPE_CHECKING

import sunray

from ray_graph.graph import ActorRemoteOptions

from .event import Event
from .graph import RayAsyncNode, RayAsyncNodeActor, handle


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from typing import Any, TypeAlias

    from ray_graph.graph import RayGraph

Epoch: TypeAlias = int
EPOCH_MANAGER_NAME = "EpochManager"


class NextEpochEvent(Event[None]):
    """Generate next epoch."""


class EpochManagerNode(RayAsyncNode):  # pragma: no cover
    """The epoch manager node."""

    # If the epoch is not -1, it will trigger all nodes to recover from that epoch.
    current_epoch: Epoch = -1
    actor_options: ActorRemoteOptions = field(
        default_factory=lambda: ActorRemoteOptions(num_cpus=0.1)
    )

    async def remote_init(self) -> Any:
        self.queue: asyncio.Queue[Epoch] = asyncio.Queue(maxsize=1)
        self.current_epoch = 0
        await self.queue.put(self.current_epoch)

    async def recovery_from_snapshot(self, epoch: int) -> None:
        self.queue = asyncio.Queue(maxsize=1)
        self.current_epoch = epoch + 1
        await self.queue.put(self.current_epoch)

    @handle(NextEpochEvent)
    async def handle_next_epoch(self, _event: NextEpochEvent) -> None:
        self.current_epoch += 1
        await self.queue.put(self.current_epoch)


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
    nodes = graph.filter(lambda node: node.name != EPOCH_MANAGER_NAME)
    for epoch_ref in actor.methods.epochs.remote():
        epoch = sunray.get(epoch_ref)
        sunray.get([node.actor.methods.update_epoch.remote(epoch) for node in nodes])
        if epoch != 0:
            # take previous epoch snapshot
            previous_epoch = epoch - 1
            sunray.get([node.actor.methods.take_snapshot.remote(previous_epoch) for node in nodes])
        try:
            from opentelemetry.trace import get_tracer

            tracer = get_tracer(__name__)
            with tracer.start_as_current_span("New Epoch", attributes={"ray_graph.epoch": epoch}):
                yield epoch
        except ImportError:
            yield epoch
