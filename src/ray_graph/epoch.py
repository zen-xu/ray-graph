# ruff: noqa: D102
from __future__ import annotations

import asyncio

from dataclasses import field
from typing import TYPE_CHECKING

import sunray

from ray_graph.graph import ActorRemoteOptions

from .event import Event
from .graph import RayAsyncNode, RayAsyncNodeActor, RayNodeRef, handle


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator
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


def epochs(
    graph: RayGraph,
    *,
    disable_snapshot: bool = False,
    update_epoch_nodes_filter: Callable[[RayNodeRef], bool] = lambda node: node.name
    != EPOCH_MANAGER_NAME,
    snapshot_nodes_filter: Callable[[RayNodeRef], bool] = lambda node: node.name
    != EPOCH_MANAGER_NAME,
) -> Generator[Epoch, None, None]:  # pragma: no cover
    """Get the epoch from EpochManager."""
    actor = sunray.get_actor[EpochManagerNodeActor](EPOCH_MANAGER_NAME)
    update_epoch_nodes = graph.filter(update_epoch_nodes_filter)
    snapshot_nodes = graph.filter(snapshot_nodes_filter)

    def handle_epoch(epoch: Epoch) -> None:
        sunray.get([node.actor.methods.update_epoch.remote(epoch) for node in update_epoch_nodes])
        if epoch != 0 and not disable_snapshot:
            # take previous epoch snapshot
            previous_epoch = epoch - 1
            sunray.get(
                [
                    node.actor.methods.take_snapshot.remote(previous_epoch)
                    for node in snapshot_nodes
                ]
            )

    for epoch_ref in actor.methods.epochs.remote():
        epoch = sunray.get(epoch_ref)
        handle_epoch(epoch)
        yield epoch
