# ruff: noqa: ARG001
from __future__ import annotations

import sunray

from ray_graph import (
    EPOCH_MANAGER_NAME,
    EpochManagerNode,
    Event,
    NextEpochEvent,
    RayAsyncNode,
    RayGraphBuilder,
    epochs,
    get_node_context,
    handle,
)


def test_handle_epoch_event(init_ray):
    class CustomEvent(Event): ...

    class CustomNode(RayAsyncNode):
        @handle(CustomEvent)
        async def handle_custom_event(self, _event: CustomEvent) -> None:
            ctx = get_node_context()
            # notify epoch manager to next epoch
            await ctx.graph.get(EPOCH_MANAGER_NAME).send(NextEpochEvent())

    graph = RayGraphBuilder(
        {"node1": CustomNode(), EPOCH_MANAGER_NAME: EpochManagerNode()}
    ).build()
    graph.start()
    epoch_generator = epochs(graph)
    assert next(epoch_generator) == 0
    sunray.get(graph.get("node1").send(CustomEvent()))
    assert next(epoch_generator) == 1
