# ruff: noqa: ARG001
from __future__ import annotations

import pytest
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


def test_get_epoch_from_context(init_ray):
    class UpdateEpochEvent(Event): ...

    class GetEpochEvent(Event[int]): ...

    class CustomNode(RayAsyncNode):
        @handle(UpdateEpochEvent)
        async def handle_update_epoch(self, _event: UpdateEpochEvent) -> None:
            ctx = get_node_context()
            # notify epoch manager to next epoch
            await ctx.graph.get(EPOCH_MANAGER_NAME).send(NextEpochEvent())

        @handle(GetEpochEvent)
        async def handle_get_epoch_event(self, _event: GetEpochEvent) -> int:
            ctx = get_node_context()
            return ctx.current_epoch

    graph = RayGraphBuilder(
        {"node1": CustomNode(), EPOCH_MANAGER_NAME: EpochManagerNode()}
    ).build()
    graph.start()
    epoch_generator = epochs(graph)
    assert next(epoch_generator) == 0
    sunray.get(graph.get("node1").send(UpdateEpochEvent()))
    assert next(epoch_generator) == 1
    assert sunray.get(graph.get("node1").send(GetEpochEvent())) == 1


def test_fail_to_get_epoch_from_context(init_ray):
    class GetEpochEvent(Event[int]): ...

    class CustomNode(RayAsyncNode):
        @handle(GetEpochEvent)
        async def handle_get_epoch_event(self, _event: GetEpochEvent) -> int:
            ctx = get_node_context()
            return ctx.current_epoch

    graph = RayGraphBuilder({"node1": CustomNode()}).build()
    graph.start()
    with pytest.raises(ValueError, match=r"current epoch is not set"):
        sunray.get(graph.get("node1").send(GetEpochEvent()))
