# ruff: noqa: ARG002
from __future__ import annotations

import asyncio

from textwrap import dedent
from typing import Any

import pytest
import ray
import rustworkx as rwx
import sunray

from ray.util.scheduling_strategies import In, NodeLabelSchedulingStrategy

from ray_graph.epoch import EPOCH_MANAGER_NAME, EpochManagerNode, NextEpochEvent, epochs
from ray_graph.event import Event
from ray_graph.graph import (
    ActorRemoteOptions,
    PlacementWarning,
    RayAsyncNode,
    RayGraphBuilder,
    RayGraphRef,
    RayNode,
    RayNodeActor,
    RayNodeRef,
    RayResources,
    RegisterHandlerError,
    _convert_ray_resources_to_placement_bundle,
    get_node_context,
    handle,
)


class CustomEvent(Event[int]):
    value: int


@pytest.fixture
def dummy_graph() -> RayGraphRef:
    return RayGraphRef(rwx.PyDAG(check_cycle=True))


class TestRayNode:
    def test_register_event_handler(self):
        class CustomNode(RayNode):
            @handle(CustomEvent)
            def handle_custom_event(self, event: CustomEvent) -> Any: ...

        assert len(CustomNode._event_handlers) == 1
        assert CustomNode._event_handlers.get(CustomEvent) is not None
        assert not hasattr(CustomNode, "handle_custom_event")

    def test_register_duplicate_event_handler(self):
        with pytest.raises(RegisterHandlerError, match="got duplicate event handler for"):

            class CustomNode(RayNode):
                @handle(CustomEvent)
                def handle_custom_event(self, event: CustomEvent) -> Any: ...

                @handle(CustomEvent)
                def handle_custom_event2(self, event: CustomEvent) -> Any: ...

    def test_register_async_handler_for_ray_node(self):
        with pytest.raises(RegisterHandlerError, match="can't be async func"):

            class CustomNode(RayNode):
                @handle(CustomEvent)
                async def handle_custom_event(self, event: CustomEvent) -> Any: ...

    def test_register_non_async_handler_for_ray_async_node(self):
        with pytest.raises(RegisterHandlerError, match="must be async func"):

            class CustomNode(RayAsyncNode):
                @handle(CustomEvent)
                def handle_custom_event(self, event: CustomEvent) -> Any: ...

    def test_ray_node_actor_handle_event(self, init_local_ray, dummy_graph):
        class CustomNode(RayNode):
            def remote_init(self) -> None:
                self.value = 1

            @handle(CustomEvent)
            def handle_custom_event(self, event: CustomEvent) -> int:
                self.value += event.value

                return self.value

        node_actor = RayNodeActor.new_actor().remote("", CustomNode(), dummy_graph)
        sunray.get(node_actor.methods.remote_init.remote())
        assert sunray.get(node_actor.methods.handle.remote(CustomEvent(value=2))) == 3

        class FakeEvent(Event): ...

        with pytest.raises(ValueError, match=r"no handler for event .*FakeEvent.*"):
            sunray.get(node_actor.methods.handle.remote(FakeEvent()))
        sunray.kill(node_actor)

    def test_ray_node_ref(self, init_ray, dummy_graph):
        class GetProcName(Event[str]): ...

        class CustomNode(RayNode):
            @handle(GetProcName)
            def handle_custom_event(self, _event: GetProcName) -> str:
                import setproctitle

                return setproctitle.getproctitle()

        name = "test_ray_node_ref"
        node_actor = (
            RayNodeActor.new_actor().options(name=name).remote(name, CustomNode(), dummy_graph)
        )
        labels = {"node": "mac"}
        node_ref = RayNodeRef(name, labels)
        assert node_ref.name == name
        assert node_ref.labels == labels
        assert sunray.get(node_ref.send(GetProcName())) == f"ray::{name}.handle[GetProcName]"
        sunray.kill(node_actor)

    def test_ray_node_context(self, init_ray):
        class GetContext(Event): ...

        class CustomNode(RayNode):
            @handle(GetContext)
            def get_context(self, _event: GetContext) -> Any:
                ctx = get_node_context()
                return {
                    "graph": ctx.graph,
                    "node_name": ctx.node_name,
                }

        dag = rwx.PyDAG()
        dag.add_node(RayNodeRef("node1"))
        node_name = "Test"
        node_actor = (
            RayNodeActor.new_actor()
            .options(name=node_name)
            .remote(node_name, CustomNode(), RayGraphRef(dag))
        )
        remote_ctx = sunray.get(node_actor.methods.handle.remote(GetContext()))
        assert remote_ctx["graph"] is not None
        remote_ctx["graph"].get("node1")
        assert remote_ctx["node_name"] == node_name


class TestRayGraph:
    def test_set_relationships(self):
        class CustomNode(RayNode):
            def remote_init(self) -> None: ...

        total_nodes = {
            "node1": CustomNode(),
            "node2": CustomNode(),
            "node3": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.set_parent("node2", "node1")
        builder.set_parent("node2", "node1")  # add duplicate edge
        builder.set_children("node2", ["node3"])

        got = builder._dag.to_dot(node_attr=lambda n: {"label": n.name})
        got = got and got.strip()
        expect = dedent(
            """
            digraph {
            0 [label="node1"];
            1 [label="node2"];
            2 [label="node3"];
            0 -> 1 ;
            1 -> 2 ;
            }
            """
        ).strip()
        assert got == expect

    @pytest.fixture
    def graph_ref(self) -> RayGraphRef:
        """
           root1  root2
             │     │
         ┌───┴───┐ │
         │       │ │
         │       │ │
         ▼       ▼ ▼
        node2   node1
                  │
             ┌────┴────┐
             ▼         ▼
            leaf1    leaf2
        """

        dag: rwx.PyDAG[RayNodeRef, None] = rwx.PyDAG()
        root1, root2, node1, node2, leaf1, leaf2 = dag.add_nodes_from(
            [
                RayNodeRef("root1", labels={"kind": "root"}),
                RayNodeRef("root2", labels={"kind": "root"}),
                RayNodeRef("node1", labels={"kind": "normal"}),
                RayNodeRef("node2", labels={"kind": "normal"}),
                RayNodeRef("leaf1", labels={"kind": "leaf"}),
                RayNodeRef("leaf2", labels={"kind": "leaf"}),
            ]
        )
        dag.add_edges_from_no_data(
            [(root1, node1), (root2, node1), (root1, node2), (node1, leaf1), (node1, leaf2)]
        )
        return RayGraphRef(dag)

    def test_get(self, graph_ref: RayGraphRef):
        assert graph_ref.get("node1").name == "node1"

    @pytest.mark.parametrize(
        ("node_name", "expect"),
        [
            ("node1", ["root1", "root2"]),
            ("node2", ["root1"]),
            ("leaf1", ["node1"]),
            ("root1", []),
        ],
    )
    def test_get_parents(self, graph_ref: RayGraphRef, node_name, expect):
        assert sorted(node.name for node in graph_ref.get_parents(node_name)) == expect

    @pytest.mark.parametrize(
        ("node_name", "expect"),
        [
            ("node1", ["leaf1", "leaf2"]),
            ("node2", []),
            ("leaf1", []),
            ("root1", ["node1", "node2"]),
        ],
    )
    def test_get_children(self, graph_ref: RayGraphRef, node_name, expect):
        assert sorted(node.name for node in graph_ref.get_children(node_name)) == expect

    @pytest.mark.parametrize(
        ("node_name", "expect"),
        [
            ("node1", ["root1", "root2"]),
            ("node2", ["root1"]),
            ("leaf1", ["root1", "root2"]),
            ("root1", []),
        ],
    )
    def test_get_roots(self, graph_ref: RayGraphRef, node_name, expect):
        assert sorted(node.name for node in graph_ref.get_roots(node_name)) == expect

    @pytest.mark.parametrize(
        ("node_name", "expect"),
        [
            ("root1", ["leaf1", "leaf2", "node2"]),
            ("node1", ["leaf1", "leaf2"]),
            ("node2", []),
            ("leaf1", []),
        ],
    )
    def test_get_leaves(self, graph_ref: RayGraphRef, node_name, expect):
        assert sorted(node.name for node in graph_ref.get_leaves(node_name)) == expect

    @pytest.mark.parametrize(
        ("node_name", "expect"),
        [
            ("root1", []),
            ("node1", ["node2"]),
            ("leaf1", ["leaf2"]),
        ],
    )
    def test_get_siblings(self, graph_ref: RayGraphRef, node_name, expect):
        assert sorted(node.name for node in graph_ref.get_siblings(node_name)) == expect

    @pytest.mark.parametrize(
        ("predicate", "expect"),
        [
            (lambda node_ref: node_ref.name == "root1", ["root1"]),
            (lambda node_ref: node_ref.labels["kind"] == "root", ["root1", "root2"]),
            (lambda node_ref: node_ref.labels["kind"] == "normal", ["node1", "node2"]),
            (lambda node_ref: node_ref.labels["kind"] == "leaf", ["leaf1", "leaf2"]),
        ],
    )
    def test_filter(self, graph_ref: RayGraphRef, predicate, expect):
        assert sorted(node.name for node in graph_ref.filter(predicate)) == expect

    def test_start_graph(self, init_ray):
        class GetNodeName(Event): ...

        class CustomNode(RayNode):
            @handle(GetNodeName)
            def handle_get_node_name(self, event: GetNodeName) -> str:
                return get_node_context().node_name

        total_nodes = {
            "node1": CustomNode(),
            "node2": CustomNode(),
            "node3": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.set_parent("node2", "node1")
        builder.set_children("node2", ["node3"])
        graph = builder.build()
        graph.start()
        assert sunray.get(graph.get("node1").send(GetNodeName())) == "node1"

    def test_set_placement(self, init_ray):
        class GetPlacementName(Event): ...

        class CustomNode(RayNode):
            @handle(GetPlacementName)
            def handle_get_placement_name(self, event: GetPlacementName) -> str | None:
                if pg_id := get_node_context().runtime_context.get_placement_group_id():
                    return ray.util.placement_group_table()[pg_id]["name"]
                return None

        total_nodes = {
            "node": CustomNode(),
            "leaf1": CustomNode(),
            "leaf2": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.set_children("node", ["leaf1", "leaf2"])
        graph = builder.build()
        graph.start(
            placement_rule=(
                lambda node_name, _: "leaf" if node_name.startswith("leaf") else None,
                {"leaf": "SPREAD"},
            )
        )
        assert sunray.get(graph.get("leaf1").send(GetPlacementName())) == "leaf"
        assert sunray.get(graph.get("node").send(GetPlacementName())) is None

    def test_ray_graph_builder_add_node(self):
        class CustomNode(RayNode): ...

        total_nodes = {
            "node": CustomNode(),
            "leaf1": CustomNode(),
            "leaf2": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.add_node("node2", CustomNode())
        assert "node2" in builder._total_nodes

    def test_missing_set_placement_strategy(self):
        class GetPlacementName(Event): ...

        class CustomNode(RayNode):
            @handle(GetPlacementName)
            def handle_get_placement_name(self, event: GetPlacementName) -> str | None:
                if pg_id := get_node_context().runtime_context.get_placement_group_id():
                    return ray.util.placement_group_table()[pg_id]["name"]
                return None

        total_nodes = {
            "node": CustomNode(),
            "leaf1": CustomNode(),
            "leaf2": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.set_children("node", ["leaf1", "leaf2"])
        graph = builder.build()
        with pytest.raises(RuntimeError, match=r"Placement \['leaf'\] missing placement strategy"):
            graph.start(
                placement_rule=(
                    lambda node_name, _: "leaf" if node_name.startswith("leaf") else None,
                    {"leaf1": "SPREAD"},
                )
            )

    def test_warning_placement_strategy(self):
        class GetPlacementName(Event): ...

        class CustomNode(RayNode):
            @handle(GetPlacementName)
            def handle_get_placement_name(self, event: GetPlacementName) -> str | None:
                if pg_id := get_node_context().runtime_context.get_placement_group_id():
                    return ray.util.placement_group_table()[pg_id]["name"]
                return None

            def actor_options(self) -> ActorRemoteOptions:
                return {
                    "num_cpus": 1,
                    "scheduling_strategy": NodeLabelSchedulingStrategy(
                        {}, soft={"region": In("us")}
                    ),
                }

        total_nodes = {
            "node": CustomNode(),
            "leaf1": CustomNode(),
            "leaf2": CustomNode(),
        }
        builder = RayGraphBuilder(total_nodes)
        builder.set_children("node", ["leaf1", "leaf2"])
        graph = builder.build()
        with pytest.warns(PlacementWarning):
            graph.start(
                placement_rule=(
                    lambda node_name, _: "leaf1" if node_name.startswith("leaf") else None,
                    {"leaf1": "SPREAD"},
                )
            )

    def test_async_node_actor(self, init_ray):
        class GetNodeName(Event): ...

        class CustomNode(RayAsyncNode):
            @handle(GetNodeName)
            async def handle_get_node_name(self, event: GetNodeName) -> str:
                return get_node_context().node_name

        total_nodes = {"node1": CustomNode()}
        builder = RayGraphBuilder(total_nodes)
        graph = builder.build()
        graph.start()
        assert sunray.get(graph.get("node1").send(GetNodeName())) == "node1"

    def test_async_node_actor_remote_init(self, init_ray):
        class GetQueue(Event[int]): ...

        class CustomNode(RayAsyncNode):
            async def remote_init(self) -> None:
                self.queue: asyncio.Queue[int] = asyncio.Queue()
                await self.queue.put(1)

            @handle(GetQueue)
            async def handle_get_node_name(self, event: GetQueue) -> int:
                return await self.queue.get()

        total_nodes = {"node1": CustomNode()}
        builder = RayGraphBuilder(total_nodes)
        graph = builder.build()
        graph.start()
        assert sunray.get(graph.get("node1").send(GetQueue())) == 1

    def test_take_snapshot(self, init_ray):
        class GetSnapshot(Event[str]): ...

        class CustomNode(RayNode):
            def __init__(self, id) -> None:
                self.id = id

            def take_snapshot(self, epoch: int) -> None:
                self.snapshot = f"epoch-{self.id}-{epoch}"

            @handle(GetSnapshot)
            def get_snapshot(self, event: GetSnapshot) -> str:
                return self.snapshot

        graph = RayGraphBuilder(
            {
                "node1": CustomNode(1),
                "node2": CustomNode(2),
                EPOCH_MANAGER_NAME: EpochManagerNode(),
            }
        ).build()
        graph.start()

        generator = epochs(graph)
        # skip epoch 0
        next(generator)
        # start epoch 1
        sunray.get(graph.get(EPOCH_MANAGER_NAME).send(NextEpochEvent()))
        next(generator)
        # will take epoch 0 snapshot
        assert sunray.get(graph.get("node1").send(GetSnapshot())) == "epoch-1-0"
        assert sunray.get(graph.get("node2").send(GetSnapshot())) == "epoch-2-0"

    def test_recovery(self, init_ray):
        class GetEpoch(Event[str]): ...

        class CustomNode(RayNode):
            def __init__(self, id) -> None:
                self.id = id

            def recovery_from_snapshot(self, epoch: int) -> Any:
                self.epoch = epoch

            @handle(GetEpoch)
            def handle_get_epoch(self, event: GetEpoch) -> str:
                return f"epoch-{self.id}-{self.epoch}"

        graph = RayGraphBuilder(
            {
                "node1": CustomNode(1),
                "node2": CustomNode(2),
                EPOCH_MANAGER_NAME: EpochManagerNode(3),
            }
        ).build()
        graph.start()

        assert sunray.get(graph.get("node1").send(GetEpoch())) == "epoch-1-3"
        assert sunray.get(graph.get("node2").send(GetEpoch())) == "epoch-2-3"
        epoch_generator = epochs(graph)
        assert next(epoch_generator) == 4


def test_convert_ray_resources_to_placement_bundle():
    ray_resources: RayResources = {"num_cpus": 1, "memory": 1000, "resources": {"disk": 1}}
    assert _convert_ray_resources_to_placement_bundle(ray_resources) == {
        "CPU": 1,
        "memory": 1000,
        "disk": 1,
    }
