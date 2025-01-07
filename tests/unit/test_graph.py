# ruff: noqa: ARG001
from __future__ import annotations

from textwrap import dedent
from typing import Any

import pytest
import rustworkx as rwx
import sunray

from ray_graph.event import Event
from ray_graph.graph import RayGraphBuilder, RayGraphRef, RayNode, RayNodeActor, RayNodeRef, handle


class CustomEvent(Event[int]):
    value: int


def test_register_event_handler():
    class CustomNode(RayNode):
        @handle(CustomEvent)
        def handle_custom_event(self, event: CustomEvent) -> Any: ...

    assert len(CustomNode._event_handlers) == 1
    assert CustomNode._event_handlers.get(CustomEvent) is not None
    assert not hasattr(CustomNode, "handle_custom_event")


def test_register_duplicate_event_handler():
    with pytest.raises(ValueError, match="got duplicate event handler for"):

        class CustomNode(RayNode):
            @handle(CustomEvent)
            def handle_custom_event(self, event: CustomEvent) -> Any: ...

            @handle(CustomEvent)
            def handle_custom_event2(self, event: CustomEvent) -> Any: ...


def test_ray_node_actor_handle_event(init_local_ray):
    class CustomNode(RayNode):
        def remote_init(self) -> None:
            self.value = 1

        @handle(CustomEvent)
        def handle_custom_event(self, event: CustomEvent) -> int:
            self.value += event.value

            return self.value

    node_actor = RayNodeActor.new_actor().remote(CustomNode())
    sunray.get(node_actor.methods.remote_init.remote())
    assert sunray.get(node_actor.methods.handle.remote(CustomEvent(value=2))) == 3

    class FakeEvent(Event): ...

    with pytest.raises(ValueError, match=r"no handler for event .*FakeEvent.*"):
        sunray.get(node_actor.methods.handle.remote(FakeEvent()))
    sunray.kill(node_actor)


def test_ray_node_ref(init_ray):
    class GetProcName(Event[str]): ...

    class CustomNode(RayNode):
        @handle(GetProcName)
        def handle_custom_event(self, _event: GetProcName) -> str:
            import setproctitle

            return setproctitle.getproctitle()

    name = "test_ray_node_ref"
    node_actor = RayNodeActor.new_actor().options(name=name).remote(CustomNode())
    labels = {"node": "mac"}
    node_ref = RayNodeRef(name, labels)
    assert node_ref.name == name
    assert node_ref.labels == labels
    assert sunray.get(node_ref.send(GetProcName())) == f"ray::{name}.handle[GetProcName]"
    sunray.kill(node_actor)


class TestRayGraphBuilder:
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

        got = builder._dag.to_dot(node_attr=lambda n: {"label": n[0]})
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


class TestRayGraphRef:
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
