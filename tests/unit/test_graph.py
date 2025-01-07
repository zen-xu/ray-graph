# ruff: noqa: ARG001
from __future__ import annotations

from typing import Any

import pytest
import sunray

from ray_graph.event import Event
from ray_graph.graph import RayNode, RayNodeActor, RayNodeRef, handle


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
