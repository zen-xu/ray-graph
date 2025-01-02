# ruff: noqa: ARG001
from __future__ import annotations

import pytest
import sunray

from ray_graph.event import Event
from ray_graph.graph import RayNode, RayNodeActor, RayNodeRef, handle


class CustomEvent(Event):
    value: int


def test_register_event_handler():
    class CustomNode(RayNode):
        @handle(CustomEvent)
        def handle_custom_event(self, event: CustomEvent) -> None: ...

    assert len(CustomNode._event_handlers) == 1
    assert CustomNode._event_handlers.get(CustomEvent) is not None
    assert not hasattr(CustomNode, "handle_custom_event")


def test_register_duplicate_event_handler():
    with pytest.raises(ValueError, match="got duplicate event handler for"):

        class CustomNode(RayNode):
            @handle(CustomEvent)
            def handle_custom_event(self, event: CustomEvent) -> None: ...

            @handle(CustomEvent)
            def handle_custom_event2(self, event: CustomEvent) -> None: ...


def test_ray_node_actor_handle_event(init_local_ray):
    class CustomNode(RayNode):
        def remote_init(self) -> None:
            self.value = 1

        @handle(CustomEvent)
        def handle_custom_event(self, event: CustomEvent) -> None:
            self.value += event.value

    class RayNodeActorProxy(RayNodeActor):
        ray_node: CustomNode

        @sunray.remote_method
        def get_value(self) -> int:
            return self.ray_node.value

    node_actor = RayNodeActorProxy.new_actor().remote(CustomNode())
    sunray.get(node_actor.methods.remote_init.remote())
    sunray.get(node_actor.methods.handle.remote(CustomEvent(value=2)))
    assert sunray.get(node_actor.methods.get_value.remote()) == 3

    class FakeEvent(Event): ...

    with pytest.raises(ValueError, match=r"no handler for event .*FakeEvent.*"):
        sunray.get(node_actor.methods.handle.remote(FakeEvent()))
    sunray.kill(node_actor)


def test_ray_node_ref(init_ray):
    class CustomNode(RayNode):
        def __init__(self) -> None:
            self.process_name = ""

        @handle(CustomEvent)
        def handle_custom_event(self, _event: CustomEvent) -> None:
            import setproctitle

            self.process_name = setproctitle.getproctitle()

    class RayNodeActorProxy(RayNodeActor):
        ray_node: CustomNode

        @sunray.remote_method
        def get_name(self) -> str:
            return self.ray_node.process_name

    node_actor = RayNodeActorProxy.new_actor().remote(CustomNode())
    name = "test_ray_node_ref"
    node_ref = RayNodeRef(name, node_actor)
    assert node_ref.name == name
    sunray.get(node_ref.send(CustomEvent(value=1)))
    assert sunray.get(node_actor.methods.get_name.remote()) == f"ray::{name}.handle[CustomEvent]"
    sunray.kill(node_actor)
