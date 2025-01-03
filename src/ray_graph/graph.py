from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

import sunray

from typing_extensions import TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sunray._internal.core import RuntimeContext

    from ray_graph.event import Event, _Rsp_co

_node_context = None


@dataclass(frozen=True)
class RayNodeContext:
    """The context of a RayGraph node."""

    runtime_context: RuntimeContext


_Event_co = TypeVar("_Event_co", bound="Event", covariant=True)
_RayNode_co = TypeVar("_RayNode_co", bound="RayNode", covariant=True, default="RayNode")


class _EventHandler(Generic[_Event_co]):
    def __init__(self, event_t: type[_Event_co]) -> None:
        self.event_t = event_t

    def __call__(
        self: _EventHandler[Event[_Rsp_co]],
        handler_func: Callable[[_RayNode_co, _Event_co], _Rsp_co],
    ) -> Any:
        self.handler_func = handler_func
        return self


def handle(event_t: type[_Event_co]) -> _EventHandler[_Event_co]:
    """Decorator to register an event handler."""
    return _EventHandler(event_t)


class _RayNodeMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs.setdefault("_event_handlers", {})
        event_handlers = attrs["_event_handlers"]
        for _, attr_value in attrs.items():
            if isinstance(attr_value, _EventHandler):
                if attr_value.event_t in event_handlers:
                    raise ValueError(
                        f"<class '{attrs["__module__"]}.{attrs["__qualname__"]}'> "
                        f"got duplicate event handler for {attr_value.event_t}"
                    )
                event_handlers[attr_value.event_t] = attr_value
        # remove event handler functions from attrs
        attrs = {k: v for k, v in attrs.items() if not isinstance(v, _EventHandler)}
        return super().__new__(cls, name, bases, attrs)


def get_node_context() -> RayNodeContext:  # pragma: no cover
    """Get the context of the current RayGraph node."""
    global _node_context
    if _node_context is None:
        _node_context = RayNodeContext(runtime_context=sunray.get_runtime_context())
    return _node_context


class RayNode(metaclass=_RayNodeMeta):
    """The base class for all RayGraph nodes."""

    _event_handlers: Mapping[type[Event], _EventHandler[Event]]

    def remote_init(self) -> None:  # pragma: no cover
        """Initialize the node in ray cluster."""


class RayNodeActor(sunray.ActorMixin):
    """The actor class for RayGraph nodes."""

    def __init__(self, ray_node: RayNode) -> None:
        self.ray_node = ray_node

    @sunray.remote_method
    def remote_init(self) -> None:
        """Invoke ray_node remote init method."""
        self.ray_node.remote_init()

    @sunray.remote_method
    def handle(self, event: Event[_Rsp_co]) -> _Rsp_co:
        """Handle the given event."""
        event_type = type(event)
        event_handler = self.ray_node._event_handlers.get(event_type)
        if event_handler:
            return event_handler.handler_func(self.ray_node, event)  # type: ignore

        raise ValueError(f"no handler for event {event_type}")


class RayNodeRef:
    """The reference to a RayGraph node."""

    def __init__(self, name: str, actor: sunray.Actor[RayNodeActor]):
        self._name = name
        self._actor = actor

    @property
    def name(self) -> str:
        """The name of the node."""
        return self._name

    def send(self, event: Event[_Rsp_co], **extra_ray_opts) -> sunray.ObjectRef[_Rsp_co]:
        """Send an event to the node."""
        return self._actor.methods.handle.options(
            name=f"{self.name}.handle[{type(event).__name__}]", **extra_ray_opts
        ).remote(event)
