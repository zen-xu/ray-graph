from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import sunray


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ray_graph.event import Event, _RspT

_node_context = None


@dataclass(frozen=True)
class RayNodeContext:
    """The context of a RayGraph node."""

    name: str
    """The name of the node."""

    namespace: str
    """The namespace of the node."""

    job_id: str
    """The job id of the node."""

    task_id: str
    """The task id of the node."""

    actor_id: str
    """The actor id of the node."""

    worker_id: str
    """The worker id of the node."""

    node_id: str
    """The id of the node."""

    placement_group_id: str | None
    """The placement group id of the node."""

    accelerator_ids: Mapping[str, list[str]]
    """The current node's visible accelerator ids."""

    assigned_resources: Mapping[str, float]
    """The current node's assigned resources."""


_EventT = TypeVar("_EventT", bound="Event")
_RayNode_co = TypeVar("_RayNode_co", bound="RayNode", covariant=True)


class _EventHandler(Generic[_EventT]):
    def __init__(self, event_t: type[_EventT]) -> None:
        self.event_t = event_t

    def __call__(self, handler_func: Callable[[_RayNode_co, _EventT], Any]) -> Any:
        self.handler_func = handler_func
        return self


def handle(event_t: type[_EventT]) -> _EventHandler[_EventT]:
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
        ctx = sunray.get_runtime_context()
        _node_context = RayNodeContext(
            name=ctx.get_actor_name() or "",
            namespace=ctx.namespace,
            job_id=ctx.get_job_id(),
            task_id=ctx.get_task_id() or "",
            actor_id=ctx.get_actor_id() or "",
            worker_id=ctx.get_worker_id(),
            node_id=ctx.get_node_id(),
            placement_group_id=ctx.get_placement_group_id(),
            accelerator_ids=ctx.get_accelerator_ids(),
            assigned_resources=ctx.get_assigned_resources(),
        )
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
    def handle(self, event: Event[_RspT]) -> _RspT:
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

    def send(self, event: Event[_RspT], **extra_ray_opts) -> sunray.ObjectRef[_RspT]:
        """Send an event to the node."""
        return self._actor.methods.handle.options(
            name=f"{self.name}.handle[{type(event).__name__}]", **extra_ray_opts
        ).remote(event)
