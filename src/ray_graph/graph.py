from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic

import rustworkx as rwx
import sunray

from typing_extensions import TypedDict, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import TypeAlias

    from sunray._internal.core import RuntimeContext

    from ray_graph.event import Event, _Rsp_co

_node_context = None

NodeName: TypeAlias = str


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
                    mod = attrs["__module__"]
                    qual = attrs["__qualname__"]
                    event_t = attr_value.event_t
                    raise ValueError(
                        f"<class '{mod}.{qual}'> got duplicate event handler for {event_t}"
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


class RayResources(TypedDict, total=False):
    """The ray actor resources."""

    num_cpus: float
    num_gpus: float
    memory: float
    custom_resources: Mapping[str, float]


class RayNode(metaclass=_RayNodeMeta):  # pragma: no cover
    """The base class for all RayGraph nodes."""

    _event_handlers: Mapping[type[Event], _EventHandler[Event]]

    def remote_init(self) -> None:
        """Initialize the node in ray cluster."""

    def labels(self) -> Mapping[str, str]:
        """The labels of the node, which will inject into its node reference."""
        return {}

    def resources(self) -> RayResources:
        """Declare the ray actor resources."""
        return {"num_cpus": 1}


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

    def __init__(self, name: str, labels: Mapping[str, str] | None = None):
        self._name = name
        self._labels = labels or {}

    @cached_property
    def _actor(self) -> sunray.Actor[RayNodeActor]:
        return sunray.get_actor[RayNodeActor](self.name)

    @property
    def name(self) -> NodeName:
        """The name of the node."""
        return self._name

    @property
    def labels(self) -> Mapping[str, str]:
        """The labels of the node."""
        return self._labels

    def send(self, event: Event[_Rsp_co], **extra_ray_opts) -> sunray.ObjectRef[_Rsp_co]:
        """Send an event to the node."""
        return self._actor.methods.handle.options(
            name=f"{self.name}.handle[{type(event).__name__}]", **extra_ray_opts
        ).remote(event)


class RayGraphBuilder:
    """The graph builder of ray nodes."""

    def __init__(self, total_nodes: Mapping[NodeName, RayNode]):
        self._dag = rwx.PyDAG(check_cycle=True)
        self._node_name_ids = {
            node_name: self._dag.add_node((node_name, node))
            for node_name, node in total_nodes.items()
        }

    def set_parent(self, child: NodeName, parent: NodeName) -> None:
        """Set the parent of the child node."""
        child_id = self._node_name_ids[child]
        parent_id = self._node_name_ids[parent]
        if not self._dag.has_edge(parent_id, child_id):
            self._dag.add_edge(parent_id, child_id, None)

    def set_children(self, parent: NodeName, children: list[NodeName]) -> None:
        """Set the children of the parent node."""
        for child in children:
            self.set_parent(child, parent)


class RayGraphRef:
    """The reference to a RayGraph."""

    def __init__(self, dag: rwx.PyDAG[RayNodeRef, None]):
        self._dag = dag
        self._node_ids: Mapping[NodeName, int] = {
            node_ref.name: node_id for node_id, node_ref in enumerate(self._dag.nodes())
        }

    def get(self, node: NodeName) -> RayNodeRef:
        """Get the node reference of the given name."""
        return self._dag.get_node_data(self._node_ids[node])

    def filter(self, predicate: Callable[[RayNodeRef], bool]) -> list[RayNodeRef]:
        """Filter the nodes by the given predicate."""
        return [node for node in self._dag.nodes() if predicate(node)]

    def get_parents(self, child: NodeName) -> list[RayNodeRef]:
        """Get the parent node references of the child node."""
        child_id = self._node_ids[child]
        return self._dag.predecessors(child_id)

    def get_children(self, parent: NodeName) -> list[RayNodeRef]:
        """Get the children node references of the parent node."""
        parent_id = self._node_ids[parent]
        return self._dag.successors(parent_id)

    def get_roots(self, node: NodeName) -> list[RayNodeRef]:
        """Get the root node references of the graph."""
        return [
            self._dag.get_node_data(ancestor_id)
            for ancestor_id in rwx.ancestors(self._dag, self._node_ids[node])
            if not rwx.ancestors(self._dag, ancestor_id)
        ]

    def get_leaves(self, node: NodeName) -> list[RayNodeRef]:
        """Get the leaf node references of current node."""
        return [
            self._dag.get_node_data(ancestor_id)
            for ancestor_id in rwx.descendants(self._dag, self._node_ids[node])
            if not rwx.descendants(self._dag, ancestor_id)
        ]

    def get_siblings(self, node: NodeName) -> list[RayNodeRef]:
        """Get the sibling node references of the current node."""
        parent_node_names = [node.name for node in self.get_parents(node)]
        return [
            sibling
            for parent_node_name in parent_node_names
            for sibling in self.get_children(parent_node_name)
            if sibling.name != node
        ]
