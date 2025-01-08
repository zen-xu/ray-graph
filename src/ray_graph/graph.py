from __future__ import annotations

import warnings

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal

import rustworkx as rwx
import sunray

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import TypedDict, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import TypeAlias

    from sunray._internal.core import RuntimeContext
    from sunray._internal.typing import RuntimeEnv, SchedulingStrategy

    from ray_graph.event import Event, _Rsp_co


NodeName: TypeAlias = str
PlacementName: TypeAlias = str
PlacementStrategy = Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"]

_node_context = None
_graph: RayGraphRef | None = None


@dataclass(frozen=True)
class RayNodeContext:
    """The context of a RayGraph node."""

    runtime_context: RuntimeContext
    "ray runtime context"

    graph: RayGraphRef
    "ray node graph"

    node_name: NodeName
    "ray node name"


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
    from ray_graph import graph

    if graph._node_context is None:
        assert graph._graph is not None
        runtime_context = sunray.get_runtime_context()
        graph._node_context = RayNodeContext(
            runtime_context=runtime_context,
            graph=graph._graph,
            node_name=runtime_context.get_actor_name() or "",
        )
    return graph._node_context


class RayResources(TypedDict, total=False):
    """The ray actor resources."""

    num_cpus: float
    num_gpus: float
    memory: float
    resources: dict[str, float]


class ActorRemoteOptions(TypedDict, total=False):
    """The ray actor remote options."""

    num_cpus: float
    num_gpus: float
    resources: dict[str, float]
    accelerator_type: str
    memory: float
    object_store_memory: float
    max_restarts: int
    max_task_retries: int
    max_pending_calls: int
    max_concurrency: int
    lifetime: Literal["detached"] | None
    runtime_env: RuntimeEnv | dict[str, Any]
    concurrency_groups: dict[str, int]
    scheduling_strategy: SchedulingStrategy


class RayNode(metaclass=_RayNodeMeta):  # pragma: no cover
    """The base class for all RayGraph nodes."""

    _event_handlers: Mapping[type[Event], _EventHandler[Event]]

    def remote_init(self) -> None:
        """Initialize the node in ray cluster."""

    def labels(self) -> Mapping[str, str]:
        """The labels of the node, which will inject into its node reference."""
        return {}

    def actor_options(self) -> ActorRemoteOptions:
        """Declare the ray actor remote options."""
        return {"num_cpus": 1}


class RayNodeActor(sunray.ActorMixin):
    """The actor class for RayGraph nodes."""

    def __init__(self, name: str, ray_node: RayNode, ray_graph: RayGraphRef) -> None:
        self.name = name
        self.ray_node = ray_node
        self.ray_graph = ray_graph
        from ray_graph import graph

        graph._graph = ray_graph

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.ray_node.__class__.__name__}({self.name})"

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

    def send(
        self, event: Event[_Rsp_co] | sunray.ObjectRef[Event[_Rsp_co]], **extra_ray_opts
    ) -> sunray.ObjectRef[_Rsp_co]:
        """Send an event to the node."""
        return self._actor.methods.handle.options(
            name=f"{self.name}.handle[{type(event).__name__}]", **extra_ray_opts
        ).remote(event)


class RayGraphBuilder:
    """The graph builder of ray nodes."""

    def __init__(self, total_nodes: Mapping[NodeName, RayNode]):
        self._dag: rwx.PyDAG[RayNodeRef, None] = rwx.PyDAG(check_cycle=True)
        self._total_nodes = total_nodes
        self._node_name_ids = {
            node_name: self._dag.add_node(RayNodeRef(node_name, node.labels()))
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

    def build(self) -> RayGraph:  # pragma: no cover
        """Build the ray graph."""
        return RayGraph(self._dag, self._total_nodes)


class PlacementWarning(Warning):
    """The placement warning."""


class RayGraph:  # pragma: no cover
    """The graph of ray nodes."""

    def __init__(
        self, dag: rwx.PyDAG[RayNodeRef, None], total_nodes: Mapping[NodeName, RayNode]
    ) -> None:
        self._graph_ref = RayGraphRef(dag)
        self._total_nodes = total_nodes
        self._node_actors: Mapping[NodeName, sunray.Actor[RayNodeActor]] | None = None

    @property
    def ref(self) -> RayGraphRef:
        """The reference of RayGraph."""
        return self._graph_ref

    def start(
        self,
        *,
        placement_rule: tuple[
            Callable[[NodeName, RayNode], PlacementName | None],
            Mapping[PlacementName, PlacementStrategy],
        ]
        | None = None,
        **_future_opts,
    ) -> None:
        """Create all ray node actors and start these actors."""
        if not self._node_actors:
            graph_obj_ref = sunray.put(self._graph_ref)
            if placement_rule:
                place_func, strategies = placement_rule
                placement_nodes: defaultdict[PlacementName, list[NodeName]] = defaultdict(list)
                node_placement_names: dict[NodeName, PlacementName] = {}

                for node_name, node in self._total_nodes.items():
                    if placement_name := place_func(node_name, node):
                        placement_nodes[placement_name].append(node_name)
                        node_placement_names[node_name] = placement_name
                placement_missing_strategies = sorted(set(placement_nodes) - set(strategies))
                if placement_missing_strategies:
                    raise RuntimeError(
                        f"Placement {placement_missing_strategies} missing placement strategy"
                    )

                # create the placement groups
                placement_groups = {
                    placement_name: placement_group(
                        [
                            _convert_ray_resources_to_placement_bundle(
                                self._total_nodes[node].actor_options()
                            )
                            for node in nodes
                        ],
                        name=placement_name,
                        strategy=strategies[placement_name],
                    )
                    for placement_name, nodes in placement_nodes.items()
                }
                # waiting placement group to be ready
                sunray.get([pg.ready() for pg in placement_groups.values()])

                def create_node_actors():
                    for node_name, node in self._total_nodes.items():
                        options = {"name": node_name, **node.actor_options()}
                        if placement_name := node_placement_names.get(node_name):
                            pg = placement_groups[placement_name]
                            if "scheduling_strategy" in options:
                                warnings.warn(
                                    f"Node {node_name} specified scheduling_strategy and will replace it with PlacementGroupSchedulingStrategy({placement_name})",  # noqa: E501
                                    category=PlacementWarning,
                                )
                            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                                pg,
                                placement_group_bundle_index=placement_nodes[placement_name].index(
                                    node_name
                                ),
                            )
                        yield (
                            node_name,
                            RayNodeActor.new_actor()
                            .options(**options)
                            .remote(node_name, node, graph_obj_ref),
                        )

                self._node_actors = dict(create_node_actors())
            else:
                self._node_actors = {
                    name: RayNodeActor.new_actor()
                    .options(name=name, **node.actor_options())
                    .remote(name, node, graph_obj_ref)
                    for name, node in self._total_nodes.items()
                }
            # remote init ray node actors
            sunray.get(
                [actor.methods.remote_init.remote() for actor in self._node_actors.values()]
            )

    def get(self, node: NodeName) -> RayNodeRef:
        """Get the node reference of the given name."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get(node)

    def filter(self, predicate: Callable[[RayNodeRef], bool]) -> list[RayNodeRef]:
        """Filter the nodes by the given predicate."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.filter(predicate)

    def get_parents(self, child: NodeName) -> list[RayNodeRef]:
        """Get the parent node references of the child node."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get_parents(child)

    def get_children(self, parent: NodeName) -> list[RayNodeRef]:
        """Get the children node references of the parent node."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get_children(parent)

    def get_roots(self, node: NodeName) -> list[RayNodeRef]:
        """Get the root node references of the graph."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get_roots(node)

    def get_leaves(self, node: NodeName) -> list[RayNodeRef]:
        """Get the leaf node references of current node."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get_leaves(node)

    def get_siblings(self, node: NodeName) -> list[RayNodeRef]:
        """Get the sibling node references of the current node."""
        assert self._node_actors, "start RayGraph first"
        return self._graph_ref.get_siblings(node)


class RayGraphRef:
    """The graph reference of ray nodes."""

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


def _convert_ray_resources_to_placement_bundle(resources: RayResources) -> dict[str, float]:
    bundle = {
        "CPU": resources.get("num_cpus"),
        "GPU": resources.get("num_gpus"),
        "memory": resources.get("memory"),
        **resources.get("resources", {}),
    }
    return {k: v for k, v in bundle.items() if v}
