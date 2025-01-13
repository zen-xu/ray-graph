from __future__ import annotations

import importlib.util
import inspect
import warnings

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

import rustworkx as rwx
import sunray

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rustworkx.visualization import graphviz_draw
from typing_extensions import TypedDict, TypeVar


_rich_enabled = importlib.util.find_spec("rich") is not None


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from typing import TypeAlias

    from opentelemetry.trace import Span
    from PIL.Image import Image
    from ray.util.placement_group import PlacementGroup
    from sunray._internal.core import RuntimeContext
    from sunray._internal.typing import RuntimeEnv, SchedulingStrategy

    from .epoch import Epoch
    from .event import Event, _Rsp_co


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
        handler_func: Callable[[_RayNode_co, _Event_co], _Rsp_co | Awaitable[_Rsp_co]],
    ) -> Any:
        self.handler_func = handler_func
        return self


def handle(event_t: type[_Event_co]) -> _EventHandler[_Event_co]:
    """Decorator to register an event handler."""
    return _EventHandler(event_t)


class RegisterHandlerError(Exception):
    """Raise Error when register handler failed."""


class _RayNodeMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs.setdefault("_event_handlers", {})
        event_handlers = attrs["_event_handlers"]
        for _, attr_value in attrs.items():
            if isinstance(attr_value, _EventHandler):
                mod = attrs["__module__"]
                qual = attrs["__qualname__"]
                if attr_value.event_t in event_handlers:
                    event_t = attr_value.event_t
                    raise RegisterHandlerError(
                        f"<class '{mod}.{qual}'> got duplicate event handler for {event_t}"
                    )
                if RayAsyncNode in bases:
                    # event_handler must be async func for RayAsyncNode
                    if not inspect.iscoroutinefunction(attr_value.handler_func):
                        raise RegisterHandlerError(
                            f"<class '{mod}.{qual}'> method '{attr_value.handler_func.__name__}' must be async func"  # noqa: E501
                        )
                else:
                    # event_handler must not be async func for RayNode
                    if inspect.iscoroutinefunction(attr_value.handler_func):
                        raise RegisterHandlerError(
                            f"<class '{mod}.{qual}'> method {attr_value.handler_func.__name__} can't be async func"  # noqa: E501
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

    def remote_init(self) -> Any:
        """Initialize the node in ray cluster."""

    def labels(self) -> Mapping[str, str]:
        """The labels of the node, which will inject into its node reference."""
        return {}

    def actor_options(self) -> ActorRemoteOptions:
        """Declare the ray actor remote options."""
        return {"num_cpus": 1}

    def take_snapshot(self, epoch: Epoch) -> None:
        """Take current epoch snapshot."""

    def recovery_from_snapshot(self, epoch: Epoch) -> Any:
        """Recovery node from the given epoch snapshot."""


class RayAsyncNode(RayNode):
    """The base async RayNode."""

    async def remote_init(self) -> Any:
        """Async initialize the node in ray cluster."""

    async def recovery_from_snapshot(self, epoch: Epoch) -> Any:
        """Recovery node from the given epoch snapshot."""


def _set_node_span_attributes(node: RayNodeActor, span: Span):  # pragma: no cover
    span.set_attribute("ray_graph.node.name", node.name)
    span.set_attribute("ray_graph.node.class", node.ray_node.__class__.__name__)
    for k, v in node.labels.items():
        span.set_attribute(f"ray_graph.node.label.{k}", v)


class RayNodeActor(sunray.ActorMixin):
    """The actor class for RayGraph nodes."""

    def __init__(self, name: str, ray_node: RayNode, ray_graph: RayGraphRef) -> None:
        self.name = name
        self.ray_node = ray_node
        self.labels = ray_node.labels()
        self.ray_graph = ray_graph
        from ray_graph import graph

        graph._graph = ray_graph

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.ray_node.__class__.__name__}({self.name})"

    @sunray.remote_method
    def remote_init(self) -> str:
        """Invoke ray_node remote init method."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        self.ray_node.remote_init()
        return self.name

    @sunray.remote_method
    def handle(self, event: Event[_Rsp_co]) -> _Rsp_co:
        """Handle the given event."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        event_type = type(event)
        span.set_attribute("ray_graph.event", event_type.__name__)
        event_handler = self.ray_node._event_handlers.get(event_type)
        if event_handler:
            return event_handler.handler_func(self.ray_node, event)  # type: ignore

        raise ValueError(f"no handler for event {event_type}")

    @sunray.remote_method
    def take_snapshot(self, epoch: Epoch) -> None:  # pragma: no cover
        """Take current epoch snapshot."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        self.ray_node.take_snapshot(epoch)

    @sunray.remote_method
    def recovery_from_snapshot(self, epoch: Epoch) -> str:  # pragma: no cover
        """Recovery node from the given epoch snapshot."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        self.ray_node.recovery_from_snapshot(epoch)
        return self.name


class RayAsyncNodeActor(RayNodeActor):  # pragma: no cover
    """The async version actor class for RayGraph nodes."""

    ray_node: RayAsyncNode

    @sunray.remote_method
    async def remote_init(self) -> str:
        """Invoke ray_node remote init method."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        await self.ray_node.remote_init()
        return self.name

    @sunray.remote_method
    async def handle(self, event: Event[_Rsp_co]) -> _Rsp_co:
        """Handle the given event."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        event_type = type(event)
        span.set_attribute("ray_graph.event", event_type.__name__)
        event_handler = self.ray_node._event_handlers.get(event_type)
        if event_handler:
            return await event_handler.handler_func(self.ray_node, event)  # type: ignore

        raise ValueError(f"no handler for event {event_type}")

    @sunray.remote_method
    async def recovery_from_snapshot(self, epoch: Epoch) -> str:
        """Recovery node from the given epoch snapshot."""
        span = get_current_span()
        _set_node_span_attributes(self, span)
        await self.ray_node.recovery_from_snapshot(epoch)
        return self.name


class RayNodeRef:
    """The reference to a RayGraph node."""

    def __init__(self, name: str, labels: Mapping[str, str] | None = None):
        self._name = name
        self._labels = labels or {}

    @cached_property
    def actor(self) -> sunray.Actor[RayNodeActor]:
        """The ray node actor."""
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
        return self.actor.methods.handle.options(
            name=f"{self.name}.handle[{type(event).__name__}]", **extra_ray_opts
        ).remote(event)


class RayGraphBuilder:
    """The graph builder of ray nodes."""

    def __init__(self, total_nodes: Mapping[NodeName, RayNode]):
        self._dag: rwx.PyDAG[RayNodeRef, None] = rwx.PyDAG(check_cycle=True)
        self._total_nodes = dict(total_nodes)
        self._node_name_ids = {
            node_name: self._dag.add_node(RayNodeRef(node_name, node.labels()))
            for node_name, node in total_nodes.items()
        }

    def add_node(self, node_name: NodeName, node: RayNode):
        """Add new ray node."""
        self._total_nodes[node_name] = node

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
        disable_rich: bool = False,
        **_future_opts,
    ) -> None:
        """Create all ray node actors and start these actors."""

        def determine_actor_class(node) -> type[RayNodeActor] | type[RayAsyncNodeActor]:
            from .epoch import EpochManagerNode, EpochManagerNodeActor

            if isinstance(node, EpochManagerNode):
                return EpochManagerNodeActor
            if isinstance(node, RayAsyncNode):
                return RayAsyncNodeActor
            return RayNodeActor

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
                _wait_placement_group_ready(placement_groups, disable_rich=disable_rich)

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
                            determine_actor_class(node)
                            .new_actor()
                            .options(**options)
                            .remote(node_name, node, graph_obj_ref),
                        )

                self._node_actors = dict(create_node_actors())
            else:
                self._node_actors = {
                    name: determine_actor_class(node)
                    .new_actor()
                    .options(name=name, **node.actor_options())
                    .remote(name, node, graph_obj_ref)
                    for name, node in self._total_nodes.items()
                }
            # remote init ray node actors
            _wait_node_init(
                {
                    name: {"actor": self._node_actors[name], "node": self._total_nodes[name]}
                    for name in self._total_nodes
                },
                disable_rich=disable_rich,
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

    @overload
    def graphviz_draw(
        self,
        node_attr_fn: Callable[[RayNodeRef], dict[str, str]] | None = None,
        edge_attr: dict[str, str] | None = None,
        graph_attr: dict[str, str] | None = None,
        *,
        image_type: Literal[
            "canon",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dia",
            "dot",
            "fig",
            "gd",
            "gd2",
            "gif",
            "hpgl",
            "imap",
            "imap_np",
            "ismap",
            "jpe",
            "jpeg",
            "jpg",
            "mif",
            "mp",
            "pcl",
            "pdf",
            "pic",
            "plain",
            "plain-ext",
            "png",
            "ps",
            "ps2",
            "svg",
            "svgz",
            "vml",
            "vmlz",
            "vrml",
            "vtx",
            "wbmp",
            "xdot",
            "xlib",
        ]
        | None = None,
        method: Literal["dot", "twopi", "neato", "circo", "fdp", "sfdp"] | None = None,
        filename: None = None,
    ) -> Image: ...

    @overload
    def graphviz_draw(
        self,
        node_attr_fn: Callable[[RayNodeRef], dict[str, str]] | None = None,
        edge_attr: dict[str, str] | None = None,
        graph_attr: dict[str, str] | None = None,
        *,
        image_type: Literal[
            "canon",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dia",
            "dot",
            "fig",
            "gd",
            "gd2",
            "gif",
            "hpgl",
            "imap",
            "imap_np",
            "ismap",
            "jpe",
            "jpeg",
            "jpg",
            "mif",
            "mp",
            "pcl",
            "pdf",
            "pic",
            "plain",
            "plain-ext",
            "png",
            "ps",
            "ps2",
            "svg",
            "svgz",
            "vml",
            "vmlz",
            "vrml",
            "vtx",
            "wbmp",
            "xdot",
            "xlib",
        ]
        | None = None,
        method: Literal["dot", "twopi", "neato", "circo", "fdp", "sfdp"] | None = None,
        filename: str,
    ) -> None: ...

    def graphviz_draw(
        self,
        node_attr_fn: Callable[[RayNodeRef], dict[str, str]] | None = None,
        edge_attr: dict[str, str] | None = None,
        graph_attr: dict[str, str] | None = None,
        *,
        filename: str | None = None,
        image_type: Literal[
            "canon",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dia",
            "dot",
            "fig",
            "gd",
            "gd2",
            "gif",
            "hpgl",
            "imap",
            "imap_np",
            "ismap",
            "jpe",
            "jpeg",
            "jpg",
            "mif",
            "mp",
            "pcl",
            "pdf",
            "pic",
            "plain",
            "plain-ext",
            "png",
            "ps",
            "ps2",
            "svg",
            "svgz",
            "vml",
            "vmlz",
            "vrml",
            "vtx",
            "wbmp",
            "xdot",
            "xlib",
        ]
        | None = None,
        method: Literal["dot", "twopi", "neato", "circo", "fdp", "sfdp"] | None = None,
    ) -> Image | None:
        """Draw RayGraph using graphviz.

        :param node_attr_fn: An optional callable object that will be passed the
            weight/data payload for every node in the graph and expected to return
            a dictionary of Graphviz node attributes to be associated with the node
            in the visualization. The key and value of this dictionary **must** be
            a string.
        :param edge_attr_fn: An optional dictionary of Graphviz edge attributes to
            be associated with the edge in the visualization file. The key and value
            of this dictionary must be a string.
        :param dict graph_attr: An optional dictionary that specifies any Graphviz
            graph attributes for the visualization. The key and value of this
            dictionary must be a string.
        :param str filename: An optional path to write the visualization to. If
            specified the return type from this function will be ``None`` as the
            output image is saved to disk.
        :param str image_type: The image file format to use for the generated
            visualization. The support image formats are:
            ``'canon'``, ``'cmap'``, ``'cmapx'``, ``'cmapx_np'``, ``'dia'``,
            ``'dot'``, ``'fig'``, ``'gd'``, ``'gd2'``, ``'gif'``, ``'hpgl'``,
            ``'imap'``, ``'imap_np'``, ``'ismap'``, ``'jpe'``, ``'jpeg'``,
            ``'jpg'``, ``'mif'``, ``'mp'``, ``'pcl'``, ``'pdf'``, ``'pic'``,
            ``'plain'``, ``'plain-ext'``, ``'png'``, ``'ps'``, ``'ps2'``,
            ``'svg'``, ``'svgz'``, ``'vml'``, ``'vmlz'``, ``'vrml'``, ``'vtx'``,
            ``'wbmp'``, ``'xdot'``, ``'xlib'``. It's worth noting that while these
            formats can all be used for generating image files when the ``filename``
            kwarg is specified, the Pillow library used for the returned object can
            not work with all these formats.
        :param str method: The layout method/Graphviz command method to use for
            generating the visualization. Available options are ``'dot'``,
            ``'twopi'``, ``'neato'``, ``'circo'``, ``'fdp'``, and ``'sfdp'``.
            You can refer to the
            `Graphviz documentation <https://graphviz.org/documentation/>`__ for
            more details on the different layout methods. By default ``'dot'`` is
            used.

        :returns: A ``PIL.Image`` object of the generated visualization, if
            ``filename`` is not specified. If ``filename`` is specified then
            ``None`` will be returned as the visualization was written to the
            path specified in ``filename``
        :rtype: PIL.Image
        """
        node_attr_fn = node_attr_fn or (
            lambda n: {
                "label": n.name,
                "shape": "box",
                "style": '"rounded,filled"',
                "fillcolor": '"#E3F2FD"',
                "fontname": "Arial",
                "fontsize": "12",
            }
        )
        return graphviz_draw(
            self._graph_ref._dag,
            node_attr_fn=node_attr_fn,
            edge_attr_fn=lambda _: edge_attr or {},
            graph_attr=graph_attr,
            filename=filename,
            image_type=image_type,
            method=method,
        )


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


def _wait_placement_group_ready(
    placement_groups: Mapping[PlacementName, PlacementGroup], *, disable_rich: bool = False
):  # pragma: no cover
    pg_readies = {pg_name: pg.ready() for pg_name, pg in placement_groups.items()}
    if not _rich_enabled or disable_rich:
        sunray.wait(list(pg_readies.values()), num_returns=len(pg_readies))

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.table import Table

    pg_status = {name: "Pending" for name in pg_readies}

    def render_table(pg_status):
        table = Table(title="Placement Group Status")
        table.add_column("Placement", justify="center")
        table.add_column("Status", justify="center")

        for name, status in pg_status.items():
            table.add_row(name, Spinner("dots", text="Pending") if status == "Pending" else "✅")
        return table

    pg_names = {pg: name for name, pg in placement_groups.items()}
    with Live(render_table(pg_status)) as live:
        not_readies = list(pg_readies.values())
        while not_readies:
            readies, not_readies = sunray.wait(not_readies)
            for ready_pg in sunray.get(readies):
                ready_pg_name = pg_names[ready_pg]
                pg_status[ready_pg_name] = "Ready"
                live.update(render_table(pg_status))


class _WaitNode(TypedDict):
    actor: sunray.Actor[RayNodeActor]
    node: RayNode


def _wait_node_init(
    node_actors: Mapping[NodeName, _WaitNode], *, disable_rich: bool = False
):  # pragma: no cover
    action = None
    from .epoch import EPOCH_MANAGER_NAME, EpochManagerNode

    if epoch_manager := node_actors.get(EPOCH_MANAGER_NAME):
        epoch_manager_node = cast(EpochManagerNode, epoch_manager["node"])
        if epoch_manager_node.current_epoch is not None:
            # recover all nodes at the given epoch snapshot
            action = f"Recovery epoch({epoch_manager_node.current_epoch}) {{task.description}}"
            not_readies = [
                item["actor"].methods.recovery_from_snapshot.remote(
                    epoch_manager_node.current_epoch
                )
                for _, item in node_actors.items()
            ]
    if action is None:
        action = "Init {{task.description}}"
        not_readies = [
            item["actor"].methods.remote_init.remote() for _, item in node_actors.items()
        ]

    if not _rich_enabled or disable_rich:
        sunray.wait(not_readies, num_returns=len(not_readies))

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    node_to_classes = {
        node_name: item["node"].__class__.__name__ for node_name, item in node_actors.items()
    }
    class_node_counter = Counter(node_to_classes.values())

    with Progress(
        TextColumn(action),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        tasks = {
            class_name: progress.add_task(f"[bold yellow]{class_name}[/]", total=total)
            for class_name, total in class_node_counter.items()
        }

        while not_readies:
            readies, not_readies = sunray.wait(not_readies)
            for node_name in sunray.get(readies):
                progress.update(tasks[node_to_classes[node_name]], advance=1)


def get_current_span() -> Span:  # pragma: no cover
    """Retrieve the current span."""
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except ImportError:

        class DummySpan:
            def end(self, end_time=None): ...
            def get_span_context(self): ...
            def set_attributes(self, attributes): ...
            def set_attribute(self, key, value): ...
            def add_event(self, name, attributes, timestamp): ...
            def add_link(self, context, attributes): ...
            def update_name(self, name): ...
            def is_recording(self): ...
            def set_status(self, status, description): ...
            def record_exception(
                self, exception, attributes=None, timestamp=None, escaped=False
            ): ...
            def __enter__(self):
                return self

        return DummySpan()  # type: ignore
