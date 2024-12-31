from __future__ import annotations

import abc

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


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


class RayNode(abc.ABC):
    """The base class for all RayGraph nodes."""

    def remote_init(self, context: RayNodeContext) -> None:
        """Initialize the node in ray cluster."""
