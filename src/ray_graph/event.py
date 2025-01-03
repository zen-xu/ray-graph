from __future__ import annotations

import dataclasses as dc
import datetime as dt

from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import Self, TypeVar, dataclass_transform


if TYPE_CHECKING:
    from collections.abc import Callable


_T = TypeVar("_T")


@overload
def field(
    *,
    default: _T,
    repr: bool = False,
    init: bool = True,
    hash: bool | None = None,
    compare: bool = True,
) -> _T: ...


@overload
def field(
    *,
    default_factory: Callable[[], _T],
    repr: bool = False,
    init: bool = True,
    hash: bool | None = None,
    compare: bool = True,
) -> _T: ...


@overload
def field(
    *,
    repr: bool = False,
    init: bool = True,
    hash: bool | None = None,
    compare: bool = True,
) -> Any: ...


def field(
    *,
    default=dc.MISSING,
    default_factory=dc.MISSING,
    repr=False,
    init=True,
    hash=None,
    compare=True,
):
    """Return an object to identify event fields."""
    return dc.field(  # type: ignore
        default=default,
        default_factory=default_factory,
        repr=repr,
        init=init,
        hash=hash,
        compare=compare,
    )


@dataclass_transform(kw_only_default=True, frozen_default=True, field_specifiers=(field,))
class _EventMeta(type):
    def __new__(cls, name: str, bases: tuple[type], dct: dict):
        annotations = dct.get("__annotations__", {})
        for field_name in annotations:
            if field_name not in dct:
                # disable repr field by default
                dct[field_name] = field(repr=False)
            elif not isinstance(dct[field_name], dc.Field):
                dct[field_name] = field(default=dct[field_name], repr=False)
        return dc.dataclass(kw_only=True, frozen=True)(super().__new__(cls, name, bases, dct))


_Rsp_co = TypeVar("_Rsp_co", default=Any, covariant=True)


class Event(Generic[_Rsp_co], metaclass=_EventMeta):
    """The base event.

    Each actor communicates by passing Events.
    """

    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: dt.datetime.now().timestamp(), init=False)

    def clone(self, **changes: Any) -> Self:
        """Return a new event replacing specified fields with new values."""
        return dc.replace(self, **changes)
