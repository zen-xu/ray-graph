import dataclasses as dc

import pytest

from ray_graph.event import Event, field


def test_event_immutable():
    class FakeEvent(Event):
        value: int

    event = FakeEvent(value=1)
    with pytest.raises(dc.FrozenInstanceError):
        event.value = 2  # type: ignore


def test_event_repr_omit_field():
    class FakeEvent(Event):
        value: int

    event = FakeEvent(value=1)
    assert "FakeEvent()" in repr(event)


def test_event_repr_with_default_value():
    class FakeEvent(Event):
        value: int = 1

    event = FakeEvent()
    assert "FakeEvent()" in repr(event)


@pytest.mark.parametrize("enable", [True, False])
def test_event_repr_with_field(enable: bool):
    class FakeEvent(Event):
        value: int = field(repr=enable)

    event = FakeEvent(value=1)
    if enable:
        assert "FakeEvent(value=1)" in repr(event)
    else:
        assert "FakeEvent()" in repr(event)


def test_event_clone():
    class FakeEvent(Event):
        value: int = 1

    event = FakeEvent()
    new_event = event.clone(value=2)
    assert new_event.value == 2
