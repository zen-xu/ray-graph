from __future__ import annotations

import pytest
import ray


@pytest.fixture
def init_ray():
    ray.init(address="local", local_mode=False, include_dashboard=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="session")
def init_session_ray():
    ray.init(address="local", local_mode=False, include_dashboard=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def init_local_ray():
    ray.init(address="local", local_mode=True, include_dashboard=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="session")
def init_local_session_ray():
    ray.init(address="local", local_mode=True, include_dashboard=True, ignore_reinit_error=True)
    yield
    ray.shutdown()
