import pytest

from diff_stream.simulate import simulate_panel

pytest_plugins = ["stream_limits.pytest_plugin"]


@pytest.fixture
def data():
    return simulate_panel(80, 5, seed=13)
