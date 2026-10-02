"""Cache lifetime, distinct query options, and normal consumers stay independent."""

import pytest

from robolab.core.omy_teleop.boundary import boundary_cached, boundary_queries


class World:
    def __init__(self):
        self.calls = 0
        self.state = 1

    @boundary_cached
    def pose(self, body, relative=True):
        self.calls += 1
        return body, relative, self.state


def test_boundary_reuses_equivalent_inputs_and_expires_without_a_counter():
    world = World()
    with boundary_queries(world):
        assert world.pose("banana") == world.pose(body="banana", relative=True)
        assert world.calls == 1
        assert world.pose("banana", relative=False) != world.pose("banana")
        assert world.calls == 2
    world.state = 2  # same physics counter, changed restored state
    with boundary_queries(world):
        assert world.pose("banana")[-1] == 2
        assert world.calls == 3
    world.pose("banana")
    world.pose("banana")
    assert world.calls == 5


def test_exception_discards_boundary_cache():
    world = World()
    with pytest.raises(ValueError):
        with boundary_queries(world):
            world.pose("banana")
            raise ValueError("reset")
    assert world._boundary_query_cache is None
