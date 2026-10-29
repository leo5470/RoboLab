"""Opt-in replay experiments; never installed by the production collector."""

def placement_cache(func, get_world, *, enabled=None):
    """Instrument the production cache implementation for replay comparisons."""
    from robolab.core.omy_teleop.boundary import single_env_condition_cache

    stats = {"computed": 0, "reused": 0, "fallback": 0}
    cached = single_env_condition_cache(
        func, get_world, enabled=lambda env: enabled is None or enabled(), stats=stats,
    )
    cached.cache_stats = stats
    return cached


def install_placement_cache(*, enabled=None):
    """Call before task registration so all consumers and audits see the probe."""
    from robolab.core.task import conditionals
    from robolab.core.world.world_state import get_world

    conditionals.object_resting_in_container = placement_cache(
        conditionals.object_resting_in_container, get_world, enabled=enabled
    )
    return conditionals.object_resting_in_container.cache_stats
