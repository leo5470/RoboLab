"""Opt-in replay experiments; never installed by the production collector."""

import inspect
from functools import wraps


def placement_cache(func, get_world, *, enabled=None):
    """Share one single-env placement evaluation within an existing boundary.

    Both consumers already share the predicate's settling timer. Include every
    argument in the key and let the boundary context own invalidation. The
    underlying condition and state machines are otherwise unchanged. This
    experiment canonicalizes single-env scalar calls to the batched evaluator;
    replay comparison must check event and termination equivalence.
    """
    from robolab.core.omy_teleop.boundary import _key

    signature = inspect.signature(func)
    stats = {"computed": 0, "reused": 0, "fallback": 0}

    @wraps(func)
    def cached(env, *args, **kwargs):
        if enabled is not None and not enabled():
            stats["fallback"] += 1
            return func(env, *args, **kwargs)
        world = get_world(env)
        cache = getattr(world, "_boundary_query_cache", None)
        if cache is None or env.num_envs != 1:
            stats["fallback"] += 1
            return func(env, *args, **kwargs)
        bound = signature.bind(env, *args, **kwargs)
        bound.apply_defaults()
        env_id = bound.arguments["env_id"]
        if env_id is not None and (type(env_id) is not int or env_id != 0):
            stats["fallback"] += 1
            return func(env, *args, **kwargs)
        bound.arguments["env_id"] = None
        values = {name: value for name, value in bound.arguments.items() if name != "env"}
        for name in ("object", "gripper_name", "exclude_contact_with"):
            if name in values:
                value = values[name]
                values[name] = (value,) if isinstance(value, str) else tuple(value or ())
        key = ("diagnostic_placement_result", func, _key(values))
        if key not in cache:
            cache[key] = func(*bound.args, **bound.kwargs).clone()
            stats["computed"] += 1
        else:
            stats["reused"] += 1
        result = cache[key]
        return bool(result[0]) if env_id is not None else result.clone()

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
