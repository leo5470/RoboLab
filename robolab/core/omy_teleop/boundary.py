"""Read-only query sharing within one OMY control boundary."""

import inspect
from contextlib import contextmanager
from functools import wraps


def _key(value):
    if isinstance(value, list):
        return (list, tuple(_key(item) for item in value))
    if isinstance(value, tuple):
        return (tuple, tuple(_key(item) for item in value))
    if isinstance(value, dict):
        return (dict, tuple(sorted((name, _key(item)) for name, item in value.items())))
    hash(value)  # Non-value-like parameters must use the original query.
    return value


def boundary_cached(func=None, *, batched=False):
    if func is None:
        return lambda method: boundary_cached(method, batched=batched)
    signature = inspect.signature(func)

    @wraps(func)
    def query(self, *args, **kwargs):
        cache = getattr(self, "_boundary_query_cache", None)
        if cache is None:
            return func(self, *args, **kwargs)
        arguments = signature.bind(self, *args, **kwargs)
        arguments.apply_defaults()
        env_id = arguments.arguments.get("env_id")
        # OMY admits one env. Its batched and indexed tensor getters are
        # equivalent; sharing prevents duplicate pose/velocity GPU copies.
        select = batched and isinstance(env_id, int) and getattr(getattr(self, "env", None), "num_envs", 0) == 1
        if select:
            arguments.arguments["env_id"] = None
        try:
            key = (func.__name__, tuple((name, _key(value)) for name, value in arguments.arguments.items()
                                      if name != "self"))
        except TypeError:
            return func(self, *args, **kwargs)
        if key not in cache:
            cache[key] = func(*arguments.args, **arguments.kwargs)
        value = cache[key]
        if select:
            return tuple(item[env_id] for item in value) if isinstance(value, tuple) else value[env_id]
        return value

    return query


def single_env_condition_cache(func, get_world, *, enabled, stats=None):
    """Share an audited condition within one existing OMY boundary.

    Canonicalize single-env scalar consumers to the batched evaluator. Every
    parameter participates in the key; the boundary/reset lifecycle owns
    invalidation. The caller must audit stateful conditions before using this.
    """
    signature = inspect.signature(func)

    def count(name):
        if stats is not None:
            stats[name] += 1

    @wraps(func)
    def cached(env, *args, **kwargs):
        if not enabled(env):
            count("fallback")
            return func(env, *args, **kwargs)
        world = get_world(env)
        cache = getattr(world, "_boundary_query_cache", None)
        if cache is None or env.num_envs != 1:
            count("fallback")
            return func(env, *args, **kwargs)
        bound = signature.bind(env, *args, **kwargs)
        bound.apply_defaults()
        env_id = bound.arguments["env_id"]
        if env_id is not None and (type(env_id) is not int or env_id != 0):
            count("fallback")
            return func(env, *args, **kwargs)
        bound.arguments["env_id"] = None
        values = {name: value for name, value in bound.arguments.items() if name != "env"}
        for name in ("object", "gripper_name", "exclude_contact_with"):
            if name in values:
                value = values[name]
                values[name] = (value,) if isinstance(value, str) else tuple(value or ())
        try:
            key = ("placement_result", func, _key(values))
        except TypeError:
            count("fallback")
            return func(env, *args, **kwargs)
        if key not in cache:
            cache[key] = func(*bound.args, **bound.kwargs).clone()
            count("computed")
        else:
            count("reused")
        result = cache[key]
        if env_id is not None:
            return bool(result[0])
        # Managers retain their simulator device; only the audited query and
        # timer evaluation may use an instance-owned CPU world.
        return result.to(getattr(env, "device", result.device), copy=True)

    return cached


def shared_placement_condition(get_world):
    """Opt in only the placement condition audited against its settling timer."""
    return lambda func: single_env_condition_cache(
        func, get_world,
        enabled=lambda env: getattr(getattr(env, "cfg", None), "omy_shared_placement_checks", False),
    )


@contextmanager
def boundary_queries(world):
    """Share geometry/pose inputs, keeping predicate parameters and transitions distinct."""
    previous = getattr(world, "_boundary_query_cache", None)
    world._boundary_query_cache = {} if previous is None else previous
    try:
        yield
    finally:
        world._boundary_query_cache = previous
