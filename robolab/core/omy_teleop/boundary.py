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


@contextmanager
def boundary_queries(world):
    """Share geometry/pose inputs, keeping predicate parameters and transitions distinct."""
    previous = getattr(world, "_boundary_query_cache", None)
    world._boundary_query_cache = {} if previous is None else previous
    try:
        yield
    finally:
        world._boundary_query_cache = previous
