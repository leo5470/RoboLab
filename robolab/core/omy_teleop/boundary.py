"""Read-only query sharing within one OMY control boundary."""

import inspect
from contextlib import contextmanager
from functools import wraps


def boundary_cached(func):
    signature = inspect.signature(func)

    @wraps(func)
    def query(self, *args, **kwargs):
        cache = getattr(self, "_boundary_query_cache", None)
        if cache is None:
            return func(self, *args, **kwargs)
        arguments = signature.bind(self, *args, **kwargs)
        arguments.apply_defaults()
        key = (func.__name__, tuple((name, value) for name, value in arguments.arguments.items() if name != "self"))
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        return cache[key]

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
