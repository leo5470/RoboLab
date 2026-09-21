"""Amortized linear recording with the ordinary Isaac Lab EpisodeData interface."""

import torch
from isaaclab.utils.datasets import EpisodeData


class BufferedEpisodeData(EpisodeData):
    """Append snapshots into geometrically growing storage, exposing only used rows.

    Upstream EpisodeData concatenates the entire history on every append. Here
    each frame is copied once, and old frames are copied only when capacity
    doubles. Consumers still see a nested dictionary of contiguous tensors.
    """

    def __init__(self):
        super().__init__()
        self._storage = {}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._storage = {}

    def add(self, key, value):
        if isinstance(value, dict):
            for name, leaf in value.items():
                self.add(f"{key}/{name}", leaf)
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Recorder leaf {key!r} must be a tensor")
        parts = key.split("/")
        parent = self._data
        for name in parts[:-1]:
            parent = parent.setdefault(name, {})
        name = parts[-1]
        previous = parent.get(name)
        size = 0 if previous is None else len(previous)
        if previous is not None and (
            previous.shape[1:] != value.shape or previous.dtype != value.dtype or previous.device != value.device
        ):
            raise ValueError(f"Recorder leaf {key!r} changed shape, dtype, or device")
        storage = self._storage.get(key)
        if storage is None or size == len(storage):
            # Small initial blocks avoid reserving many full-resolution images.
            capacity = max(2, 2 * size)
            storage = torch.empty((capacity, *value.shape), dtype=value.dtype, device=value.device)
            if size:
                storage[:size].copy_(previous)
            self._storage[key] = storage
        storage[size].copy_(value)
        parent[name] = storage[:size + 1]
