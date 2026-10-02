"""Fixed-capacity tensor snapshots for OMY state-only episodes; no simulator imports."""

import torch


class EpisodeStorage:
    def __init__(self, max_steps, max_bytes):
        if max_steps < 1 or max_bytes < 1:
            raise ValueError("Episode capacity and byte budget must be positive")
        self.max_steps = max_steps
        self.max_bytes = max_bytes
        self.allocated_bytes = 0
        self.data = {}
        self.storage = {}
        self.lengths = {}

    def reserve(self, key, value):
        if isinstance(value, dict):
            for name, leaf in value.items():
                self.reserve(f"{key}/{name}", leaf)
            return
        if key in self.storage:
            existing = self.storage[key]
            if existing.shape[1:] != value.shape or existing.dtype != value.dtype or existing.device != value.device:
                raise ValueError(f"Recorder leaf {key!r} changed shape, dtype, or device")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Recorder leaf {key!r} must be a tensor")
        # Sparse task status may receive one final status row at finalization.
        capacity = 1 if key.startswith("initial_state/") else self.max_steps + int(key.startswith("subtask/"))
        required = capacity * value.numel() * value.element_size()
        if self.allocated_bytes + required > self.max_bytes:
            raise MemoryError(f"Compact episode exceeds {self.max_bytes} byte budget at {key}")
        self.storage[key] = torch.empty((capacity, *value.shape), dtype=value.dtype, device=value.device)
        self.lengths[key] = 0
        self.allocated_bytes += required

    def add(self, key, value):
        if isinstance(value, dict):
            for name, leaf in value.items():
                self.add(f"{key}/{name}", leaf)
            return
        self.reserve(key, value)
        storage = self.storage[key]
        size = self.lengths[key]
        if size == len(storage):
            raise BufferError(f"Compact episode capacity exceeded at {key}")
        storage[size].copy_(value)
        self.lengths[key] = size + 1
        parts = key.split("/")
        parent = self.data
        for name in parts[:-1]:
            parent = parent.setdefault(name, {})
        parent[parts[-1]] = storage[:size + 1]
