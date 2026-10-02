"""Owned snapshots, hard memory limits, and no cross-episode storage aliasing."""

import pytest
import torch

from robolab.core.omy_teleop.buffers import EpisodeStorage


def test_fixed_capacity_snapshot_survives_source_mutation_and_reset():
    source = torch.arange(13, dtype=torch.float32)
    episode = EpisodeStorage(max_steps=2, max_bytes=104)
    episode.reserve("states/robot/joint_position", source)
    pointer = episode.storage["states/robot/joint_position"].data_ptr()
    episode.add("states/robot/joint_position", source)
    source.zero_()
    episode.add("states/robot/joint_position", source)
    states = episode.data["states"]["robot"]["joint_position"]
    torch.testing.assert_close(states[0], torch.arange(13, dtype=torch.float32))
    torch.testing.assert_close(states[1], source)
    assert episode.allocated_bytes == 104
    assert episode.storage["states/robot/joint_position"].data_ptr() == pointer
    with pytest.raises(BufferError):
        episode.add("states/robot/joint_position", source)
    fresh = EpisodeStorage(2, 104)
    fresh.add("states/robot/joint_position", source)
    assert fresh.storage["states/robot/joint_position"].data_ptr() != pointer


def test_byte_budget_checked_before_allocation():
    episode = EpisodeStorage(10, 39)
    with pytest.raises(MemoryError, match="byte budget"):
        episode.reserve("actions", torch.zeros(1))
    assert episode.allocated_bytes == 0 and not episode.storage


def test_initial_state_and_final_sparse_status_have_distinct_capacities():
    episode = EpisodeStorage(2, 100)
    episode.add("initial_state/robot/q", torch.tensor(1.0))
    with pytest.raises(BufferError):
        episode.add("initial_state/robot/q", torch.tensor(2.0))
    for value in (1, 2, 3):
        episode.add("subtask/status", torch.tensor(value, dtype=torch.uint16))
    assert episode.data["subtask"]["status"].tolist() == [1, 2, 3]


def test_changed_schema_is_rejected():
    episode = EpisodeStorage(2, 100)
    episode.add("actions", torch.zeros(8))
    with pytest.raises(ValueError, match="changed shape"):
        episode.add("actions", torch.zeros(7))
