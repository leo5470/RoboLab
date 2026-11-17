"""PPS export checks for episodes generated with seven-axis IK: they may move panda_joint3, six-axis episodes must hold
it at home, and one file never mixes the two."""

import h5py
import numpy as np
import pytest

from robolab.core.omy_postprocess import pps_hdf5


def _write(path, episodes):
    """``episodes``: list of (seven_axis, moves_joint3)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs.update(fps=pps_hdf5.FPS, task="SimplePnp230MilkjugA01", prompt="p", training_ready=False)
        for i, (seven_axis, moves) in enumerate(episodes):
            n = 6
            g = f.create_group(f"data/episode_{i:03d}")
            arrays = {name: rng.uniform(-1, 1, (n, *shape)).astype(dtype)
                      for name, (dtype, shape) in pps_hdf5.DATASETS.items()}
            for name in ("actions", "original_actions"):
                arrays[name][:, 2] = rng.uniform(-0.2, 0.2, n) if moves else 0.0
            arrays["source_step"] = np.arange(n, dtype=pps_hdf5.DATASETS["source_step"][0])
            arrays["timestamp"] = (np.arange(n) / pps_hdf5.FPS).astype(pps_hdf5.DATASETS["timestamp"][0])
            for name, values in arrays.items():
                g[name] = values
            for wire in pps_hdf5.CAMERAS:
                g[wire] = np.zeros((n, *pps_hdf5.IMAGE_SIZE, 3), np.uint8)
            g.attrs.update(num_samples=n, source_family="f", seven_axis=seven_axis)
    return path


def test_six_axis_file_keeps_the_held_joint(tmp_path):
    report = pps_hdf5.verify_file(_write(tmp_path / "pps.hdf5", [(False, False), (False, False)]))
    assert not report["seven_axis"] and report["constant_channels"]["actions"] == [2]
    assert report["held_joint"]["hold_value_rad"] == 0.0


def test_seven_axis_file_may_move_joint3(tmp_path):
    report = pps_hdf5.verify_file(_write(tmp_path / "pps.hdf5", [(True, True), (True, True)]))
    assert report["seven_axis"]
    assert 2 not in report["constant_channels"].get("actions", [])
    assert report["held_joint"]["held"] is False


def test_six_axis_episode_moving_joint3_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not held"):
        pps_hdf5.verify_file(_write(tmp_path / "pps.hdf5", [(False, True)]))


def test_file_mixing_six_and_seven_axis_episodes_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mixes six- and seven-axis"):
        pps_hdf5.verify_file(_write(tmp_path / "pps.hdf5", [(False, False), (True, True)]))


def test_files_without_the_attribute_are_six_axis(tmp_path):
    path = _write(tmp_path / "pps.hdf5", [(False, False)])
    with h5py.File(path, "a") as f:
        del f["data/episode_000"].attrs["seven_axis"]
    assert not pps_hdf5.verify_file(path)["seven_axis"]
