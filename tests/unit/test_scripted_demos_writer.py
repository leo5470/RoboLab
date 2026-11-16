"""The PPS writer against the PPS reader's checks (pps src/pps/droid/data.py:35-95 at d9c50fd)."""

import h5py
import numpy as np
import pytest

from robolab.core.scripted_demos.writer import EpisodeBuffer, Writer, check_episode


PROMPT = "Pick up the pitcher and place it in the grey bin."


def fake_episode(n=6, seed=0):
    rng = np.random.default_rng(seed)
    buf = EpisodeBuffer(source_sim_origin_s=12.5)
    for t in range(n):
        request = {
            "observation/joint_position": rng.normal(size=7).astype(np.float32),
            "observation/gripper_position": np.array([min(1.0, t / n)], dtype=np.float32),
            "observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        }
        action = np.r_[rng.normal(size=7), float(t >= n // 2)].astype(np.float32)
        buf.add(request, action, raw_finger=0.7 * t / n, sim_time=12.5 + t / 15)
    return buf


def reader_checks(path, task):
    """What PPS's inspect_source(layout="episodes") requires of a file, re-stated."""
    with h5py.File(path, "r") as f:
        assert float(f.attrs["fps"]) == 15.0
        assert f.attrs["task"] == task and f.attrs["prompt"] == PROMPT
        for name in sorted(f["data"]):
            g = f["data"][name]
            n = len(g["actions"])
            for key, width in (("joint_position", 7), ("gripper_position", 1), ("actions", 8), ("original_actions", 8),
                               ("raw_finger_position", 1), ("normalized_gripper_position", 1)):
                assert g[key].shape == (n, width) and g[key].dtype == np.float32
                assert np.isfinite(g[key][:]).all()
            for key in ("exterior_image_1_left", "wrist_image_left"):
                assert g[key].dtype == np.uint8 and g[key].shape == (n, 224, 224, 3)
            assert 0 <= g["gripper_position"][:].min() and g["gripper_position"][:].max() <= 1
            assert set(np.unique(g["actions"][:, 7])) <= {0.0, 1.0}
            assert np.array_equal(g["source_step"][:], np.arange(n)) and g["source_step"].dtype == np.int64
            assert np.allclose(g["timestamp"][:], np.arange(n) / 15.0, atol=1e-8, rtol=0)
            assert g["source_sim_before_s"].dtype == np.float64 and g["source_sim_origin_s"].shape == (n,)
            assert int(g.attrs["num_samples"]) == n
            assert bool(g.attrs["success"]) is True
            assert "source_family" in g.attrs and "layout_task" in g.attrs


def test_writer_round_trip_and_reader_checks(tmp_path):
    path = tmp_path / "pps.hdf5"
    writer = Writer(path, task="SimplePnp303Pitcher", prompt=PROMPT)
    bufs = [fake_episode(6, 0), fake_episode(9, 1)]
    for i, buf in enumerate(bufs):
        writer.append(buf.arrays(), {"layout_task": f"SimplePnp303Pitcher0{i + 1}Task",
                                     "source_family": f"SimplePnp303Pitcher0{i + 1}Task", "seed": i})
    reader_checks(path, "SimplePnp303Pitcher")
    with h5py.File(path, "r") as f:
        names = sorted(f["data"])
        assert names == ["episode_000000", "episode_000001"]
        g = f["data"][names[1]]
        a = bufs[1].arrays()
        for key in ("joint_position", "gripper_position", "actions", "exterior_image_1_left", "wrist_image_left"):
            assert np.array_equal(g[key][:], a[key])
        assert np.array_equal(g["original_actions"][:], g["actions"][:])
        assert np.allclose(g["normalized_gripper_position"][:], g["raw_finger_position"][:] / (np.pi / 4))
        assert np.allclose(g["source_sim_origin_s"][:], 12.5)
    # Re-opening appends; a different prompt is refused.
    Writer(path, task="SimplePnp303Pitcher", prompt=PROMPT).append(fake_episode(3, 2).arrays(), {"layout_task": "x", "source_family": "x"})
    assert len(Writer(path, task="SimplePnp303Pitcher", prompt=PROMPT).episodes()) == 3
    with pytest.raises(ValueError):
        Writer(path, task="SimplePnp303Pitcher", prompt="other")


def test_check_episode_rejects_bad_rows():
    good = fake_episode(5).arrays()
    check_episode(good)
    bad = dict(good)
    bad["actions"] = good["actions"].copy()
    bad["actions"][0, 7] = 1.5
    with pytest.raises(ValueError, match="gripper"):
        check_episode(bad)
    bad = dict(good)
    bad["joint_position"] = good["joint_position"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        check_episode(bad)
    bad = dict(good)
    bad["joint_position"] = good["joint_position"].copy()
    bad["joint_position"][2, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        check_episode(bad)
