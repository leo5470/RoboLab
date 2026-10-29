"""Real CPU encoding checks for individual and combined verification videos."""

import json
import shutil

import h5py
import numpy as np
import pytest

from robolab.core.omy_postprocess import videos
from robolab.core.omy_postprocess.cameras import FRAME_MAPPING_GROUP
from robolab.core.omy_postprocess.io import sha256_file


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe required")


@pytest.fixture
def dataset(tmp_path):
    episodes = []
    with h5py.File(tmp_path / "images.hdf5", "w") as file:
        for i, count in enumerate((3, 6)):
            group = file.create_group(f"data/demo_{i}")
            for camera in ("over_shoulder_left_camera", "wrist_cam"):
                group.create_dataset(f"obs/image_obs/{camera}",
                                     data=np.full((count, 48, 64, 3), 50 + 100*i, dtype=np.uint8))
            group.create_dataset(f"{FRAME_MAPPING_GROUP}/timestamp_s", data=np.arange(count) / 15)
            episodes.append(dict(outcome="success", attempt_id=f"attempt_{i}", images="images.hdf5",
                                 source_group=group.name, sample_count=count))
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dict(episodes=episodes)))
    return path


def test_combined_video_retains_all_frames_and_episode_ranges(dataset):
    output = videos.export_success_videos(dataset)
    assert sorted(p.name for p in output.glob("*.mp4")) == [
        "all_successes.mp4", "success_001_attempt_0.mp4", "success_002_attempt_1.mp4"]
    assert not (output / "concat.txt").exists()
    index = json.loads((output / "index.json").read_text())
    assert index["video_retention"] == "combined_and_individual"
    assert int(videos._probe(output / "all_successes.mp4", 9)["nb_read_frames"]) == 9
    assert [(c["combined_video"], c["start_frame"], c["end_frame_exclusive"], c["start_time_s"])
            for c in index["clips"]] == [("all_successes.mp4", 0, 3, 0), ("all_successes.mp4", 3, 9, .2)]
    for clip in index["clips"]:
        videos._probe(output / clip["video"], clip["frame_count"])
    assert len(list(output.glob("success_*_frames.jpg"))) == 2
    for filename, expected in index["artifact_sha256"].items():
        assert sha256_file(output / filename) == expected
    assert videos.export_success_videos(dataset) == output
    with (output / "all_successes.mp4").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        videos.export_success_videos(dataset)


def test_failed_combined_verification_preserves_individual_videos(dataset, monkeypatch):
    probe = videos._probe

    def reject_reel(path, count):
        if path.name == "all_successes.mp4":
            raise ValueError("Combined verification failed")
        return probe(path, count)

    monkeypatch.setattr(videos, "_probe", reject_reel)
    with pytest.raises(ValueError, match="Combined verification failed"):
        videos.export_success_videos(dataset)
    assert not (dataset.parent / "verification_videos").exists()
    partial, = dataset.parent.glob(".verification_videos.partial-*")
    assert len(list(partial.glob("success_*.mp4"))) == 2
    assert (partial / "concat.txt").exists()
