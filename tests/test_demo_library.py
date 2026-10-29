"""Live discovery, selected-only admission, persistent jobs and safe media serving."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.demonstrations import register_demonstrations
from robolab.core.omy_postprocess.library import DemoCatalog, identity, save_json, VIDEO_RENDER_PROFILE
from robolab.core.omy_postprocess.library_worker import VideoQueue


def recording(path, count=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["robolab_omy_teleop"] = json.dumps({"task":"SimplePnp082WoodenBowl00Task"})
        ep = data.create_group("demo_0")
        ep.attrs["success"] = True
        ep.create_dataset("actions", data=np.zeros((count,8)))
    save_json(path.parent / "env_cfg.json", {})


@pytest.fixture
def library(tmp_path):
    collected, generated = tmp_path / "collected", tmp_path / "generated"
    collected.mkdir(); generated.mkdir()
    return DemoCatalog([collected], [generated], tmp_path / "state", refresh_s=.01)


def selection(catalog, name="future_run", count=1):
    root = catalog.generated[0] / name
    source = root / "selected/task/shard/omy_demos.hdf5"
    recording(source)
    ep = dict(file=str(source), candidate_file=str(root / "jobs/shard/data/omy_demos.hdf5"),
              family="wooden_bowl", task="SimplePnp082WoodenBowl00Task", episode="demo_0")
    save_json(root / "selection.json", dict(selected_count=count, episodes=[ep]))
    return root, ep


def test_future_discovery_selection_change_and_incomplete_manifest(library):
    library.refresh(); assert not library.rows
    raw = library.collected[0] / "20990101_120000_new_object/Task/part_000/omy_demos.hdf5"
    recording(raw)
    root, ep = selection(library)
    # An unselected retained recording never becomes a generated catalog entry.
    recording(root / "jobs/unselected/data/omy_demos.hdf5")
    library.refresh(force=True)
    assert len(library.rows) == 2
    assert sum(r["kind"] == "generated" for r in library.rows.values()) == 1
    assert identity(raw, "/data/demo_0") == identity(raw,"data/demo_0")
    # An in-progress selection update keeps the last complete snapshot.
    save_json(root / "selection.json", dict(selected_count=2, episodes=[ep]))
    library.refresh(force=True)
    assert len(library.rows) == 2
    assert library.warnings
    save_json(root / "selection.json", dict(selected_count=0, episodes=[]))
    library.refresh(force=True)
    assert len(library.rows) == 1


def test_named_recording_path_resolves_generation_parent(library):
    root, ep = selection(library)
    ep["job_id"] = "00001_task_near"
    ep["candidate_file"] = str(root / "recordings/original_session/Task/00001_task_near/part_000/omy_demos.hdf5")
    parent = library.collected[0] / "original_session/Task/part_000"
    recording(parent / "omy_demos.hdf5")
    save_json(root / "jobs" / ep["job_id"] / "job.json", {"input": str(parent), "episode": "demo_0"})
    save_json(root / "selection.json", {"selected_count": 1, "episodes": [ep]})
    library.refresh(force=True)
    generated = [row for row in library.rows.values() if row["kind"] == "generated"]
    assert len(generated) == 1
    assert generated[0]["source_id"] == identity(parent / "omy_demos.hdf5", "data/demo_0")


def test_queue_recovers_interruption_retries_and_deselection(library):
    root, ep = selection(library)
    q = VideoQueue(library, max_attempts=2); q.reconcile()
    claimed = q.claim(); assert claimed["kind"] == "generated"
    assert q.claim() is None

    # Simulated restart after acquiring exclusive service ownership.
    q.recover(); assert q.claim()["id"] == claimed["id"]
    q.finish(claimed["id"], "test failure")
    assert q.claim()["id"] == claimed["id"]
    q.finish(claimed["id"], "second failure")
    assert q.claim() is None  # bounded retries survive restarts
    assert q.states()[claimed["id"]]["render_attempts"] == 2
    save_json(root / "selection.json", dict(selected_count=0, episodes=[]))
    q.reconcile(); assert not q.states()


def test_pool_source_link_uses_chosen_demo_instead_of_target_scene(library):
    root, ep = selection(library)
    source = library.collected[0] / "session/AnotherLayout/part_000/omy_demos.hdf5"
    recording(source)
    save_json(Path(ep["file"]).parent / "mimicgen_summary.json", {"attempts": [{
        "demo": ep["episode"], "steps": 3,
        "source_reference": {"file": str(source), "episode": "demo_0"}}]})
    library.refresh(force=True)
    generated = next(row for row in library.rows.values() if row["kind"] == "generated")
    assert generated["source_id"] == identity(source, "data/demo_0")
    assert generated["source_id"] in library.rows


def test_queue_picks_up_future_selection_during_backfill(library, monkeypatch):
    from robolab.core.omy_postprocess import library_worker, batch
    selection(library)
    q = VideoQueue(library); q.reconcile()
    rendered = []

    class Renderer:
        def __init__(self, *args): pass
        def close(self): pass

    def render(catalog, row, device, renderer):
        rendered.append(row['id'])
        if len(rendered) == 1:
            selection(library, 'created_while_rendering')
            q.reconcile()

    monkeypatch.setattr(batch, 'BatchRenderer', Renderer)
    monkeypatch.setattr(library_worker, 'render_episode', render)
    library_worker.drain(q, [0])
    assert len(rendered) == len(set(rendered)) == 2
    assert all(job['video_status'] == 'ready' for job in q.states().values())


def test_old_preview_remains_playable_but_is_queued_for_resolution_upgrade(library):
    selection(library)
    library.refresh(); row, = library.rows.values()
    output = library.state_dir / 'videos' / row['id']; output.mkdir(parents=True)
    (output / 'video.mp4').write_bytes(b'old preview')
    info = dict(source_signature=library.signature(row),video='video.mp4',poster='poster.jpg',
                verified_stream=dict(width=640,height=252))
    save_json(output / 'ready.json',info)
    q = VideoQueue(library); q.reconcile()
    refreshed = library.rows[row['id']]
    assert refreshed['video_status'] == 'ready'
    assert refreshed['video_upgrade_pending']
    assert q.states()[row['id']]['video_status'] == 'pending'
    info.update(render_profile=VIDEO_RENDER_PROFILE,verified_stream=dict(width=2560,height=792))
    save_json(output / 'ready.json',info);q.reconcile()
    assert q.states()[row['id']]['video_status'] == 'ready'
    assert not library.rows[row['id']]['video_upgrade_pending']


def test_low_resolution_images_are_rerendered_without_override(library, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from robolab.core.omy_postprocess import library_worker, render
    from robolab.core.omy_postprocess.cameras import FRAME_MAPPING_GROUP
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg and ffprobe required')
    selection(library);library.refresh();row, = library.rows.values()
    low = library.state_dir / 'old_preview/images.hdf5';low.parent.mkdir(parents=True)
    low.write_bytes(b'low resolution cache must not be used')
    save_json(low.parent/'manifest.json',{'profile':{'camera_width':320,'camera_height':180,'camera_scale':1.0}})
    row['_images'] = str(low)
    calls=[]

    def native_render(source, folder, **kwargs):
        calls.append(kwargs);folder.mkdir(parents=True)
        recording(folder/'images.hdf5')
        with h5py.File(folder/'images.hdf5','a') as f:
            group=f['data/demo_0']
            for camera in ['over_shoulder_left_camera','wrist_cam']:
                group.create_dataset('obs/image_obs/'+camera,data=np.zeros((3,720,1280,3),dtype=np.uint8))
            group.create_dataset(FRAME_MAPPING_GROUP+'/timestamp_s',data=np.arange(3)/15)
        return folder

    monkeypatch.setattr(render,'render_recording',native_render)
    library_worker.render_episode(library,row,'cuda:0',SimpleNamespace(run=lambda *a,**k:None))
    assert len(calls)==1 and calls[0]['camera_scale']==1.0
    assert calls[0]['hdf5_compression']=='none'
    assert 'camera_width' not in calls[0] and 'camera_height' not in calls[0]
    ready=json.loads((library.state_dir/'videos'/row['id']/'ready.json').read_text())
    assert ready['render_profile']==VIDEO_RENDER_PROFILE
    assert ready['verified_stream']['width']==2560
    assert ready['verified_stream']['height']==792
    assert ready['verified_stream']['nb_read_frames']=='3'


def test_api_media_ranges_live_refresh_and_root_restriction(library):
    root, ep = selection(library)
    library.refresh()
    row, = library.rows.values()
    output = library.state_dir / "videos" / row["id"]
    output.mkdir(parents=True)
    (output / "video.mp4").write_bytes(b"0123456789")
    (output / "poster.jpg").write_bytes(b"poster")
    save_json(output / "ready.json", dict(source_signature=library.signature(row), video="video.mp4",poster="poster.jpg"))
    app = FastAPI(); register_demonstrations(app, library)
    with TestClient(app) as client:
        assert client.get('/demonstrations').status_code == 200
        client.post('/api/demonstrations/refresh')
        episodes = client.get('/api/demonstrations?kind=generated').json()['episodes']
        assert len(episodes) == 1 and episodes[0]['video_status'] == 'ready'
        response = client.get(episodes[0]['video'], headers={'Range':'bytes=2-5'})
        assert response.status_code == 206 and response.content == b'2345'
        assert client.get('/api/demonstrations/not-an-id/video').status_code == 404
        # A malicious manifest cannot expose an arbitrary host file.
        save_json(output / "ready.json", dict(source_signature=library.signature(row),video="/etc/passwd",poster="poster.jpg"))
        client.post('/api/demonstrations/refresh')
        assert client.get(episodes[0]['video']).status_code == 404
        new_root, _ = selection(library, "another_future_run")
        client.post('/api/demonstrations/refresh')
        assert client.get('/api/demonstrations?kind=generated').json()['total'] == 2
