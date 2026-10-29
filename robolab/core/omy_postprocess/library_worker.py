"""Persistent selected-only video queue; the web process never imports Isaac."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import time

from .library import save_json, VIDEO_RENDER_PROFILE


class VideoQueue:
    def __init__(self, catalog, max_attempts=3):
        self.catalog, self.max_attempts = catalog, max_attempts
        catalog.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = catalog.state_dir / "queue.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs
                (id TEXT PRIMARY KEY, signature TEXT, status TEXT, attempts INTEGER DEFAULT 0,
                 error TEXT, updated REAL, active INTEGER DEFAULT 1)""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reconcile(self):
        self.catalog.refresh(force=True)
        with self.connect() as db:
            db.execute("UPDATE jobs SET active=0")
            for row in self.catalog.rows.values():
                if row["kind"] != "generated":
                    continue
                try:
                    signature = json.dumps({"source": self.catalog.signature(row), "render_profile": VIDEO_RENDER_PROFILE})
                except OSError:
                    signature = "missing"
                status = "ready" if row.get("_render_current") else "pending"
                old = db.execute("SELECT signature,status,attempts FROM jobs WHERE id=?", (row["id"],)).fetchone()
                if old and old[0] == signature:
                    db.execute("UPDATE jobs SET active=1,status=? WHERE id=?",
                               ("ready" if status == "ready" else ("pending" if old[1] == "ready" else old[1]), row["id"]))
                else:
                    db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,0,NULL,?,1)",
                               (row["id"], signature, status, time.time()))

    def recover(self):
        # Called only while holding the single worker-service lock.
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='pending',attempts=MAX(0,attempts-1) WHERE status='running'")

    def claim(self, run=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            pending = db.execute("SELECT id,attempts FROM jobs WHERE active=1 AND status IN ('pending','failed') "
                                 "AND attempts<?", (self.max_attempts,)).fetchall()
            pending.sort(key=lambda item: (item[1], self.catalog.rows.get(item[0], {}).get("family", ""),
                self.catalog.rows.get(item[0], {}).get("_source", ""), item[0]))
            for key, _ in pending:
                row = self.catalog.rows.get(key)
                if not row or (run and Path(row["_run"]).resolve() != Path(run).resolve()):
                    continue
                db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated=? WHERE id=?", (time.time(), key))
                return dict(row)

    def finish(self, key, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,error=?,updated=? WHERE id=?",
                       ("failed" if error else "ready", str(error) if error else None, time.time(), key))

    def states(self):
        with self.connect() as db:
            return {key: {"video_status": status, "render_error": error, "render_attempts": attempts}
                    for key, status, error, attempts in db.execute("SELECT id,status,error,attempts FROM jobs WHERE active=1")}

    def report(self):
        states = self.states()
        runs = {}
        for row in self.catalog.rows.values():
            if row["kind"] == "generated":
                run = runs.setdefault(row["_run"], {"selected": 0, "ready": 0, "pending": 0, "running": 0, "failed": 0})
                run["selected"] += 1
                run[states.get(row["id"], {}).get("video_status", row["video_status"])] += 1
        for root, counts in runs.items():
            save_json(Path(root) / "video_status.json", {**counts, "updated_unix_s": time.time(),
                "status": "complete" if counts["ready"] == counts["selected"] else "rendering"})
        return runs


@contextmanager
def service_lock(state_dir):
    path = Path(state_dir) / "worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def render_episode(catalog, row, device, renderer):
    from .render import render_recording
    from .videos import export_success_videos
    from .io import publish_directory, sha256_file

    signature = catalog.signature(row)
    render_identity = {"source": signature, "render_profile": VIDEO_RENDER_PROFILE, "hdf5_compression": "none"}
    revision = hashlib.sha256(json.dumps(render_identity).encode()).hexdigest()[:16]
    folder = catalog.state_dir / "renders" / row["id"] / revision
    # A low-resolution preview must never satisfy a configured-resolution job.
    images = None
    if row.get("_images") and Path(row["_images"]).is_file():
        candidate = Path(row["_images"])
        manifest = candidate.parent / "manifest.json"
        if manifest.is_file():
            recorded = json.loads(manifest.read_text())
            profile = recorded.get("profile", {})
            if (profile.get("camera_width") is None and profile.get("camera_height") is None
                    and profile.get("camera_scale") == 1.0
                    and profile.get("renderer") == VIDEO_RENDER_PROFILE["renderer"]
                    and profile.get("renders_per_frame") == VIDEO_RENDER_PROFILE["renders_per_frame"]
                    and profile.get("source_sha256") == catalog.file_sha(row["_source"])
                    and profile.get("config_sha256") == catalog.file_sha(Path(row["_source"]).parent / "env_cfg.json")):
                images = candidate
    if images is None:
        folder.parent.mkdir(parents=True, exist_ok=True)
        render_recording(row["_source"], folder, group=row["_group"], device=device,
                         camera_scale=1.0, renders_per_frame=2, hdf5_compression="none", runner=renderer.run)
        images = folder / "images.hdf5"
    import h5py
    with h5py.File(images, "r") as f:
        count = len(f[row["_group"] + "/actions"])
    destination = catalog.state_dir / "videos" / row["id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{row['id']}.", dir=destination.parent))
    save_json(temporary / "dataset.json", {"episodes": [{"outcome": "success", "attempt_id": row["id"],
        "images": str(images), "source_group": row["_group"], "sample_count": count}]})
    output = export_success_videos(temporary / "dataset.json", temporary / "encoded")
    index = json.loads((output / "index.json").read_text())
    clip, = index["clips"]
    shutil.copy2(output / clip["video"], temporary / "video.mp4")
    shutil.copy2(output / clip["frames"], temporary / "poster.jpg")
    if catalog.signature(row) != signature:
        raise RuntimeError("Recording changed during rendering")
    save_json(temporary / "ready.json", {"source_signature": signature, "video": "video.mp4", "poster": "poster.jpg",
        "render_profile": VIDEO_RENDER_PROFILE,
        "verified_stream": clip["verified_stream"], "video_sha256": sha256_file(temporary / "video.mp4"),
        "source_images": str(images), "completed_unix_s": time.time()})
    shutil.rmtree(output)
    if destination.exists():
        destination.rename(destination.with_name(destination.name + f".previous-{time.time_ns()}"))
    publish_directory(temporary, destination)


def drain(queue, gpus, run=None, stop=None):
    from .batch import BatchRenderer
    stop = stop or threading.Event()

    def work(gpu):
        # Release GPU ownership regularly so waiting generation jobs can proceed.
        while not stop.is_set():
            renderer = BatchRenderer(queue.catalog.state_dir, f"cuda:{gpu}")
            worked = False
            try:
                for _ in range(8):
                    if stop.is_set():
                        return
                    row = queue.claim(run)
                    if row is None:
                        return
                    worked = True
                    try:
                        render_episode(queue.catalog, row, f"cuda:{gpu}", renderer)
                        queue.finish(row["id"])
                        print(f"Video ready: {row['family']} / {row['task']} / {row['id']}", flush=True)
                    except Exception as error:
                        queue.finish(row["id"], error)
                        print(f"Video failed: {row['id']}: {error}", flush=True)
                    queue.report()
            finally:
                try:
                    renderer.close()
                except Exception as error:
                    print(f"Renderer shutdown: {error}", flush=True)
            if not worked:
                return

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(work, gpus))
    queue.report()
