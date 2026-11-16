"""Local web viewer for scripted demonstrations: every recorded episode as a video, to check demo quality by eye.

    scripts/scripted_demos/run.sh --cpu scripts/scripted_demos/demo_viewer.py              # then open http://localhost:8765
    scripts/scripted_demos/run.sh --cpu scripts/scripted_demos/demo_viewer.py --prerender  # also encode every episode in the background

Families are the directories under ``$SCRIPTED_DEMOS_DIR`` (and one level below, e.g. ``_phase5_gate/<family>``)
that hold a ``pps.hdf5``. An episode's video is the pair of camera streams the policy is trained on (exterior | wrist,
224 px, 15 fps, one frame per recorded row), encoded with ffmpeg on first request and cached under ``--cache-dir``.
The collector's hi-res renders (``videos/<task>_s<seed>.mp4``: exterior, wrist and viewport, also one frame per row)
are offered on the page of the episode they belong to.

Review marks (good / bad / unsure, with a note) are written to ``<family>/review.json``, keyed by episode name and
carrying the episode's layout task and seed.

The page polls the server every 15 s while it is visible, so episodes appended by a running collection or MimicGen
generation (any generator writing this layout) appear without a reload; MimicGen episodes also show their level
and pick / place source episodes.

The HDF5 files are opened read-only, without HDF5 file locking and only for the length of a request, so the viewer
never makes a running collection fail to open its file for appending. An episode being appended at that moment may
fail to load; reload the page.
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse

import h5py
import numpy as np

from robolab.core.scripted_demos import paths
from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos.records import read_json, read_jsonl

FPS = 15
HTML = pathlib.Path(__file__).with_name("demo_viewer.html")
EPISODE_RE = re.compile(r"^episode_\d+$")
HD_RE = re.compile(r"^(?P<task>.+)_s(?P<seed>\d+)(?P<failed>\.failed)?\.mp4$")
LAYOUT_RE = re.compile(r"(\d+)Task$")
REVIEW_STATUSES = ("good", "bad", "unsure")


def open_h5(path: pathlib.Path) -> h5py.File:
    return h5py.File(path, "r", locking=False)


def attr_str(attrs, key, default=""):
    value = attrs.get(key, default)
    return value.decode() if isinstance(value, bytes) else value


def attr_json(attrs, key) -> dict:
    value = attr_str(attrs, key, None)
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def finite(obj):
    """NaN and inf (e.g. in plan costs) as null, which JSON.parse accepts."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def layout_of(task: str) -> str:
    m = LAYOUT_RE.search(task)
    return m.group(1) if m else ""


def phases(num_rows: int, executor: dict) -> list[dict]:
    """Row ranges of the scripted segments (``collect.Session.run``, ``motion.Executor``).

    The first rows hold the arm while the object comes to rest (``rest``); then each executor segment has
    ``segment_steps_taken[name]`` rows. A successful episode ends on the env's success termination, which breaks the
    loop after the row is recorded but before ``Executor.update``: the executor recorded ``total_steps + 1`` rows, and
    the segment active at termination (absent from ``segment_steps_taken``) has the rows left over.
    """
    if not executor or "total_steps" not in executor:
        return [{"name": "episode", "start": 0, "end": num_rows}]
    taken = executor.get("segment_steps_taken") or {}
    order = list(executor.get("segment_steps_planned") or {}) + ["settle"]
    start = max(num_rows - int(executor["total_steps"]) - 1, 0)
    out = [{"name": "rest", "start": 0, "end": start}] if start else []
    for name in order:
        if start >= num_rows:
            break
        n = taken.get(name)
        end = num_rows if n is None else min(start + int(n), num_rows)
        out.append({"name": name, "start": start, "end": end})
        start = end
    if out and out[-1]["end"] < num_rows:
        out[-1]["end"] = num_rows
    return out


def episode_row(name: str, g: h5py.Group) -> dict:
    a = g.attrs
    plan, executor = attr_json(a, "plan"), attr_json(a, "executor")
    q = g["joint_position"][:]
    actions = g["actions"][:]
    n = len(actions)
    task = attr_str(a, "layout_task") or attr_str(a, "source_family")
    planned, taken = executor.get("segment_steps_planned") or {}, executor.get("segment_steps_taken") or {}
    mimic = attr_json(a, "mimic")
    pick, place = mimic.get("pick") or {}, mimic.get("place") or {}
    scene = attr_json(a, "scene")
    offsets, settled = scene.get("offsets") or {}, scene.get("settled") or {}
    return {
        # Sampled scenes (MimicGen --random-scenes): anchor layout, handle angle to the shoulder camera, offsets.
        "scene": {"anchor": scene.get("anchor"), "toward_deg": settled.get("toward_deg", scene.get("toward_deg")),
                  "pitcher_dx": offsets.get("pitcher_dx"), "pitcher_dy": offsets.get("pitcher_dy"),
                  "bin_dx": offsets.get("bin_dx"), "bin_dy": offsets.get("bin_dy"),
                  "eval_nearest": scene.get("eval_nearest")} if scene else None,
        # MimicGen episodes (waypoint MimicGen over scripted plans): level, pick and place sources, held-object offset.
        "mimic": {"level": mimic.get("level"), "stitch": mimic.get("stitch"), "twin": mimic.get("twin"),
                  "pick_episode": pick.get("episode"), "pick_layout": pick.get("layout"),
                  "pick_grasp": pick.get("grasp_id"), "place_episode": place.get("episode"),
                  "place_layout": place.get("layout"), "place_grasp": place.get("grasp_id"),
                  "offset": (mimic.get("held") or {}).get("centre_offset_m")} if mimic else None,
        "name": name,
        "task": task,
        "layout": layout_of(task),
        "seed": int(a.get("seed", -1)),
        "steps": n,
        "grasp_id": int(a["grasp_id"]) if "grasp_id" in a else None,
        "cost": (plan.get("costs") or {}).get("total"),
        "closedness": executor.get("closedness_after_close"),
        "contact": executor.get("held_finger_contact_fraction"),
        "extra_steps": sum(int(v) - int(planned.get(k, v)) for k, v in taken.items()),
        "max_speed": float(np.abs(np.diff(q, axis=0)).max() * FPS) if n > 1 else 0.0,
        "max_lead": float(np.abs(actions[:, :7] - q).max()),
    }


def hd_videos(family_dir: pathlib.Path) -> dict[tuple[str, int], dict]:
    """The collector's renders by (layout task, seed)."""
    out = {}
    videos = family_dir / "videos"
    if videos.is_dir():
        for p in sorted(videos.glob("*.mp4")):
            m = HD_RE.match(p.name)
            if m and not p.name.endswith(".partial.mp4"):
                out[(m["task"], int(m["seed"]), bool(m["failed"]))] = p.name
    return out


class Store:
    """Family discovery, per-family episode tables (cached on the HDF5 file's mtime) and review marks."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self._tables: dict[str, tuple[tuple, dict]] = {}
        self._lock = threading.Lock()
        self._review_lock = threading.Lock()

    def families(self) -> dict[str, pathlib.Path]:
        found = {}
        for pattern in ("*/pps.hdf5", "*/*/pps.hdf5"):
            for p in sorted(self.root.glob(pattern)):
                found[p.parent.relative_to(self.root).as_posix()] = p.parent
        return found

    def family_dir(self, fid: str) -> pathlib.Path:
        fams = self.families()
        if fid not in fams:
            raise KeyError(f"unknown family {fid!r}")
        return fams[fid]

    def table(self, fid: str) -> dict:
        family_dir = self.family_dir(fid)
        h5 = family_dir / "pps.hdf5"
        st = h5.stat()
        key = (st.st_mtime_ns, st.st_size)
        with self._lock:
            cached = self._tables.get(fid)
            if cached and cached[0] == key:
                return cached[1]
        with open_h5(h5) as f:
            attrs = {k: attr_str(f.attrs, k) for k in ("task", "prompt")}
            episodes = [episode_row(name, f["data"][name]) for name in sorted(f["data"])]
        table = {"attrs": attrs, "episodes": episodes}
        with self._lock:
            self._tables[fid] = (key, table)
        return table

    def reviews(self, fid: str) -> dict:
        return read_json(self.family_dir(fid) / "review.json", {})

    def set_review(self, fid: str, episode: str, status: str | None, note: str) -> dict:
        if status is not None and status not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of {REVIEW_STATUSES} or null")
        rows = {e["name"]: e for e in self.table(fid)["episodes"]}
        if episode not in rows:
            raise KeyError(f"unknown episode {episode!r}")
        path = self.family_dir(fid) / "review.json"
        with self._review_lock:
            reviews = read_json(path, {})
            if status is None and not note:
                reviews.pop(episode, None)
            else:
                reviews[episode] = {"status": status, "note": note, "layout_task": rows[episode]["task"],
                                    "seed": rows[episode]["seed"],
                                    "updated": datetime.datetime.now().isoformat(timespec="seconds")}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(dict(sorted(reviews.items())), indent=1) + "\n")
            os.replace(tmp, path)
        return reviews.get(episode)

    def overview(self) -> list[dict]:
        out = []
        for fid, family_dir in self.families().items():
            summary = read_json(family_dir / "summary.json", {})
            try:
                table = self.table(fid)
            except (OSError, KeyError) as e:  # being written, or not a PPS file
                out.append({"id": fid, "error": str(e)})
                continue
            steps = [e["steps"] for e in table["episodes"]]
            attempts = read_jsonl(family_dir / "attempts.jsonl", skip_bad=True)  # a running job may be mid-row
            failures: dict[str, int] = {}
            for r in attempts:
                if not r.get("success"):
                    failures[r.get("failure") or "unknown"] = failures.get(r.get("failure") or "unknown", 0) + 1
            reviews = self.reviews(fid)
            out.append({
                "id": fid,
                "object": summary.get("object", ""),
                "prompt": table["attrs"]["prompt"],
                "episodes": len(steps),
                "attempts": len(attempts) or summary.get("attempts"),
                "successes": sum(1 for r in attempts if r.get("success")) if attempts else summary.get("successes"),
                "failures": failures or summary.get("failure_taxonomy", {}),
                "layouts": len({e["task"] for e in table["episodes"]}),
                "step_min": min(steps) if steps else None,
                "step_max": max(steps) if steps else None,
                "hd_videos": len(hd_videos(family_dir)),
                "reviews": {s: sum(1 for r in reviews.values() if r.get("status") == s) for s in REVIEW_STATUSES},
            })
        return out

    def family(self, fid: str) -> dict:
        family_dir = self.family_dir(fid)
        table = self.table(fid)
        hd = hd_videos(family_dir)
        episodes = [e | {"hd": hd.get((e["task"], e["seed"], False))} for e in table["episodes"]]
        failures = []
        for r in read_jsonl(family_dir / "attempts.jsonl", skip_bad=True):  # a running job may be mid-row
            if r.get("success"):
                continue
            executor = r.get("executor") or {}
            failures.append({"task": r.get("task"), "layout": r.get("layout"), "seed": r.get("seed"),
                             "level": r.get("level"), "pair": (r.get("mimic") or {}).get("key"),
                             "failure": r.get("failure"), "segment": executor.get("failure_segment"),
                             "detail": executor.get("failure_detail") or (r.get("selection") or {}).get("detail", ""),
                             "steps": r.get("steps"), "video": hd.get((r.get("task"), r.get("seed"), True))})
        return {"id": fid, **table["attrs"], "episodes": episodes, "failures": failures, "reviews": self.reviews(fid),
                "summary": read_json(family_dir / "summary.json", {})}

    def episode(self, fid: str, name: str) -> dict:
        family_dir = self.family_dir(fid)
        with open_h5(family_dir / "pps.hdf5") as f:
            g = f["data"][name]
            row = episode_row(name, g)
            plan, executor = attr_json(g.attrs, "plan"), attr_json(g.attrs, "executor")
            q = g["joint_position"][:]
            gripper = g["gripper_position"][:, 0]
            actions = g["actions"][:]
        names = [e["name"] for e in self.table(fid)["episodes"]]
        i = names.index(name) if name in names else -1
        hd = hd_videos(family_dir).get((row["task"], row["seed"], False))
        return {
            **row, "family": fid, "fps": FPS, "hd": hd,
            "prev": names[i - 1] if i > 0 else None, "next": names[i + 1] if 0 <= i < len(names) - 1 else None,
            "index": i, "count": len(names),
            "phases": phases(len(actions), executor),
            "plan": {k: plan.get(k) for k in ("grasp_id", "backoff", "lift_extra", "preplace_yaw", "costs",
                                              "held_object_bottom_offset", "release_drop") if k in plan},
            "executor": executor,
            "review": self.reviews(fid).get(name),
            "state": np.round(np.c_[q, gripper], 4).T.tolist(),  # 8 x N: joints 1-7, gripper
            "command": np.round(actions, 4).T.tolist(),  # 8 x N
        }


class Renderer:
    """Encodes an episode's exterior | wrist frames to H.264 once, then serves the cached file."""

    def __init__(self, store: Store, cache_dir: pathlib.Path, workers: int):
        self.store, self.cache_dir = store, cache_dir
        self._slots = threading.Semaphore(workers)
        self._locks: dict[pathlib.Path, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def path(self, fid: str, name: str) -> pathlib.Path:
        h5 = self.store.family_dir(fid) / "pps.hdf5"
        with open_h5(h5) as f:
            g = f["data"][name]
            n, seed = len(g["actions"]), int(g.attrs.get("seed", -1))
        out = self.cache_dir / fid.replace("/", "__") / f"{name}_n{n}_s{seed}.mp4"
        if out.exists():
            return out
        with self._locks_lock:
            lock = self._locks.setdefault(out, threading.Lock())
        with lock, self._slots:
            if not out.exists():
                with open_h5(h5) as f:
                    g = f["data"][name]
                    frames = np.concatenate([g["exterior_image_1_left"][:], g["wrist_image_left"][:]], axis=2)
                encode(frames, out)
        return out


def encode(frames: np.ndarray, out: pathlib.Path) -> None:
    n, h, w, _ = frames.shape
    frames = np.pad(frames, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)))  # yuv420p needs even sides
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=out.parent)
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", str(FPS), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-threads", "2", tmp]
    try:
        proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
        os.replace(tmp, out)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


class Handler(http.server.BaseHTTPRequestHandler):
    store: Store
    renderer: Renderer
    server_version = "DemoViewer/1"
    protocol_version = "HTTP/1.1"  # keep-alive: a grid of videos reuses a few connections; every reply sets a length

    def log_message(self, fmt, *args):  # quiet: errors are reported by send_error/log_error
        pass

    def log_error(self, fmt, *args):
        print(f"[demo_viewer] {self.command} {self.path}: {fmt % args}", flush=True)

    def query(self) -> dict[str, str]:
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(finite(obj), allow_nan=False, default=lambda o: None).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: pathlib.Path, content_type: str, cache: str) -> None:
        """With byte ranges: browsers need them to seek and loop video."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if m and (m[1] or m[2]):
            if m[1]:
                start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
            else:
                start = max(size - int(m[2]), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", cache)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        q = self.query()
        try:
            if route in ("/", "/index.html"):
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/families":
                self.send_json(self.store.overview())
            elif route == "/api/family":
                self.send_json(self.store.family(q["f"]))
            elif route == "/api/episode":
                self.send_json(self.store.episode(q["f"], self.episode_name(q)))
            elif route == "/video":  # ?f=&e=&v= (v only busts the browser cache when an episode is re-collected)
                path = self.renderer.path(q["f"], self.episode_name(q))
                self.send_file(path, "video/mp4", "private, max-age=86400")
            elif route == "/hd":  # ?f=&name=: the collector's renders, by file name
                family_dir = self.store.family_dir(q["f"])
                name = q["name"]
                if not HD_RE.match(name) or "/" in name or name.startswith("."):
                    raise KeyError(f"bad video name {name!r}")
                path = family_dir / "videos" / name
                if not path.is_file():
                    raise KeyError(f"no video {name!r}")
                self.send_file(path, "video/mp4", "private, max-age=3600")
            else:
                self.send_json({"error": f"no route {route}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except KeyError as e:
            self.send_json({"error": f"not found: {e}"}, 404)
        except Exception as e:  # noqa: BLE001  shown in the page, logged here
            self.log_error("%s", repr(e))
            try:
                self.send_json({"error": repr(e)}, 500)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            if route != "/api/review":
                return self.send_json({"error": f"no route {route}"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            review = self.store.set_review(body["f"], self.episode_name(body), body.get("status"),
                                           str(body.get("note") or "").strip())
            self.send_json({"review": review})
        except KeyError as e:
            self.send_json({"error": f"not found: {e}"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self.log_error("%s", repr(e))
            self.send_json({"error": repr(e)}, 500)

    @staticmethod
    def episode_name(q: dict) -> str:
        name = q["e"]
        if not EPISODE_RE.match(name):
            raise KeyError(f"bad episode name {name!r}")
        return name


def prerender(store: Store, renderer: Renderer) -> None:
    done = 0
    for fid in store.families():
        for e in store.table(fid)["episodes"]:
            try:
                renderer.path(fid, e["name"])
                done += 1
            except Exception as exc:  # noqa: BLE001  keep going; the page reports per-video errors
                print(f"[demo_viewer] prerender {fid}/{e['name']}: {exc!r}", flush=True)
    print(f"[demo_viewer] prerender finished: {done} episodes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demos-dir", type=pathlib.Path, default=objects.scripted_demos_root(),
                        help="scripted demos root (default: $SCRIPTED_DEMOS_DIR)")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=paths.demo_viewer_cache(),
                        help="where encoded episode videos are kept (default: $DEMO_VIEWER_CACHE, else output/demo_viewer_cache)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; 0.0.0.0 exposes the viewer (and the review endpoint) to the network")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=4, help="concurrent ffmpeg encodes")
    parser.add_argument("--prerender", action="store_true", help="encode every episode in the background at start")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    store = Store(args.demos_dir)
    renderer = Renderer(store, args.cache_dir, args.workers)
    Handler.store, Handler.renderer = store, renderer
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    fams = store.families()
    print(f"[demo_viewer] {len(fams)} families in {args.demos_dir}; videos cached in {args.cache_dir}", flush=True)
    print(f"[demo_viewer] serving on http://localhost:{args.port}", flush=True)
    if args.prerender:
        threading.Thread(target=prerender, args=(store, renderer), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
