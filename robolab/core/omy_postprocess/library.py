"""Incremental, simulator-free catalog for collected and selected demonstrations."""

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import h5py


# Match collected-session postprocessing: preserve the configured camera sizes.
VIDEO_RENDER_PROFILE = {"schema_version": 1, "camera_resolution": "configured",
                        "camera_scale": 1.0, "renderer": "realtime", "renders_per_frame": 2,
                        "fps": 15, "codec": "h264", "crf": 18}


def identity(path, group):
    return hashlib.sha256(f"{Path(path).resolve()}::{group.strip('/')}".encode()).hexdigest()[:24]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class DemoCatalog:
    def __init__(self, collected, generated, state_dir, refresh_s=30):
        self.collected = [Path(p).resolve() for p in collected]
        self.generated = [Path(p).resolve() for p in generated]
        self.state_dir = Path(state_dir).resolve()
        self.roots = self.collected + self.generated + [self.state_dir]
        self.refresh_s = refresh_s
        self.lock = threading.RLock()
        self.rows, self.warnings, self.updated = {}, [], 0
        self.cache = {}
        try:
            self.cache = json.loads((self.state_dir / "manifest_cache.json").read_text())
        except (OSError, ValueError):
            pass

    def allowed(self, path):
        path = Path(path).resolve()
        return any(path.is_relative_to(root) for root in self.roots)

    def read(self, path, loader=None):
        path = Path(path).resolve()
        if not self.allowed(path):
            raise ValueError(f"Artifact outside configured roots: {path}")
        st = path.stat()
        signature = [st.st_mtime_ns, st.st_size]
        cached = self.cache.get(str(path), {})
        if cached.get("signature") == signature:
            return cached["value"]
        try:
            value = loader(path) if loader else json.loads(path.read_text())
        except (OSError, ValueError, KeyError) as error:
            if "value" not in cached:
                raise
            self.warnings.append(f"Reading last complete version of {path}: {error}")
            return cached["value"]
        self.cache[str(path)] = {"signature": signature, "value": value}
        return value

    def file_sha(self, path):
        path = Path(path).resolve()
        if not self.allowed(path):
            raise ValueError("Source outside configured roots")
        st = path.stat()
        key, signature = str(path) + "::sha256", [st.st_mtime_ns, st.st_size]
        cached = self.cache.get(key, {})
        if cached.get("signature") != signature:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.cache[key] = {"signature": signature, "value": digest.hexdigest()}
        return self.cache[key]["value"]

    @staticmethod
    def recording(path):
        with h5py.File(path, "r") as f:
            data = f["data"]
            meta = json.loads(data.attrs.get("robolab_omy_teleop", "{}"))
            return {"task": meta.get("task", path.parent.parent.name), "episodes": [
                {"group": g.name, "samples": int(g.attrs.get("num_samples", len(g.get("actions", [])))),
                 "outcome": "success" if g.attrs.get("success", False) else "failure"}
                for g in data.values() if isinstance(g, h5py.Group) and "actions" in g]}

    def paths(self, roots, pattern):
        for root in roots:
            if root.exists():
                for p in sorted(root.rglob(pattern)):
                    if not any(part.startswith(".") for part in p.relative_to(root).parts):
                        yield p

    @staticmethod
    def selection(path):
        value = json.loads(path.read_text())
        if value.get("selected_count") != len(value["episodes"]):
            raise ValueError("Selection is incomplete")
        for episode in value["episodes"]:
            for key in ("file", "episode", "family", "task"):
                if not isinstance(episode.get(key), str):
                    raise ValueError(f"Selection episode is missing {key}")
            if "catalog_id" in episode and (not isinstance(episode["catalog_id"], str)
                                            or not re.fullmatch(r"[0-9a-f]{24}", episode["catalog_id"])):
                raise ValueError("Invalid persistent catalog identity")
        return value

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.updated < self.refresh_s:
                return
            self.warnings = []
            rows, media = {}, {}
            # Video indexes are authoritative; filenames alone never imply membership.
            for index in self.paths(self.collected + self.generated, "index.json"):
                if not index.parent.name.startswith("verification_videos"):
                    continue
                try:
                    contents = self.read(index)
                    for clip in contents.get("clips", []):
                        video = index.parent / clip.get("video", "")
                        if video.name == "all_successes.mp4" or not video.is_file():
                            continue
                        poster = index.parent / clip.get("frames", clip.get("preview", ""))
                        media[str(clip["attempt_id"])] = (str(video.resolve()),
                            str(poster.resolve()) if poster.is_file() else None)
                except (OSError, ValueError, KeyError) as error:
                    self.warnings.append(f"{index}: {error}")
            for root in self.collected:
                for path in self.paths([root], "omy_demos.hdf5"):
                    if "postprocess" in path.relative_to(root).parts:
                        continue
                    try:
                        info = self.read(path, self.recording)
                        parts = path.relative_to(root).parts
                        session = next((s for s in parts if re.match(r"\d{8}_\d{6}", s)), parts[0])
                        family = re.sub(r"^\d{8}_\d{6}_", "", session)
                        if family == session or family.isdigit() or family.startswith("SimplePnp"):
                            task_label = re.sub(r"^SimplePnp\d+|\d{2}Task$|Task$", "", info["task"])
                            family = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", task_label).lower()
                        for ep in info["episodes"]:
                            row = self.base(path, ep["group"], "collected", family, info["task"], session)
                            row.update(outcome=ep["outcome"], sample_count=ep["samples"])
                            rows[row["id"]] = row
                    except (OSError, ValueError, KeyError) as error:
                        self.warnings.append(f"{path}: {error}")
            # Enrich raw recordings with postprocessing outcomes and media.
            for dataset in self.paths(self.collected, "dataset.json"):
                try:
                    for ep in self.read(dataset).get("episodes", []):
                        if not ep.get("source"):
                            continue
                        source = (dataset.parent / ep["source"]).resolve()
                        key = identity(source, ep["source_group"])
                        if key not in rows:
                            continue
                        row = rows[key]
                        row.update(attempt_id=ep["attempt_id"], outcome=ep.get("outcome", row["outcome"]))
                        row["_video"], row["_poster"] = media.get(ep["attempt_id"], (None, None))
                        image = dataset.parent / ep.get("images", "")
                        if image.is_file():
                            row["_images"] = str(image.resolve())
                except (OSError, ValueError, KeyError) as error:
                    self.warnings.append(f"{dataset}: {error}")
            # Only selection.json admits generated episodes to the library.
            for selection in self.paths(self.generated, "selection.json"):
                try:
                    selected = self.read(selection, self.selection)
                    review = selected.get("quality_review", {})
                    for ep in selected["episodes"]:
                        source = Path(ep["file"])
                        if not source.is_absolute():
                            source = selection.parent / source
                        group = "data/" + ep["episode"].removeprefix("data/")
                        canonical = ep.get("candidate_file", source)
                        if not Path(canonical).is_absolute():
                            canonical = selection.parent / canonical
                        if not self.allowed(source) or not self.allowed(canonical):
                            raise ValueError("Selection source is outside configured roots")
                        row = self.base(source, group, "generated", ep["family"], ep["task"], selection.parent.name)
                        row["id"] = ep.get("catalog_id") or identity(canonical, group)
                        row["_canonical"] = str(Path(canonical).resolve())
                        row.update(seed=ep.get("seed"), tier=ep.get("tier"), selected=True,
                                   _run=str(selection.parent), outcome="success")
                        if source.exists():
                            recorded = self.read(source, self.recording)
                            entry = next((item for item in recorded["episodes"] if item["group"].strip('/') == group), None)
                            if entry:
                                row["sample_count"] = entry["samples"]
                                row["outcome"] = entry["outcome"]
                        if ep["family"] in review.get("held_families", []):
                            row["review"] = "contact_review_required"
                        job = selection.parent / "jobs" / ep.get("job_id", Path(canonical).parent.parent.name) / "job.json"
                        if job.exists():
                            job_info = self.read(job)
                            row["source_id"] = identity(Path(job_info["input"]) / "omy_demos.hdf5", "data/" + job_info["episode"])
                        summary = source.parent / "mimicgen_summary.json"
                        if summary.exists():
                            attempt = next((a for a in self.read(summary).get("attempts", [])
                                            if a.get("demo") == ep["episode"]), {})
                            row["sample_count"] = attempt.get("steps", row["sample_count"])
                            origin = attempt.get("source_reference")
                            if origin and self.allowed(origin["file"]):
                                row["source_id"] = identity(origin["file"], "data/" + origin["episode"])
                        rows[row["id"]] = row
                except (OSError, ValueError, KeyError) as error:
                    self.warnings.append(f"{selection}: {error}")
            # Reuse existing generated previews by their exact source + episode.
            for manifest in self.paths(self.generated, "manifest.json"):
                try:
                    info = self.read(manifest)
                    if info.get("status") != "rendered_for_review":
                        continue
                    for row in rows.values():
                        if row["kind"] != "generated" or row["_source"] != info.get("source"):
                            continue
                        profile = info.get("profile", {})
                        if (profile.get("source_sha256") != self.file_sha(row["_source"])
                                or profile.get("config_sha256") != self.file_sha(Path(row["_source"]).parent / "env_cfg.json")):
                            continue
                        name = f"{row['family']}_{row['task']}_{row['_group'].split('/')[-1]}"
                        if name in media:
                            row["_video"], row["_poster"] = media[name]
                            row["_images"] = str(manifest.parent / "images.hdf5")
                except (OSError, ValueError, KeyError) as error:
                    self.warnings.append(f"{manifest}: {error}")
            # Queue output contains only verified, atomically published episodes.
            for row in rows.values():
                ready = self.state_dir / "videos" / row["id"] / "ready.json"
                row["_render_current"] = False
                if ready.exists():
                    try:
                        info = self.read(ready)
                        if info["source_signature"] == self.signature(row):
                            row["_video"] = str(ready.parent / info["video"])
                            row["_poster"] = str(ready.parent / info["poster"])
                            row["_render_current"] = info.get("render_profile") == VIDEO_RENDER_PROFILE
                            row["video_width"] = info.get("verified_stream", {}).get("width")
                            row["video_height"] = info.get("verified_stream", {}).get("height")
                            if info.get("source_images"):
                                row["_images"] = info["source_images"]
                    except (OSError, ValueError, KeyError) as error:
                        self.warnings.append(f"{ready}: {error}")
                for field in ("_video", "_poster", "_source", "_images"):
                    if row.get(field) and not self.allowed(row[field]):
                        row[field] = None
                row["video_status"] = ("ready" if row.get("_video") and Path(row["_video"]).is_file()
                                       else "unavailable" if row["kind"] == "collected" else "pending")
                row["duration_s"] = row.get("sample_count", 0) / 15
                row["video_upgrade_pending"] = (row["kind"] == "generated" and row["video_status"] == "ready"
                                                and not row["_render_current"])
            self.rows, self.updated = rows, time.time()
            save_json(self.state_dir / "manifest_cache.json", self.cache)

    @staticmethod
    def signature(row):
        path = Path(row["_source"])
        st = path.stat()
        cfg = path.parent / "env_cfg.json"
        return [str(path), row["_group"], st.st_size, st.st_mtime_ns,
                cfg.stat().st_mtime_ns if cfg.exists() else 0]

    @staticmethod
    def base(source, group, kind, family, task, session):
        match = re.search(r"(\d{2})Task$", task)
        return {"id": identity(source, group), "kind": kind, "family": family, "task": task,
                "layout": match[1] if match else task, "session": session,
                "review": "unverified", "selected": False, "sample_count": 0,
                "_source": str(Path(source).resolve()), "_group": group.strip('/'),
                "_video": None, "_poster": None}

    def public(self, row):
        item = {k: v for k, v in row.items() if not k.startswith("_")}
        for field in ("video", "poster"):
            item[field] = f"/api/demonstrations/{row['id']}/{field}" if row.get("_" + field) else None
        return item
