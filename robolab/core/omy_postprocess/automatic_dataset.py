"""Discover pending sessions and publish minimal, camera-equipped HDF5 datasets."""

import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

from .batch import BatchRenderer
from .cameras import FRAME_MAPPING_GROUP
from .io import publish_directory, sha256_file
from .render import compare_raw_channels, render_recording
from .validation import validate_recording
from robolab.core.omy_teleop.collection_handoff import replace_report
from robolab.core.omy_teleop.dataset_paths import job_recording, source_names
from robolab.core.omy_teleop.final_dataset import candidates, choose, gather_candidates, read_json
from robolab.core.omy_teleop.quality import THRESHOLDS


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def discover_sessions(collected, generated):
    names = {p.parent.parent.parent.name for p in collected.glob("*/*/part_*/omy_demos.hdf5")}
    names.update(p.parts[-5] for p in generated.glob("*/*/*/*/omy_demos.hdf5")
                 if p.relative_to(generated).parts[1] not in ("jobs", "selected", "recordings")
                 and (p.parent / "generation_finalized.json").exists())
    for manifest in generated.glob("*/source_manifest.json"):
        for source in read_json(manifest)["sources"]:
            names.add(source_names(source["input"], source["task"])[0])
    return sorted(names)


def discover_recordings(collected, generated, session):
    records, reviews = {}, []
    batch_paths = {str(job_recording(result.parent.parent.parent, read_json(result)).resolve())
                   for result in generated.glob("*/jobs/*/result.json")}
    for path in sorted((collected / session).glob("*/part_*/omy_demos.hdf5")):
        records[str(path.resolve())] = {"kind": "collected", "session": session}
    # Legacy batches are joined through source metadata, not their timestamped run names.
    for manifest in sorted(generated.glob("*/source_manifest.json")):
        sources = read_json(manifest)["sources"]
        if any(source_names(s["input"], s["task"])[0] == session for s in sources):
            records.update(candidates(manifest.parent, session, collected))
            selection = manifest.parent / "selection.json"
            review = read_json(selection).get("quality_review", {}) if selection.exists() else {}
            if review:
                review = dict(review)
                report = review.pop("report", None)
                if report and (manifest.parent / report).is_file():
                    review["evidence"] = read_json(manifest.parent / report)
                reviews.append({"run": str(manifest.parent), **review})
    # New standalone generations and batch recordings share original session/task names.
    roots = [generated / session, *sorted(generated.glob(f"*/recordings/{session}"))]
    for root in roots:
        for path in sorted(root.glob("*/*/*/omy_demos.hdf5")):
            if str(path.resolve()) in records:
                continue
            if str(path.resolve()) in batch_paths:
                continue  # A rejected batch job must not reenter through directory discovery.
            finalized = path.parent / "generation_finalized.json"
            if not finalized.exists():
                continue
            status = read_json(finalized)
            if not status.get("recording_closed") or status.get("generation_exit_status") not in (0, 1):
                continue
            with h5py.File(path) as stream:
                metadata = json.loads(stream["data"].attrs["robolab_omy_mimicgen"])
            sources = metadata.get("sources", [])
            if not sources:
                continue
            # Refuse a misleading folder containing descendants of a different session.
            with h5py.File(path) as stream:
                task = json.loads(stream["data"].attrs["robolab_omy_teleop"])["task"]
            if metadata.get("target_input"):
                if source_names(metadata["target_input"], task)[0] != session:
                    continue
            elif any(source_names(Path(s["file"]).parent, s.get("task", task))[0] != session for s in sources):
                continue
            records[str(path.resolve())] = {"kind": "generated", "session": session,
                                           "generation_sources": sources}
    families = set()
    for filename in records:
        cfg = read_json(Path(filename).parent / "env_cfg.json")
        families.add(cfg["terminations"]["success"]["params"]["object"])
    if len(families) != 1:
        raise ValueError(f"Session {session} must contain one object family; found {sorted(families)}")
    return records, next(iter(families)), reviews


def verify_final(output, options):
    manifest = read_json(output / "dataset.json")
    if manifest.get("status") != "complete" or manifest.get("options") != options:
        raise ValueError(f"Existing output has different options or is incomplete: {output}")
    if manifest["selected_count"] != options["count"] or len(manifest["episodes"]) != options["count"]:
        raise ValueError("Final dataset episode count is inconsistent")
    if sha256_file(output / "training.hdf5") != manifest["sha256"]:
        raise ValueError(f"Final dataset checksum mismatch: {output}")
    with h5py.File(output / "training.hdf5") as stream:
        if len(stream["data"]) != options["count"]:
            raise ValueError("Final HDF5 episode count is inconsistent")
        for row in manifest["episodes"]:
            if len(stream[row["group"]]["actions"]) != row["num_samples"]:
                raise ValueError("Final HDF5 sample count is inconsistent")
    return manifest


def cache_index(roots):
    result = {}
    for root in roots:
        for path in root.rglob("manifest.json") if root.exists() else []:
            try:
                item = read_json(path)
                profile = item.get("profile", {})
                if item.get("status") == "rendered_for_review":
                    result.setdefault(profile.get("source_fingerprint"), []).append((path.parent, item))
            except (OSError, ValueError):
                continue
    return result


def rendered_episode(row, cache, worker, existing, device):
    source = Path(row["file"])
    group = "data/" + row["episode"]
    if sha256_file(source) != row["source_sha256"] or sha256_file(source.parent / "env_cfg.json") != row["config_sha256"]:
        raise ValueError(f"Selected source changed: {source}")
    for folder, manifest in existing.get(row["fingerprint"], []):
        profile = manifest["profile"]
        if (profile.get("source_group") != group or profile.get("config_sha256") != row["config_sha256"]
                or profile.get("camera_scale") != 1 or profile.get("camera_width") is not None
                or profile.get("renderer") != "realtime" or profile.get("renders_per_frame") != 2):
            continue
        try:
            # Revalidate the exact cache profile, dependencies, and numerical channels.
            render_recording(manifest["source"], folder, group=group, device=device,
                             hdf5_compression=profile.get("hdf5_compression", "gzip"), runner=worker.run)
            compare_raw_channels(source, folder / "images.hdf5", group)
            return folder / "images.hdf5"
        except (OSError, ValueError, KeyError):
            continue
    folder = cache / "rendered" / row["source_sha256"] / row["episode"]
    return render_recording(source, folder, group=group, device=device, runner=worker.run) / "images.hdf5"


def write_training_episode(destination, images, row):
    """Keep training tensors only, with explicit native robot conventions."""
    report = validate_recording(images)
    ep = next(e for e in report["episodes"] if e["name"] == row["episode"])
    if report["errors"] or not ep["structural_valid"] or not ep["native_15_hz"] or ep["fingerprint"] != row["fingerprint"]:
        raise ValueError("Rendered trajectory does not match the selected episode")
    n = ep["sample_count"]
    with h5py.File(images) as source:
        group = source["data"][row["episode"]]
        alignment = group[FRAME_MAPPING_GROUP]
        if (not np.array_equal(alignment["source_step"][:], np.arange(n))
                or not np.array_equal(alignment["post_state_row"][:], np.arange(n) - 1)
                or not np.allclose(alignment["timestamp_s"][:], np.arange(n) / 15, rtol=0, atol=1e-8)):
            raise ValueError("Policy-camera frames are not aligned to pre-action rows")
        source.copy(group["actions"], destination, "actions")
        obs = destination.create_group("obs")
        source.copy(group["obs/proprio_obs/arm_joint_pos"], obs, "arm_joint_pos")
        finger = ep["joint_names"].index("finger_joint")
        initial = group["initial_state/articulation/robot/joint_position"][0, finger]
        subsequent = group["states/articulation/robot/joint_position"][:-1, finger]
        obs["gripper_pos"] = np.r_[initial, subsequent].reshape(n, 1)
        cameras = obs.create_group("images")
        for name in ("over_shoulder_left_camera", "wrist_cam"):
            image = group[f"obs/image_obs/{name}"]
            if image.ndim != 4 or image.shape[0] != n or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Invalid training camera {name}")
            target = cameras.create_dataset(name, shape=image.shape, dtype=image.dtype,
                                           chunks=(1, *image.shape[1:]), compression="lzf")
            for i in range(n):
                target[i] = image[i]
        destination["timestamp"] = np.arange(n, dtype=np.float64) / 15
        destination.attrs.update(num_samples=n, task=row["task"], task_instruction=row["task_instruction"])


def process_one(session, collected, generated, output_root, cache_root, *, count=50,
                collected_count=None, device="cuda:1", dry_run=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("--session must be an original session folder name")
    options = {"session": session, "count": count, "collected_count": collected_count,
               "selection_policy": "up_to_10_humans_v1",
               "collected_root": str(collected), "generated_root": str(generated),
               "format": "omy_training_hdf5_v1", "camera_profile": "native_realtime_2"}
    output, cache = output_root / session, cache_root / session
    if output.exists():
        verify_final(output, options)
        print(f"{session}: already complete; verified and skipped", flush=True)
        return "skipped"
    with locked(cache / ".processing.lock"):
        if output.exists():
            verify_final(output, options)
            return "skipped"
        plan_path = cache / "selection.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan["options"] != options:
                raise ValueError("Cached selection uses different options; use another cache/output root")
        else:
            # Exclude active collection while inspecting recorded parts.
            with locked(collected / session / ".session.lock"):
                recordings, family, reviews = discover_recordings(collected, generated, session)
                rows, rejected = gather_candidates(recordings, family, {k: v[0] for k, v in THRESHOLDS.items()})
                selected, _ = choose(rows, count, collected_count)
            for row in selected:
                row.pop("blocks")
                cfg = read_json(Path(row["file"]).parent / "env_cfg.json")
                row["task_instruction"] = cfg.get("instruction") or row["task"]
            plan = {"options": options, "family": family, "episodes": selected,
                    "quality_review": reviews, "rejected": rejected}
            if not dry_run:
                replace_report(plan_path, plan)
        rows = plan["episodes"]
        print(f"{session}: selected {len(rows)} {dict(Counter(r['kind'] for r in rows))}", flush=True)
        if dry_run:
            return "pending"
        output_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{session}.", dir=output_root))
        worker = BatchRenderer(cache, device)
        try:
            existing = cache_index([collected / session / "postprocess", generated,
                                    generated.parent / "demo_library" / "renders", cache / "rendered"])
            manifest_rows = []
            with h5py.File(staging / "training.hdf5", "x") as target:
                target.attrs.update(fps=15, format="omy_training_hdf5_v1",
                                    action_encoding="7 absolute arm joint targets (rad), gripper +1=open/-1=closed",
                                    state_encoding="7 measured arm joints (rad), measured finger_joint (rad)",
                                    image_encoding="RGB uint8; pre-action", training_ready=False)
                data = target.create_group("data")
                for i, row in enumerate(rows):
                    print(f"{session}: preparing training episode {i + 1}/{count}", flush=True)
                    images = rendered_episode(row, cache, worker, existing, device)
                    name = f"demo_{i}"
                    write_training_episode(data.create_group(name), images, row)
                    manifest_rows.append({"group": f"data/{name}", "kind": row["kind"],
                                          "generation_source": row.get("generation_source"),
                                          "task": row["task"], "source_sha256": row["source_sha256"],
                                          "source_group": row["episode"], "fingerprint": row["fingerprint"],
                                          "num_samples": row["num_samples"], "warnings": row["warnings"]})
                data.attrs["total"] = sum(r["num_samples"] for r in rows)
            manifest = {"schema_version": 1, "status": "complete", "options": options,
                        "selected_count": len(rows), "episodes": manifest_rows,
                        "file": "training.hdf5", "sha256": sha256_file(staging / "training.hdf5"),
                        "quality_review": [{k: v for k, v in r.items() if k != "evidence"}
                                           for r in plan["quality_review"]], "training_ready": False}
            replace_report(staging / "dataset.json", manifest)
            verify_final(staging, options)
            publish_directory(staging, output)
            print(f"{session}: final dataset {output / 'training.hdf5'}", flush=True)
            return "complete"
        finally:
            worker.close()
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", help="Original session name; repeatable; default: all pending sessions")
    parser.add_argument("--collected-root", type=Path, default=Path("output/omy_teleop"))
    parser.add_argument("--generated-root", type=Path, default=Path("output/omy_mimicgen"))
    parser.add_argument("--output-root", type=Path, default=Path("output/omy_training"))
    parser.add_argument("--cache-root", type=Path, default=Path("output/omy_training_cache"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--collected-count", type=int)
    parser.add_argument("--device", default="cuda:1", help="Isolated host GPU for missing policy-camera renders")
    parser.add_argument("--dry-run", action="store_true", help="Discover and select only; no rendering or dataset publication")
    args = parser.parse_args(argv)
    if args.count < 1 or (args.collected_count is not None and not 0 <= args.collected_count <= args.count):
        parser.error("Invalid count or collected quota")
    if not re.fullmatch(r"cuda:[0-9]+", args.device):
        parser.error("--device must be cuda:N")
    roots = [getattr(args, name).resolve() for name in ("collected_root", "generated_root", "output_root", "cache_root")]
    sessions = args.session or discover_sessions(*roots[:2])
    failures = 0
    for session in sessions:
        try:
            process_one(session, *roots, count=args.count, collected_count=args.collected_count,
                        device=args.device, dry_run=args.dry_run)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            failures += 1
            print(f"{session}: not published: {error}", flush=True)
    print(f"Checked {len(sessions)} sessions; {failures} incomplete.", flush=True)
    return 1 if failures else 0
