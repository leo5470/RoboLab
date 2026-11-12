"""Single-source OMY dataset pipeline: freeze → qualify → generate → select → render → status.

See docs/omy_mimicgen.md. Each stage leaves an artifact on disk, so a rerun
resumes where the previous run stopped:

0. Freeze the session's eligible human demos (``STATE/SESSION/sources.json``).
1. Qualify each human with one non-randomized single-source reproduction
   under the current generator implementation.
2. Generate single-source candidates, round-robin across qualified humans,
   until each has ``candidate_factor`` × its share of the generated slots or
   its attempt budget is spent. Quality, timing and IK checks run per job.
3. Select up to 10 humans plus generated fill balanced across parents, check
   that every generated demo has one qualified parent, and publish the pool.
4. Render missing cameras and publish ``training.hdf5`` for the frozen pool.
   Then write the PPS file (``PPS/SESSION/pps.hdf5``, pi05 DROID wire convention;
   ``omy_postprocess.pps_hdf5``).
5. Keep the website summary, pool index and verification report current.

GPUs come from ``gpu_pool.resolve_gpu_pool``. Jobs take whichever pool GPU is
free; there is no dedicated rendering GPU.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import h5py
import psutil

from robolab.core.omy_postprocess import pps_hdf5
from robolab.core.omy_postprocess.automatic_dataset import (WRITE_WORKERS, discover_recordings, locked, process_one,
                                                            verify_final)
from robolab.core.omy_postprocess.io import sha256_file
from robolab.core.omy_postprocess.validation import validate_recording
from .dataset_paths import job_recording, source_names
from .diversity import descriptor
from . import grasp_groups
from .final_dataset import choose, export, gather_candidates, parent_key, read_json
from .quality import THRESHOLDS, assess

REPO = Path(__file__).resolve().parents[3]
LIMITS = {k: v[0] for k, v in THRESHOLDS.items()}
TIERS = {
    "near": {"object_pose_range": .025, "object_yaw_range": math.radians(5), "pose_range": .01, "yaw_range": 0.},
    "moderate": {"object_pose_range": .05, "object_yaw_range": math.radians(15), "pose_range": .02, "yaw_range": 0.},
    "wide": {"object_pose_range": .1, "object_yaw_range": math.radians(30), "pose_range": .03,
             "yaw_range": math.radians(5)},
}
TIER_WEIGHTS = (("near", .5), ("moderate", .25), ("wide", .25))
IMPLEMENTATION_FILES = ("robolab/core/omy_teleop/mimicgen.py", "robolab/core/omy_teleop/mimic_sources.py",
                        "robolab/core/omy_teleop/repairs.py", "robolab/core/omy_teleop/target_sampler.py")
GENERATION_VERSION = "omy_dataset_pipeline_v1"
POOL_METHOD = ("single source only; up to 10 eligible humans; generated fill balanced across qualified "
               "human parents, then layout-balanced farthest-first")
JOB_PREFIX = "ss"
STOP_AFTER = ("qualify", "generate", "select")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def implementation(profile="none"):
    """Generator identity: source hashes, plus the generator profile when one is used."""
    impl = {name: sha256_file(REPO / name) for name in IMPLEMENTATION_FILES}
    if profile != "none":
        from .repairs import GENERATOR_PROFILES
        impl["generator_profile"] = {"name": profile, "args": list(GENERATOR_PROFILES[profile])}
    return impl


def implementation_tag(impl):
    """Short stable tag of an implementation record, for job ids."""
    return hashlib.sha256(json.dumps(impl, sort_keys=True).encode()).hexdigest()[:8]


def allocate(total, weights):
    """Split ``total`` in proportion to ``weights`` (largest remainder; ties to the earlier entry)."""
    raw = [total * w / sum(weights) for w in weights]
    counts = [math.floor(r + 1e-9) for r in raw]
    for index in sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))[:total - sum(counts)]:
        counts[index] += 1
    return counts


def tier_of(ranges):
    """The ``TIERS`` entry whose reset ranges a generated recording used, or None."""
    obj, container = (ranges or {}).get("object_pose_range"), (ranges or {}).get("pose_range")
    if not obj or not container:
        return None
    recorded = (obj["x"][1], obj["yaw"][1], container["x"][1], container["yaw"][1])
    for name, spec in TIERS.items():
        expected = (spec["object_pose_range"], spec["object_yaw_range"], spec["pose_range"], spec["yaw_range"])
        if all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(recorded, expected)):
            return name
    return None


def source_key(source):
    """(resolved part directory, episode): the same key as ``final_dataset.parent_key``."""
    return str(Path(source["input"]).resolve()), source["episode"]


def label(key):
    return f"{Path(key[0]).parent.name}/{Path(key[0]).name}:{key[1]}"


class Interrupted(RuntimeError):
    """Raised inside a worker when the pipeline is asked to stop."""


def process_job(root, job, gpu, stop, thresholds=LIMITS):
    """Run one isolated single-source generator child and check its output independently.

    ``root`` is the session's generated folder. A completed job is verified and
    reused. An interrupted job writes no result; its folder is quarantined by
    the next run's preparation.
    """
    if job.get("sources"):
        raise ValueError("Multisource generation is not supported")
    folder = root / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        saved = read_json(result_path)
        payload = job_recording(root, saved)
        for name, expected in saved.get("artifact_sha256", {}).items():
            if sha256_file(payload.parent / name) != expected:
                raise RuntimeError(f"Completed job artifact changed: {payload.parent / name}")
        return saved
    output = job_recording(root, job).parent
    if output.exists():
        raise RuntimeError(f"Unfinalized job output requires recovery: {output}")
    if job.get("target_manifest") and job["stage"] != "generate":
        raise ValueError("Explicit targets are generation jobs")
    command = [sys.executable, "-B", "-u", str(REPO / "examples/generate_omy_mimic.py"), "--worker-exit",
               "--host-gpu", str(gpu), "--input", job["input"], "--target-input", job["input"],
               "--target-episode", job["episode"].split("_")[-1], "--episodes", job["episode"].split("_")[-1],
               "--output-dir", str(output), "--num-demos", str(job["num_demos"]),
               "--max-attempts", str(job["max_attempts"]), "--seed", str(job["seed"])]
    if job["stage"] == "validate":
        command += ["--no-randomize", "--keep-failed"]
    elif job.get("target_manifest"):
        command += ["--no-randomize", "--keep-failed", "--target-manifest", job["target_manifest"],
                    "--target-id", job["target_id"]]
    else:
        command += ["--keep-failed"]
        for key, value in job["ranges"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
    profile = job.get("generator_profile", "none")
    if profile != "none":
        from .repairs import GENERATOR_PROFILES
        command += list(GENERATOR_PROFILES[profile])
    impl = implementation(profile)
    atomic_json(folder / "job.json", {**job, "gpu": gpu, "command": command, "implementation": impl})
    started = time.time()
    with (folder / "run.log").open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "PYTHONUNBUFFERED": "1"})
        atomic_json(folder / "process.json", {"pid": child.pid, "gpu": gpu, "started": started})
        code = None
        while code is None:
            try:
                code = child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    terminate_group(child)
                    raise Interrupted(f"Stopped job {job['id']}")
                if time.time() - started > 3600:
                    terminate_group(child)
                    code = child.returncode
    summary_path = output / "mimicgen_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"attempts": []}
    result = {**job, "gpu": gpu, "implementation": impl, "returncode": code, "wall_s": time.time() - started,
              "attempts": len(summary["attempts"]), "accepted": [], "rejections": {}, "errors": [], "warnings": []}
    finalized_path = output / "generation_finalized.json"
    if finalized_path.exists():
        finalized = read_json(finalized_path)
        result["finalized"] = finalized
        if finalized.get("generation_exit_status") not in (0, 1):
            result["errors"].append("Generation was interrupted before completing its trial loop")
        elif code not in (0, 1):
            result["warnings"].append(f"Worker exited {code} after recording finalization; outputs independently verified")
    else:
        result["errors"].append("Missing generation finalization record")
    path = output / "omy_demos.hdf5"
    if path.exists():
        try:
            quality = assess(path, thresholds, summary)
            validation = validate_recording(path)
            atomic_json(output / "quality.json", quality)
            atomic_json(output / "validation.json", validation)
            valid = {e["name"]: e for e in validation["episodes"]}
            with h5py.File(path, "r") as stream:
                meta = json.loads(stream["data"].attrs["robolab_omy_mimicgen"])
                if len(meta.get("sources") or []) != 1:
                    result["errors"].append("Generator loaded more than one source")
                for name, item in quality.items():
                    reasons = list(item["failed"])
                    if validation["errors"]:
                        reasons.append("recording_structure")
                    if not valid[name]["structural_valid"] or not valid[name]["native_15_hz"]:
                        reasons += [e["code"] for e in valid[name]["errors"]] or ["invalid_structure_or_cadence"]
                    if not stream["data"][name].attrs.get("success", False):
                        reasons.append("unsuccessful")
                    generated = next((a for a in summary["attempts"] if a.get("demo") == name), {})
                    if generated.get("ik_position_error_m") is None or generated.get("events") is None:
                        reasons.append("missing_generation_checks")
                    try:
                        descriptor(stream["data"][name], meta["object"], meta["container"])
                    except (ValueError, KeyError) as error:
                        reasons.append("diversity_features:" + str(error))
                    if reasons:
                        result["rejections"][name] = sorted(set(reasons))
                    else:
                        result["accepted"].append(name)
        except Exception:
            result["errors"].append(traceback.format_exc())
    elif not summary["attempts"]:
        result["errors"].append("Worker produced no finalized recording or attempt summary")
    result["artifact_sha256"] = {name: sha256_file(output / name)
                                 for name in ("omy_demos.hdf5", "env_cfg.json", "mimicgen_summary.json",
                                              "generation_finalized.json") if (output / name).is_file()}
    atomic_json(result_path, result)
    return result


def terminate_group(child, grace_s=30):
    try:
        os.killpg(child.pid, signal.SIGINT)
        child.wait(timeout=grace_s)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def signal_descendants(signum):
    """Reach renderer and generator children, which run in their own sessions."""
    for process in psutil.Process().children(recursive=True):
        try:
            process.send_signal(signum)
        except psutil.Error:
            pass


def verify_pool(output, count):
    """A published raw pool has exactly ``count`` unique, checksum-verified demos."""
    manifest = read_json(output / "dataset.json")
    rows = manifest["episodes"]
    if manifest.get("selected_count") != count or len(rows) != count or len({r["fingerprint"] for r in rows}) != count:
        raise ValueError(f"Pool count or identity mismatch: {output}")
    observed = Counter(r["kind"] for r in rows)
    if observed["collected"] > min(10, count) or dict(observed) != manifest["counts_by_kind"]:
        raise ValueError("Pool origin counts are inconsistent")
    checked = set()
    for row in rows:
        file = output / row["file"]
        if file not in checked:
            if sha256_file(file) != row["sha256"]:
                raise ValueError(f"Pool checksum mismatch: {file}")
            if sha256_file(file.parent / "env_cfg.json") != row["config_sha256"]:
                raise ValueError(f"Pool configuration checksum mismatch: {file}")
            checked.add(file)
    return manifest


def verify_lineage(output, manifest, qualified):
    """Every generated demo comes from exactly one qualified human of this session."""
    problems = []
    loaded = {}
    for row in manifest["episodes"]:
        if row["kind"] != "generated":
            continue
        file = output / row["file"]
        with h5py.File(file, "r") as stream:
            if file not in loaded:
                loaded[file] = json.loads(stream["data"].attrs.get("robolab_omy_mimicgen", "{}")).get("sources") or []
            parents = json.loads(stream["data"][row["episode"]].attrs.get("mimicgen_source", "{}")).get("parents") or []
        if len(loaded[file]) != 1:
            problems.append(f"{row['episode']} in {row['file']}: generated from {len(loaded[file])} loaded sources")
            continue
        if len(set(parents)) > 1:
            problems.append(f"{row['episode']} in {row['file']}: mixed parents {parents}")
            continue
        try:
            parent = parent_key(row)
        except ValueError as error:
            problems.append(str(error))
            continue
        if parent not in qualified:
            problems.append(f"{row['episode']} in {row['file']}: parent {label(parent)} is not qualified")
    if problems:
        raise ValueError("Pool is not single-source: " + "; ".join(problems[:5]) +
                         (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))


def training_options(session, roots, count, collected_count):
    """Must equal ``automatic_dataset.process_one``'s options for the same roots."""
    return {"session": session, "count": count, "collected_count": collected_count,
            "selection_policy": "up_to_10_humans_v1", "collected_root": str(roots["collected"]),
            "generated_root": str(roots["generated"]), "format": "omy_training_hdf5_v1",
            "camera_profile": "native_realtime_2"}


def freeze_training_plan(pool, roots, count, collected_count):
    """Write the builder's frozen selection from the published pool, or check an existing one."""
    options = training_options(pool["session"], roots, count, collected_count)
    rows = []
    for item in pool["episodes"]:
        row = {**item, "file": item["source_file"]}
        cfg = read_json(Path(row["file"]).parent / "env_cfg.json")
        row["task_instruction"] = cfg.get("instruction") or row["task"]
        rows.append(row)
    plan = {"options": options, "family": pool["family"], "episodes": rows,
            "quality_review": pool.get("quality_review", []), "rejected": pool.get("rejected", [])}
    folder = roots["cache"] / pool["session"]
    with locked(folder / ".processing.lock"):
        path = folder / "selection.json"
        if path.exists():
            previous = read_json(path)
            if (previous["options"] != options
                    or {r["fingerprint"] for r in previous["episodes"]} != {r["fingerprint"] for r in rows}):
                raise ValueError(f"Training cache selection differs from the published pool: {path}")
        else:
            atomic_json(path, plan)
    return options


class Status:
    """Single writer for the website summary, the pool index and the verification report."""

    def __init__(self, roots, dry_run=False):
        self.roots, self.dry_run = roots, dry_run
        self.lock = threading.Lock()
        self.summary_path = roots["summary"]
        self.verification_path = roots["summary"].parent / "pool_verification.json"
        self.index_path = roots["pools"] / "index.json"
        self.status_path = roots["state"] / "status.json"

    @staticmethod
    def _read(path, default):
        return read_json(path) if path.exists() else default

    def session(self, work, **values):
        with self.lock:
            work.status_fields.update(values)
            if self.dry_run:
                return
            summary = self._read(self.summary_path, {"schema_version": 1, "sessions": {}})
            previous = summary["sessions"].get(work.name, {})
            entry = {k: v for k, v in previous.items() if k == "multisource_archive"}
            entry.update(work.status_fields)
            summary["sessions"][work.name] = entry
            atomic_json(self.summary_path, summary)
            state = self._read(self.status_path, {"schema_version": 1, "sessions": {}})
            state["sessions"][work.name] = {**work.status_fields, "updated_unix_s": time.time()}
            atomic_json(self.status_path, state)

    def pool(self, work, manifest):
        """Record a verified published pool in the index and verification reports."""
        dataset = self.roots["pools"] / work.name / "dataset.json"
        if self.dry_run:
            return
        digest = sha256_file(dataset)
        with self.lock:
            index = self._read(self.index_path, {"status": "complete", "sessions": []})
            entry = {"session": work.name, "family": manifest["family"], "selected_count": manifest["selected_count"],
                     "counts_by_kind": manifest["counts_by_kind"], "manifest": f"{work.name}/dataset.json",
                     "manifest_sha256": digest, "generation_mode": "single_source"}
            index["sessions"] = [s for s in index["sessions"] if s["session"] != work.name] + [entry]
            index["sessions"].sort(key=lambda s: s["session"])
            index["demo_count"] = sum(s["selected_count"] for s in index["sessions"])
            atomic_json(self.index_path, index)
            report = self._read(self.verification_path, {"schema_version": 1, "sessions": {}})
            rows = manifest["episodes"]
            report["sessions"][work.name] = {
                "family": manifest["family"], "selected_count": manifest["selected_count"],
                "counts_by_kind": manifest["counts_by_kind"],
                "source_lineage_count": sum(r["kind"] == "generated" for r in rows),
                "task_layouts": len({r["task"] for r in rows}), "samples": sum(r["num_samples"] for r in rows),
                "manifest_sha256": digest, "generation_mode": "single_source"}
            atomic_json(self.verification_path, report)


class SessionWork:
    """Mutable per-session scheduling state. Guarded by the pipeline's condition lock."""

    def __init__(self, name):
        self.name = name
        self.family = None
        self.folder = None          # generated/SESSION: generation jobs and recordings
        self.sources = []
        self.qualified, self.failed = set(), set()
        self.validations = {}       # qualified parent -> its accepted qualification result
        self.pending_qualify = []
        self.rows = []              # gated, quality-passing candidates (humans and generated)
        self.rejected = []
        self.reviews = []
        self.results = defaultdict(list)    # parent -> this pipeline's generate results
        self.active = {}            # job id -> job
        self.stage = "prepare"      # prepare, qualify, generate, select, export, done, failed
        self.next_index = 0
        self.tier_shortfall = {}    # tier -> candidates still missing when generation stopped
        self.status_fields = {}
        self.error = None


class Pipeline:
    def __init__(self, roots, gpus, *, count=50, collected_count=None, max_attempts_per_source=150,
                 candidate_factor=1.5, dry_run=False, stop_after=None, replace_invalid=False, reverify=False,
                 write_workers=WRITE_WORKERS, log=print, generator_profile="none"):
        if stop_after not in (None, *STOP_AFTER):
            raise ValueError(f"stop_after must be one of {STOP_AFTER}")
        self.roots, self.gpus = roots, gpus
        self.count, self.collected_count = count, collected_count
        self.max_attempts_per_source, self.candidate_factor = max_attempts_per_source, candidate_factor
        self.dry_run, self.stop_after, self.replace_invalid = dry_run, stop_after, replace_invalid
        self.reverify = reverify
        self.write_workers = write_workers
        self.log = log
        self.status = Status(roots, dry_run)
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.running = 0
        self.work = []
        from .repairs import GENERATOR_PROFILES
        if generator_profile not in GENERATOR_PROFILES:
            raise ValueError(f"Unknown generator profile {generator_profile!r}; choose from {sorted(GENERATOR_PROFILES)}")
        self.generator_profile = generator_profile
        self.impl = implementation(generator_profile)

    # ----- discovery and preparation (CPU, before any GPU work) -----

    def profile_field(self):
        """Job field naming the generator profile; absent for the default, so old job records stay equal."""
        return {} if self.generator_profile == "none" else {"generator_profile": self.generator_profile}

    def say(self, work, message):
        self.log(f"{work.name}: {message}", flush=True)

    def fail(self, work, error):
        work.stage, work.error = "failed", str(error)
        self.say(work, f"FAILED: {error}")
        self.status.session(work, status="failed", error=str(error))

    def humans_quota(self, work):
        humans = sum(r["kind"] == "collected" for r in work.rows)
        quota = self.collected_count if self.collected_count is not None else min(10, self.count, humans)
        return quota, self.count - quota

    def evidence(self, session):
        """Every accepted qualification of this session's humans, by any implementation."""
        found = set()
        for path in self.roots["generated"].glob("*/jobs/*/result.json"):
            result = read_json(path)
            if (result.get("stage") == "validate" and result.get("accepted") and not result.get("errors")
                    and source_names(result["input"], result["task"])[0] == session):
                found.add(source_key(result))
        return found

    def published(self, work):
        """'complete', 'export' or None (nothing published). Raises if a published pool is invalid."""
        pool_dir = self.roots["pools"] / work.name
        if not pool_dir.exists():
            return None
        manifest = verify_pool(pool_dir, self.count)
        verify_lineage(pool_dir, manifest, self.evidence(work.name))
        work.family = manifest["family"]
        self.status.pool(work, manifest)
        training = self.roots["training"] / work.name
        if not training.exists():
            return "export"
        self.verify_training(work, manifest)
        if not (self.roots["pps"] / work.name).exists():
            return "export"
        pps_hdf5.verify_session(self.roots["pps"] / work.name, pool_dir, training)
        return "complete"

    def verify_training(self, work, manifest):
        """Full check of a published training file, or a stamp check if it is unchanged since one."""
        training = self.roots["training"] / work.name
        options = training_options(work.name, self.roots, self.count, self.collected_count)
        stamp_path = self.roots["state"] / work.name / "training_verified.json"
        stat = (training / "training.hdf5").stat()
        dataset = sha256_file(training / "dataset.json")
        current = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "dataset_json_sha256": dataset,
                   "pool_sha256": sha256_file(self.roots["pools"] / work.name / "dataset.json"), "options": options}
        if not self.reverify and stamp_path.exists() and read_json(stamp_path) == current:
            return
        final = verify_final(training, options)
        if {r["fingerprint"] for r in final["episodes"]} != {r["fingerprint"] for r in manifest["episodes"]}:
            raise ValueError(f"Training file does not match the published pool: {training}")
        if not self.dry_run:
            atomic_json(stamp_path, current)

    def archive(self, work, reason):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target = self.roots["state"] / "archive" / stamp / work.name
        for name, root in (("pool", self.roots["pools"]), ("training", self.roots["training"]),
                           ("cache", self.roots["cache"])):
            if (root / work.name).exists():
                target.mkdir(parents=True, exist_ok=True)
                os.rename(root / work.name, target / name)
        atomic_json(target / "reason.json", {"reason": reason, "archived_unix_s": time.time()})
        self.say(work, f"archived the invalid published pool to {target}")

    def quarantine_unfinished(self, work):
        """Move this pipeline's interrupted job folders aside, preserving them."""
        for folder in sorted((work.folder / "jobs").glob(f"{JOB_PREFIX}*")):
            if (folder / "result.json").exists():
                continue
            job = read_json(folder / "job.json") if (folder / "job.json").exists() else {"id": folder.name}
            target = work.folder / "interrupted" / f"{folder.name}_{int(time.time())}"
            target.mkdir(parents=True)
            os.rename(folder, target / "job")
            if job.get("output_dir") and job_recording(work.folder, job).parent.exists():
                os.rename(job_recording(work.folder, job).parent, target / "output")
            self.say(work, f"moved interrupted job {folder.name} to {target}")

    def prepare(self, session):
        work = SessionWork(session)
        self.work.append(work)
        try:
            state = self.published(work)
        except (OSError, ValueError, KeyError) as error:
            if not self.replace_invalid:
                self.fail(work, f"published pool failed verification ({error}); rerun with --replace-invalid "
                                "to archive it and rebuild")
                return
            if not self.dry_run:
                self.archive(work, str(error))
            state = None
        if state == "complete":
            work.stage = "done"
            self.say(work, "published pool, training file and PPS file verified; skipped")
            self.status.session(work, status="complete", family=work.family, generation_mode="single_source",
                                selected_count=self.count, pool=str(self.roots["pools"] / session / "dataset.json"),
                                training_file=str(self.roots["training"] / session / "training.hdf5"),
                                pps_file=str(self.roots["pps"] / session / pps_hdf5.FILE_NAME),
                                counts_by_kind=read_json(self.roots["pools"] / session / "dataset.json")["counts_by_kind"])
            return
        if state == "export":
            work.stage = "export" if self.stop_after is None else "done"
            self.say(work, "published pool verified; training or PPS file pending")
            if not self.dry_run:
                self.status.session(work, status="pool_complete", family=work.family, generation_mode="single_source",
                                    selected_count=self.count, pool=str(self.roots["pools"] / session / "dataset.json"))
            return
        try:
            with locked(self.roots["collected"] / session / ".session.lock"):
                recordings, work.family, work.reviews = discover_recordings(
                    self.roots["collected"], self.roots["generated"], session)
            rows, work.rejected = gather_candidates(recordings, work.family, LIMITS)
        except BlockingIOError:
            self.fail(work, "collection is active; session skipped")
            return
        except (OSError, ValueError, KeyError) as error:
            self.fail(work, f"discovery failed: {error}")
            return
        humans = [r for r in rows if r["kind"] == "collected"]
        work.sources = [{"family": work.family, "session": session, "task": r["task"],
                         "input": str(Path(r["file"]).parent), "episode": r["episode"],
                         "sha256": r["source_sha256"], "fingerprint": r["fingerprint"]} for r in humans]
        if not work.sources:
            self.fail(work, "no eligible human demonstrations")
            return
        frozen = self.roots["state"] / session / "sources.json"
        if frozen.exists():
            saved = read_json(frozen)["sources"]
            key = lambda s: (source_key(s), s["sha256"])  # noqa: E731
            if sorted(map(key, saved)) != sorted(map(key, work.sources)):
                self.fail(work, f"eligible human demos changed since {frozen} was frozen; "
                                "use a new --state-root to start over")
                return
        elif not self.dry_run:
            atomic_json(frozen, {"session": session, "family": work.family, "sources": work.sources})
        work.folder = self.roots["generated"] / session
        marker = work.folder / "generator_profile.json"
        recorded = read_json(marker)["name"] if marker.exists() else ("none" if (work.folder / "jobs").exists() else None)
        if recorded not in (None, self.generator_profile):
            self.fail(work, f"{work.folder} holds generator profile {recorded!r} candidates; use a new "
                            f"--generated-root for profile {self.generator_profile!r}")
            return
        if not self.dry_run:
            (work.folder / "jobs").mkdir(parents=True, exist_ok=True)
            if recorded is None:
                atomic_json(marker, {"name": self.generator_profile, "implementation": self.impl})
            if not (work.folder / "source_manifest.json").exists():
                atomic_json(work.folder / "source_manifest.json", {"sources": work.sources})
            self.quarantine_unfinished(work)
        results = [read_json(p) for p in sorted((work.folder / "jobs").glob("*/result.json"))]
        current = {s: None for s in map(source_key, work.sources)}
        for result in results:
            key = source_key(result)
            if result["id"].startswith(JOB_PREFIX):
                work.next_index = max(work.next_index, int(result.get("index", -1)) + 1)
            if key not in current or not result["id"].startswith(JOB_PREFIX):
                continue
            if result["stage"] == "validate" and result.get("implementation") == self.impl:
                ok = bool(result["accepted"]) and not result["errors"]
                (work.qualified if ok else work.failed).add(key)
                if ok:
                    work.validations[key] = result
            elif result["stage"] == "generate":
                work.results[key].append(result)
        work.pending_qualify = [s for s in work.sources if source_key(s) not in work.qualified | work.failed]
        work.rows = rows
        work.stage = "qualify"
        quota, generated = self.humans_quota(work)
        self.say(work, f"{len(work.sources)} eligible humans ({len(work.qualified)} qualified, "
                       f"{len(work.pending_qualify)} to qualify); {sum(r['kind'] == 'generated' for r in rows)} "
                       f"single-source candidates; need {quota} human + {generated} generated")
        if not self.dry_run:
            self.status.session(work, status="qualifying", family=work.family, generation_mode="single_source")

    def record_grasp_groups(self, work):
        """Write ``STATE/SESSION/grasp_groups.json`` from the qualified humans' reproductions.

        Mixed-parent generation reads it to restrict placement parents; it refuses
        an ``ambiguous`` or ``unavailable`` record. Single-source runs never read it.
        """
        if self.dry_run or not work.qualified:
            return
        try:
            recordings = {label(key): (job_recording(work.folder, work.validations[key]), work.validations[key]["accepted"][0])
                          for key in sorted(work.qualified)}
            record = grasp_groups.session_groups(recordings, work.family)
        except (OSError, KeyError, ValueError) as error:
            record = {"schema_version": 1, "object": work.family, "status": "unavailable", "reason": str(error),
                      "groups": None}
        record["implementation"] = self.impl
        grasp_groups.write(self.roots["state"] / work.name / "grasp_groups.json", record)
        if record["status"] != "grouped":
            self.say(work, f"grasp groups {record['status']}: {record['reason']}")

    # ----- scheduling (all under self.cv) -----

    def parent_counts(self, work):
        have = Counter()
        for row in work.rows:
            if row["kind"] == "generated":
                try:
                    have[parent_key(row)] += 1
                except ValueError:
                    continue
        return have

    def tier_targets(self, work):
        """Candidate targets by (parent, tier): global tier totals, then each tier split across qualified parents.

        ``candidate_factor`` × the generated quota is split over ``TIER_WEIGHTS``
        first, so every tier is requested however many parents there are.
        """
        _, generated = self.humans_quota(work)
        parents = sorted(work.qualified)
        totals = dict(zip((t for t, _ in TIER_WEIGHTS),
                          allocate(math.ceil(self.candidate_factor * generated), [w for _, w in TIER_WEIGHTS])))
        targets = {}
        for tier, total in totals.items():
            for parent, count in zip(parents, allocate(total, [1] * len(parents))):
                targets[parent, tier] = count
        return totals, targets

    def tier_counts(self, work):
        """Quality-passing candidates by (parent, tier).

        A row from one of this pipeline's jobs takes that job's tier; any other
        generated row takes the tier its recorded reset ranges match, if any.
        """
        own = {}
        if work.folder is not None:
            for results in work.results.values():
                for result in results:
                    own[str(job_recording(work.folder, result).resolve())] = result["tier"]
        have = Counter()
        for row in work.rows:
            if row["kind"] == "generated":
                try:
                    parent = parent_key(row)
                except ValueError:
                    continue
                tier = own.get(str(Path(row["file"]).resolve())) if "file" in row else None
                have[parent, tier or tier_of(row.get("generation_ranges"))] += 1
        return have

    def plan(self, work):
        """Next generate job, 'select', 'shortfall' or None (wait for active jobs).

        The most underfilled tier, relative to its total, is served first, by
        the qualified parent furthest below its share of that tier. A parent
        may exceed its share only when no parent below it can still generate
        that tier, so an exhausted parent's share moves to the others instead
        of to another tier.
        """
        quota, generated = self.humans_quota(work)
        if not generated:
            return "select"
        if not work.qualified:
            return "shortfall"
        totals, targets = self.tier_targets(work)
        have = self.tier_counts(work)
        for job in work.active.values():
            have[source_key(job), job["tier"]] += job["num_demos"]  # reserved until the job finishes
        busy = {source_key(job) for job in work.active.values()}

        def budget(key):
            return self.max_attempts_per_source - sum(r["attempts"] for r in work.results[key])

        def dead(key, tier):
            previous = [r for r in work.results[key] if r["tier"] == tier]
            return sum(r["attempts"] for r in previous) >= 20 and not any(r["accepted"] for r in previous)

        best = None
        for priority, (tier, _) in enumerate(TIER_WEIGHTS):
            deficit = totals[tier] - sum(have[k, tier] for k in work.qualified)
            if deficit <= 0:
                continue
            alive = [s for s in work.sources if source_key(s) in work.qualified
                     and budget(source_key(s)) >= 1 and not dead(source_key(s), tier)]
            below = [s for s in alive if have[source_key(s), tier] < targets[source_key(s), tier]]
            for source in below or alive:
                key = source_key(source)
                if key in busy:
                    continue
                own = have[key, tier] - targets[key, tier]
                order = (-deficit / totals[tier], own, priority, source["task"], source["episode"])
                if best is None or order < best[0]:
                    best = (order, source, tier, budget(key), min(deficit, -own) if own < 0 else deficit)
        if best is not None:
            _, source, tier, remaining, missing = best
            index = work.next_index
            work.next_index += 1
            demos = min(8, missing)
            return {"id": f"{JOB_PREFIX}{index:05d}_{source['task']}_{tier}", "index": index,
                    "family": work.family, "task": source["task"], "input": source["input"],
                    "episode": source["episode"], "stage": "generate", "tier": tier, "num_demos": demos,
                    "max_attempts": min(remaining, max(24, 3 * demos)), "seed": 3_000_000 + index * 1000,
                    "ranges": TIERS[tier], **self.profile_field(),
                    "output_dir": str(Path(source["task"]) / f"{JOB_PREFIX}{index:05d}_{source['task']}_{tier}"
                                      / Path(source["input"]).name)}
        if work.active:
            return None
        # Selection may still fill the total, but a missing tier is reported, never hidden.
        work.tier_shortfall = {tier: totals[tier] - sum(have[k, tier] for k in work.qualified)
                               for tier in totals if sum(have[k, tier] for k in work.qualified) < totals[tier]}
        eligible = sum(n for (k, _), n in have.items() if k in work.qualified)
        return "select" if eligible >= generated else "shortfall"

    def next_task(self):
        for work in self.work:
            if self.stop.is_set():
                return None
            if work.stage == "qualify":
                if work.pending_qualify:
                    source = work.pending_qualify.pop(0)
                    tag = (self.impl[IMPLEMENTATION_FILES[0]][:8] if self.generator_profile == "none"
                           else implementation_tag(self.impl))
                    job_id = f"{JOB_PREFIX}q_{source['task']}_{source['episode']}_{tag}"
                    job = {"id": job_id, "index": -1, "family": work.family, "task": source["task"],
                           "input": source["input"], "episode": source["episode"], "stage": "validate",
                           "tier": "baseline", "num_demos": 1, "max_attempts": 2, "seed": 2_000_000, "ranges": {},
                           **self.profile_field(),
                           "output_dir": str(Path(source["task"]) / job_id / Path(source["input"]).name)}
                    work.active[job["id"]] = job
                    return ("job", work, job)
                if work.active:
                    continue
                self.record_grasp_groups(work)
                self.say(work, f"qualified {len(work.qualified)}/{len(work.sources)} humans"
                               + (f"; not qualified: {', '.join(label(k) for k in sorted(work.failed))}"
                                  if work.failed else ""))
                if self.stop_after == "qualify":
                    work.stage = "done"
                    self.status.session(work, status="qualified", qualified_human_sources=len(work.qualified))
                    continue
                work.stage = "generate"
                self.status.session(work, status="generating", qualified_human_sources=len(work.qualified))
            if work.stage == "generate":
                step = self.plan(work)
                if isinstance(step, dict):
                    work.active[step["id"]] = step
                    return ("job", work, step)
                if step == "select":
                    if self.stop_after == "generate":
                        work.stage = "done"
                        self.status.session(work, status="generated")
                        continue
                    work.stage = "selecting"
                    return ("select", work, None)
                if step == "shortfall":
                    have = self.parent_counts(work)
                    _, generated = self.humans_quota(work)
                    self.fail(work, f"not enough single-source candidates: "
                                    f"{sum(n for k, n in have.items() if k in work.qualified)}/{generated} from "
                                    f"{len(work.qualified)} qualified humans after their attempt budgets")
                continue
            if work.stage == "export":
                work.stage = "exporting"
                return ("export", work, None)
        return None

    # ----- task bodies (outside the lock) -----

    def run_job(self, work, job, gpu):
        self.say(work, f"{job['stage']} {job['id']} on GPU {gpu}")
        result = process_job(work.folder, job, gpu, self.stop)
        rows = []
        if result["stage"] == "generate" and result["accepted"] and not result["errors"]:
            path = job_recording(work.folder, result).resolve()
            origin = {"kind": "generated", "session": work.name, "job_id": result["id"],
                      "accepted": result["accepted"], "parent_source": result["input"],
                      "parent_episode": result["episode"]}
            fresh, rejected = gather_candidates({str(path): origin}, work.family, LIMITS)
            known = {r["fingerprint"] for r in work.rows}
            rows = [r for r in fresh if r["fingerprint"] not in known]
            work.rejected += rejected
        self.say(work, f"{job['id']}: {len(result['accepted'])} accepted / {result['attempts']} attempts"
                       + (f"; errors: {result['errors'][0][:200]}" if result["errors"] else ""))
        return result, rows

    def finish_job(self, work, job, result, rows):
        work.active.pop(job["id"], None)
        key = source_key(job)
        if job["stage"] == "validate":
            ok = bool(result["accepted"]) and not result["errors"]
            (work.qualified if ok else work.failed).add(key)
            if ok:
                work.validations[key] = result
        else:
            work.results[key].append(result)
            work.rows += rows
            if result["errors"]:
                recent = [r for results in work.results.values() for r in results][-3:]
                if len(recent) == 3 and all(r["errors"] for r in recent):
                    raise RuntimeError(f"three consecutive generation jobs failed; see {work.folder / 'jobs'}")

    def run_select(self, work):
        quota, generated = self.humans_quota(work)
        rows = [r for r in work.rows if r["kind"] == "collected"
                or (r["kind"] == "generated" and parent_key(r) in work.qualified)]
        selected, normalization = choose(rows, self.count, self.collected_count, balance_parents=True)
        counts = Counter(r["kind"] for r in selected)
        parents = Counter(label(parent_key(r)) for r in selected if r["kind"] == "generated")
        self.say(work, f"selected {dict(counts)}; generated per human: {dict(sorted(parents.items()))}")
        if self.dry_run:
            return None
        report = {"schema_version": 1, "status": "complete", "session": work.name, "family": work.family,
                  "selected_count": self.count, "counts_by_kind": dict(counts), "candidate_count": len(rows),
                  "quality_review": work.reviews, "rejected": work.rejected, "normalization": normalization,
                  "thresholds": LIMITS, "training_ready": False, "method": POOL_METHOD,
                  "generation_mode": "single_source", "generation_version": GENERATION_VERSION,
                  "qualified_human_sources": len(work.qualified),
                  "qualified_sources": [{"input": k[0], "episode": k[1]} for k in sorted(work.qualified)],
                  "generated_per_parent": dict(sorted(parents.items())),
                  "candidate_tier_shortfall": work.tier_shortfall}
        output = self.roots["pools"] / work.name
        export(output, selected, report)
        manifest = verify_pool(output, self.count)
        verify_lineage(output, manifest, work.qualified)
        self.status.pool(work, manifest)
        return manifest

    def run_export(self, work, gpu):
        manifest = verify_pool(self.roots["pools"] / work.name, self.count)
        freeze_training_plan(manifest, self.roots, self.count, self.collected_count)
        training = self.roots["training"] / work.name
        if not training.exists():
            self.status.session(work, status="processing_cameras", gpu=gpu)
            process_one(work.name, self.roots["collected"], self.roots["generated"], self.roots["training"],
                        self.roots["cache"], count=self.count, collected_count=self.collected_count,
                        device=f"cuda:{gpu}", write_workers=self.write_workers)
        self.verify_training(work, manifest)
        pps = self.roots["pps"] / work.name
        if not pps.exists():
            self.status.session(work, status="exporting_pps")
            pps_hdf5.export_session(self.roots["pools"] / work.name, training, self.roots["pps"])
        pps_hdf5.verify_session(pps, self.roots["pools"] / work.name, training)

    def worker(self, gpu):
        while True:
            with self.cv:
                while True:
                    task = None if self.stop.is_set() else self.next_task()
                    if task is not None or self.running == 0 or self.stop.is_set():
                        break
                    self.cv.wait(timeout=5)
                if task is None:
                    self.cv.notify_all()
                    return
                self.running += 1
            kind, work, job = task
            outcome, error = None, None
            try:
                if kind == "job":
                    outcome = self.run_job(work, job, gpu)
                elif kind == "select":
                    outcome = self.run_select(work)
                else:
                    self.run_export(work, gpu)
            except Interrupted as stopped:
                error = stopped
            except Exception as failure:  # one session's failure must not stop the others
                error = failure
                traceback.print_exc()
            with self.cv:
                self.running -= 1
                try:
                    if isinstance(error, Interrupted):
                        work.active.pop(job["id"], None) if job else None
                    elif error is not None:
                        work.active.pop(job["id"], None) if job else None
                        self.fail(work, f"{kind}: {error}")
                    elif kind == "job":
                        self.finish_job(work, job, *outcome)
                    elif kind == "select":
                        if self.stop_after == "select":
                            work.stage = "done"
                            self.status.session(work, status="pool_complete", selected_count=self.count,
                                                counts_by_kind=outcome["counts_by_kind"],
                                                pool=str(self.roots["pools"] / work.name / "dataset.json"))
                        else:
                            work.stage = "export"
                            self.status.session(work, status="pool_complete", selected_count=self.count,
                                                counts_by_kind=outcome["counts_by_kind"],
                                                pool=str(self.roots["pools"] / work.name / "dataset.json"))
                    else:
                        work.stage = "done"
                        self.say(work, f"final dataset {self.roots['training'] / work.name / 'training.hdf5'}; "
                                       f"PPS file {self.roots['pps'] / work.name / pps_hdf5.FILE_NAME}")
                        self.status.session(work, status="complete", generation_mode="single_source",
                                            training_file=str(self.roots["training"] / work.name / "training.hdf5"),
                                            pps_file=str(self.roots["pps"] / work.name / pps_hdf5.FILE_NAME))
                except Exception as failure:
                    self.fail(work, failure)
                self.cv.notify_all()

    def run(self, sessions):
        for session in sessions:
            self.prepare(session)
        if self.dry_run:
            for work in self.work:
                if work.stage == "qualify":
                    _, generated = self.humans_quota(work)
                    # Preview with past qualifications of this session's frozen eligible humans; a result
                    # under the current implementation overrides older ones.
                    known = ((self.evidence(work.name) & {source_key(s) for s in work.sources})
                             | work.qualified) - work.failed
                    eligible = sum(n for k, n in self.parent_counts(work).items() if k in known)
                    if eligible >= generated:
                        try:
                            work.qualified = known
                            self.run_select(work)
                        except ValueError as error:
                            self.say(work, f"selection not possible yet: {error}")
                    else:
                        self.say(work, f"would qualify/generate: {eligible}/{generated} generated candidates "
                                       "from previously qualified humans")
            return 0 if all(w.stage != "failed" for w in self.work) else 1
        threads = [threading.Thread(target=self.worker, args=(gpu,), name=f"gpu{gpu}") for gpu in self.gpus]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=1)
        except KeyboardInterrupt:
            self.log("Stopping: interrupting running jobs (Ctrl+C again to kill)...", flush=True)
            self.stop.set()
            signal_descendants(signal.SIGINT)
            try:
                for thread in threads:
                    thread.join()
            except KeyboardInterrupt:
                signal_descendants(signal.SIGKILL)
                raise
        incomplete = [w.name for w in self.work if w.stage not in ("done",)]
        for work in self.work:
            if work.stage not in ("done", "failed"):
                self.status.session(work, status="interrupted" if self.stop.is_set() else "incomplete")
        self.log(f"Checked {len(self.work)} sessions; {len(incomplete)} incomplete"
                 + (f": {', '.join(incomplete)}" if incomplete else "."), flush=True)
        return 0 if not incomplete else 1
