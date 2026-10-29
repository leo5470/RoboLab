"""Per-attempt, resumable camera jobs around the existing state renderer."""

from contextlib import contextmanager
from contextlib import nullcontext
import fcntl
import json
import os
from itertools import product
from numbers import Integral
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from .archive import ATTEMPT_ATTR
from .cameras import FRAME_MAPPING_GROUP
from .gpu_isolation import isolated_gpu_launch
from .io import fsync_directory, publish_directory, sha256_file, sync_file, write_json
from .validation import validate_recording


@contextmanager
def _gpu_slot(device, *, lock_dir=None):
    """Serialize this user's OMY renderer children on each selected GPU."""
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if match is None and device != "cpu":
        raise ValueError("Rendering device must be cpu, cuda or cuda:N")
    # CPU simulation still renders with Kit's default GPU 0.
    index = int(match.group(1) or 0) if match is not None else 0
    path = Path(lock_dir or tempfile.gettempdir()) / f"robolab_omy_render_{os.getuid()}_gpu{index}.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for another OMY renderer to release GPU {index}...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def compare_raw_channels(source, rendered, group="data/demo_0", *, max_block_bytes=1024**2):
    """Verify every original raw channel/attribute, not just trajectory identity.

    The deduplication fingerprint intentionally omits wall timing and event
    summaries. Those still belong to the immutable recording and must survive
    camera materialization exactly. Existing rendered image groups and derived
    frame-mapping provenance are regenerated; new camera/provenance nodes may
    be added. All other existing source nodes are required, including unknown
    legacy channels. Fixed-size array reads are tiled to the byte budget;
    a scalar or variable-length element necessarily remains indivisible.
    """
    if (isinstance(max_block_bytes, bool) or not isinstance(max_block_bytes, Integral)
            or max_block_bytes < 1):
        raise ValueError("Raw preservation read budget must be a positive integer")
    result = {"datasets_checked": 0, "attributes_checked": 0, "max_block_bytes": max_block_bytes}
    replaced = ("obs/image_obs", "obs/viewport_cam", FRAME_MAPPING_GROUP)

    def same_value(before, after):
        # No approximate comparisons: even diagnostic timestamp/validity
        # channels and event summaries must retain their original values.
        return np.array_equal(np.asarray(before), np.asarray(after))

    def attrs_equal(before, after, *, skip=()):
        for name in before.attrs:
            if name in skip:
                continue
            if name not in after.attrs:
                raise ValueError(f"Materialization dropped raw attribute {before.name}@{name}")
            if (before.attrs.get_id(name).dtype != after.attrs.get_id(name).dtype
                    or not same_value(before.attrs[name], after.attrs[name])):
                raise ValueError(f"Materialization changed raw attribute {before.name}@{name}")
            result["attributes_checked"] += 1

    def selections(shape, itemsize):
        if not shape:
            yield ()
            return
        if any(length == 0 for length in shape):
            return
        remaining = max(1, max_block_bytes // max(1, itemsize))
        tile = []
        for length in reversed(shape):
            width = min(length, max(1, remaining))
            tile.append(width)
            remaining = max(1, remaining // width)
        tile.reverse()
        starts = (range(0, length, width) for length, width in zip(shape, tile))
        for origin in product(*starts):
            yield tuple(slice(start, min(start + width, length))
                        for start, width, length in zip(origin, tile, shape))

    def inspect(before, after, relative=""):
        if relative and any(relative == path or relative.startswith(path + "/") for path in replaced):
            return
        if isinstance(before, h5py.Group):
            if not isinstance(after, h5py.Group):
                raise ValueError(f"Materialization changed raw group type {before.name}")
            attrs_equal(before, after)
            for name, child in before.items():
                child_path = f"{relative}/{name}" if relative else name
                if any(child_path == path or child_path.startswith(path + "/") for path in replaced):
                    continue
                if name not in after:
                    raise ValueError(f"Materialization dropped raw channel {child.name}")
                inspect(child, after[name], child_path)
            return
        if not isinstance(before, h5py.Dataset) or not isinstance(after, h5py.Dataset):
            raise ValueError(f"Materialization changed raw dataset type {before.name}")
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Materialization changed raw shape/dtype {before.name}")
        attrs_equal(before, after)
        for selection in selections(before.shape, before.dtype.itemsize):
            if not same_value(before[selection], after[selection]):
                raise ValueError(f"Materialization changed raw channel values {before.name}")
        result["datasets_checked"] += 1

    with h5py.File(source, "r") as original, h5py.File(rendered, "r") as materialized:
        attrs_equal(original, materialized)
        attrs_equal(original["data"], materialized["data"], skip=("total", "robolab_materialized"))
        if group not in original or group not in materialized:
            raise ValueError(f"Materialization lacks the original raw episode {group}")
        inspect(original[group], materialized[group])
    return result


def verify_render_cache(output, profile):
    """Reuse only acknowledged artifacts and unchanged resolved dependencies."""
    output = Path(output)
    if output.is_symlink() or (output / "manifest.json").is_symlink():
        raise ValueError("Symlink render caches/manifests are not trusted")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") != "rendered_for_review" or manifest.get("profile") != profile:
        raise FileExistsError(f"Existing render cache has a different source/profile: {output}")
    artifacts = manifest.get("artifact_sha256", {})
    if not {"images.hdf5", "env_cfg.json", "quality.json", "render.log"}.issubset(artifacts):
        raise ValueError("Render cache is missing required artifact acknowledgments")
    for name, expected in artifacts.items():
        path = output / name
        if Path(name).name != name or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"Existing render artifact failed checksum verification: {name}")
    dependencies = manifest.get("asset_dependencies")
    if not isinstance(dependencies, dict) or not isinstance(dependencies.get("files"), dict):
        raise ValueError("Render cache lacks its resolved asset-dependency snapshot")
    for filename, value in dependencies["files"].items():
        if sha256_file(filename) != value["sha256"]:
            raise ValueError(f"Resolved render dependency changed: {filename}")
    return manifest


def render_recording(source, output, *, group="data/demo_0", python=None, device="cuda:0",
                     renderer="realtime", camera_scale=1.0, camera_width=None, camera_height=None,
                     renders_per_frame=2, min_free_bytes=64 * 1024**2, runner=None, hdf5_compression="gzip"):
    """Render a native-15 attempt for review, never replaying physics actions.

    Warnings are retained: a successful render is not training certification.
    The simulator lives in a child process using the known-working environment.
    Failed temporary output and its log are retained, not advertised as complete.
    """
    import shutil

    source = Path(source).resolve()
    output = Path(output).absolute()
    if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, Integral) or min_free_bytes < 0:
        raise ValueError("min_free_bytes must be nonnegative")
    report = validate_recording(source)
    if report["errors"]:
        raise ValueError(f"Invalid source: {report['errors']}")
    matches = [e for e in report["episodes"] if e["group"] == group or e["name"] == group]
    if len(matches) != 1:
        raise ValueError(f"Unknown episode selection: {group}")
    episode = matches[0]
    if not episode["structural_valid"] or not episode["native_15_hz"]:
        raise ValueError(f"Rendering profile requires a complete native-15 attempt: {episode['errors']}")
    config_path = source.parent / "env_cfg.json"
    if not config_path.is_file():
        raise ValueError("env_cfg.json is required; no fallback to today's different task")
    if (renderer not in ("realtime", "pathtracing") or not 0 < camera_scale <= 1
            or isinstance(renders_per_frame, bool) or not isinstance(renders_per_frame, Integral)
            or renders_per_frame < 1):
        raise ValueError("Invalid rendering profile")
    if hdf5_compression not in ("gzip", "lzf", "none"):
        raise ValueError("Invalid HDF5 compression")
    if (camera_width is None) != (camera_height is None):
        raise ValueError("Camera width and height must be provided together")
    if camera_width is not None and (any(isinstance(value, bool) or not isinstance(value, Integral)
                                       or value < 1 for value in (camera_width, camera_height))
                                     or camera_scale != 1):
        raise ValueError("Explicit camera dimensions must be positive and cannot be combined with scaling")
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/materialize_demo_images.py"
    profile = {"schema_version": 1, "source_sha256": report["source"]["sha256"],
               "source_fingerprint": episode["fingerprint"], "source_group": episode["group"],
               "config_sha256": sha256_file(config_path), "materializer_sha256": sha256_file(script),
               "batch_worker_sha256": sha256_file(root / "robolab/core/omy_postprocess/batch_worker.py"),
               "renderer_helper_sha256": sha256_file(root / "robolab/core/replay/materialize.py"),
               "omy_camera_helper_sha256": sha256_file(root / "robolab/core/omy_postprocess/cameras.py"),
               "renderer": renderer, "camera_scale": camera_scale,
               "camera_width": camera_width, "camera_height": camera_height,
               "renders_per_frame": renders_per_frame,
               "hdf5_compression": hdf5_compression,
               "initial_warmup_renders": 8,
               "fps": 15, "alignment": "initial_state[0], then states[t-1]", "physics_replay": False,
               "profile_status": "provisional_review", "training_ready": False}
    # Capture top-level assets for cache invalidation. The materializer captures
    # resolved USD dependencies inside Kit; historical identity is not invented.
    config = json.loads(config_path.read_text())
    top_assets = {}

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("usd_path", "texture_file") and isinstance(child, str) and child.startswith("/"):
                    path = Path(child)
                    if not path.is_file():
                        raise FileNotFoundError(f"Recorded asset is unavailable: {child}; use verified explicit remapping")
                    top_assets[str(path)] = sha256_file(path)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(config)
    profile["top_level_asset_sha256"] = top_assets
    if output.exists() or output.is_symlink():
        verify_render_cache(output, profile)
        compare_raw_channels(source, output / "images.hdf5", episode["group"])
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < min_free_bytes:
        raise OSError("Insufficient space for rendering; source retained")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        gpu = isolated_gpu_launch(device)
        command = [*gpu.prefix, str(python or sys.executable), "-u", str(script), "--input", str(source),
                   "--output", str(temporary / "images.hdf5"), "--episode", episode["name"].removeprefix("demo_"),
                   "--device", gpu.child_device, "--headless", "--require-native15", "--renderer", renderer,
                   "--camera-scale", str(camera_scale), "--renders-per-frame", str(renders_per_frame),
                   "--hdf5-compression", hdf5_compression,
                   "--kit_args", "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
                                 "--/renderer/activeGpu=0 --/physics/cudaDevice=0"]
        if camera_width is not None:
            command.extend(["--camera-width", str(camera_width), "--camera-height", str(camera_height)])
        print("Rendering recorded states in a separate headless process; see", temporary / "render.log", flush=True)
        with (_gpu_slot(device) if runner is None else nullcontext()), (temporary / "render.log").open("x") as log:
            log.write(f"GPU isolation: {json.dumps(gpu.metadata, sort_keys=True)}\n")
            log.flush()
            (runner or subprocess.run)(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, env=gpu.env)
        if sha256_file(source) != profile["source_sha256"] or sha256_file(config_path) != profile["config_sha256"]:
            raise RuntimeError("Source/configuration changed during rendering")
        for path, expected in top_assets.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"Asset changed during rendering: {path}")
        rendered_report = validate_recording(temporary / "images.hdf5")
        if rendered_report["errors"] or len(rendered_report["episodes"]) != 1:
            raise ValueError("Rendered file failed structural validation")
        rendered_episode = rendered_report["episodes"][0]
        if not rendered_episode["structural_valid"] or rendered_episode["fingerprint"] != episode["fingerprint"]:
            raise ValueError("Materialization changed the raw numerical trajectory")
        preserved_raw = compare_raw_channels(source, temporary / "images.hdf5", episode["group"])
        with h5py.File(temporary / "images.hdf5", "r") as file:
            demo = file[episode["group"]]
            mapping = demo[FRAME_MAPPING_GROUP]
            if (mapping.attrs.get("schema_version") != 1
                    or mapping.attrs.get("source_episode") != episode["name"]
                    or mapping.attrs.get("source_sha256") != profile["source_sha256"]
                    or mapping.attrs.get("fps") != 15
                    or not mapping.attrs.get("native_15_hz")
                    or bool(mapping.attrs.get("physics_replay", True))):
                raise ValueError("Camera mapping provenance differs from the native-15 source/profile")
            if not np.array_equal(mapping["source_step"][:], np.arange(episode["sample_count"])):
                raise ValueError("Camera/source-step mapping differs from native-15 actions")
            if not np.array_equal(mapping["post_state_row"][:], np.arange(episode["sample_count"]) - 1):
                raise ValueError("Camera states are not pre-action aligned")
            if not np.allclose(mapping["timestamp_s"][:], episode["timing"]["timestamps_s"], rtol=0, atol=1e-8):
                raise ValueError("Camera timestamps differ from validated simulation-time grid")
            for name in ("before", "after"):
                if not np.array_equal(mapping[f"recorded_sim_{name}_s"][:], demo[f"timing/sim_{name}_s"][:]):
                    raise ValueError("Camera mapping discarded or changed original simulation timestamps")
            images = demo.get("obs/image_obs")
            if images is None or set(images) != {"over_shoulder_left_camera", "wrist_cam"}:
                raise ValueError("Both policy cameras must be rendered")
            camera_shapes = {key: list(value.shape) for key, value in images.items()}
            if any(shape[0] != episode["sample_count"] for shape in camera_shapes.values()):
                raise ValueError("Camera frame counts differ from action count")
            for name, value in images.items():
                if value.dtype != np.dtype("uint8") or value.ndim != 4 or value.shape[-1] != 3:
                    raise ValueError(f"Camera {name} must have uint8 RGB [N,H,W,3] data")
                if camera_width is not None and value.shape[1:3] != (camera_height, camera_width):
                    raise ValueError(f"Camera {name} resolution differs from the explicit profile")
            identity = file["data"].attrs.get(ATTEMPT_ATTR)
            provenance = json.loads(file["data"].attrs["robolab_materialized"])
            if (provenance.get("alignment") != "pre_action"
                    or provenance.get("source_sha256") != profile["source_sha256"]
                    or not provenance.get("omy_recording")):
                raise ValueError("Materialization provenance does not match the pre-action OMY source")
            dependencies = provenance["asset_dependencies"]
        for filename, value in dependencies["files"].items():
            if sha256_file(filename) != value["sha256"]:
                raise RuntimeError(f"Resolved asset changed during rendering: {filename}")
        write_json(temporary / "quality.json", rendered_episode)
        artifacts = {p.name: sha256_file(p) for p in temporary.iterdir() if p.is_file()}
        manifest = {"schema_version": 1, "status": "rendered_for_review", "training_ready": False,
                    "profile": profile, "source": str(source), "camera_shapes": camera_shapes,
                    "gpu_execution": gpu.metadata,
                    "asset_dependencies": dependencies,
                    "raw_preservation": preserved_raw,
                    "attempt": json.loads(identity) if identity else None,
                    "artifact_sha256": artifacts}
        write_json(temporary / "manifest.json", manifest)
        for artifact in temporary.iterdir():
            if artifact.is_file():
                sync_file(artifact)
        fsync_directory(temporary)
        verify_render_cache(temporary, profile)
        if sha256_file(source) != profile["source_sha256"] or sha256_file(config_path) != profile["config_sha256"]:
            raise RuntimeError("Source/configuration changed before render publication")
        publish_directory(temporary, output)
        return output
    except BaseException as error:
        if temporary.exists() and not (temporary / "failure.json").exists():
            write_json(temporary / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
