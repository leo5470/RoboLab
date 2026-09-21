#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render camera observations for a state-only recording, offline.

Keyboard demos collected with ``--defer-images`` hold actions and per-step scene
states but no images, so teleoperation never waits on RTX renders. This script
turns such a recording into an ordinary RoboLab image demonstration: it rebuilds
the environment with its policy cameras, walks the recorded states, renders each
one, and writes the observations under the usual ``obs/image_obs/<camera>``
paths. Every other recorded channel is copied through unchanged.

Frames are rendered from the recorded states, never by re-stepping the recorded
actions: contact-rich trajectories diverge under open-loop replay, and the
images would then drift away from the trajectory stored beside them.

Usage:
    python scripts/materialize_demo_images.py \\
        --input output/keyboard_demos/state_only/keyboard_demos.hdf5 \\
        --output output/keyboard_demos/final/keyboard_demos.hdf5 \\
        --device cuda:0 --hdf5-compression gzip

    # One episode, at half the configured camera resolution
    python scripts/materialize_demo_images.py --input in.hdf5 --output out.hdf5 \\
        --episode 0 --camera-scale 0.5

The recording's ``env_cfg.json`` sidecar must sit next to the input file; it is
overlaid onto a freshly built config so the scene matches what was recorded
(see docs/replay.md).
"""

# isort: skip_file
import argparse
import os
import shutil
import sys
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # Must be imported before isaaclab. Do not remove.  # noqa: F401
import h5py
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Render policy-camera observations for a state-only RoboLab recording."
)
parser.add_argument("--input", "-i", required=True, help="State-only recording (HDF5) to materialize.")
parser.add_argument("--output", "-o", required=True, help="Path of the image demonstration to write.")
parser.add_argument(
    "--episode",
    type=int,
    action="append",
    dest="episodes",
    help="Demo index to materialize; repeatable. Default: every episode in the input.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace the output file if it already exists (refused by default).",
)
parser.add_argument(
    "--hdf5-compression",
    choices=("lzf", "gzip", "none"),
    default="gzip",
    help="Compression for the rendered image datasets (default: gzip). Copied channels keep "
         "the compression they were recorded with.",
)
parser.add_argument(
    "--camera-scale",
    type=float,
    default=1.0,
    help="Scale policy-camera width and height (default: 1.0, the configured resolution).",
)
parser.add_argument("--task", help="Task class name. Default: the task named in the recording.")
parser.add_argument(
    "--record-viewport-camera",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Also render the third-person viewport_cam group. Default: whatever the recording asked for.",
)
parser.add_argument(
    "--renderer",
    choices=("realtime", "pathtracing"),
    default="realtime",
    help="RTX renderer for the offline pass (default: realtime). Materialization is not "
         "latency-bound, so 'pathtracing' is affordable here.",
)
parser.add_argument(
    "--renders-per-frame",
    type=int,
    default=1,
    help="sim.render() calls per frame (default: 1, matching the one render per control step a "
         "live recording performs). More passes let the renderer's temporal accumulation converge "
         "further on each frame, at proportional cost.",
)
parser.add_argument(
    "--gui",
    action="store_true",
    help="Show the Isaac Sim window. Off by default: this is a batch tool and rendering is offscreen.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Fail fast on bad paths: a typo should cost a second, not a minute of Isaac Sim
# startup. These are a UX fast path only — the output file is written through
# ``atomic_hdf5_output``, which enforces the same overwrite rule when it opens
# the destination, so a file that appears in between is still refused.
if not os.path.isfile(args_cli.input):
    parser.error(f"--input file not found: {args_cli.input}")
if os.path.abspath(args_cli.input) == os.path.abspath(args_cli.output):
    parser.error("--output must differ from --input; materialization never writes in place")
if os.path.exists(args_cli.output) and not args_cli.overwrite:
    parser.error(f"--output already exists: {args_cli.output}. Pass --overwrite to replace it.")
if args_cli.renders_per_frame < 1:
    parser.error("--renders-per-frame must be positive")
if not 0 < args_cli.camera_scale <= 1:
    parser.error("--camera-scale must be greater than 0 and at most 1")

# Cameras must render; the window is opt-in.
args_cli.enable_cameras = True
if not args_cli.gui and not getattr(args_cli, "livestream", 0):
    args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from tqdm import tqdm  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.replay import apply_recorded_env_cfg, load_recorded_env_cfg  # noqa: E402
from robolab.core.replay.materialize import (  # noqa: E402
    IMAGE_OBS_GROUPS,
    RecordedEpisode,
    atomic_hdf5_output,
    camera_extrinsics_from_env,
    copy_episode,
    episode_names,
    materialize_episode,
    read_deferred_metadata,
    restore_nulled_cameras,
    scale_camera_resolutions,
    stamp_materialization_provenance,
    write_camera_extrinsics,
)
from robolab.core.utils.version_utils import warn_on_stack_mismatch  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import (  # noqa: E402
    auto_register_droid_rel_ik_envs,
)


def _selected_episodes(available: list[str]) -> list[str]:
    """Resolve --episode indices to demo names, preserving file order."""
    if not args_cli.episodes:
        return available
    selected = []
    for index in args_cli.episodes:
        name = f"demo_{index}"
        if name not in available:
            raise ValueError(f"--episode {index} not in {args_cli.input} (has {', '.join(available)})")
        if name not in selected:
            selected.append(name)
    return [name for name in available if name in selected]


def _build_env(metadata: dict):
    """Rebuild the recording's environment with its policy cameras restored.

    The state-only config has no camera entries to restore, so the cameras come
    from a fresh registration; the recorded ``env_cfg.json`` is then overlaid so
    every other value (object poses, physics, terminations, instruction) is the
    one the episode was recorded with.
    """
    task = args_cli.task or metadata.get("task")
    if not task:
        raise ValueError(
            "the recording does not name its task and --task was not given; "
            "pass --task <TaskName> to rebuild the environment"
        )
    viewport = args_cli.record_viewport_camera
    if viewport is None:
        viewport = bool(metadata.get("viewport_camera", False))

    auto_register_droid_rel_ik_envs(
        task=task,
        env_postfix=metadata.get("env_postfix", "RelIK"),
        include_viewport_camera=viewport,
        include_policy_cameras=True,
    )
    env_names = get_envs(task=task)
    if not env_names:
        raise ValueError(f"No relative-IK environment found for task '{task}'")
    recorded_env_name = metadata.get("env_name")
    env_name = recorded_env_name if recorded_env_name in env_names else env_names[0]
    if recorded_env_name and recorded_env_name != env_name:
        print(f"\033[93mNOTE: the recording names env '{recorded_env_name}', which is not "
              f"registered here; rendering with '{env_name}' instead.\033[0m")

    env_cfg = parse_env_cfg(
        env_name,
        device=args_cli.device,
        seed=metadata.get("base_seed", 0),
        num_envs=1,
        use_fabric=True,
    )

    loaded = load_recorded_env_cfg(args_cli.input)
    if loaded is None:
        print(f"\033[93mWARNING: no env_cfg.json next to {args_cli.input}; rendering with the "
              "current repo's task definitions. If task or scene definitions changed since "
              "recording, the rendered scene is not the one that was recorded.\033[0m")
    else:
        recorded_cfg, sidecar_path = loaded
        # The state-only config removed its cameras by setting them to null;
        # keep the ones we just registered instead of overlaying those nulls.
        restored = restore_nulled_cameras(env_cfg, recorded_cfg)
        skipped = apply_recorded_env_cfg(env_cfg, recorded_cfg)
        print(f"Restored recorded env config from {sidecar_path}")
        if restored:
            print(f"Re-enabled cameras the recording omitted: {', '.join(sorted(restored))}")
        if skipped:
            print(f"\033[93mWARNING: {len(skipped)} recorded config field(s) no longer match the "
                  "current config schema and were kept at their current values:\n  "
                  + "\n  ".join(skipped) + "\033[0m")

    resolutions = scale_camera_resolutions(env_cfg, args_cli.camera_scale)
    if not resolutions:
        raise ValueError(
            "the rebuilt environment has no camera sensors, so there is nothing to render. "
            "Check that the task registers policy cameras."
        )
    env, _ = create_env(
        env_cfg,
        policy="materialize_demo_images",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_mode,
    )
    print("Cameras: " + ", ".join(f"{name}={w}x{h}" for name, (w, h) in sorted(resolutions.items())))
    return env, resolutions


def _describe(demo_group: h5py.Group) -> str:
    """One-line-per-dataset dump of a demo's final structure."""
    lines = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            lines.append(f"    {name}: {tuple(obj.shape)} {obj.dtype}")

    demo_group.visititems(visit)
    return "\n".join(sorted(lines))


def main() -> int:
    compression = None if args_cli.hdf5_compression == "none" else args_cli.hdf5_compression
    output_dir = os.path.dirname(os.path.abspath(args_cli.output)) or "."
    os.makedirs(output_dir, exist_ok=True)
    # The env writes its own env_cfg.json (and any stray recorder output) into a
    # scratch directory, so materialization never clobbers the recording's
    # sidecar; the config is copied next to the output on success.
    scratch_dir = os.path.join(output_dir, f".materialize_{os.getpid()}")
    robolab.constants.set_output_dir(scratch_dir)
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False

    warn_on_stack_mismatch(args_cli.input)
    metadata = read_deferred_metadata(args_cli.input)
    if metadata:
        print(f"Recording: state-only, task '{metadata.get('task')}', "
              f"control {metadata.get('control_hz', float('nan')):.1f} Hz "
              f"(sim dt {metadata.get('sim_dt')}, decimation {metadata.get('decimation')})")
    else:
        print(f"\033[93mNOTE: {args_cli.input} carries no deferred-image metadata (not collected "
              "with --defer-images). Its recorded states will still be rendered; pass --task if "
              "the environment cannot be resolved.\033[0m")

    available = episode_names(args_cli.input)
    selected = _selected_episodes(available)
    if not selected:
        print(f"{args_cli.input} contains no episodes; nothing to materialize.")
        return 1
    print(f"Episodes: materializing {len(selected)} of {len(available)} ({', '.join(selected)})")

    env = None
    materialized = 0
    try:
        env, resolutions = _build_env(metadata)
        env.reset()

        # The output is built beside its destination and moved into place only
        # once every episode has been rendered, so an interrupted run never
        # leaves a half-rendered demonstration behind.
        with h5py.File(args_cli.input, "r") as src, \
                atomic_hdf5_output(args_cli.output, overwrite=args_cli.overwrite) as dst:
            src_data = src["data"]
            dst_data = dst.create_group("data")
            for key, value in src_data.attrs.items():
                dst_data.attrs[key] = value

            total_samples = 0
            for name in selected:
                episode = RecordedEpisode.load(src_data[name])
                recorded_samples = int(src_data[name].attrs.get("num_samples", episode.num_actions))
                if recorded_samples != episode.num_actions:
                    print(f"\033[93mNOTE: {name} records num_samples={recorded_samples} but has "
                          f"{episode.num_actions} actions; using the action count.\033[0m")

                # Reset with the recorded seed so reset-time randomization
                # (lighting, camera pose) reproduces the recorded scene before
                # the recorded states are written over it.
                if episode.seed is not None:
                    env.reset(seed=episode.seed)
                else:
                    env.reset()

                demo = copy_episode(src_data[name], dst_data, name)
                progress = tqdm(total=episode.num_actions, desc=f"{name} frames", unit="frame")
                try:
                    shapes = materialize_episode(
                        env, episode, demo,
                        compression=compression,
                        renders_per_frame=args_cli.renders_per_frame,
                        progress=lambda _step: progress.update(1),
                    )
                finally:
                    progress.close()
                write_camera_extrinsics(demo, camera_extrinsics_from_env(env))
                demo.attrs["num_samples"] = episode.num_actions
                total_samples += episode.num_actions
                materialized += 1

                image_frames = {path: shape[0] for path, shape in shapes.items()}
                print(f"{name}: {episode.num_actions} actions, "
                      f"{episode.num_actions} state rows, "
                      f"{sorted(set(image_frames.values()))} image frames, "
                      f"success={episode.success}, seed={episode.seed}")
                for path, shape in sorted(shapes.items()):
                    print(f"    obs/{path}: {shape}")

            dst_data.attrs["total"] = total_samples
            stamp_materialization_provenance(
                dst_data, args_cli.input,
                extra={
                    "episodes": selected,
                    "camera_resolutions": {name: list(res) for name, res in resolutions.items()},
                    "renders_per_frame": args_cli.renders_per_frame,
                    "image_groups": list(IMAGE_OBS_GROUPS),
                },
            )

        print(f"\nWrote {materialized} materialized episode(s) to {args_cli.output}")

        sidecar = os.path.join(output_dir, "env_cfg.json")
        scratch_cfg = os.path.join(scratch_dir, "env_cfg.json")
        recording_dir = os.path.dirname(os.path.abspath(args_cli.input))
        if os.path.isfile(scratch_cfg) and os.path.abspath(output_dir) != recording_dir:
            shutil.copyfile(scratch_cfg, sidecar)
            print(f"Wrote {sidecar} (the env config the images were rendered with)")
        else:
            print(f"\033[93mNOTE: output shares a directory with the recording, so its "
                  f"env_cfg.json was left as recorded. Replay of {args_cli.output} uses the "
                  "state-only config, which has no camera entries.\033[0m")

        with h5py.File(args_cli.output, "r") as final:
            print("\nFinal HDF5 structure:")
            print(f"  data attrs: {sorted(final['data'].attrs)}")
            for name in selected:
                print(f"  {name} attrs: {dict(final['data'][name].attrs)}")
                print(_describe(final["data"][name]))
        return 0
    finally:
        if env is not None:
            env.close()
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    except Exception as exc:
        print(f"Materialization failed: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
