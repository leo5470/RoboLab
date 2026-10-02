"""Render the reset scene without stepping physics or accepting robot actions."""

import json
import time
from datetime import datetime
from pathlib import Path


def prepare_output(path):
    path = Path(path) if path else Path("output/quest_view") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def render_loop(app, render, view, *, frames=0, timeout=60.0, clock=time.monotonic):
    """No step()/advance()/controller calls; host updates are not headset FPS."""
    count = 0
    start = report_start = clock()
    report_count = 0
    while app.is_running() and (not frames or count < frames):
        render()
        count += 1
        now = clock()
        if view is not None:
            if view.update():
                print("XR display active; initial view placed." if view.active else
                      "XR display inactive; scene remains frozen.", flush=True)
            if not view.ever_active and now - start > timeout:
                raise RuntimeError("XR display did not become active. Check SteamVR Home in the headset, "
                                   "ALVR USB mode, and the SteamVR OpenXR runtime. Scene load alone is not XR success.")
        if now - report_start >= 5:
            print(f"Viewer: {(count - report_count) / (now - report_start):.1f} host render updates/s "
                  f"(not headset FPS); XR active={bool(view and view.active)}; physics frozen", flush=True)
            report_start, report_count = now, count
    if view is not None and not view.ever_active:
        raise RuntimeError("Viewer ended before an XR display became active; headset connection was not validated")
    return count


def run(args, app, preflight):
    import carb
    import torch

    import robolab.constants
    from robolab.core.environments.runtime import create_env

    from .native import create_native_view
    from .scene import build_scene_cfg

    output = prepare_output(args.output_dir)
    robolab.constants.set_output_dir(str(output))
    robolab.constants.RECORD_IMAGE_DATA = False
    # Render the Kit viewport in headless desktop diagnostics as well. This does
    # not create a sensor or enable Replicator camera observations.
    if args.desktop and args.headless:
        carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
    cfg = build_scene_cfg(args)
    env = view = None
    result = {"mode": preflight["mode"], "runtime_json": preflight["runtime_json"],
              "task": args.task, "robot_control": "disabled", "physics": "frozen_after_reset",
              "eye": args.eye, "lookat": args.lookat, "render_gpu": args.render_gpu,
              "xr_resolution_multiplier_override": args.xr_resolution_multiplier,
              "xr_display_seen": False, "completed": False}
    try:
        with torch.inference_mode():
            env, _ = create_env(cfg, policy="view_only", rendering_mode="performance")
            if env.action_manager.total_action_dim != 0:
                raise RuntimeError("Viewing-only environment unexpectedly has robot actions")
            env.reset(seed=args.seed)
            # Refresh transforms once; from this point only render() is called.
            env.sim.forward()
            settings = carb.settings.get_settings()
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
            settings.set_bool("/app/player/playSimulations", False)
            if not args.desktop:
                view = create_native_view(args, preflight["runtime_json"])
                view.start()
            print(f"Scene ready: {args.task}. No robot control, camera sensors, or physics stepping.\n"
                  f"Output: {output}", flush=True)
            result["host_render_updates"] = render_loop(
                app, env.sim.render, view, frames=args.frames, timeout=args.xr_timeout,
            )
            result["completed"] = True
        return 0
    finally:
        try:
            if view is not None:
                result["xr_display_seen"] = view.ever_active
                view.close()
                # Process the native disable request before destroying the scene.
                if env is not None:
                    env.sim.render()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                (output / "viewer_status.json").write_text(json.dumps(result, indent=2) + "\n")
