"""Viewer options and read-only OpenXR preflight; no simulator imports."""

import argparse
import json
import math
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="View a frozen RoboLab scene through SteamVR/OpenXR; no robot control.")
    parser.add_argument("--task", default="BananaInBowlTask")
    add_view_arguments(parser)
    parser.add_argument("--device", default="cpu", help="Physics initialization device; no physics steps run after reset.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N render updates (0: run until closed).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="New/empty directory for configuration and viewer diagnostics.")
    return parser


def add_view_arguments(parser, *, desktop_help="Scene-only preview without starting XR; not a headset test."):
    """Share runtime, renderer and viewpoint options with the L100 XR launcher."""
    parser.add_argument("--check", action="store_true", help="Check runtime/display setup without launching Isaac Sim.")
    parser.add_argument("--runtime-json", type=Path, help="SteamVR steamxr_linux64.json (including custom Steam libraries).")
    parser.add_argument("--desktop", action="store_true", help=desktop_help)
    parser.add_argument("--headless", action="store_true", help="Hide Isaac's window; SteamVR still needs its host desktop session.")
    parser.add_argument("--render-gpu", type=int, default=0, help="Kit renderer GPU index; use the same GPU as SteamVR/ALVR.")
    parser.add_argument("--xr-resolution-multiplier", type=float, default=None,
                        help="Native XR render-buffer multiplier (0.1–2.0); omitted preserves Kit's current setting. "
                             "Independent of desktop/camera resolution and SteamVR's percentage.")
    parser.add_argument("--cpu-workers", type=int, default=8, help="Kit and TBB worker limits (default: 8).")
    parser.add_argument("--physics-workers", type=int, default=2, help="CPU physics worker limit (default: 2).")
    # The robot starts at X=0 and reaches toward the task table around X=0.5.
    # Put the standing viewer just behind the base, with the work area in a
    # comfortable forward/downward gaze instead of far below the headset.
    # Raise both eye and target by 0.30 m to preserve the viewing direction.
    parser.add_argument("--eye", nargs=3, type=float, default=(-0.25, 0.0, 0.95), metavar=("X", "Y", "Z"))
    parser.add_argument("--lookat", nargs=3, type=float, default=(0.53, 0.0, 0.48), metavar=("X", "Y", "Z"))
    parser.add_argument("--xr-timeout", type=float, default=60.0, help="Seconds after scene load to wait for an active XR display.")


def validate_args(parser, args):
    if args.xr_resolution_multiplier is not None and (
        not math.isfinite(args.xr_resolution_multiplier) or not 0.1 <= args.xr_resolution_multiplier <= 2.0
    ):
        parser.error("--xr-resolution-multiplier must be finite and between 0.1 and 2.0")
    if args.desktop and args.xr_resolution_multiplier is not None:
        parser.error("--xr-resolution-multiplier requires XR (omit --desktop)")
    if args.cpu_workers < 1 or args.physics_workers < 1:
        parser.error("--cpu-workers and --physics-workers must be positive")
    if getattr(args, "frames", 0) < 0 or args.render_gpu < 0:
        parser.error("--frames and --render-gpu must be nonnegative")
    if not math.isfinite(args.xr_timeout) or args.xr_timeout <= 0:
        parser.error("--xr-timeout must be finite and positive")
    if not all(math.isfinite(x) for x in (*args.eye, *args.lookat)):
        parser.error("--eye and --lookat must be finite")
    delta = [b - a for a, b in zip(args.eye, args.lookat)]
    if math.hypot(delta[0], delta[1]) < 1e-6:
        parser.error("Viewing direction must not be zero or parallel to world Z (the up axis)")
    if args.device != "cpu" and not (args.device == "cuda" or (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    )):
        parser.error("--device must be cpu, cuda, or cuda:N")


def find_runtime(explicit=None, *, environ=None, home=None):
    """Find a manifest without changing the system-wide default runtime.

    An explicit path or XR_RUNTIME_JSON must be valid; never silently fall back
    from an invalid override. Standard Steam locations are only fallbacks.
    """
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    override = explicit or environ.get("XR_RUNTIME_JSON")
    if override:
        candidates = [Path(override).expanduser()]
    else:
        config = Path(environ.get("XDG_CONFIG_HOME", home / ".config"))
        data = Path(environ.get("XDG_DATA_HOME", home / ".local/share"))
        candidates = [config / "openxr/1/active_runtime.json"]
        candidates += [Path(p) / "openxr/1/active_runtime.json" for p in
                       environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":") if p]
        candidates += [root / "steamapps/common/SteamVR/steamxr_linux64.json" for root in (
            data / "Steam", home / ".steam/steam", home / ".steam/debian-installation",
        )]
    for candidate in candidates:
        if not candidate.is_file() and not override:
            continue
        try:
            manifest = candidate.resolve(strict=True)
            data = json.loads(manifest.read_text())
            library = data["runtime"]["library_path"]
            if not isinstance(library, str) or not library:
                raise ValueError("runtime.library_path must be a nonempty string")
            library_path = Path(library)
            if not library_path.is_absolute():
                library_path = manifest.parent / library_path
            if not library_path.is_file():
                raise ValueError(f"Runtime library not found: {library_path}")
            # This entry point is intentionally for SteamVR, not bundled CloudXR.
            identity = f"{manifest} {library} {data['runtime'].get('name', '')}".lower()
            if "steam" not in identity or "cloudxr" in identity:
                raise ValueError("Expected a SteamVR OpenXR manifest; pass --runtime-json to select it explicitly")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid OpenXR runtime {candidate}: {exc}") from exc
    raise ValueError("SteamVR OpenXR runtime not found. Install/start SteamVR, then pass "
                     "--runtime-json /path/to/SteamVR/steamxr_linux64.json. See docs/quest_viewing.md.")


def preflight(args, *, environ=None):
    environ = os.environ if environ is None else environ
    errors, warnings = [], []
    runtime = None
    if not args.desktop:
        try:
            runtime = find_runtime(args.runtime_json, environ=environ)
        except ValueError as exc:
            errors.append(str(exc))
        if not environ.get("DISPLAY") and not environ.get("WAYLAND_DISPLAY"):
            errors.append("No host desktop session detected. Run beside SteamVR on the simulation host's "
                          "GPU-backed desktop, not a bare SSH shell (even with --headless).")
        display = environ.get("DISPLAY", "")
        if display.startswith(("localhost:", "127.0.0.1:")):
            errors.append("DISPLAY looks SSH-forwarded. SteamVR needs the host GPU-backed desktop, not SSH X11 forwarding.")
    warnings.append("A valid manifest does not prove SteamVR, ALVR, USB, or the headset is connected.")
    return {"mode": "desktop" if args.desktop else "steamvr", "runtime_json": str(runtime) if runtime else None,
            "errors": errors, "warnings": warnings}


def launcher_options(args):
    # Use Isaac Lab's own XR experience, add native VR, and suppress its AR
    # autostart. No new stereo renderer, cameras, or encoder is implemented here.
    settings = {
        "/app/settings/persistent": "false",
        "/renderer/multiGpu/enabled": "false",
        "/renderer/activeGpu": args.render_gpu,
        "/plugins/carb.tasking.plugin/threadCount": args.cpu_workers,
        "/plugins/omni.tbb.globalcontrol/maxThreadCount": args.cpu_workers,
        "/persistent/physics/numThreads": args.physics_workers,
        "/xr/profile/ar/enabled": "false",
        "/xr/profile/vr/enabled": "false",
    }
    if not args.desktop:
        # Apply before scene loading: toggling texture streaming later requires
        # a scene reload. Keep textures resident to avoid XR texture pop-in;
        # this trades increased VRAM usage for stable surface detail.
        settings["/rtx-transient/resourcemanager/enableTextureStreaming"] = "false"
    kit_args = " ".join(f"--{key}={value}" for key, value in settings.items())
    if not args.desktop:
        kit_args += " --enable omni.kit.xr.profile.vr"
    # The no-camera headless experience has no renderer. Use the native desktop
    # experience with its window hidden for a real rendered smoke test, rather
    # than enabling only a viewport without its RTX renderer dependencies.
    experience = "isaaclab.python.kit" if args.desktop and args.headless else ""
    return dict(headless=args.headless, device=args.device, device_explicit=True,
                xr=not args.desktop, enable_cameras=False, video=True,
                rendering_mode="performance", kit_args=kit_args, experience=experience)
