#!/usr/bin/env python3
"""Compare OMY run summaries without launching Isaac Sim."""

import argparse
import json
from pathlib import Path


def load_run(path):
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    report = json.loads((directory / "omy_performance.json").read_text())
    cfg = json.loads((directory / "env_cfg.json").read_text())
    window = report["last_window"]
    if not window:
        raise ValueError(f"{directory}: no completed control-step timing window")
    settings = report["settings"]
    regions = window["region_ms_mean"]
    xr_path = directory / "xr_status.json"
    xr = json.loads(xr_path.read_text()) if xr_path.is_file() else {}
    sensor_count = settings.get("contact_sensor_count")
    contact_cache = report.get("contact_cache") or {}
    return {
        "run": directory.name,
        "task": cfg.get("_task_name", "?"),
        "gravity": cfg["sim"]["gravity"],
        "dt": cfg["sim"]["dt"],
        "decimation": cfg["decimation"],
        "render_interval": cfg["sim"]["render_interval"],
        "physics_device": settings.get("physics_device"),
        "render_gpu": settings.get("kit", {}).get("/renderer/activeGpu"),
        "xr_scale": xr.get("xr_resolution_multiplier_override"),
        "profile": settings.get("contact_sensor_profile", "full"),
        "sensors": sensor_count if sensor_count is not None else len(settings.get("contact_history_lengths", {})),
        "hz": window["control_hz"],
        "rtf": window["real_time_factor"],
        "p95_ms": window["work_ms_p95"],
        "physics_ms": regions.get("physics", 0.0),
        "render_ms": regions.get("render", 0.0),
        "sensors_ms": regions.get("sensors", 0.0),
        "checks_ms": regions.get("task_checks", 0.0),
        "batch_computed": contact_cache.get("computed", 0),
        "batch_reused": contact_cache.get("reused", 0),
        "batch_fallback": contact_cache.get("fallback", 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="OMY output directories or omy_performance.json files")
    args = parser.parse_args()
    runs = [load_run(path) for path in args.runs]
    keys = ("run", "task", "profile", "sensors", "hz", "rtf", "p95_ms",
            "physics_ms", "render_ms", "sensors_ms", "checks_ms",
            "batch_computed", "batch_reused", "batch_fallback")
    print("\t".join(keys))
    for run in runs:
        print("\t".join(f"{run[key]:.3f}" if isinstance(run[key], float) else str(run[key])
                        for key in keys))
    if len(runs) > 1:
        first = runs[0]
        for run in runs[1:]:
            comparable = ("task", "gravity", "dt", "decimation", "render_interval",
                          "physics_device", "render_gpu", "xr_scale")
            differences = [key for key in comparable if run[key] != first[key]]
            if differences:
                print(f"WARNING: {run['run']} differs from {first['run']} in: {', '.join(differences)}")


if __name__ == "__main__":
    main()
