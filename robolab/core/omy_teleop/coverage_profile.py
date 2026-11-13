"""Explicit-target profiles for any object, derived from its evaluation layouts' overall distribution.

``target_sampler.PROFILES`` holds the hand-measured plastic pail profile. This
module derives the same kind of profile for another object from the 50
authored Simple-PnP v2 layouts of that object, using only distribution-level
numbers, so no individual evaluation layout is targeted:

- the XY boxes of the object and the container, and the range of their distance;
- the arc of object yaw the layouts use (a full circle unless they leave a gap
  of ``FULL_CIRCLE_GAP_DEG`` or more, as the handle-visibility rule does for jugs);
- the share of layouts in each object-x band, with band weights calibrated so
  that targets surviving the v1 layout screens keep those shares.

Yaw is measured relative to the v1 authored rotation in ``geometry``, which is
what ``Sampler.pose`` composes a sampled yaw with.

CLI (needs USD; run with tools/simple_pnp_v2/usd_python.sh -m ...):
  layouts  write the authored poses of scenes (evaluation layouts)
  profile  derive a profile from a layouts file and the v1 scenes
  targets  sample a target manifest from a profile file
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .target_sampler import (EDGE_CLEARANCE_M, EXCLUDE_M, PAIR_CLEARANCE_M, SCHEMA_VERSION, Sampler, _sha256,
                             common_geometry, layouts_manifest, quat_mul, validate_profile)

X_BAND_EDGES_M = (0.35, 0.45, 0.55, 0.65)  # interior band edges, as in the pail profile
BAND_MARGIN_M = 0.03                       # an interior edge this close to the box limit is dropped
FULL_CIRCLE_GAP_DEG = 60.0
CALIBRATION_DRAWS = 6000


def relative_yaw_deg(quat, base):
    """Yaw of ``quat`` relative to ``base`` (both upright), in degrees."""
    w, x, y, z = quat_mul(np.asarray(quat, dtype=float), np.asarray(base, dtype=float) * [1, -1, -1, -1])
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_arc(yaws_deg, full_gap=FULL_CIRCLE_GAP_DEG):
    """Smallest arc [low, high] (high may pass 360) holding every yaw; the full circle if no gap reaches ``full_gap``."""
    yaws = np.sort(np.mod(np.asarray(yaws_deg, dtype=float), 360.0))
    if len(yaws) < 2:
        return [0.0, 360.0]
    gaps = np.diff(np.r_[yaws, yaws[0] + 360.0])
    widest = int(gaps.argmax())
    if gaps[widest] < full_gap:
        return [0.0, 360.0]
    start, end = yaws[(widest + 1) % len(yaws)], yaws[widest]
    low = math.floor(start)
    return [float(low), float(low + math.ceil((end - start) % 360.0 + (start - low)))]


def outward(low, high, step=0.001):
    return [math.floor(low / step + 1e-9) * step, math.ceil(high / step - 1e-9) * step]


def calibrate(profile, geometry, draws=CALIBRATION_DRAWS, seed=0):
    """Band weights that give ``target_share`` among targets passing the layout screens; also the pass rates."""
    strata = profile["object_x_strata"]
    bands = len(strata["target_share"])
    rng = np.random.default_rng(seed)
    rates = []
    for band in range(bands):
        only = {**profile, "object_x_strata": {**strata, "weights": [float(i == band) for i in range(bands)]}}
        sampler = Sampler(only, geometry, exclude_m=0)
        per_band = max(1, draws // bands)
        rates.append(sum(sampler.screen(sampler.draw(rng)) is None for _ in range(per_band)) / per_band)
    weights = np.array([share / rate if rate > 0 else 0.0 for share, rate in zip(strata["target_share"], rates)])
    if weights.sum() <= 0:
        raise ValueError(f"No object-x band passes the layout screens: {rates}")
    return [round(float(w), 4) for w in weights / weights.sum()], [round(r, 4) for r in rates]


def derive_profile(object_name, container_name, layouts, geometry, *, draws=CALIBRATION_DRAWS, seed=0):
    """A sampler profile from evaluation layouts (``layouts_manifest`` rows) and v1 ``geometry``."""
    poses = [row["poses"] for row in layouts]
    obj = np.array([p[object_name] for p in poses], dtype=float)
    cont = np.array([p[container_name] for p in poses], dtype=float)
    distance = np.linalg.norm(obj[:, :2] - cont[:, :2], axis=1)
    box = {}
    for key, values in (("object_xy", obj), ("container_xy", cont)):
        x, y = outward(values[:, 0].min(), values[:, 0].max()), outward(values[:, 1].min(), values[:, 1].max())
        box[key] = [[round(x[0], 3), round(y[0], 3)], [round(x[1], 3), round(y[1], 3)]]
    arcs = {key: yaw_arc([relative_yaw_deg(p[3:], geometry[name]["pose"][3:]) for p in values])
            for key, name, values in (("object_yaw_deg", object_name, obj), ("container_yaw_deg", container_name, cont))}
    if max(abs(relative_yaw_deg(p[3:], geometry[container_name]["pose"][3:])) for p in cont) < 0.5:
        arcs["container_yaw_deg"] = [0.0, 0.0]
    low, high = box["object_xy"][0][0], box["object_xy"][1][0]
    edges = [low, *[e for e in X_BAND_EDGES_M if low + BAND_MARGIN_M < e < high - BAND_MARGIN_M], high]
    counts = np.histogram(np.clip(obj[:, 0], low, high), bins=edges)[0]
    profile = {"object": object_name, "container": container_name, **box, **arcs,
               "relative_xy_m": [round(v, 3) for v in outward(distance.min(), distance.max())],
               "object_x_strata": {"edges": [round(e, 3) for e in edges], "weights": [1.0] * len(counts),
                                   "target_share": [round(float(c) / len(obj), 4) for c in counts],
                                   "layout_counts": [int(c) for c in counts]},
               "source": f"derived from {len(obj)} evaluation layouts of {object_name} by coverage_profile.derive_profile"}
    weights, rates = calibrate(profile, geometry, draws, seed)
    profile["object_x_strata"].update(weights=weights, screen_pass_rate=rates)
    validate_profile(profile)
    return profile


def build_manifest(profile_name, profile, scenes, count, seed, exclude_paths=(), exclude_m=EXCLUDE_M):
    """``target_sampler.build_manifest`` for a profile given as data."""
    names = (profile["object"], profile["container"], "table")
    geometry = common_geometry([str(s) for s in scenes], names)
    exclude = [row for path in exclude_paths for row in json.loads(Path(path).read_text())["targets"]]
    targets, counts = Sampler(profile, geometry, exclude=exclude, exclude_m=exclude_m).sample(count, seed)
    from . import target_sampler
    return {
        "schema_version": SCHEMA_VERSION, "construction": "simple_pnp_v1",
        "sampler": {"module": "robolab.core.omy_teleop.target_sampler", "sha256": _sha256(target_sampler.__file__),
                    "profile_module_sha256": _sha256(__file__), "profile_name": profile_name, "profile": profile,
                    "seed": seed, "count": count, "pair_clearance_m": PAIR_CLEARANCE_M,
                    "edge_clearance_m": EDGE_CLEARANCE_M, "bounds": "legacy", "exclude_m": exclude_m},
        "scenes": [{"path": str(Path(s).resolve()), "sha256": _sha256(s)} for s in scenes],
        "geometry": geometry,
        "excluded": [{"path": str(Path(p).resolve()), "sha256": _sha256(p)} for p in exclude_paths],
        "counts": counts, "targets": targets,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("layouts", "profile", "targets"))
    parser.add_argument("--object")
    parser.add_argument("--container", default="grey_bin")
    parser.add_argument("--scene", nargs="+", required=True, type=Path,
                        help="layouts: scenes to read; profile/targets: the v1 scenes generation uses")
    parser.add_argument("--layouts", type=Path, help="profile: evaluation layouts file")
    parser.add_argument("--profile-file", type=Path, help="targets: profile written by 'profile'")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exclude", nargs="*", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; it is frozen once written")
    if args.command == "layouts":
        result = layouts_manifest(args.scene, (args.object, args.container))
        summary = {"layouts": len(result["targets"])}
    elif args.command == "profile":
        geometry = common_geometry([str(s) for s in args.scene], (args.object, args.container, "table"))
        result = derive_profile(args.object, args.container, json.loads(args.layouts.read_text())["targets"], geometry)
        summary = {k: result[k] for k in ("object_xy", "object_yaw_deg", "object_x_strata")}
    else:
        profile = json.loads(args.profile_file.read_text())
        result = build_manifest(f"derived:{profile['object']}", profile, args.scene, args.count, args.seed, args.exclude)
        summary = {"counts": result["counts"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
