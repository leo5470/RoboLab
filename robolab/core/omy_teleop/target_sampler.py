"""Sample explicit generation targets in v1 (Simple-PnP) scene construction.

Training targets are new object/container arrangements for MimicGen's
``--target-manifest``. They are built from v1 scene geometry and checked with
v1 preparation's layout rules (legacy bounds, pair and edge clearance); no
Simple-PnP v2 scene or tool is used for collection. A named range profile
bounds where arrangements may fall; arrangements close to excluded layouts
(development and final-test sets) are refused.

Sampling is independent per target: object XY, container XY and object yaw are
drawn uniformly from the profile, then screened. Every rejection is counted by
stage so a screen cannot silently narrow the distribution.

The core is pure numpy; reading bounds from a scene needs ``pxr``
(``tools/simple_pnp_v2/usd_python.sh`` or Isaac Sim's Python).
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from robolab.core.scenes.placement import bounds_corners, flat_layout_issues, footprint_separation

SCHEMA_VERSION = 1

# Ranges spanned by the 50 authored Simple-PnP v2 plastic pail layouts, the
# deployment (evaluation) distribution, measured 2026-11-11 and rounded outward to 1 mm: root XY
# of the pail and bin, their XY distance, and the bin's fixed zero yaw. Only
# these numbers are taken from v2; arrangements are v1-construction scenes.
PROFILES = {
    "pail_v2_deployment_box": {
        "object": "plasticpail_a01",
        "container": "grey_bin",
        "object_xy": [[0.287, -0.379], [0.748, 0.403]],
        "container_xy": [[0.425, -0.345], [0.666, 0.343]],
        "object_yaw_deg": [0.0, 360.0],
        "container_yaw_deg": [0.0, 0.0],
        "relative_xy_m": [0.280, 0.755],
        "source": "authored simple_pnp_v2 pnp_242_plasticpail_a01_* (50 layouts), measured 2026-11-11",
    },
}
PAIR_CLEARANCE_M = 0.02  # v1 PreparationSettings.pair_clearance_m
EDGE_CLEARANCE_M = 0.01  # v1 PreparationSettings.edge_clearance_m
EXCLUDE_M = 0.02         # joint-XY distance below which a target duplicates an excluded layout


def yaw_quat(degrees):
    half = math.radians(degrees) / 2
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def joint_xy_distance(first, second, object_name, container_name):
    """Larger of the object and container XY offsets between two arrangements (the R2 metric)."""
    return max(float(np.linalg.norm(np.asarray(first[name][:2]) - np.asarray(second[name][:2])))
               for name in (object_name, container_name))


def validate_profile(profile):
    for key in ("object_xy", "container_xy"):
        lo, hi = np.asarray(profile[key], dtype=float)
        if lo.shape != (2,) or np.any(hi < lo):
            raise ValueError(f"Profile {key} must be [[xmin, ymin], [xmax, ymax]]")
    for key in ("object_yaw_deg", "container_yaw_deg", "relative_xy_m"):
        lo, hi = profile[key]
        if not lo <= hi:
            raise ValueError(f"Profile {key} must be [low, high]")


class Sampler:
    """Draw arrangements and screen them with v1 layout rules.

    ``geometry`` maps the object, container and ``table`` to their v1 scene
    data: ``lower``/``upper`` local bounds (legacy, as v1 preparation reads
    them), ``scale`` and authored ``pose`` [x, y, z, qw, qx, qy, qz]. A target
    keeps each asset's authored height and composes the sampled yaw with its
    authored rotation (orientation = yaw x authored), so support is unchanged.
    """

    def __init__(self, profile, geometry, *, pair_clearance=PAIR_CLEARANCE_M, edge_clearance=EDGE_CLEARANCE_M,
                 exclude=(), exclude_m=EXCLUDE_M):
        validate_profile(profile)
        self.profile, self.geometry = profile, geometry
        self.object, self.container = profile["object"], profile["container"]
        for name in (self.object, self.container, "table"):
            if name not in geometry:
                raise ValueError(f"No geometry for {name!r}")
        self.pair_clearance, self.edge_clearance = pair_clearance, edge_clearance
        self.exclude, self.exclude_m = [row["poses"] for row in exclude], exclude_m
        self.table = self.corners("table", geometry["table"]["pose"])

    def corners(self, name, pose):
        g = self.geometry[name]
        return bounds_corners(g["lower"], g["upper"], pose, g["scale"])

    def pose(self, name, xy, yaw_deg):
        authored = np.asarray(self.geometry[name]["pose"], dtype=float)
        return [float(xy[0]), float(xy[1]), float(authored[2]),
                *(float(v) for v in quat_mul(yaw_quat(yaw_deg), authored[3:]))]

    def draw(self, rng):
        p = self.profile
        poses = {}
        for name, key, yaw_key in ((self.object, "object_xy", "object_yaw_deg"),
                                   (self.container, "container_xy", "container_yaw_deg")):
            lo, hi = np.asarray(p[key], dtype=float)
            poses[name] = self.pose(name, rng.uniform(lo, hi), float(rng.uniform(*p[yaw_key])))
        return poses

    def screen(self, poses, accepted=()):
        """None if the arrangement passes, else the stage that rejected it."""
        lo, hi = self.profile["relative_xy_m"]
        distance = float(np.linalg.norm(np.subtract(poses[self.object][:2], poses[self.container][:2])))
        if not lo <= distance <= hi:
            return "relative_distance"
        objects = {name: self.corners(name, poses[name]) for name in (self.object, self.container)}
        issues = flat_layout_issues(objects, self.table, pair_clearance=self.pair_clearance,
                                    edge_clearance=self.edge_clearance)
        if any("edge clearance" in issue for issue in issues):
            return "edge_clearance"
        if issues:
            return "pair_clearance"
        if any(joint_xy_distance(poses, other, self.object, self.container) < self.exclude_m
               for other in [*self.exclude, *accepted]):
            return "duplicate_or_excluded"
        return None

    def sample(self, count, seed, max_draws=None, prefix="t"):
        rng = np.random.default_rng(seed)
        max_draws = max_draws or 1000 * count
        targets, counts = [], Counter()
        while len(targets) < count and counts["proposed"] < max_draws:
            poses = self.draw(rng)
            counts["proposed"] += 1
            reason = self.screen(poses, [t["poses"] for t in targets])
            if reason:
                counts[reason] += 1
                continue
            counts["accepted"] += 1
            object_pose, container_pose = (np.asarray(poses[n]) for n in (self.object, self.container))
            targets.append({
                "id": f"{prefix}{seed}_{len(targets):04d}", "poses": poses,
                "pair_gap_m": round(footprint_separation(self.corners(self.object, object_pose),
                                                         self.corners(self.container, container_pose)), 6),
                "relative_xy_m": round(float(np.linalg.norm(object_pose[:2] - container_pose[:2])), 6),
            })
        if len(targets) < count:
            raise RuntimeError(f"Only {len(targets)} of {count} targets after {counts['proposed']} draws: {dict(counts)}")
        return targets, dict(counts)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_geometry(scene, names):
    """Legacy local bounds, scale and authored pose of ``/world/<name>`` prims, as v1 preparation reads them."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    out = {}
    for name in names:
        prim = stage.GetPrimAtPath(f"/world/{name}")
        if not prim:
            raise ValueError(f"{scene} has no /world/{name}")
        bound = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
        q = transform.GetRotation().GetQuat()
        out[name] = {"lower": list(bound.GetMin()), "upper": list(bound.GetMax()),
                     "scale": list(transform.GetScale()),
                     "pose": [*transform.GetTranslation(), q.GetReal(), *q.GetImaginary()]}
    return out


def common_geometry(scenes, names, atol=1e-6):
    """Geometry shared by every scene; poses of the object and container may differ, sizes may not."""
    geometries = [scene_geometry(scene, names) for scene in scenes]
    first = geometries[0]
    for scene, other in zip(scenes[1:], geometries[1:]):
        for name in names:
            keys = ("lower", "upper", "scale") + (("pose",) if name == "table" else ())
            for key in keys:
                if not np.allclose(first[name][key], other[name][key], atol=atol):
                    raise ValueError(f"{scene}: {name} {key} differs from {scenes[0]}")
            if abs(first[name]["pose"][2] - other[name]["pose"][2]) > atol:
                raise ValueError(f"{scene}: {name} authored height differs from {scenes[0]}")
    return first


def build_manifest(profile_name, scenes, count, seed, exclude_paths=(), exclude_m=EXCLUDE_M):
    profile = PROFILES[profile_name]
    names = (profile["object"], profile["container"], "table")
    geometry = common_geometry([str(s) for s in scenes], names)
    exclude = [row for path in exclude_paths for row in json.loads(Path(path).read_text())["targets"]]
    sampler = Sampler(profile, geometry, exclude=exclude, exclude_m=exclude_m)
    targets, counts = sampler.sample(count, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "construction": "simple_pnp_v1",
        "sampler": {"module": "robolab.core.omy_teleop.target_sampler", "sha256": _sha256(__file__),
                    "profile_name": profile_name, "profile": profile, "seed": seed, "count": count,
                    "pair_clearance_m": PAIR_CLEARANCE_M, "edge_clearance_m": EDGE_CLEARANCE_M,
                    "bounds": "legacy", "exclude_m": exclude_m},
        "scenes": [{"path": str(Path(s).resolve()), "sha256": _sha256(s)} for s in scenes],
        "geometry": geometry,
        "excluded": [{"path": str(Path(p).resolve()), "sha256": _sha256(p)} for p in exclude_paths],
        "counts": counts,
        "targets": targets,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", default="pail_v2_deployment_box", choices=sorted(PROFILES))
    parser.add_argument("--scene", nargs="+", required=True, type=Path,
                        help="v1 scene(s) the targets will be generated in; asset sizes must agree")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], type=Path,
                        help="Target manifests of layouts no target may duplicate (development, final test)")
    parser.add_argument("--exclude-m", type=float, default=EXCLUDE_M)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; target manifests are frozen once written")
    manifest = build_manifest(args.profile, args.scene, args.count, args.seed, args.exclude, args.exclude_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
