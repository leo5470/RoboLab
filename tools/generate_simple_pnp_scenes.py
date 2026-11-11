#!/usr/bin/env python3
"""Create robust two-object tabletop PnP scene variants.

Each scene contains one catalog target and RoboLab's established grey-bin
destination.  The target is initially on the table, outside the bin; task
definitions can use the manifest to specify pick-and-place into /world/grey_bin.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from robolab.scene_gen.llm_scene_gen import (
    FeedbackSystem,
    ObjectState,
    SpatialSolver,
    parse_predicates_from_dict,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "assets/objects/object_catalog.json"
BASE_PATH = ROOT / "assets/scenes/base_empty.usda"
OUT_DIR = ROOT / "assets/scenes/simple_pnp"
VARIANTS_PER_TARGET = 50

# grey_bin is a project fixture rather than a catalog object.  Existing scenes
# use this scale; the conservative footprint below is used only by the spatial
# solver and pairwise safety checks.
BIN_NAME = "grey_bin"
BIN_PATH = "assets/fixtures/grey_bin.usd"
BIN_SCALE = 0.007
BIN_DIMS = (0.23, 0.17, 0.13)
BIN_Z = 0.005
TABLE_BOUNDS = (0.30, 0.80, -0.40, 0.40)
EDGE_MARGIN = 0.01
COLLISION_MARGIN = 0.05


def rotated_half_extents(dims: tuple[float, float, float], yaw_deg: float) -> tuple[float, float]:
    """Return conservative axis-aligned XY half extents for a yawed object."""
    yaw = math.radians(yaw_deg)
    return (
        (abs(math.cos(yaw)) * dims[0] + abs(math.sin(yaw)) * dims[1]) / 2,
        (abs(math.sin(yaw)) * dims[0] + abs(math.cos(yaw)) * dims[1]) / 2,
    )


def halton(index: int, base: int) -> float:
    value, fraction = 0.0, 1.0
    while index:
        fraction /= base
        value += fraction * (index % base)
        index //= base
    return value


def point_in_bounds(u: float, v: float, hx: float, hy: float) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = TABLE_BOUNDS
    lo_x, hi_x = min_x + hx + EDGE_MARGIN, max_x - hx - EDGE_MARGIN
    lo_y, hi_y = min_y + hy + EDGE_MARGIN, max_y - hy - EDGE_MARGIN
    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("object cannot fit within tabletop safety bounds")
    return lo_x + u * (hi_x - lo_x), lo_y + v * (hi_y - lo_y)


def separated(a: tuple[float, float], ah: tuple[float, float], b: tuple[float, float], bh: tuple[float, float]) -> bool:
    return (
        abs(a[0] - b[0]) >= ah[0] + bh[0] + COLLISION_MARGIN
        or abs(a[1] - b[1]) >= ah[1] + bh[1] + COLLISION_MARGIN
    )


def make_pair(dims: tuple[float, float, float], target_index: int, variant: int) -> tuple[dict, dict]:
    """Use a low-discrepancy sequence to maximize location and yaw diversity."""
    bin_half = rotated_half_extents(BIN_DIMS, 0.0)
    for attempt in range(400):
        sequence = (target_index + 1) * 100_003 + variant * 401 + attempt
        yaw = (360.0 * halton(sequence, 7) + 137.507764 * variant) % 360.0
        target_half = rotated_half_extents(dims, yaw)
        target_xy = point_in_bounds(halton(sequence, 2), halton(sequence, 3), *target_half)
        bin_xy = point_in_bounds(halton(sequence, 5), halton(sequence, 11), *bin_half)
        if separated(target_xy, target_half, bin_xy, bin_half):
            return (
                {"x": target_xy[0], "y": target_xy[1], "yaw": yaw},
                {"x": bin_xy[0], "y": bin_xy[1], "yaw": 0.0},
            )
    raise RuntimeError(f"could not find a collision-free pair for target index {target_index}, variant {variant}")


def rewrite_base_for_subdirectory(base: str) -> str:
    # The target directory is two levels below assets/, so inherited payloads
    # need one extra ../ segment.
    return re.sub(r"@(\.\./)", r"@../../", base)


def prim_block(name: str, payload: str, x: float, y: float, z: float, yaw: float, scale: float = 1.0) -> str:
    half_yaw = math.radians(yaw) / 2
    qw, qz = math.cos(half_yaw), math.sin(half_yaw)
    return f'''\n    def "{name}" (\n        prepend payload = @{payload}@\n    )\n    {{\n        quatf xformOp:orient = ({qw:.9f}, 0, 0, {qz:.9f})\n        float3 xformOp:scale = ({scale:.9f}, {scale:.9f}, {scale:.9f})\n        double3 xformOp:translate = ({x:.9f}, {y:.9f}, {z:.9f})\n        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n    }}\n'''


def solve_and_validate(target_name: str, target_dims: tuple[float, float, float], target_pose: dict, bin_pose: dict) -> None:
    """Run the required predicate parser/solver pipeline for every variant."""
    predicate_data = {
        "objects": [{"name": target_name}, {"name": BIN_NAME}],
        "predicates": [
            {"type": "place-on-base", "object": target_name, **target_pose},
            {"type": "random-rot", "object": target_name},
            {"type": "place-on-base", "object": BIN_NAME, **bin_pose},
            {"type": "random-rot", "object": BIN_NAME},
        ],
    }
    states = {name: ObjectState(name=name) for name in (target_name, BIN_NAME)}
    for data in predicate_data["predicates"]:
        predicate = parse_predicates_from_dict(data)
        states[predicate.target_object].predicates.append(predicate)
    solved, message = SpatialSolver(TABLE_BOUNDS, collision_margin=COLLISION_MARGIN).solve(
        states, {target_name: target_dims, BIN_NAME: BIN_DIMS}, allow_relaxation=False
    )
    grammar = FeedbackSystem.generate_grammar_feedback(states)
    if not solved or grammar:
        raise RuntimeError(f"solver validation failed: {message}; {grammar or ''}")


def safe_target(entry: dict) -> bool:
    dims = entry["dims"]
    # A target must fit separately from the 23x17cm bin while retaining a 5cm
    # clearance. Fixtures and malformed/oversized assets are intentionally
    # listed in the manifest instead of being silently dropped.
    return (
        entry.get("class") != "fixture"
        and not entry["name"].startswith("bin_")
        and all(isinstance(dimension, (int, float)) and 0.005 <= dimension <= 0.30 for dimension in dims)
    )


def exclusion_reason(entry: dict) -> str:
    if entry["name"].startswith("bin_"):
        return "bin target excluded; grey_bin is reserved as the placement destination"
    if entry.get("class") == "fixture":
        return "fixture"
    return "does not fit safely beside the grey-bin destination"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    base = rewrite_base_for_subdirectory(BASE_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    accepted = [(index, entry) for index, entry in enumerate(catalog) if safe_target(entry)]
    rejected = [
        {
            "catalog_index": index,
            "name": entry["name"],
            "usd_path": entry["usd_path"],
            "dims": entry["dims"],
            "class": entry.get("class", ""),
            "reason": exclusion_reason(entry),
        }
        for index, entry in enumerate(catalog)
        if not safe_target(entry)
    ]
    manifest = {
        "format_version": 1,
        "description": "50 robust two-object tabletop PnP variants per compatible catalog entry",
        "catalog": str(CATALOG_PATH.relative_to(ROOT)),
        "target_count": len(accepted),
        "variants_per_target": VARIANTS_PER_TARGET,
        "scene_count": len(accepted) * VARIANTS_PER_TARGET,
        "destination": {
            "name": BIN_NAME,
            "usd_path": BIN_PATH,
            "scale": BIN_SCALE,
            "assumed_solver_dims": BIN_DIMS,
        },
        "table_bounds": TABLE_BOUNDS,
        "excluded_targets": rejected,
        "scenes": [],
    }

    for catalog_index, entry in accepted:
        target_name = entry["name"]
        dims = tuple(entry["dims"])
        stem = f"pnp_{catalog_index:03d}_{target_name}"
        for variant in range(VARIANTS_PER_TARGET):
            target_pose, bin_pose = make_pair(dims, catalog_index, variant)
            solve_and_validate(target_name, dims, target_pose, bin_pose)
            filename = f"{stem}_{variant:02d}.usda"
            target_payload = "../../" + entry["usd_path"].removeprefix("assets/")
            contents = base[:-2]
            contents += prim_block(target_name, target_payload, target_pose["x"], target_pose["y"], dims[2] / 2 + 0.002, target_pose["yaw"])
            contents += prim_block(BIN_NAME, "../../fixtures/grey_bin.usd", bin_pose["x"], bin_pose["y"], BIN_Z, 0.0, BIN_SCALE)
            contents += "}\n"
            (OUT_DIR / filename).write_text(contents)
            manifest["scenes"].append(
                {
                    "scene": filename,
                    "catalog_index": catalog_index,
                    "target": {"name": target_name, "usd_path": entry["usd_path"], "dims": dims, **target_pose},
                    "destination": {"name": BIN_NAME, **bin_pose},
                    "predicates": [
                        {"type": "place-on-base", "object": target_name, **target_pose},
                        {"type": "random-rot", "object": target_name},
                        {"type": "place-on-base", "object": BIN_NAME, **bin_pose},
                        {"type": "random-rot", "object": BIN_NAME},
                    ],
                }
            )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {manifest['scene_count']} scenes for {manifest['target_count']} targets in {OUT_DIR}")
    print(f"Excluded {len(rejected)} catalog entries; see {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
