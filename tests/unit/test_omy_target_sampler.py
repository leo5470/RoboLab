"""Explicit-target sampling in v1 construction, on synthetic geometry (no USD)."""

import math

import numpy as np
import pytest

from robolab.core.omy_teleop.target_sampler import PROFILES, Sampler, joint_xy_distance, quat_mul, yaw_quat
from robolab.core.scenes.placement import bounds_corners, flat_layout_issues

PROFILE = {**PROFILES["pail_v2_deployment_box"], "object": "obj", "container": "bin"}
# Table top spans x 0.1..0.9, y -0.5..0.5; object 0.1 m cube, bin 0.3 x 0.4 m, both resting on it.
GEOMETRY = {
    "table": {"lower": [-0.4, -0.5, -0.8], "upper": [0.4, 0.5, 0.0], "scale": [1, 1, 1], "pose": [0.5, 0, 0, 1, 0, 0, 0]},
    "obj": {"lower": [-0.05, -0.05, 0], "upper": [0.05, 0.05, 0.1], "scale": [1, 1, 1],
            "pose": [0.3, 0.3, 0.002, *yaw_quat(30)]},
    "bin": {"lower": [-0.15, -0.2, 0], "upper": [0.15, 0.2, 0.1], "scale": [1, 1, 1], "pose": [0.5, 0, 0.003, 1, 0, 0, 0]},
}


def _yaw(q):
    return math.degrees(2 * math.atan2(q[3], q[0])) % 360


def test_targets_are_seeded_in_range_and_pass_v1_layout_checks():
    sampler = Sampler(PROFILE, GEOMETRY)
    targets, counts = sampler.sample(40, seed=3)
    again, _ = Sampler(PROFILE, GEOMETRY).sample(40, seed=3)
    assert targets == again and targets != Sampler(PROFILE, GEOMETRY).sample(40, seed=4)[0]
    assert len({t["id"] for t in targets}) == 40
    assert counts["proposed"] == sum(v for k, v in counts.items() if k != "proposed")
    table = bounds_corners(**{k: GEOMETRY["table"][k] for k in ("lower", "upper", "scale", "pose")})
    for t in targets:
        poses = t["poses"]
        objects = {n: bounds_corners(GEOMETRY[n]["lower"], GEOMETRY[n]["upper"], poses[n], GEOMETRY[n]["scale"])
                   for n in ("obj", "bin")}
        assert flat_layout_issues(objects, table) == []
        for name, key in (("obj", "object_xy"), ("bin", "container_xy")):
            lo, hi = np.asarray(PROFILE[key])
            assert np.all(lo <= poses[name][:2]) and np.all(poses[name][:2] <= hi)
            assert poses[name][2] == GEOMETRY[name]["pose"][2]  # authored height kept; preparation re-seats
            assert len(poses[name]) == 7 and math.isclose(np.linalg.norm(poses[name][3:]), 1)
        assert PROFILE["relative_xy_m"][0] <= t["relative_xy_m"] <= PROFILE["relative_xy_m"][1]
        assert t["pair_gap_m"] >= 0.02
        assert _yaw(poses["bin"][3:]) == pytest.approx(0, abs=1e-9)


def test_sampled_yaw_composes_with_the_authored_rotation():
    sampler = Sampler({**PROFILE, "object_yaw_deg": [90, 90]}, GEOMETRY)
    q = sampler.pose("obj", [0.3, 0.3], 90)[3:]
    assert np.allclose(q, quat_mul(yaw_quat(90), GEOMETRY["obj"]["pose"][3:]))
    assert _yaw(q) == pytest.approx(120)


def test_layouts_near_excluded_ones_are_refused():
    targets, _ = Sampler(PROFILE, GEOMETRY).sample(30, seed=0)
    excluded = [{"id": "dev", "poses": targets[0]["poses"]}]
    sampler = Sampler(PROFILE, GEOMETRY, exclude=excluded)
    assert sampler.screen(targets[0]["poses"]) == "duplicate_or_excluded"
    near = {n: list(p) for n, p in targets[0]["poses"].items()}
    near["obj"][0] += 0.015  # joint-XY 15 mm < 20 mm
    assert sampler.screen(near) == "duplicate_or_excluded"
    fresh, _ = sampler.sample(30, seed=0)
    assert all(joint_xy_distance(t["poses"], excluded[0]["poses"], "obj", "bin") >= 0.02 for t in fresh)
    assert all(joint_xy_distance(a["poses"], b["poses"], "obj", "bin") >= 0.02
               for i, a in enumerate(fresh) for b in fresh[i + 1:])


def test_screen_reasons():
    sampler = Sampler(PROFILE, GEOMETRY)
    def poses(obj, bin_xy):
        return {"obj": sampler.pose("obj", obj, 0), "bin": sampler.pose("bin", bin_xy, 0)}
    assert sampler.screen(poses([0.3, 0.3], [0.5, 0.0])) is None
    assert sampler.screen(poses([0.5, 0.2], [0.5, 0.0])) == "relative_distance"  # 0.20 m < 0.28 m
    assert sampler.screen(poses([0.87, 0.3], [0.5, -0.1])) == "edge_clearance"
    assert sampler.screen(poses([0.36, 0.30], [0.55, 0.05])) == "pair_clearance"  # 0.32 m apart, gap 13 mm


def test_an_infeasible_profile_fails_with_counts():
    impossible = {**PROFILE, "relative_xy_m": [2.0, 3.0]}
    with pytest.raises(RuntimeError, match="relative_distance"):
        Sampler(impossible, GEOMETRY).sample(5, seed=0, max_draws=100)
    with pytest.raises(ValueError, match="object_xy"):
        Sampler({**PROFILE, "object_xy": [[1, 0], [0, 0]]}, GEOMETRY)
    with pytest.raises(ValueError, match="No geometry"):
        Sampler(PROFILE, {k: v for k, v in GEOMETRY.items() if k != "table"})
