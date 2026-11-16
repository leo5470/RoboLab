"""The grasp selector: geometry filters, ranking, command replay, part classification, and the screening
configuration stored with qualified grasps."""

import itertools

import numpy as np
import pytest

from robolab.core.scripted_demos.grasps import frames, library
from robolab.core.scripted_demos import kinematics, motion, select
from scripted_demos_helpers import box_obb, random_pose, side_grasp, top_down_grasp, toy_library, toy_scene


# ---- selector geometry ---------------------------------------------------------------------------------------------

def test_obb_helpers():
    corners = box_obb([0.5, 0.2, 0.06], [0.1, 0.15, 0.06])
    c, axes, half = select.obb_frame(corners)
    assert np.allclose(c, [0.5, 0.2, 0.06]) and np.allclose(sorted(half), sorted([0.1, 0.15, 0.06]))
    assert select.points_in_obb(np.array([[0.5, 0.2, 0.06]]), corners)[0]
    assert not select.points_in_obb(np.array([[0.5, 0.2, 0.2]]), corners)[0]
    assert select.points_in_obb(np.array([[0.5, 0.2, 0.125]]), corners, inflate=0.01)[0]


def test_corridor_detects_object_in_the_approach_path():
    obj = np.array([[0.0, 0.0, 0.0]])
    grasp = top_down_grasp([0.0, 0.0, -0.05])  # pads 5 cm below the point: the palm sweeps through it
    assert select.corridor_hits(grasp[None], obj, backoff=0.08, inset=0.0)[0]
    clear = top_down_grasp([0.0, 0.0, 0.0])  # the point sits between the open pads
    assert not select.corridor_hits(clear[None], obj, backoff=0.08, inset=0.0)[0]


class ToyChain:
    """Every target reachable: the selector's IK step returns converged for everything."""

    def ik(self, T, q0, **_):
        n = T.shape[:-2]
        return np.zeros(n + (7,)), np.ones(n, dtype=bool), np.zeros(n), np.zeros(n)


def test_selector_filters_table_bin_and_bottom_approaches_and_prefers_top_down():
    scene = toy_scene()
    grasps = {
        "top": top_down_grasp([0.0, 0.0, 0.07]),
        "side": side_grasp([-0.02, 0.0, 0.0], np.array([-1.0, 0.0])),  # pads 2 cm past the -x face
        "too_deep": side_grasp([0.0, 0.0, 0.0], np.array([-1.0, 0.0])),  # palm/linkage would enter the box
        "below_table": side_grasp([0.0, 0.0, -0.095], np.array([-1.0, 0.0])),  # palm under the table top
        "from_below": frames.homogeneous(frames.rot(1, -np.pi / 2)),  # fingers pointing up
        "into_bin": side_grasp([0.0, 0.0, 0.0], np.array([0.0, 1.0])),  # palm reaches into the bin side
    }
    grasps["into_bin"][:3, 3] += [0.0, 0.25, 0.0]
    names = list(grasps)
    lib = toy_library([grasps[n] for n in names])
    found = select.evaluate(lib, scene, ToyChain(), np.random.default_rng(0), select.Params(backoff=(0.05, 0.05), emulate=False))
    chosen = [names[p.grasp_id] for p in found.plans]
    assert chosen[0] == "top"
    assert "side" in chosen
    for bad in ("below_table", "from_below", "into_bin", "too_deep"):
        assert bad not in chosen
    plan = found.plans[0]
    assert set(plan.waypoints) == {"pregrasp", "grasp", "lift", "preplace", "seat"}
    # Pregrasp backs off along -approach (up, for a top-down grasp); preplace puts the object over the bin centre.
    assert np.isclose(plan.waypoints["pregrasp"][2, 3] - plan.waypoints["grasp"][2, 3], 0.05)
    binfo = select.bin_geometry(scene.bin_corners, select.Params().bin_floor_offset)
    held_bottom = plan.waypoints["lift"][2, 3] - plan.held_object_bottom_offset
    assert held_bottom >= binfo["rim_z"] + select.Params().rim_clearance - 1e-9
    seat_bottom = plan.waypoints["seat"][2, 3] - plan.held_object_bottom_offset
    assert np.isclose(seat_bottom, binfo["floor_z"] + select.Params().seat_height)
    assert plan.release_drop == 0.0
    # The side-held box cannot be lowered to the floor without the wrist crossing the wall: it is released higher.
    side = found.plans[chosen.index("side")]
    assert 0.0 < side.release_drop <= select.Params().max_release_drop


def test_select_draws_from_top_k_only():
    scene = toy_scene()
    lib = toy_library([top_down_grasp([0.0, 0.0, 0.07], yaw) for yaw in np.linspace(0, np.pi, 12)])
    params = select.Params(top_k=3, emulate=False)
    found = select.evaluate(lib, scene, ToyChain(), np.random.default_rng(0), params)
    top = {p.grasp_id for p in found.plans[:3]}
    draws = {select.select(lib, scene, ToyChain(), np.random.default_rng(s), params)[0].grasp_id for s in range(40)}
    assert draws <= top and len(draws) > 1


@pytest.mark.parametrize("allowed", [[1], [0, 2]])
def test_select_respects_allowed_ids(allowed):
    scene = toy_scene()
    lib = toy_library([top_down_grasp([0.0, 0.0, 0.07], yaw) for yaw in (0.0, 0.5, 1.0)])
    plan, _ = select.select(lib, scene, ToyChain(), np.random.default_rng(0), select.Params(emulate=False), allowed=allowed)
    assert plan.grasp_id in allowed


def test_orientation_error_matches_rotation_angle():
    rng = np.random.default_rng(4)
    for _ in range(20):
        A, B = random_pose(rng)[:3, :3], random_pose(rng)[:3, :3]
        e = kinematics.orientation_error(A, B)
        assert np.isclose(np.linalg.norm(e), frames.rotation_angle(A, B), atol=1e-6)


def synthetic_chain():
    """A 7-joint arm (revolute about each joint's local X) with axes cycling through three directions."""
    rots = [np.eye(3), frames.rot(1, np.pi / 2), frames.rot(2, np.pi / 2)]
    T0 = np.stack([frames.homogeneous(rots[i % 3], [0.0, 0.0, 0.12 if i else 0.0]) for i in range(7)])
    T1inv = np.repeat(np.eye(4)[None], 7, 0)
    end = frames.homogeneous(t=[0.0, 0.0, 0.1])
    return kinematics.Chain(T0, T1inv, np.ones(7, dtype=bool), np.tile([-2.8, 2.8], (7, 1)), np.eye(4), end)


def test_emulate_replays_dls_commands_to_convergence():
    chain = synthetic_chain()
    q0 = np.full(7, 0.1)
    T0 = chain.fk(q0)
    reachable = chain.fk(q0 + np.array([0.3, -0.2, 0.25, 0.1, -0.3, 0.2, 0.1]))
    unreachable = reachable.copy()
    unreachable[:3, 3] += [2.0, 0.0, 0.0]
    params = motion.MotionParams()
    out = motion.emulate(chain, q0, np.stack([T0, T0]), {"pregrasp": np.stack([reachable, unreachable])},
                             ["pregrasp"], {"pregrasp": 30}, params, chain.joint_limits)
    assert out["converged"].tolist() == [True, False]
    assert out["first_fail"][1] == "pregrasp"
    assert out["poses"]["pregrasp"].shape == (2, 15, 4, 4)
    # The commanded pose actually ends at the target.
    assert np.linalg.norm(chain.fk(out["q"][0])[:3, 3] - reachable[:3, 3]) < params.converge_pos
    # And it moves at most 0.04 rad per step: 30 + allowance steps cannot cover a 2 rad move.
    far = chain.fk(q0 + 2.0)
    slow = motion.emulate(chain, q0, T0[None], {"pregrasp": far[None]}, ["pregrasp"], {"pregrasp": 10}, params,
                              chain.joint_limits)
    assert not slow["converged"][0]


def pitcher_like(handle_deg=45.0, spout_deg=225.0, n=4000, seed=0):
    """A 6 cm-radius, 24 cm cylinder with a densely meshed handle (37-79 % of its height) and a sparse top spout."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, 0.24, n)
    body = np.c_[0.06 * np.cos(a), 0.06 * np.sin(a), z]
    h = np.radians(handle_deg)
    t = rng.uniform(0, 1, 3 * n)  # handle: many more vertices than the body, like the YCB pitcher's
    handle = np.c_[(0.08 + 0.01 * t) * np.cos(h), (0.08 + 0.01 * t) * np.sin(h), 0.09 + 0.10 * t]  # above the body slice
    sp = np.radians(spout_deg)
    spout = np.c_[np.full(200, 0.085 * np.cos(sp)), np.full(200, 0.085 * np.sin(sp)), rng.uniform(0.22, 0.24, 200)]
    return np.vstack([body, handle, spout])


def grasp_at(point):
    T = np.eye(4)
    T[:3, 3] = np.asarray(point) - frames.TCP
    return T


def test_classify_finds_the_handle_not_the_spout():
    v = pitcher_like()
    assert abs(library.handle_direction(v) - 45.0) < 5
    grasps = np.stack([grasp_at([0.085 * np.cos(np.radians(45)), 0.085 * np.sin(np.radians(45)), 0.12]),  # handle
                       grasp_at([0.085 * np.cos(np.radians(225)), 0.085 * np.sin(np.radians(225)), 0.23]),  # spout
                       grasp_at([0.0, 0.0, 0.235]),  # rim
                       grasp_at([0.06, 0.0, 0.05])])  # body
    assert library.classify(grasps, v, catalog_index=303).tolist() == ["handle", "body", "rim", "body"]


def test_classify_handles_coarse_meshes():
    box = np.array(list(itertools.product((-0.04, 0.04), (-0.08, 0.08), (0.0, 0.21))))  # vertices only at corners
    kinds = library.classify(np.stack([grasp_at([0.0, 0.0, 0.2])]), box, catalog_index=294)
    assert kinds.tolist() == ["rim"]


# ---- screening configuration ---------------------------------------------------------------------------------------

def test_params_round_trip_through_json():
    import json as _json

    from robolab.core.scripted_demos import motion

    p = select.Params(use_corridor=False, max_candidates_for_ik=4096, preplace_yaws=(0.0, 0.4),
                      motion=motion.MotionParams(retreat_back=0.12, segment_scale={"preplace": 1.5}))
    q = select.Params.from_json(_json.loads(_json.dumps(p.to_json())))
    assert q == p
    assert isinstance(q.backoff, tuple) and isinstance(q.motion.duration_scale, tuple)


def test_qualified_grasps_return_the_screening_configuration(tmp_path, monkeypatch):
    import json as _json

    from robolab.core.scripted_demos import collect

    monkeypatch.setenv("GRASP_DATA_DIR", str(tmp_path))
    (tmp_path / "qualified").mkdir()
    screened = select.Params(use_corridor=False, max_candidates_for_ik=4096)
    trials = [{"seed": 0, "grasp_id": 7, "success": True}, {"seed": 0, "grasp_id": 3, "success": False},
              {"seed": 0, "grasp_id": 2, "success": True}]
    (tmp_path / "qualified" / "T.json").write_text(_json.dumps({"params": screened.to_json(), "trials": trials}))
    ids, params = collect.qualified_grasps("T")
    assert ids == [2, 7] and params == screened and params.use_corridor is False
    assert collect.qualified_grasps("missing") is None
    (tmp_path / "qualified" / "Old.json").write_text(_json.dumps({"trials": trials}))  # screened before the change
    with pytest.raises(ValueError, match="re-screen"):
        collect.qualified_grasps("Old")
