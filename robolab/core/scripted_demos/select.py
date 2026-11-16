"""Grasp selection at reset time (Phase 4 step 1), after MolmoSpaces' ``select_grasp_pose``.

Steps, all in numpy on the env-local (= robot root) frame:

1. Library grasps → world: ``T_world_grasp = T_world_obj @ T_obj_grasp``.
2. Hard filters: approach not from below; gripper boxes above the table and outside the inflated bin at pregrasp and
   grasp; wrist above the table; the pregrasp→grasp corridor free of object points.
3. Cost ranking (MolmoSpaces terms: TCP distance and rotation from the current gripper, verticality, jaw
   horizontality, distance to the object centre) plus a carry term and a preference for grasps that also held
   under the default 1 g test.
4. Sequential numpy IK from the current joints through pregrasp, grasp, lift, preplace and seat (the carry
   constraint: at lift height the held object's bottom clears the bin rim, and the arm still reaches preplace).
5. ``select``: a uniform draw among the ``top_k`` cheapest feasible grasps.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from robolab.core.scripted_demos.grasps import frames
from robolab.core.scripted_demos import motion
from robolab.core.scripted_demos.kinematics import Chain

# DROID Robotiq 2F-85 (open) as 6 mm slice boxes in base_link, (min xyz, max xyz), exported from the robot USD by
# scripts/scripted_demos/grasps/export_robot_geometry.py into gripper_geometry.json next to this file.
_GEOMETRY = json.loads((pathlib.Path(__file__).with_name("gripper_geometry.json")).read_text())["parts"]
_boxes = {k: [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in v] for k, v in _GEOMETRY.items()}
PALM, PADS, FINGERS, OUTER, KNUCKLES = (_boxes["palm"][0], _boxes["pads"], _boxes["fingers"], _boxes["outer"],
                                         _boxes["knuckles"])
GRIPPER_BOXES = [PALM, *PADS, *FINGERS, *OUTER, *KNUCKLES]
WRIST = (np.array([-0.20, -0.045, -0.045]), np.array([-0.004, 0.045, 0.045]))  # flange and link 7, approximate
TCP = frames.TCP


@dataclasses.dataclass
class Scene:
    """What the selector needs to know about one reset, all in the env-local frame."""

    T_world_obj: np.ndarray  # (4, 4) object pose
    object_points: np.ndarray  # (P, 3) object mesh points in the object frame
    bin_corners: np.ndarray  # (8, 3) bin OBB corners
    table_z: float  # table top height
    q_current: np.ndarray  # (7,) arm joints
    T_world_ee: np.ndarray  # (4, 4) current base_link pose
    mass: float = 0.3  # simulated object mass (kg), for the torque term


@dataclasses.dataclass
class Params:
    top_k: int = 8
    backoff: tuple[float, float] = (0.04, 0.10)  # pregrasp distance along -approach (m)
    lift_extra: tuple[float, float] = (0.0, 0.05)
    rim_clearance: float = 0.08  # object bottom above the bin rim while carrying (5 cm caught rims: Phase 4)
    seat_height: float = 0.02  # object bottom above the bin floor at release
    bin_floor_offset: float = 0.012  # bin floor above the bin OBB bottom (wall thickness)
    table_margin: float = 0.008
    bin_inflate: float = 0.01
    corridor_inset: float = 0.002
    use_corridor: bool = True
    max_approach_up: float = 0.3  # approach axis z-component limit (0 = horizontal, -1 = straight down)
    max_pad_penetration: float = 0.002  # DROID's open pads sit 1.75 mm further in than GraspDataGen's
    preplace_yaws: tuple[float, ...] = (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2)
    # MolmoSpaces' terms, plus terms for what DROID's simulated Robotiq actually holds (Phase 3 transfer test):
    # pinches that close far on a thin feature (GraspDataGen closure >= 0.3 rad) slip (pitcher 0/22, mug 4/50) while
    # wide-section grasps near the centre axis hold (pitcher 15/25); RoboLab's omy_bowl_failures_20261029.md reports
    # 1-2 N finger forces on thin walls. "torque" is the horizontal pad-centre offset from the object centre, by mass.
    weights: dict = dataclasses.field(default_factory=lambda: {
        "tcp_distance": 1.0, "rotation": 0.5, "verticality": 0.6, "horizontality": 0.3, "com_distance": 1.0,
        "carry": 0.5, "default_force": 0.3, "closure": 2.0, "torque": 1.0})
    max_candidates_for_ik: int = 256
    wall_clearance: float = 0.01  # gripper/wrist above the rim, or this far inside the walls
    max_release_drop: float = 0.12
    lift_only: float | None = None  # transfer test: lift straight up by this much, no place waypoints
    motion: motion.MotionParams = dataclasses.field(default_factory=motion.MotionParams)
    emulate: bool = True  # replay the executor's commands and keep only grasps that converge and clear the object
    emulate_top: int = 48  # cheapest feasible candidates replayed per selection
    path_inset: float = 0.003

    def to_json(self) -> dict:
        """Plain JSON form (tuples become lists), stored with screening results so later draws reuse it."""
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def from_json(cls, data: dict) -> "Params":
        data = dict(data)
        m = dict(data.pop("motion", {}))
        tuples = {f.name for f in dataclasses.fields(motion.MotionParams) if str(f.type).startswith("tuple")}
        motion_params = motion.MotionParams(**{k: tuple(v) if k in tuples else v for k, v in m.items()})
        tuples = {f.name for f in dataclasses.fields(cls) if str(f.type).startswith("tuple")}
        return cls(**{k: tuple(v) if k in tuples else v for k, v in data.items()}, motion=motion_params)


@dataclasses.dataclass
class Plan:
    grasp_id: int
    waypoints: dict  # name → (4, 4) base_link pose
    joints: dict  # name → (7,) numpy-IK solution (diagnostics)
    backoff: float
    lift_extra: float
    preplace_yaw: float | None
    costs: dict
    held_object_bottom_offset: float  # object bottom below base_link while held, world z (m)
    release_drop: float = 0.0  # how much higher than the nominal seat height the object is released (m)
    steps: dict | None = None  # segment durations the commands were checked with (the executor uses them)

    def to_json(self) -> dict:
        return {"grasp_id": self.grasp_id, "backoff": self.backoff, "lift_extra": self.lift_extra,
                "preplace_yaw": self.preplace_yaw, "costs": self.costs,
                "held_object_bottom_offset": self.held_object_bottom_offset, "release_drop": self.release_drop,
                "steps": self.steps,
                "waypoints": {k: v.round(5).tolist() for k, v in self.waypoints.items()},
                "joints": {k: np.round(v, 5).tolist() for k, v in self.joints.items()}}


@dataclasses.dataclass
class Candidates:
    plans: list  # feasible Plans in cost order
    report: dict  # filter counts and the failure class when nothing is feasible


def box_points(box, n: int = 2) -> np.ndarray:
    """An n x n x n grid over a box (n=2: its corners)."""
    lo, hi = box
    g = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


_GRIPPER_POINTS = np.vstack([box_points(b) for b in GRIPPER_BOXES])
_WRIST_POINTS = box_points(WRIST, 3)


def obb_frame(corners: np.ndarray):
    """Centre, axes (3, 3; columns) and half extents of an OBB given its 8 corners."""
    corners = np.asarray(corners, dtype=np.float64)
    c = corners.mean(0)
    d = corners[1:] - corners[0]
    edges: list[np.ndarray] = []
    for v in d[np.argsort(np.linalg.norm(d, axis=1))]:  # shortest first; keep the mutually orthogonal ones
        u = v / np.linalg.norm(v)
        if all(abs(u @ (e / np.linalg.norm(e))) < 1e-3 for e in edges):
            edges.append(v)
        if len(edges) == 3:
            break
    edges = np.asarray(edges)
    axes = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    return c, axes.T, np.linalg.norm(edges, axis=1) / 2


def points_in_obb(points: np.ndarray, corners: np.ndarray, inflate: float = 0.0) -> np.ndarray:
    c, axes, half = obb_frame(corners)
    return np.all(np.abs((points - c) @ axes) <= half + inflate, axis=-1)


def _apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 4, 4) × (P, 3) → (N, P, 3)."""
    return np.einsum("nij,pj->npi", T[:, :3, :3], points) + T[:, None, :3, 3]


def backed_off(T: np.ndarray, d) -> np.ndarray:
    out = np.array(T, copy=True)
    out[..., :3, 3] -= np.asarray(d)[..., None] * out[..., :3, frames.APPROACH_AXIS]
    return out


def corridor_hits(T_world_grasp: np.ndarray, points_world: np.ndarray, backoff: float, inset: float,
                  max_points: int = 1500, chunk: int = 256) -> np.ndarray:
    """Whether object points lie in the volume the open gripper sweeps between pregrasp and grasp, per grasp."""
    pts = points_world[:: max(1, len(points_world) // max_points)]
    lo = np.stack([b[0] for b in GRIPPER_BOXES]) + inset
    hi = np.stack([b[1] for b in GRIPPER_BOXES]) - inset
    lo[:, 0] -= backoff  # swept along -approach
    env_lo, env_hi = lo.min(0), hi.max(0)
    hit = np.zeros(len(T_world_grasp), dtype=bool)
    for start in range(0, len(T_world_grasp), chunk):
        T = T_world_grasp[start : start + chunk]
        local = np.einsum("nji,npj->npi", T[:, :3, :3], pts[None] - T[:, None, :3, 3])
        near = np.all((local >= env_lo) & (local <= env_hi), axis=-1)
        for r in np.nonzero(near.any(axis=1))[0]:
            q = local[r][near[r]][:, None]
            if np.all((q >= lo) & (q <= hi), axis=-1).any():
                hit[start + r] = True
    return hit


def pad_penetration(T_obj_grasp: np.ndarray, points: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Deepest object point inside either open fingertip pad (m), per grasp; 0 if none."""
    depth = np.zeros(len(T_obj_grasp))
    for start in range(0, len(T_obj_grasp), chunk):
        T = T_obj_grasp[start : start + chunk]
        local = np.einsum("nji,npj->npi", T[:, :3, :3], points[None] - T[:, None, :3, 3])
        for lo, hi in PADS:
            inside = np.all((local >= lo) & (local <= hi), axis=-1)
            d = np.minimum(local - lo, hi - local).min(-1)
            depth[start : start + chunk] = np.maximum(depth[start : start + chunk], np.where(inside, d, 0.0).max(1))
    return depth


def bin_geometry(bin_corners: np.ndarray, floor_offset: float) -> dict:
    c, axes, half = obb_frame(bin_corners)
    vertical = int(np.argmax(np.abs(axes[2])))
    order = [i for i in range(3) if i != vertical] + [vertical]  # horizontal axes first, vertical last
    z = np.asarray(bin_corners)[:, 2]
    return {"center": c, "rim_z": float(z.max()), "floor_z": float(z.min()) + floor_offset, "half": half[order],
            "axes": axes[:, order]}


def seat_clearance(T_seat: np.ndarray, binfo: dict, params: Params) -> np.ndarray:
    """How much each seat pose must rise so gripper and wrist points outside the bin opening clear the rim."""
    pts = _apply(T_seat, np.vstack([_GRIPPER_POINTS, _WRIST_POINTS]))
    local = (pts - binfo["center"]) @ binfo["axes"]
    inner = binfo["half"][:2] - params.bin_floor_offset - params.wall_clearance
    outside = np.any(np.abs(local[..., :2]) > inner, axis=-1)
    need = np.where(outside, binfo["rim_z"] + params.wall_clearance - pts[..., 2], 0.0)
    return np.clip(need.max(axis=1), 0.0, None)


def evaluate(library: dict, scene: Scene, chain: Chain, rng: np.random.Generator, params: Params = Params(),
             allowed: np.ndarray | None = None) -> Candidates:
    """All feasible grasps for this reset, cheapest first, each with its waypoints."""
    report: dict = {"counts": {}}
    n = len(library["T_obj_grasp"])
    ids = np.arange(n) if allowed is None else np.asarray(allowed, dtype=np.int64)
    report["counts"]["library"], report["counts"]["allowed"] = int(n), int(len(ids))
    T = scene.T_world_obj @ library["T_obj_grasp"][ids]
    approach = T[:, :3, frames.APPROACH_AXIS]
    closing = T[:, :3, frames.CLOSING_AXIS]

    ok = approach[:, 2] <= params.max_approach_up
    if "pad_penetration" in library:
        ok &= library["pad_penetration"][ids] <= params.max_pad_penetration
    report["counts"]["approach_and_pads"] = int(ok.sum())

    backoff = float(rng.uniform(*params.backoff))
    T_pre = backed_off(T, backoff)
    table_top = scene.table_z + params.table_margin
    for pose in (T, T_pre):
        pts = _apply(pose, _GRIPPER_POINTS)
        ok &= pts[..., 2].min(axis=1) >= table_top
        ok &= ~points_in_obb(pts.reshape(-1, 3), scene.bin_corners, params.bin_inflate).reshape(len(T), -1).any(1)
        ok &= _apply(pose, _WRIST_POINTS)[..., 2].min(axis=1) >= table_top
    report["counts"]["table_and_bin"] = int(ok.sum())

    obj_world = frames.transform_points(scene.T_world_obj, scene.object_points)
    if params.use_corridor:
        idx = np.nonzero(ok)[0]
        if len(idx):
            ok[idx[corridor_hits(T[idx], obj_world, backoff, params.corridor_inset)]] = False
    report["counts"]["corridor"] = int(ok.sum())

    # Costs (lower is better), each roughly in [0, 1].
    com = obj_world.mean(0)
    size = float(np.linalg.norm(obj_world.max(0) - obj_world.min(0)))
    tcp = _apply(T, TCP[None])[:, 0]
    ee_tcp = frames.transform_points(scene.T_world_ee, TCP)
    rel = np.einsum("ji,njk->nik", scene.T_world_ee[:3, :3], T[:, :3, :3])
    rot = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    horizontal = np.linalg.norm(tcp[:, :2] - com[:2], axis=1)
    costs = {
        "closure": np.clip(library["finger_joint"][ids] / 0.8, 0, 1) if "finger_joint" in library else np.zeros(len(T)),
        "torque": np.clip(horizontal / 0.1, 0, 2) * min(scene.mass / 0.3, 2.0),
        "tcp_distance": np.linalg.norm(tcp - ee_tcp, axis=1) / 0.5,
        "rotation": rot / np.pi,
        "verticality": (1 + approach[:, 2]) / 2,
        "horizontality": np.abs(closing[:, 2]),
        "com_distance": np.linalg.norm(tcp - com, axis=1) / max(size, 1e-3),
    }
    binfo = bin_geometry(scene.bin_corners, params.bin_floor_offset)
    # Where the object hangs while held: GraspDataGen's post-test pose of the gripper relative to the object.
    obj_at_grasp = T @ frames.invert(library["T_obj_held"][ids])
    sub = scene.object_points[:: max(1, len(scene.object_points) // 800)]
    held_pts = _apply(obj_at_grasp, sub)
    bottom_offset = T[:, 2, 3] - held_pts[..., 2].min(axis=1)
    if params.lift_only is not None:
        lift_z = T[:, 2, 3] + params.lift_only
    else:
        lift_z = np.maximum(binfo["rim_z"] + params.rim_clearance + bottom_offset, T[:, 2, 3])
    costs["carry"] = np.clip((lift_z - T[:, 2, 3]) / 0.4, 0, 1)
    if "passed_1g" in library:
        costs["default_force"] = (~library["passed_1g"][ids]).astype(float)
    total = sum(params.weights.get(k, 0.0) * v for k, v in costs.items())

    cand = np.array([i for i in np.argsort(total, kind="stable") if ok[i]][: params.max_candidates_for_ik], dtype=int)
    report["counts"]["ranked"] = int(len(cand))
    if len(cand) == 0:
        report["failure"] = "no_feasible_grasp"
        return Candidates([], report)

    lift_extra = 0.0 if params.lift_only is not None else float(rng.uniform(*params.lift_extra))
    Tg, Tp = T[cand], T_pre[cand]
    Tl = Tg.copy()
    Tl[:, 2, 3] = lift_z[cand] + lift_extra
    q_pre, c1, *_ = chain.ik(Tp, scene.q_current)
    q_grasp, c2, *_ = chain.ik(Tg, q_pre)
    q_lift, c3, *_ = chain.ik(Tl, q_grasp)
    feasible = c1 & c2 & c3
    report["counts"]["ik_pick"] = int(feasible.sum())

    yaw_of = np.full(len(cand), np.nan)
    drop_of = np.zeros(len(cand))
    Tpp_all, Ts_all = np.zeros_like(Tg), np.zeros_like(Tg)
    q_pp_all, q_s_all = np.zeros_like(q_lift), np.zeros_like(q_lift)
    if params.lift_only is None:
        # Preplace and seat: object centre over the bin centre, upright as grasped, yaw about world z in order.
        centre = _apply(obj_at_grasp[cand], sub.mean(0)[None])[:, 0]
        seat_z = binfo["floor_z"] + params.seat_height + bottom_offset[cand]
        pre_z = lift_z[cand] + lift_extra
        yaws = list(params.preplace_yaws)
        yaws = yaws[:1] + list(rng.permutation(yaws[1:]))
        remaining = feasible.copy()
        for yaw in yaws:
            if not remaining.any():
                break
            Rz = frames.rot(2, yaw)
            r = np.nonzero(remaining)[0]
            Tpp = Tg[r].copy()
            Tpp[:, :3, :3] = Rz @ Tg[r, :3, :3]
            Tpp[:, :3, 3] = binfo["center"] + (Tg[r, :3, 3] - centre[r]) @ Rz.T
            Ts = Tpp.copy()
            Tpp[:, 2, 3] = pre_z[r]
            Ts[:, 2, 3] = seat_z[r]
            # Lower only as far as the gripper and wrist stay clear of the walls: points outside the bin's opening
            # must stay above the rim; a side-held short object is released from higher up.
            drop = seat_clearance(Ts, binfo, params)
            Ts[:, 2, 3] += drop
            q_pp, c4, *_ = chain.ik(Tpp, q_lift[r])
            q_s, c5, *_ = chain.ik(Ts, q_pp)
            good = c4 & c5 & (drop <= params.max_release_drop) & (Ts[:, 2, 3] <= Tpp[:, 2, 3] + 1e-6)
            hit = r[good]
            yaw_of[hit] = yaw
            drop_of[hit] = drop[good]
            Tpp_all[hit], Ts_all[hit], q_pp_all[hit], q_s_all[hit] = Tpp[good], Ts[good], q_pp[good], q_s[good]
            remaining[hit] = False
        feasible &= ~np.isnan(yaw_of)
        report["counts"]["ik_place"] = int(feasible.sum())
    if not feasible.any():
        report["failure"] = "ik_unreachable"
        return Candidates([], report)

    order_idx = np.nonzero(feasible)[0]
    steps = None
    if params.emulate:
        # Replay the executor's joint commands for the cheapest candidates (see motion.emulate).
        order_idx = order_idx[: params.emulate_top]
        mode = "transfer" if params.lift_only is not None else "pnp"
        steps = dict(motion.TRANSFER) if mode == "transfer" else motion.segment_steps(rng, params.motion)
        seq = list(motion.TRANSFER) if mode == "transfer" else list(motion.NOMINAL)
        wp = {"pregrasp": Tp[order_idx], "grasp": Tg[order_idx], "lift": Tl[order_idx]}
        if mode == "pnp":
            wp |= {"preplace": Tpp_all[order_idx], "seat": Ts_all[order_idx]}
        targets = motion.segment_targets(wp, params.motion, mode)
        replay = motion.emulate(chain, scene.q_current, np.repeat(scene.T_world_ee[None], len(order_idx), 0),
                                    targets, seq, steps, params.motion, chain.joint_limits)
        good = replay["converged"].copy()
        report["counts"]["commands_converge"] = int(good.sum())
        fails = [f for f in replay["first_fail"][~good] if f]
        report["command_failures"] = {f: fails.count(f) for f in set(fails)}
        # The commanded path to pregrasp must not run through the object.
        path = replay["poses"]["pregrasp"]
        if path.shape[1] and good.any():
            flat = path[good].reshape(-1, 4, 4)
            hits = corridor_hits(flat, obj_world, 0.0, params.path_inset).reshape(int(good.sum()), -1).any(1)
            good[np.nonzero(good)[0][hits]] = False
        report["counts"]["path_clear"] = int(good.sum())
        if mode == "pnp" and good.any():
            # Backing out after the release must not drag the gripper through a bin wall.
            g = np.nonzero(good)[0]
            good[g[seat_clearance(targets["retreat"][g], binfo, params) > 0.005]] = False
            report["counts"]["retreat_clear"] = int(good.sum())
        order_idx = order_idx[good]
        if len(order_idx) == 0:
            report["failure"] = "ik_unreachable"
            return Candidates([], report)

    plans = []
    for j in order_idx:
        i = cand[j]
        waypoints = {"pregrasp": Tp[j], "grasp": Tg[j], "lift": Tl[j]}
        joints = {"pregrasp": q_pre[j], "grasp": q_grasp[j], "lift": q_lift[j]}
        if params.lift_only is None:
            waypoints |= {"preplace": Tpp_all[j], "seat": Ts_all[j]}
            joints |= {"preplace": q_pp_all[j], "seat": q_s_all[j]}
        plans.append(Plan(
            grasp_id=int(ids[i]), waypoints=waypoints, joints=joints, backoff=backoff, lift_extra=lift_extra,
            preplace_yaw=None if np.isnan(yaw_of[j]) else float(yaw_of[j]),
            costs={k: float(v[i]) for k, v in costs.items()} | {"total": float(total[i])},
            held_object_bottom_offset=float(bottom_offset[i]), release_drop=float(drop_of[j]), steps=steps))
    report["bin"] = {"rim_z": binfo["rim_z"], "floor_z": binfo["floor_z"], "center": binfo["center"].round(4).tolist()}
    return Candidates(plans, report)


def select(library: dict, scene: Scene, chain: Chain, rng: np.random.Generator, params: Params = Params(),
           allowed: np.ndarray | None = None) -> tuple[Plan | None, dict]:
    """A uniform draw among the ``top_k`` cheapest feasible grasps. Returns ``(plan or None, report)``."""
    found = evaluate(library, scene, chain, rng, params, allowed)
    if not found.plans:
        return None, found.report
    pool = found.plans[: params.top_k]
    plan = pool[int(rng.integers(len(pool)))]
    return plan, found.report | {"top_k_pool": [p.grasp_id for p in pool], "chosen": plan.grasp_id}
