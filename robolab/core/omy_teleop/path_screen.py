"""Screen explicit targets by six-joint reachability of each source's transformed path, on CPU.

MimicGen replays a source's commanded hand poses relative to a reference
object: the grasp segment follows the object, the placement segment follows
the container. At a new target those poses move rigidly with the reference.
The generator's arm has six free joints (``panda_joint3`` stays at its
recorded home value), so a full hand pose has no spare freedom: a transformed
path either has joint solutions within limits or it does not.

For each (target, source) this solves converged inverse kinematics along both
transformed segments, starting from the source's own joints. It ignores
collisions and tracking lag, and it does not solve the connecting motions:
it only reports their length (reset pose to the first grasp pose, and the last
grasp pose to the first placement pose), which the generator crosses in a
fixed number of steps. A pass is necessary for generation, not sufficient.

Reading the robot needs ``pxr``; the rest is NumPy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .quality import POSITION_LIMITS

REPO = Path(__file__).resolve().parents[3]
ROBOT = "assets/robots/franka_robotiq_2f_85_flattened.usd"
ARM = tuple(f"panda_joint{i}" for i in range(1, 8))
LOCKED = 2            # panda_joint3, index within ARM
EE_BODY = "base_link"
POSITION_M = 0.001    # a pose counts as reached within these residuals
ROTATION_RAD = 0.01
SCHEMA_VERSION = 1


def pose_matrix(pose):
    """4x4 transform of [x, y, z, qw, qx, qy, qz]."""
    w, x, y, z = np.asarray(pose[3:7], dtype=float) / np.linalg.norm(pose[3:7])
    out = np.eye(4)
    out[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    out[:3, 3] = pose[:3]
    return out


def rotation_vector(rotation):
    """Axis-angle vector of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # angle 0, or pi (not reached by small IK residuals)
        return np.zeros(3)
    return axis / norm * angle


class Arm:
    """Kinematic chain from the robot root to the hand body: joints rotate about their local X."""

    def __init__(self, chain, limits=POSITION_LIMITS):
        self.chain = chain  # [(parent-side frame, inverse child-side frame, revolute)]
        self.limits = np.asarray(limits, dtype=float)
        if sum(revolute for _, _, revolute in chain) != len(self.limits):
            raise ValueError("Joint limits do not match the chain")

    @classmethod
    def from_usd(cls, path=REPO / ROBOT, limits=POSITION_LIMITS):
        from pxr import Gf, Usd, UsdPhysics

        stage = Usd.Stage.Open(str(path))
        prims = {}
        for prim in stage.Traverse():  # names repeat in this asset; keep the first, as Traverse orders them
            prims.setdefault(prim.GetName(), prim)

        def frame(position, quat):
            quat = Gf.Quatd(quat.GetReal(), *quat.GetImaginary()).GetNormalized()
            out = np.eye(4)
            out[:3, :3] = np.array(Gf.Matrix3d(quat)).T
            out[:3, 3] = position
            return out

        chain = []
        for name in [*ARM, "panda_joint8", "panda_hand_joint"]:
            prim = prims[name]
            joint, revolute = UsdPhysics.Joint(prim), prim.IsA(UsdPhysics.RevoluteJoint)
            if revolute and prim.GetAttribute("physics:axis").Get() != "X":
                raise ValueError(f"{name}: expected joint axis X")
            chain.append((frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                          np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())),
                          bool(revolute)))
        return cls(chain, limits)

    def fk(self, joints, jacobian=False):
        """Hand pose in the root frame; with ``jacobian``, also the 6x7 geometric Jacobian."""
        pose, k, axes = np.eye(4), 0, []
        for parent, child_inverse, revolute in self.chain:
            pose = pose @ parent
            if revolute:
                axes.append((pose[:3, 0].copy(), pose[:3, 3].copy()))
                c, s = np.cos(joints[k]), np.sin(joints[k])
                turn = np.eye(4)
                turn[1:3, 1:3] = [[c, -s], [s, c]]
                pose = pose @ turn
                k += 1
            pose = pose @ child_inverse
        if not jacobian:
            return pose
        columns = [np.r_[np.cross(axis, pose[:3, 3] - origin), axis] for axis, origin in axes]
        return pose, np.array(columns).T

    def solve(self, target, seed, iterations=60, damping=0.02, max_step=0.3):
        """Six-joint IK from ``seed`` (the locked joint keeps its seed value).

        Returns the joints with the smallest residual found within limits, and
        that residual as (position m, rotation rad).
        """
        free = [i for i in range(len(self.limits)) if i != LOCKED]
        joints = np.clip(np.asarray(seed, dtype=float), self.limits[:, 0], self.limits[:, 1])
        joints[LOCKED] = seed[LOCKED]
        best = (np.inf, joints.copy(), (np.inf, np.inf))
        for _ in range(iterations):
            pose, jac = self.fk(joints, jacobian=True)
            error = np.r_[target[:3, 3] - pose[:3, 3], rotation_vector(target[:3, :3] @ pose[:3, :3].T)]
            residual = (float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
            score = residual[0] + 0.1 * residual[1]
            if score < best[0]:
                best = (score, joints.copy(), residual)
            if residual[0] < POSITION_M / 10 and residual[1] < ROTATION_RAD / 10:
                break
            jac = jac[:, free]
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), error)
            step *= min(1.0, max_step / max(np.abs(step).max(), 1e-12))
            joints[free] = np.clip(joints[free] + step, self.limits[free, 0], self.limits[free, 1])
        return best[1], best[2]


def grasp_index(gripper_command, object_z, lift=0.02):
    """First step at which the closed gripper has raised the object by ``lift`` meters (as the generator)."""
    held = (np.asarray(gripper_command) < 0) & (np.asarray(object_z) - object_z[0] > lift)
    if not held.any():
        raise ValueError("the object is never lifted with the gripper closed")
    return int(np.argmax(held))


def load_source(path, episode, object_name, container_name, arm):
    """Commanded hand poses, pre-action joints, first placement step (``grasp``) and reference poses of one source."""
    with h5py.File(path, "r") as stream:
        group = stream["data"][episode]
        actions = np.asarray(group["actions"])
        initial, states = group["initial_state"], group["states"]

        def rows(kind, name, field):
            return np.concatenate((np.asarray(initial[kind][name][field])[:1], np.asarray(states[kind][name][field])[:-1]))

        joints = rows("articulation", "robot", "joint_position")[:, :len(ARM)]
        poses = {name: rows("rigid_object", name, "root_pose") for name in (object_name, container_name)}
    # The generator's grasp segment includes the step that raises the signal, so placement starts one later.
    grasp = grasp_index(actions[:, -1], poses[object_name][:, 2]) + 1
    if not 0 < grasp < len(actions):
        raise ValueError("the grasp boundary leaves an empty segment")
    commanded = actions[:, :len(ARM)]
    return {"hand": np.array([arm.fk(row) for row in commanded]), "joints": joints, "grasp": grasp,
            "achieved": np.array([arm.fk(row) for row in joints]), "actions": actions,
            "object_pose": poses[object_name], "container_pose": poses[container_name],
            "references": (pose_matrix(poses[object_name][0]), pose_matrix(poses[container_name][grasp])),
            "steps": len(actions)}


def transformed_path(source, target, object_name, container_name):
    """Hand poses at the target: the grasp segment moves with the object, the placement segment with the container."""
    grasp, hand = source["grasp"], source["hand"]
    moves = [pose_matrix(target[name]) @ np.linalg.inv(reference)
             for name, reference in zip((object_name, container_name), source["references"])]
    return np.concatenate((moves[0] @ hand[:grasp], moves[1] @ hand[grasp:]))


def screen_pair(arm, source, target, object_name, container_name, stride=2):
    """Reachability of one source's path at one target."""
    path, grasp = transformed_path(source, target, object_name, container_name), source["grasp"]
    worst = {"position_m": 0.0, "rotation_rad": 0.0, "step": None}
    unreached, joints, previous_step, jump = 0, None, None, 0.0
    steps = sorted({*range(0, len(path), stride), grasp - 1, grasp, len(path) - 1} - {-1})
    for step in steps:
        # Continue from the previous solution within a segment; each segment restarts from the source's joints.
        seeds = [source["joints"][step]] if joints is None or step == grasp else [joints, source["joints"][step]]
        solution, residual = min((arm.solve(path[step], seed) for seed in seeds),
                                 key=lambda item: item[1][0] + 0.1 * item[1][1])
        reached = residual[0] <= POSITION_M and residual[1] <= ROTATION_RAD
        unreached += not reached
        if residual[0] + 0.1 * residual[1] > worst["position_m"] + 0.1 * worst["rotation_rad"]:
            worst = {"position_m": residual[0], "rotation_rad": residual[1], "step": int(step)}
        if joints is not None and step != grasp and reached:
            jump = max(jump, float(np.abs(solution - joints).max()) / (step - previous_step))
        joints, previous_step = solution, step
    start = arm.fk(source["joints"][0])
    bridges = {}
    for name, first, second in (("start", start, path[0]), ("placement", path[grasp - 1], path[grasp])):
        bridges[f"{name}_bridge_m"] = round(float(np.linalg.norm(second[:3, 3] - first[:3, 3])), 4)
        bridges[f"{name}_bridge_rad"] = round(float(np.arccos(np.clip(
            (np.trace(second[:3, :3] @ first[:3, :3].T) - 1) / 2, -1, 1))), 4)
    return {"reachable": unreached == 0, **bridges, "unreached_steps": int(unreached), "checked_steps": len(steps),
            "worst_position_m": round(worst["position_m"], 5), "worst_rotation_rad": round(worst["rotation_rad"], 5),
            "worst_step": worst["step"], "worst_segment": None if worst["step"] is None else int(worst["step"] >= grasp),
            "max_joint_change_per_step_rad": round(jump, 4)}


def _rotz(angle):
    out = np.eye(4)
    out[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return out


def _tilt_only(pose):
    """4x4 pose of a root pose with its heading about world z removed (as repairs.remove_yaw)."""
    out = pose_matrix(pose)
    yaw = np.arctan2(out[1, 0], out[0, 0])
    out[:3, :3] = _rotz(-yaw)[:3, :3] @ out[:3, :3]
    return out


def interpolate(start, end, steps):
    """``steps`` interpolation points then ``end``, as Isaac Lab emits a connection (start excluded)."""
    rotation = end[:3, :3] @ start[:3, :3].T
    vector = rotation_vector(rotation)
    angle = np.linalg.norm(vector)
    axis = vector / angle if angle > 1e-12 else np.zeros(3)
    out = []
    for k in range(1, steps + 2):
        f = k / (steps + 1)
        k_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        turn = np.eye(3) + np.sin(f * angle) * k_matrix + (1 - np.cos(f * angle)) * k_matrix @ k_matrix
        pose = np.eye(4)
        pose[:3, :3] = turn @ start[:3, :3]
        pose[:3, 3] = start[:3, 3] + f * (end[:3, 3] - start[:3, 3])
        out.append(pose)
    return out


def connection_time(start, end, speed, rate):
    distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
    angle = float(np.linalg.norm(rotation_vector(end[:3, :3] @ start[:3, :3].T)))
    return max(distance / speed, angle / rate), distance, angle


def emulate_controller(arm, poses, joints, damping=0.01, max_command_step=0.5, max_velocity=2.61, hz=15.0,
                       response=1.0):
    """Kinematic emulation of the generator's six-joint differential IK, one damped least-squares step per pose.

    The command is the current joints plus one DLS step (scaled to at most
    ``max_command_step``, as in ``mimicgen.target_eef_pose_to_action``); the
    arm then moves toward it at most at ``max_velocity``. Returns the largest
    target-to-command pose error (the quality gate's IK errors), the largest
    command beyond the joint limits and the final joints.
    """
    free = [i for i in range(len(arm.limits)) if i != LOCKED]
    joints = np.asarray(joints, dtype=float).copy()
    worst_p = worst_r = excess = 0.0
    for pose in poses:
        current, jac = arm.fk(joints, jacobian=True)
        error = np.r_[pose[:3, 3] - current[:3, 3], rotation_vector(pose[:3, :3] @ current[:3, :3].T)]
        jac = jac[:, free]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), error)
        delta /= max(1.0, np.abs(delta).max() / max_command_step)
        command = joints.copy()
        command[free] += delta
        reached = arm.fk(command)
        worst_p = max(worst_p, float(np.linalg.norm(pose[:3, 3] - reached[:3, 3])))
        worst_r = max(worst_r, float(np.linalg.norm(rotation_vector(pose[:3, :3] @ reached[:3, :3].T))))
        excess = max(excess, float(np.maximum(arm.limits[:, 0] - command, command - arm.limits[:, 1]).max()))
        # ``response``: fraction of the gap to the command the arm closes per step (0.28 measured in simulation).
        move = np.clip(response * (command - joints), -max_velocity / hz, max_velocity / hz)
        joints = np.clip(joints + move, arm.limits[:, 0], arm.limits[:, 1])
    return worst_p, worst_r, excess, joints


def repaired_screen(arm, source, target, object_name, container_name, geometry, settings, offsets_deg=None,
                    max_position_m=0.02, max_rotation_rad=0.07, hz=15.0, minimum_steps=15, response=0.28):
    """Screen one (target, source) along the repaired path, optionally over symmetric grasp headings.

    Follows the generator with ``settings`` (cuts, adaptive timing): reset pose
    to the pre-grasp pose, the grasp segment to the lift cut, the carry
    connection, then the placement from the over-bin cut. With
    ``offsets_deg`` None the grasp segment moves with the object's recorded
    yaw, as without ``--symmetric-object-yaw``; otherwise each candidate
    heading turns it about the object relative to the source. (GPU pilot
    2026-11-12: the best-ranked turned headings were accepted 4/10 against
    7/10 for the recorded yaw, so headings are not chosen by default.)
    A heading is feasible when the emulated controller (``emulate_controller``)
    stays within the IK error margins and the joint limits on every step.
    Against the GPU pilot (2026-11-12) this rejects only headings that failed
    there, but it misses failures from contacts and payload sag, so a chosen
    heading still needs generation to confirm it.
    Feasible headings are ranked by the time both connections take at the
    timing limits. Returns the best heading's record and all candidates.
    """
    from .repairs import source_cuts

    speed, rate = settings.connection_speed_m_s, settings.connection_rate_rad_s
    cuts = source_cuts(source["actions"], source["achieved"][:, :3, 3], source["object_pose"],
                       source["container_pose"], source["grasp"], geometry, settings)
    start = cuts["approach"] or 0
    lift = cuts["lift"] if cuts["lift"] is not None else source["grasp"]
    place = cuts["over_bin"] if cuts["over_bin"] is not None else lift
    move_container = pose_matrix(target[container_name]) @ np.linalg.inv(pose_matrix(source["container_pose"][place]))
    placement = [move_container @ pose for pose in source["hand"][place:]]
    source_ref = _tilt_only(source["object_pose"][start])
    reset = arm.fk(source["joints"][0])

    def steps_for(a, b):
        seconds = connection_time(a, b, speed, rate)[0]
        return max(minimum_steps, int(np.ceil(seconds * hz - 1e-9)))

    candidates = []
    for offset in [None] if offsets_deg is None else offsets_deg:
        if offset is None:
            move_object = pose_matrix(target[object_name]) @ np.linalg.inv(pose_matrix(source["object_pose"][start]))
        else:
            target_ref = _tilt_only(target[object_name])
            target_ref[:3, :3] = _rotz(np.radians(offset))[:3, :3] @ target_ref[:3, :3]
            move_object = target_ref @ np.linalg.inv(source_ref)
        # First subtask: the achieved pose at the segment start, then the commanded targets.
        grasp_path = [move_object @ source["achieved"][start]] + [move_object @ pose for pose in source["hand"][start:lift]]
        path = (interpolate(reset, grasp_path[0], steps_for(reset, grasp_path[0])) + grasp_path[1:]
                + interpolate(grasp_path[-1], placement[0], steps_for(grasp_path[-1], placement[0])) + placement[1:])
        position, rotation, excess, _ = emulate_controller(arm, path, source["joints"][0], response=response)
        ok = position <= max_position_m and rotation <= max_rotation_rad and excess <= 0.0
        t1, d1, a1 = connection_time(reset, grasp_path[0], speed, rate)
        t2, d2, a2 = connection_time(grasp_path[-1], placement[0], speed, rate)
        candidates.append({"offset_deg": None if offset is None else float(offset), "feasible": bool(ok), "connection_s": round(t1 + t2, 3),
                           "approach_m": round(d1, 4), "approach_rad": round(a1, 4),
                           "carry_m": round(d2, 4), "carry_rad": round(a2, 4),
                           "ik_position_m": round(position, 4), "ik_rotation_rad": round(rotation, 4),
                           "limit_excess_rad": round(excess, 4)})
    feasible = [c for c in candidates if c["feasible"]]
    best = min(feasible, key=lambda c: c["connection_s"]) if feasible else None
    return {"cuts": cuts, "best": best, "candidates": candidates}


def _screen_target(job):
    arm, loaded, target, object_name, container_name, stride = job
    pairs = {key: screen_pair(arm, source, target["poses"], object_name, container_name, stride)
             for key, source in loaded.items()}
    return {"id": target["id"], "reachable_sources": [key for key, pair in pairs.items() if pair["reachable"]],
            "sources": pairs}


def screen(manifest, sources, arm, stride=2, workers=1):
    """Screen every target of a manifest against every source: ``{key: (file, episode)}``."""
    object_name, container_name = manifest["sampler"]["profile"]["object"], manifest["sampler"]["profile"]["container"]
    loaded = {key: load_source(path, episode, object_name, container_name, arm)
              for key, (path, episode) in sorted(sources.items())}
    jobs = [(arm, loaded, target, object_name, container_name, stride) for target in manifest["targets"]]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_screen_target, jobs))
    else:
        rows = [_screen_target(job) for job in jobs]
    return rows, {"targets": len(rows), "targets_with_a_reachable_source": sum(bool(r["reachable_sources"]) for r in rows)}


def _rank_target(job):
    arm, loaded, target, object_name, container_name, geometry, settings = job
    rows = []
    for key, source in loaded.items():
        try:
            candidate = repaired_screen(arm, source, target["poses"], object_name, container_name, geometry,
                                        settings)["candidates"][0]
        except ValueError as error:  # a source without the profile's cuts
            candidate = {"feasible": False, "error": str(error)}
        rows.append({"key": key, **candidate})
    ranked = [row["key"] for row in sorted((r for r in rows if r["feasible"]), key=lambda r: (r["connection_s"], r["key"]))]
    return {"id": target["id"], "ranked_sources": ranked, "sources": rows}


def rank(manifest, sources, arm, profile, workers=1):
    """Per target, sources whose repaired path passes the emulated controller, by connection time."""
    from .repairs import placement_geometry, profile_settings

    profile_ = manifest["sampler"]["profile"]
    object_name, container_name = profile_["object"], profile_["container"]
    geometry = placement_geometry(manifest["geometry"][object_name], manifest["geometry"][container_name])
    settings = profile_settings(profile)
    loaded = {key: load_source(path, episode, object_name, container_name, arm)
              for key, (path, episode) in sorted(sources.items())}
    jobs = [(arm, loaded, target, object_name, container_name, geometry, settings) for target in manifest["targets"]]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(_rank_target, jobs))
    return [_rank_target(job) for job in jobs]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="Target manifest from target_sampler")
    parser.add_argument("--sources", type=Path, required=True,
                        help='JSON {"sources": [{"file": omy_demos.hdf5 | "input": recording dir, "episode": "demo_N"}]}: '
                             "a --source-selection file or the pipeline's frozen sources.json")
    parser.add_argument("--stride", type=int, default=2, help="Check every Nth step, plus the segment ends")
    parser.add_argument("--rank-profile",
                        help="Instead of the plain reachability screen, rank sources per target along the path "
                             "this generator profile (repairs.GENERATOR_PROFILES) produces")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    references = json.loads(args.sources.read_text())["sources"]
    sources = {}
    for row in references:
        file = Path(row["file"]) if row.get("file") else Path(row["input"]) / "omy_demos.hdf5"
        sources[f"{file.parent.parent.name}:{row['episode']}"] = (str(file), row["episode"])
    if args.output.exists():
        parser.error(f"{args.output} exists; screen reports are frozen once written")
    if args.rank_profile:
        from .repairs import GENERATOR_PROFILES
        rows = rank(manifest, sources, Arm.from_usd(), args.rank_profile, args.workers)
        counts = {"targets": len(rows), "targets_with_a_feasible_source": sum(bool(r["ranked_sources"]) for r in rows)}
        method = {"rank_profile": args.rank_profile, "profile_args": list(GENERATOR_PROFILES[args.rank_profile]),
                  "controller": "emulate_controller (DLS lambda 0.01, step <= 0.5 rad, response 0.28)",
                  "order": "feasible sources by total connection time at the timing limits"}
    else:
        rows, counts = screen(manifest, sources, Arm.from_usd(), args.stride, args.workers)
        method = {}
    report = {"schema_version": SCHEMA_VERSION, "method": method,
              "screen": {"module": "robolab.core.omy_teleop.path_screen", "sha256": _sha256(__file__),
                         "robot": ROBOT, "robot_sha256": _sha256(REPO / ROBOT), "locked_joint": ARM[LOCKED],
                         "position_m": POSITION_M, "rotation_rad": ROTATION_RAD, "stride": args.stride,
                         "joint_position_limits": [list(v) for v in POSITION_LIMITS]},
              "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest)},
              "sources": [{"key": key, "file": str(Path(path).resolve()), "episode": episode, "sha256": _sha256(path)}
                          for key, (path, episode) in sorted(sources.items())],
              "counts": counts, "targets": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "counts": counts}))


if __name__ == "__main__":
    main()
