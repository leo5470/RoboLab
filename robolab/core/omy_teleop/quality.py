"""Quality metrics and filtering of OMY demonstrations, independent of Isaac Sim.

Works on any ``omy_demos.hdf5``: teleoperated, generated, or rendered. Generated
runs also record IK errors and tracked events in ``mimicgen_summary.json``; those
checks are skipped for recordings without them. See docs/omy_mimicgen.md.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .options import ARM_JOINTS

# Franka joint position limits in radians, for recordings that do not carry the simulated ones.
POSITION_LIMITS = ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
                   (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973))

# StatusCode names of the task event tracker that mean an unintended contact or disturbance.
# TARGET_OBJECT_DROPPED is not one of them: every release into the container raises it.
COLLISION_EVENTS = ("GRIPPER_HIT_TABLE", "GRIPPER_HIT_OBJECT", "OBJECT_BUMPED", "OBJECT_MOVED",
                    "OBJECT_OUT_OF_SCENE", "OBJECT_TIPPED_OVER", "MULTIPLE_OBJECTS_GRABBED", "WRONG_OBJECT_GRABBED")

# Largest accepted value of each metric. Accelerations and jerks are finite differences at the
# control rate, so their defaults suit 15 Hz recordings.
THRESHOLDS = {
    "command_velocity": (5.0, "rad/s", "change of the commanded joint targets per second"),
    "command_acceleration": (100.0, "rad/s^2", "second difference of the commanded joint targets"),
    "command_jerk": (2500.0, "rad/s^3", "third difference of the commanded joint targets"),
    "joint_velocity": (2.61, "rad/s", "recorded arm joint velocity"),
    "joint_acceleration": (20.0, "rad/s^2", "second difference of the measured arm joints"),
    "joint_jerk": (500.0, "rad/s^3", "third difference of the measured arm joints"),
    "limit_excess": (0.0, "rad", "distance of a commanded joint target beyond its position limit"),
    "displacement": (0.03, "m", "movement of any object other than the grasped one"),
    "ik_position_error": (0.03, "m", "distance between a target pose and the pose of its joint command"),
    "ik_orientation_error": (0.1, "rad", "rotation between a target pose and the pose of its joint command"),
}


def difference_peaks(values, hz):
    """Largest absolute first, second, and third finite difference of each row sequence, per second."""
    return tuple(float(np.abs(np.diff(values, order, axis=0)).max(initial=0.0)) * hz ** order for order in (1, 2, 3))


def demo_metrics(episode, hz, position_limits=POSITION_LIMITS, grasped=None):
    """Motion and joint-limit metrics of one recorded episode, and scene disturbance given the grasped objects."""
    arm = len(ARM_JOINTS)
    actions = np.asarray(episode["actions"])[:, :arm]
    robot, initial = episode["states"]["articulation"]["robot"], episode["initial_state"]
    # The arm joints lead the articulation's joint vector, in the order of the action.
    joints = np.concatenate((np.asarray(initial["articulation"]["robot"]["joint_position"])[:1, :arm],
                             np.asarray(robot["joint_position"])[:, :arm]))
    limits = np.asarray(position_limits)
    metrics = dict(zip(("command_velocity", "command_acceleration", "command_jerk"), difference_peaks(actions, hz)))
    metrics["joint_velocity"] = float(np.abs(np.asarray(robot["joint_velocity"])[:, :arm]).max())
    metrics["joint_acceleration"], metrics["joint_jerk"] = difference_peaks(joints, hz)[1:]
    metrics["limit_excess"] = float(np.maximum(limits[:, 0] - actions, actions - limits[:, 1]).max())
    # Without the grasped object's name, its own transport would count as a disturbance.
    moved = {} if grasped is None else {
        name: float(np.linalg.norm(np.asarray(state["root_pose"])[:, :3]
                                   - np.asarray(initial["rigid_object"][name]["root_pose"])[0, :3], axis=1).max())
        for name, state in episode["states"]["rigid_object"].items() if name not in grasped}
    if moved:
        metrics["displaced_object"] = max(moved, key=moved.get)
        metrics["displacement"] = moved[metrics["displaced_object"]]
    return metrics


def generation_records(summary):
    """IK errors and tracked events of generated demonstrations, by episode name."""
    records = {}
    for attempt in (summary or {}).get("attempts", []):
        if attempt.get("demo") is not None:
            records[attempt["demo"]] = {
                "ik_position_error": attempt.get("ik_position_error_m"),
                "ik_orientation_error": attempt.get("ik_orientation_error_rad"),
                "events": attempt.get("events"),
            }
    return records


def assess(path, thresholds, summary=None, grasped=None, events=COLLISION_EVENTS):
    """Metrics and failed checks of every episode in a recording, by episode name."""
    records = generation_records(summary)
    results = {}
    with h5py.File(path, "r") as file:
        data = file["data"]
        hz = json.loads(data.attrs["robolab_omy_teleop"])["control_hz"]
        generated = json.loads(data.attrs.get("robolab_omy_mimicgen", "{}"))
        limits = generated.get("joint_position_limits", POSITION_LIMITS)
        if grasped is None:
            grasped = [generated["object"]] if "object" in generated else None
        for name in sorted(data, key=lambda key: int(key.rsplit("_", 1)[1])):
            metrics = demo_metrics(data[name], hz, limits, grasped)
            record = records.get(name, {})
            metrics.update({key: record[key] for key in ("ik_position_error", "ik_orientation_error")
                            if record.get(key) is not None})
            failed = [key for key, limit in thresholds.items() if metrics.get(key, -np.inf) > limit]
            if record.get("events") is not None:
                metrics["events"] = sorted({event["name"] for event in record["events"]} & set(events))
                failed += metrics["events"]
            if not data[name].attrs.get("success", False):
                failed.append("unsuccessful")
            results[name] = {"metrics": metrics, "failed": failed}
    return results


def write_kept(source, output, kept, provenance):
    """Copy the kept episodes, under their original names, to a new recording."""
    with h5py.File(source, "r") as src, h5py.File(output, "x") as dst:
        data = dst.create_group("data")
        for key, value in src["data"].attrs.items():
            data.attrs[key] = value
        for name in kept:
            src.copy(src["data"][name], data, name)
        data.attrs["total"] = sum(int(data[name].attrs["num_samples"]) for name in kept)
        data.attrs["robolab_omy_quality"] = json.dumps(provenance)


def build_parser():
    parser = argparse.ArgumentParser(description="Report and filter OMY demonstrations by motion quality.")
    parser.add_argument("input", help="Recording (omy_demos.hdf5) to assess.")
    parser.add_argument("--output", help="New recording holding only the accepted episodes.")
    parser.add_argument("--report", help="JSON file for per-episode metrics and failed checks.")
    parser.add_argument("--summary", help="Generation summary with IK errors and events; "
                                          "default: mimicgen_summary.json beside the input.")
    parser.add_argument("--object", action="append", dest="grasped",
                        help="Object excluded from the displacement check; repeatable. "
                             "Default: the grasped object named by a generated recording.")
    parser.add_argument("--allow-event", action="append", default=[], choices=COLLISION_EVENTS,
                        help="Tracked event that does not reject an episode; repeatable.")
    for name, (default, unit, text) in THRESHOLDS.items():
        parser.add_argument(f"--max-{name.replace('_', '-')}", type=float, default=default,
                            help=f"Largest accepted {text} in {unit} (default: {default}; inf disables).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary) if args.summary else Path(args.input).parent / "mimicgen_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    thresholds = {name: getattr(args, f"max_{name}") for name in THRESHOLDS}
    events = [name for name in COLLISION_EVENTS if name not in args.allow_event]
    results = assess(args.input, thresholds, summary, args.grasped, events)
    kept = [name for name, result in results.items() if not result["failed"]]
    counts = {}
    for name, result in results.items():
        if result["failed"]:
            details = ", ".join(f"{key} {result['metrics'][key]:.3g}" if key in thresholds else key
                                for key in result["failed"])
            print(f"reject {name}: {details}")
        for key in result["failed"]:
            counts[key] = counts.get(key, 0) + 1
    print(f"\n{'metric':24s}{'median':>10s}{'p90':>10s}{'max':>10s}{'limit':>10s}{'rejected':>10s}")
    for name, limit in thresholds.items():
        values = [result["metrics"][name] for result in results.values() if name in result["metrics"]]
        if not values:
            print(f"{name:24s}{'not recorded':>30s}")
            continue
        median, high = np.percentile(values, (50, 90))
        print(f"{name:24s}{median:10.3g}{high:10.3g}{max(values):10.3g}{limit:10.3g}{counts.get(name, 0):10d}")
    for name in sorted(set(counts) - set(thresholds)):
        print(f"{name:24s}{'':40s}{counts[name]:10d}")
    print(f"\nAccepted {len(kept)} of {len(results)} episodes.")
    provenance = {"schema_version": 1, "source": str(Path(args.input).resolve()), "thresholds": thresholds,
                  "rejecting_events": events, "accepted": len(kept), "rejected": counts}
    if args.report:
        Path(args.report).write_text(json.dumps({**provenance, "episodes": results}, indent=2) + "\n")
    if args.output:
        write_kept(args.input, args.output, kept, provenance)
        print(f"Wrote {args.output}")
    return 0
