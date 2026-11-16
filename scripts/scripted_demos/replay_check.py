"""Phase 5 step 3 / spec §5.2: replay recorded ``actions`` in the same (stock) registration and compare.

For each episode: reset the episode's layout task (resets are deterministic), step the recorded actions, and compare
the measured arm joints before every step with the recorded ``joint_position`` (RMS, rad) and the final outcome with
the recorded success. Rendering is off: it does not affect physics.

    scripts/scripted_demos/run.sh scripts/scripted_demos/replay_check.py --file <pps.hdf5> [--episodes 5 | --names episode_000005 ...]
        [--out DIR] [--repeat 2]

Episodes on sampled scenes (``scene`` attribute) get the same object placement after the reset as when they were
recorded. Writes ``<out>/replay_check.json`` (default ``--out``: the file's directory). ``--repeat`` replays each episode that
many times, which separates simulator run-to-run differences from differences in the commands.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np

from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos import collect, sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--names", nargs="+", default=None, help="these episodes instead of the first --episodes")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="report directory (default: the file's)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fresh", action="store_true",
                        help="a new env for every replay: the simulator's outcome depends on the env's history of "
                             "resets, so an episode recorded as its env's first reset replays exactly only this way")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    out_dir = args.out or args.file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file, "r") as f:
        names = args.names or sorted(f["data"])[: args.episodes]
        episodes = [{"name": n, "task": f["data"][n].attrs["layout_task"], "seed": int(f["data"][n].attrs["seed"]),
                     "actions": f["data"][n]["actions"][:], "joint_position": f["data"][n]["joint_position"][:],
                     "gripper_position": f["data"][n]["gripper_position"][:],
                     "scene": json.loads(f["data"][n].attrs["scene"]) if "scene" in f["data"][n].attrs else None}
                    for n in names]
    sessions: dict[str, collect.Session] = {}
    rows = []
    for ep in [e for e in episodes for _ in range(args.repeat)]:
        task = ep["task"]
        layout = task[-6:-4]
        obj = next(o for o in objects.BATCH if task.startswith(o.family))
        if task not in sessions or args.fresh:
            for s in sessions.values():
                s.close()
            sessions = {task: collect.Session(obj, layout, render=args.render,
                                              robolab_output=out_dir / "robolab_output_replay")}
        session = sessions[task]
        session.reset(ep["seed"])
        if ep["scene"]:  # sampled scene (MimicGen --random-scenes): the same write and hold as the attempt
            session.place_objects({name: np.asarray(T) for name, T in ep["scene"]["placed"].items()},
                                  hold_steps=int(ep["scene"]["hold_steps"]))
        joints, grippers, terminated, success = [], [], None, False
        for t, action in enumerate(ep["actions"]):
            proprio = session.obs["proprio_obs"]
            joints.append(proprio["arm_joint_pos"][0].cpu().numpy())
            grippers.append(proprio["gripper_pos"][0].cpu().numpy())
            term, trunc = session.step(action)
            if term or trunc:
                terminated = t + 1
                success = bool(session.env_success())
                break
        joints = np.asarray(joints, dtype=np.float32)
        n = len(joints)
        err = joints - ep["joint_position"][:n]
        row = {"episode": ep["name"], "task": task, "seed": ep["seed"], "recorded_rows": len(ep["actions"]),
               "replayed_rows": n, "terminated_at": terminated, "success": success,
               "joint_rms_rad": float(np.sqrt(np.mean(err**2))), "joint_max_abs_rad": float(np.abs(err).max()),
               "gripper_max_abs": float(np.abs(np.asarray(grippers) - ep["gripper_position"][:n]).max()),
               "success_agrees": bool(success and terminated == len(ep["actions"]))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    for s in sessions.values():
        s.close()
    report = {"file": str(args.file), "episodes": rows,
              "max_joint_rms_rad": max(r["joint_rms_rad"] for r in rows),
              "all_success_agree": all(r["success_agrees"] for r in rows)}
    report["passed"] = bool(report["max_joint_rms_rad"] < 1e-3 and report["all_success_agree"])
    (out_dir / "replay_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        sim.close_app(status)
