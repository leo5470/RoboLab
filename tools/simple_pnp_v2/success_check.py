# isort: skip_file
"""Task-goal sanity check for Simple-PnP v2 drafts (deferred plan item: task goal semantics).

    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 \
        tools/simple_pnp_v2/success_check.py --headless --tasks <TaskA> <TaskB> ... \
        --output output/simple_pnp_v2_validation/success_check

Per task, in the real RobolabEnv with the task's own `success` termination:
  A. untouched: the robot holds its start joint targets (gripper open) for --hold-s;
     the success term must never fire.
  B. placed:    the target is teleported (orientation kept, velocities zeroed) so its
     footprint centre is over the bin centre and its bottom --drop-m above the bin
     rim, then released while the robot keeps holding; the success term must fire
     within --place-s.
Pass = A never succeeds and B succeeds. This checks the success predicate against
the exact scene; it is not a policy or grasp test.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401  (must import before isaaclab)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--tasks", nargs="+", required=True)
parser.add_argument("--task-dirs", nargs="+", default=["single_pnp_v2_drafts"])
parser.add_argument("--inputs", type=Path, default=Path("output/simple_pnp_v2_validation/inputs/inputs.json"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--hold-s", type=float, default=3.0)
parser.add_argument("--place-s", type=float, default=5.0)
parser.add_argument("--drop-m", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def success_now(env):
    return bool(env.termination_manager.get_term("success")[0].item())


def main():
    inputs = json.loads((ROOT / args.inputs).read_text())
    entries = {e["prim_name"]: e for e in inputs["entries"]}
    top = inputs["table"]["top_z"]
    bg = inputs["bin"]["geometry"]
    auto_register_droid_envs(task=args.tasks, task_dirs=args.task_dirs)
    args.output.mkdir(parents=True, exist_ok=True)
    for task in get_envs(task=args.tasks):
        out = args.output / f"{task}.json"
        if out.exists():
            continue
        row = {"task": task}
        env = None
        try:
            env, cfg = create_env(task, device=args.device, num_envs=1, use_fabric=True)
            env.reset(); env.reset()
            step_s = cfg.sim.dt * cfg.decimation
            robot = env.scene["robot"]
            target = next(n for n in cfg.contact_object_list if n not in ("grey_bin", "table")) \
                if hasattr(cfg, "contact_object_list") else None
            if target is None or target not in entries:
                target = next(n for n in env.scene.rigid_objects.keys() if n in entries)
            row["target"] = target
            n_act = env.action_manager.total_action_dim
            hold = torch.zeros(1, n_act, device=env.device)
            hold[0, :7] = robot.data.joint_pos[0, :7]
            # A: untouched
            fired_a = None
            for i in range(int(round(args.hold_s / step_s))):
                env.step(hold)
                if success_now(env) or env.all_terminated:
                    fired_a = i + 1; break
            row["untouched"] = {"success_fired_at_step": fired_a, "steps": i + 1}
            if fired_a is None:
                # B: place into the bin
                obj, binobj = env.scene[target], env.scene["grey_bin"]
                g = entries[target]["geometry"]
                pose = obj.data.root_state_w[0, :7].cpu().numpy()
                bpose = binobj.data.root_state_w[0, :7].cpu().numpy()
                R = quat_matrix(pose[3:]) @ quat_matrix(g["root_rotation_wxyz"]).T
                Rb = quat_matrix(bpose[3:]) @ quat_matrix(bg["root_rotation_wxyz"]).T
                centre = R @ np.array(g["rest_center_offset_m"]) + pose[:3]
                bin_c = Rb @ np.array(bg["rest_center_offset_m"]) + bpose[:3]
                bin_top = bpose[2] + bg["rest_bounds_at_origin"]["upper"][2]
                bottom = top  # target starts resting on the tabletop
                shift = np.array([bin_c[0] - centre[0], bin_c[1] - centre[1], bin_top + args.drop_m - bottom])
                new = pose.copy(); new[:3] += shift
                obj.write_root_pose_to_sim(torch.tensor(new, dtype=torch.float32, device=env.device)[None])
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
                fired_b = None
                for j in range(int(round(args.place_s / step_s))):
                    env.step(hold)
                    if env.all_terminated or success_now(env):
                        fired_b = j + 1; break
                final = obj.data.root_state_w[0, :3].cpu().numpy().tolist()
                res = env.get_env_results()[0]
                row["placed"] = {"shift_m": shift.round(4).tolist(), "success_fired_at_step": fired_b,
                                 "env_result": res, "final_target_pos": final}
                row["status"] = "passed" if fired_b is not None and res.get("success") in (True, None) else "failed"
            else:
                row["status"] = "failed"
                row["reason"] = "success fired in the untouched scene"
        except Exception:
            row["status"] = "error"; row["error"] = traceback.format_exc()
        finally:
            out.write_text(json.dumps(row, indent=1, default=str))
            print(f"{task}: {row['status']}", flush=True)
            if env is not None:
                env.close()
    app.close()


if __name__ == "__main__":
    main()
