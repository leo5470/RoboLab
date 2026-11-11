# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate `pi05_droid_jointpos` under PPS steering.

Identical to ``policies/pi0_family/run.py`` — same env registration, same eval loop, same
flags — except that the client tells the server how to steer each chunk. That is deliberate:
the steered arm and the base arm must be measured by the same protocol, so this file is a copy
of that one plus four flags rather than a second harness.

The server is the PPS repo's ``scripts/serve_pps.py``. Point ``--remote-port`` at it.

    # base arm (gamma=0), recording the observation corpus
    python policies/pps/run.py --task SmallPumpkinInBinTask --num-envs 5 --headless

    # steered arm
    python policies/pps/run.py --task SmallPumpkinInBinTask --mode pps --gamma 0.8 \
        --arm pumpkin --num-envs 5 --headless
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PPS_MODES = ["none", "pps", "naive"]

parser = argparse.ArgumentParser(description="Evaluate pi05_droid_jointpos under PPS steering.")
parser.add_argument("--mode", choices=PPS_MODES, default="none",
                    help=("Steering mode (default: none, i.e. the frozen base). 'pps' adds "
                          "gamma*(v_task - v_ref); 'naive' is the w/o-ref ablation, "
                          "gamma*(v_task - v_base)."))
parser.add_argument("--gamma", type=float, default=0.0,
                    help="Steering strength. The gamma curve's independent variable.")
parser.add_argument("--arm", type=str, default=None,
                    help=("Which (v_ref, v_task) pair the server should use. Required unless "
                          "--mode none; the server holds several and will not guess."))
parser.add_argument("--pps-task", "--pps_task", type=str, default=None,
                    help=("Label that salts the per-chunk noise seed. Defaults to the task's "
                          "instruction, which is stable across arms — the property the pairing "
                          "needs. Override only to deliberately re-randomise."))
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                    help="Remote host for policy server (default: localhost).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Remote port for policy server (default: 8000).")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help=("Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                          "Overrides --remote-host and --remote-port when set."))
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help=("Number of actions to execute from each predicted chunk before "
                          "requesting a new one. If omitted, the client uses its per-variant "
                          "default. Must match the model's action_horizon for best performance."))
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False).")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False).")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help=("Sample a random non-default background per task at registration time. "
                          "Each registered env gets one fixed background; the chosen texture is "
                          "recorded in the per-task env_cfg.json."))
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling. Used with --randomize-background.")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from policies.pps.client import PpsClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> PpsClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant="pi05",
        mode=args.mode,
        gamma=args.gamma,
        arm=args.arm,
        task=args.pps_task,
    )
    return PpsClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    # Every result row carries mode, gamma AND arm. Two arms swept at the same gamma would
    # otherwise be indistinguishable after the fact, and — worse — land in the same default output
    # folder, where run_evaluation SKIPS tasks it finds already complete and silently reports the
    # earlier arm's numbers as the later one's.
    if args_cli.mode == "none":
        label = "pps_base"
    else:
        label = f"pps_{args_cli.mode}_g{args_cli.gamma:g}_{args_cli.arm}"
    run_evaluation(args_cli, policy=label, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
