# OMY-L100 teleoperation

This is a dedicated OMY path for RoboLab's simulated DROID robot (Franka arm +
Robotiq 2F-85). It uses
[charlie8612/omy_leader_isaaclab](https://github.com/charlie8612/omy_leader_isaaclab)
directly, pinned to commit `ed4608bd8c49133e3b9fc9d231bee651d8489b65` in
[`requirements-omy.txt`](../requirements-omy.txt).

There is no dependency on the keyboard/SO-101 collectors, relative IK, or deferred
image collection. For Quest 3S head-tracked viewing with this same controller,
use [`examples/teleop_omy_xr.py`](../examples/teleop_omy_xr.py), documented in
[Quest viewing and L100 control](quest_viewing.md#l100-control-in-xr).
These runners control the simulated DROID, not a physical Franka.

## Install

Start with a working RoboLab/Isaac Lab environment, then install the optional
upstream package into that same environment:

```bash
source .venv/bin/activate
uv pip install -r requirements-omy.txt
# Alternatively: python -m pip install -r requirements-omy.txt
python scripts/patch_omy.py
export OMNI_KIT_ACCEPT_EULA=Y
```

Use the activated environment's `python` and `omy-leader-*` executables below.
The OMY dependency is separate from the base lockfile: `uv sync` can remove it;
reinstall it afterward, or use `uv run --no-sync` to run these commands.
No LeRobot installation is required for the native serial reader. Upstream's
optional LeRobot publisher remains available if separately installed.

The tracked `patches/omy-leader-trigger.patch` corrects upstream's serial trigger
and is applied idempotently by `scripts/patch_omy.py`. Reapply after reinstalling
the dependency. It uses `Drive_Mode=1`, `Homing_Offset=100` with torque disabled,
and the plugin's RANGE_0_100 conversion (`apply_drive_mode=False`). The default
`--gripper-open 48.2` gives `int(48.2/100*4095) = 1973` goal ticks. The goal is
written before enabling spring torque, then reasserted and checked after enabling
(the XC330 resets its goal on torque enable). Background readers stop before torque-off.

Gripper values retain the legacy publisher's **half-angle** units:
`(Present_Position - goal_ticks) * pi / 4095`, zero released and positive squeezed.
Thus upstream's example calibration and mirror's 10-degree threshold retain their
units. Arm readings remain full radians. Calibration captured with the old broken
serial reader is invalid for the gripper: redo only its endpoints, keeping arm zeros.
The `--gripper-open` option is passed through the RoboLab runner and publisher.
To check the physical trigger without Isaac Sim:

```bash
python -m omy_leader_isaaclab.omy_serial /dev/robotis_left --gripper-open 48.2
# Ctrl+C disables gripper torque and closes the port.
omy-leader-calib --gripper-only --out omy_calib_robolab.json
```

## Run with a local leader

### Select a task folder without importing unrelated tasks

OMY supports the evaluation CLI spelling `--task-dirs` (alias `--task_dirs`).
Folders are relative to `robolab/tasks`; the default remains `benchmark`.
For generated tasks, use e.g. `--task-dirs single_pnp --task SimplePnp064ComputerMouse02Task`.
The runner reads that folder's `manifest.json`, validates the selected file and
class, and imports only that task. Folders without a manifest use static Python
declaration lookup, not task execution. Multiple folders are accepted, but
ambiguous matches are refused. A manifest-backed folder's task list is
authoritative; regenerate its manifest after adding or renaming a task.

The active `single_pnp` pack comes from
`origin/datasets/simple-pnp-launch-filtered-20261017` at commit `71ec4d2c`.
Use its 6,754 retained task/scene pairs and committed manifests as the source of
truth. The previous local set had 7,200 layouts; origin excludes 446, including
computer-mouse layout `00`. Layout `02` in the examples is retained. See
`robolab/tasks/single_pnp/README.md` and `exclusions.json` for filtering details.

Lookup happens before SimulationApp startup. Logs report lookup, selected-task
import, configuration registration, and environment startup separately. This
does not change task settings, physics, control timing, or default policy
evaluation registrations. No `python -c` folder workaround is needed.

Recordings retain the canonical task name plus the selected directories and a
portable task-file path/SHA-256. Offline image rendering reuses that selection
and refuses a changed task definition. Older recordings without these optional
fields remain supported and default to the benchmark folder.

```bash
python examples/teleop_omy.py \
  --task BananaInBowlTask \
  --source serial --port /dev/robotis_left \
  --calib /absolute/path/to/omy_calib.json
```

The serial device must be accessible to your user. Calibration is the upstream
JSON format; use the upstream `omy-leader-calib` / `omy-leader-monitor` tools and
commission each joint direction and gripper endpoint before collecting data.
Omitting `--calib` uses upstream defaults, not a calibration inferred for your
specific leader.

Control starts immediately. Ctrl+C stops, closes the reader, and finalizes any
active recording as an unsuccessful attempt. `--steps N` stops after N total
control steps. There are no added clutch, reanchor, or keyboard-reset controls.
Task success/failure conditions remain RoboLab's; the time budget is 10,000
simulation seconds, matching upstream teleop. Set `--episode-length-s N` to use
a shorter budget. `--num-episodes N` resets after each completed attempt, using
consecutive seeds. Ctrl+C ends the run instead of starting another attempt.

The serial reader configures the leader hardware: upstream disables arm torque
and enables the gripper's current-based spring. It is not a software gravity
compensator. Support/position the arm appropriately before connecting.
The current `omy_calib_robolab.json` anchors the new hand-held READY captured on
2026-09-24 to the default DROID pose: Franka J1–J7 =
`[0°, −36°, 0°, −144°, 0°, +108°, 0°]`, with the gripper released/open.
Keep the leader supported in that READY pose while the simulator starts. The
software does not latch a physical pose: once teleoperation begins, moving or
releasing the unsupported leader moves the simulated arm.

This replaces the earlier READY calibration, including its J6 = +90° reference.
The L100 joint_6 direction was inverted after operator testing found the
gripper wrist rotation reversed. Its software zero was recalculated from the
same READY reading, so Franka J7 still starts at 0°. The other joint directions,
gripper endpoints, and motor settings were preserved. The read-only capture and
previous calibration are saved under `output/omy_ready_captures/` as
`ready_20260924_132932.json` and
`omy_calib_robolab.before_droid_20260924_132932.json`.
Use the saved calibration without `--auto-zero` to reproduce this READY pose.

### Preserved control contract

| Property | Behavior |
|---|---|
| Device | Local `RoboLabOmyLeaderDevice` home/reset adapter over native `OmyLeaderDevice`, `target="franka"` |
| Physics / control | 120 Hz physics; `--control-hz 60` (default), `30`, or `15` selects 2, 4, or 8 physics steps per action |
| Action | `[panda_joint1 … panda_joint7, gripper]`, absolute radians, no default offset or scaling |
| Gripper | Upstream `+1` open / `-1` closed; Robotiq actuator targets 0 / π/4 |
| Reset pose | Selected DROID environment's initial joint positions; default `[0°, −36°, 0°, −144°, 0°, +108°, 0°]` |
| Mapping | Upstream calibration, direct/ZYZ wrist mode, clamps, and per-step rate limiting |
| Missing/stale input | Hold the last command after `--stale-s` (default 0.5 s); resume automatically |
| Startup auto-zero | Optional `--auto-zero` anchors the held pose to DROID home; direct wrist mode only, with `--auto-zero-settle-s` (default 3 s) |
| Episode reset | Environment, mapper and cached command reset to DROID home/open; calibration is retained |

With the default direct wrist mode and calibrated leader coordinates
`u = sign × (raw − zero)`, the arm target before limits is
`[u1, u2, 0, −u3, u5, r6 − u4, π/4 − u6]`, where `r6` is the calibration's
`j6_reset_deg` converted to radians (upstream default 180°, current calibration
108°). Runtime mapping and rate limiting still call upstream unchanged. The
local calibration helper inverts this map to compute READY offsets; in
particular, the joint-6 zero includes the wrist offset needed for Franka J7 = 0°.

The home adapter initializes the limiter and startup command at the same pose
as the environment, instead of upstream's cube-demo home. A stale input holds
the last command during an episode; an explicit episode reset clears that
cache to DROID home/open until fresh readings arrive. Optional startup
auto-zero captures once after settling, adjusts all six offsets to the selected
home, and retains them across episode resets. It does not change the saved
calibration file. ZYZ auto-zero is rejected before opening the input source;
pre-calibrated ZYZ mapping remains available without auto-zero.
Disconnect/hold is not an emergency stop. Do not leave the operator setup
unattended.

The robot asset, Robotiq mount, DROID actuators, table, task objects, and task
success checking remain RoboLab's. This preserves upstream command semantics;
it does not claim identical contact dynamics to upstream's Panda stack scene.
The runner reports rolling wall-clock control/render/produced-video rates,
host timing breakdowns, and joint tracking error every five seconds.
Rendering/recording can make it slower than nominal 60 Hz; the
upstream limiter uses the simulated 1/60 s step, not elapsed wall time.

## OMY performance controls

For the measured single-robot CPU-physics / GPU-rendering profile:

```bash
python examples/teleop_omy.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib omy_calib_robolab.json \
  --headless --stream --performance-profile cpu
```

This does not open the GUI; connect the existing `omy-leader-viewer` below.
No change to the JPEG protocol or the viewer is required. X11 forwarding still
works, but a locally displayed viewer through an SSH tunnel avoids forwarding
the remote GUI's decoded pixels.

The four OMY-only changes are:

1. **Bounded rolling telemetry.** Reports cover the last 300 control steps and
   print every `--stats-interval` seconds (default 5). Control Hz includes pacing;
   mean/p95 work time excludes deliberate pacing sleep. Region timings include
   physics, rendering, sensor updates, observations, task checks, recording,
   camera readback/composition, JPEG encoding, and remaining loop work. They are
   exclusive host elapsed times, including synchronization, not GPU kernel
   timings. No extra CUDA synchronization is inserted. Initialization and episode
   resets are excluded. Video FPS means produced JPEGs, **not** received or X11
   displayed frames; network/presentation latency is not measured here.
2. **Opt-in CPU profile.** `--performance-profile cpu` selects CPU physics and
   tensors, with RTX rendering on GPU 0, disables renderer multi-GPU, and supplies
   eight Kit/TBB workers and two physics workers (bounded by CPU affinity). These
   are process-local settings; saving persistent preferences is disabled. An
   explicit `--device` wins: `--device cuda:1` keeps CUDA and skips CPU profile
   tuning. Explicit `--kit_args` follow and override profile defaults. The default
   `--performance-profile standard` retains Isaac Lab's device/worker selection.
3. **Current contact forces without unused history.** OMY runs default to
   `--contact-history-length 0`, instead of six historical force samples. Current
   forces and task predicates are retained for active sensors.
   Air-time-tracking sensors are left unchanged. Use `--contact-history-length 6`
   for an A/B comparison or a custom task consuming force history. This override
   is applied to the selected OMY instance, never shared DROID/task defaults.
4. **Render/capture alignment.** For stream-only runs, automatic render interval
   is `2 * --video-every` physics steps: the default sends and renders once every
   two control steps (four physics steps). Capture follows an actual render on
   the simulator's counter, including across episode resets. With
   `--video-every 1`, automatic rendering returns to every control step. An
   explicit `--render-interval` must be a positive multiple of two, and must
   divide `2 * --video-every` when streaming. `--record-images` always defaults
   to interval two and rejects other explicit intervals, preserving a render
   every recorded control step; video sending can still be decimated.
   For decimated streaming, Kit's additional render-rate limiter is disabled
   through session-only launch defaults: the existing OMY control-loop pacer
   already limits real-time execution. Physics and render timesteps are not
   changed by this pacing adjustment. An explicit
   `--kit_args="--/app/runLoops/main/rateLimitEnabled=true"` can override it.

`--contact-sensors required` (default) derives used pairs from the task's
termination/subtask functions and observation/recorder/event consumers, rather
than its name. All gripper/distractor sensors and required force-vector pairs
are retained. Unknown consumers retain the full set; `--contact-sensors full`
restores it explicitly for A/B comparison. Validated batch contact queries
fall back to individual sensors if the PhysX filter layout differs. Held-target
and compact-loop capabilities have independent checks and are logged per feature.
Neither sensor pruning nor query reuse removes colliders or changes physics.
The comparison procedure is in
[the XR collection guide](omy_xr_collection.md).

For the earlier CUDA simulation settings, with the new telemetry only:

```bash
python examples/teleop_omy.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib omy_calib_robolab.json --headless --stream \
  --performance-profile standard --device cuda:1 \
  --contact-history-length 6 --render-interval 2
```

Every run saves effective simulation configuration in `env_cfg.json` and, on
normal/interrupted shutdown, a final rolling summary and session counts in
`omy_performance.json`. The latter includes actual worker/renderer settings.
Recording metadata also includes the performance configuration. Neither profile
changes the physics timestep, action decimation, solver iteration settings,
actuator gains, mapping, calibration, observation contents, or JPEG quality.

Earlier short stationary probes reached approximately 42 Hz with CPU physics,
bounded workers, zero contact history, and render interval four. That is not a
guarantee for active grasping, full-resolution image recording, or X11 display.
Check the rolling rates and test grasp/lift/hold/release before collecting data;
CPU and CUDA solvers are not assumed to produce identical trajectories.

The implemented profile was also checked through this actual CLI with a
synthetic TCP leader and a local JPEG receiver: a 600-step stationary run
produced/received all 300 frames, with approximately 41 Hz control in the final
300-step window and 20.8 FPS steady local decoding. This is not a measurement of
remote X11 presentation. A pre-fix run reached only 27 Hz because initializing
render interval four also configured Kit's render-rate limiter; disabling that
second limiter was necessary to obtain the improvement without altering dt.

## Remote leader and operator view

On the operator computer with the leader connected and upstream installed:

```bash
omy-leader-publisher --port /dev/robotis_left --tcp-port 5555
```

In another operator terminal, forward the leader to the simulator and the video
back to the operator (replace `sim-host` with your SSH host):

```bash
ssh -N -R 5555:127.0.0.1:5555 -L 5556:127.0.0.1:5556 sim-host
```

On the simulation host:

```bash
python examples/teleop_omy.py \
  --task BananaInBowlTask --source tcp \
  --calib /absolute/path/to/omy_calib.json --headless --stream
```

Then on the operator computer:

```bash
omy-leader-viewer --host 127.0.0.1 --port 5556
```

The upstream JPEG/TCP protocol is unchanged: a 640×480 overview with a 240×180
wrist inset, sent every two control steps by default (`--video-every`). The wrist
camera uses the DROID/Robotiq mounting transform and lens. Operator cameras are
separate from policy observation cameras. Both default TCP endpoints are
loopback; use SSH rather than exposing the unauthenticated protocols publicly.
This is a 2D operator viewer, not Quest stereo/WebXR support. The upstream video
server uses daemon threads and lives for the duration of this CLI process.

## Record demonstrations

```bash
python examples/teleop_omy.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib /absolute/path/to/omy_calib.json \
  --record --save-failures --output-dir output/omy_banana_001
```

An output directory must be new or empty; existing data is never resumed or
overwritten by this runner. Without `--output-dir`, a unique timestamped
directory is created under `output/omy_teleop/`. Each run saves `env_cfg.json`
and a shutdown performance summary, `omy_performance.json`.
With `--record`, it also saves:

- `omy_demos.hdf5`: successful episodes. With `--save-failures`, also includes
  failed, timed-out, and interrupted episodes, identified by their `success`
  attribute. Otherwise those episodes are discarded.
- Native eight-value actions, initial state, post-step states, proprioception,
  end-effector/root poses, and available RoboLab subtask progress.
- `data.attrs["robolab_omy_teleop"]`: JSON describing the upstream revision,
  action ordering/units/gripper signs, device settings, and calibration file
  content/hash, selected home, and reset behavior. Startup auto-zero changes are held inside the device;
  this metadata records the supplied calibration, not the resulting live zero
  offsets. The actual commands remain recorded verbatim.

Add `--record-images` to record live DROID wrist and over-shoulder images in the
same episodes. It requires `--record`, enables cameras, and uses LZF compression
with periodic flushing. This does not use deferred image rendering. Full-size
policy images can substantially reduce throughput and increase disk use; check
the reported control rate. `--stream` alone does not record policy images.

Replay must start from a matching `register_omy_envs(...)` configuration and
restore the saved `env_cfg.json` and initial state. The generic
`examples/run_recorded.py` currently constructs standard DROID configurations;
it is not an OMY replay entry point. Never replay OMY actions with an unmodified
standard DROID action config: its gripper has a different numeric convention,
and optional camera groups also differ. See [Replaying Recorded Episodes](replay.md)
for the reusable sidecar/state-restoration APIs and simulator-version limitations.

## Hardware-free checks

The upstream fake publisher exercises the same TCP device path:

```bash
# Terminal 1: upstream synthetic leader; no physical hardware
omy-leader-fake --profile franka --tcp-port 5555

# Terminal 2: simulate and save the interrupted smoke-test attempt
python examples/teleop_omy.py --headless --source tcp \
  --steps 120 --record --save-failures --output-dir /tmp/omy_smoke_001
```

The fake publisher's `franka` profile was defined for upstream's wrist model;
do not assume that its rest readings map to DROID home in the default direct
wrist mode. It is a transport/motion check, not a physical calibration.

```bash
python -m pytest --confcutdir=tests/unit tests/unit/test_omy_calibration.py tests/unit/test_omy_options.py tests/unit/test_omy_launcher.py tests/unit/test_omy_performance.py tests/unit/test_omy_trigger.py -q
python -m pytest tests/test_omy_teleop.py -q
```

The integration tests require Isaac Sim and the optional OMY dependency, but
never open a physical serial device. They check factory/config compatibility,
upstream mapping, the DROID home/reset adapter and auto-zero, action/gripper
interpretation in simulation, and recording provenance. Physical leader commissioning is still
required before collecting real demonstrations.

Validated on Isaac Sim 5.0 / Isaac Lab 2.2 with CPU and CUDA physics, the upstream
fake TCP publisher/viewer, live camera recording, and consecutive episode resets.
The 65-step CPU-physics run with full-resolution recording and operator video ran
at roughly 4–6 Hz; nominal 60 Hz is a control configuration, not a demonstrated
end-to-end throughput guarantee. The local leader's trigger and READY calibration
have been measured separately; live grasping with the new READY remains an
operator check. Quest remains untested.
