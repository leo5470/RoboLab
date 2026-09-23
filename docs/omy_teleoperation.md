# OMY-L100 teleoperation

This is a dedicated OMY path for RoboLab's simulated DROID robot (Franka arm +
Robotiq 2F-85). It uses
[charlie8612/omy_leader_isaaclab](https://github.com/charlie8612/omy_leader_isaaclab)
directly, pinned to commit `ed4608bd8c49133e3b9fc9d231bee651d8489b65` in
[`requirements-omy.txt`](../requirements-omy.txt).

There is no dependency on the keyboard/SO-101 collectors, relative IK, or deferred
image collection. Quest 3S integration is not part of this first phase. This
runner does not control a physical Franka.

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
For this leader, the calibrated J6 = +90° READY pose is hand-held; its L100
joint_4 settles roughly 58° away when released. Keep the leader supported in
READY while the simulator starts. Once teleoperation begins, physical movement
of joint_4 changes Franka J6; the software does not latch the READY pose.

### Preserved control contract

| Property | Behavior |
|---|---|
| Device | Native `OmyLeaderDevice`, `target="franka"` |
| Physics / control | 120 Hz physics, two physics steps per action, nominal 60 Hz control |
| Action | `[panda_joint1 … panda_joint7, gripper]`, absolute radians, no default offset or scaling |
| Gripper | Upstream `+1` open / `-1` closed; Robotiq actuator targets 0 / π/4 |
| Reset pose | Upstream `FRANKA_HOME`: `(0, −π/4, 0, −3π/4, 0, π/2, π/4)` |
| Mapping | Upstream calibration, direct/ZYZ wrist mode, clamps, and per-step rate limiting |
| Missing/stale input | Hold the last command after `--stale-s` (default 0.5 s); resume automatically |
| Startup auto-zero | Optional upstream `--auto-zero`, with `--auto-zero-settle-s` (default 3 s) |
| Episode reset | Environment resets to home; `device.reset()` resets the mapper, not calibration or its last-command cache |

With the default direct wrist mode and calibrated leader coordinates
`u = sign × (raw − zero)`, the arm target before limits is
`[u1, u2, 0, −u3, u5, π − u4, π/4 − u6]`. The RoboLab integration does not copy
or reimplement that calculation; it calls upstream.

Important upstream behavior is deliberately retained: direct-mode auto-zero
zeros all three wrist readings, so with the default `j6_reset_deg=180` its
neutral J6 target is π, not the home pose's π/2. Choose/validate your upstream
calibration accordingly. Also, if the stream is stale during a subsequent
episode reset, the retained pre-reset target can be sent until fresh readings
arrive. Disconnect/hold is not an emergency stop. Do not leave the operator
setup unattended.

The robot asset, Robotiq mount, DROID actuators, table, task objects, and task
success checking remain RoboLab's. This preserves upstream command semantics;
it does not claim identical contact dynamics to upstream's Panda stack scene.
The runner reports actual wall-clock control rate and joint tracking error every
five seconds. Rendering/recording can make it slower than nominal 60 Hz; the
upstream limiter uses the simulated 1/60 s step, not elapsed wall time.

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
directory is created under `output/omy_teleop/`. Each run saves `env_cfg.json`.
With `--record`, it also saves:

- `omy_demos.hdf5`: successful episodes. With `--save-failures`, also includes
  failed, timed-out, and interrupted episodes, identified by their `success`
  attribute. Otherwise those episodes are discarded.
- Native eight-value actions, initial state, post-step states, proprioception,
  end-effector/root poses, and available RoboLab subtask progress.
- `data.attrs["robolab_omy_teleop"]`: JSON describing the upstream revision,
  action ordering/units/gripper signs, device settings, and calibration file
  content/hash. Startup auto-zero changes are held inside the upstream device;
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
do not assume that its rest readings map to `FRANKA_HOME` in the default direct
wrist mode. It is a transport/motion check, not a physical calibration.

```bash
python -m pytest --confcutdir=tests/unit tests/unit/test_omy_options.py tests/unit/test_omy_launcher.py -q
python -m pytest tests/test_omy_teleop.py -q
```

The integration tests require Isaac Sim and the optional OMY dependency, but
never open a physical serial device. They check factory/config compatibility,
upstream mapping and reset/stale semantics, action/gripper interpretation in
simulation, and recording provenance. Physical leader commissioning is still
required before collecting real demonstrations.

Validated on Isaac Sim 5.0 / Isaac Lab 2.2 with CPU and CUDA physics, the upstream
fake TCP publisher/viewer, live camera recording, and consecutive episode resets.
The 65-step CPU-physics run with full-resolution recording and operator video ran
at roughly 4–6 Hz; nominal 60 Hz is a control configuration, not a demonstrated
end-to-end throughput guarantee. Physical serial hardware and Quest remain untested.
