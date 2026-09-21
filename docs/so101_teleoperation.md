# SO-101 Leader Teleoperation

Use [examples/collect_so101_demos.py](../examples/collect_so101_demos.py) to control the simulated DROID arm using an SO-101 leader plugged into the RoboLab server by USB.

The connected leader passed calibration-file/device matching and a 15-second hardware reader check. Physical motion direction, gripper endpoints, and the mapping to simulated DROID still need operator validation. The implementation also includes a hardware-free smoke mode and configuration for tuning the mapping.

## Connections and Wi-Fi

```text
Server USB leader -> local reader process -> Unix-domain IPC -> simulated DROID
Server USB keyboard ---------------------------------------> episode controls
Simulated scene -> WebRTC video over Wi-Fi -> viewing computer
```

With the default `--control-source evdev`, both motion and episode commands come from USB devices on the server. The collector does not subscribe to the viewing computer's keyboard for teleoperation. Wi-Fi is used for viewing the scene; WebRTC still has its normal signaling and transport traffic. The viewer can retain its normal camera navigation.

Start, retry, clutch, and inset controls are read from the physical keyboard specified by `--control-device`. The command runs in a server terminal or tmux session; that terminal does not need keyboard focus for evdev controls.

An optional `--control-source terminal` reads single keys from the foreground terminal. Using that terminal over SSH would send those keys over the network, so use evdev for the intended local-control setup.

## Test before connecting hardware

From the existing RoboLab environment:

```bash
python scripts/so101_leader_reader.py --mock --duration 2

python examples/collect_so101_demos.py \
  --leader-mock --no-record --smoke-steps 40 \
  --task BananaInBowlTask --device cpu \
  --headless --livestream 0 --defer-images \
  --output-dir /tmp/robolab-so101-smoke
```

The second command launches Isaac Sim, starts a separate synthetic reader over local IPC, performs 40 control steps, and exits. It needs no leader, keyboard event device, LeRobot installation, or URDF.

`--leader-mock` requires `--no-record`. Synthetic input cannot be used to collect production demonstrations. Commissioning mode creates the usual environment configuration/output directory but no HDF5 demonstrations.

For an interactive mock session, omit `--smoke-steps` and supply a local control device, or explicitly select `--control-source terminal`. Add `--livestream 2` to view it remotely.

## Prepare a separate reader environment

The simulator and LeRobot run in separate processes. Install the helper requirements in a separate environment; the collector selects its interpreter through `--leader-python`.

```bash
uv venv --python .venv/bin/python .venv-so101
uv pip install --python .venv-so101/bin/python --torch-backend cpu -r requirements-so101.txt
```

Run these commands from the repository root. The reader needs no GPU; the CPU PyTorch build satisfies LeRobot's dependencies. The requirements pin `lerobot[feetech]==0.4.4`; the adapter uses the [SO leader API from that release](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/teleoperators/so_leader/so_leader.py). Installation and hardware communication must be verified on the deployment machine.

The helper computes forward kinematics from the URDF's fixed/revolute joint chain using NumPy/SciPy. It does not load meshes or require an inverse-kinematics package.

## Device paths, calibration, and model

Identify the leader USB path and the server keyboard:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/input/by-id/*-event-kbd
```

Use stable device paths when available. The account running the collector needs read/write access to the serial device and read access to the keyboard event device. Arrange narrowly scoped device permissions with the machine's administrator if necessary.

Calibrate the leader separately with LeRobot, using a consistent ID. The collector never invokes an interactive calibration routine or commands leader goal positions. It checks that saved calibration exists and matches the device; connecting the LeRobot reader configures the motors and disables torque. [SO-101 setup and calibration](https://huggingface.co/docs/lerobot/so101#calibrate)

```bash
/path/to/so101-env/bin/lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/YOUR_LEADER \
  --teleop.id=my_so101_leader
```

Use a local calibrated SO-101 URDF such as the upstream [so101_new_calib.urdf](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf). Keep a specific revision with the setup. The helper records its SHA-256 checksum and the calibration checksum; it never downloads a model during collection.

The arm chain must contain `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, and `wrist_roll`, ending at `gripper_frame_link`. Validate the resulting pose at known physical configurations. Adjust explicit `--leader-joint-signs` and `--leader-joint-offsets-deg` only if the encoder/model convention requires it.

Before starting Isaac Sim, check hardware readings:

```bash
/path/to/so101-env/bin/python scripts/so101_leader_reader.py \
  --port /dev/serial/by-id/YOUR_LEADER \
  --id my_so101_leader \
  --urdf /path/to/so101_new_calib.urdf \
  --duration 15
```

This prints named joint ordering, degree readings, normalized gripper readings, and the computed pose. Add `--calibration-dir /path/to/calibration` if the calibration is outside LeRobot's default location. Do not infer that this FK matches your assembly until you check signs and zero positions physically.

## Commission with recording disabled

Run from the RoboLab simulator environment:

```bash
python examples/collect_so101_demos.py \
  --task BananaInBowlTask \
  --leader-port /dev/serial/by-id/YOUR_LEADER \
  --leader-id my_so101_leader \
  --leader-python /path/to/so101-env/bin/python \
  --leader-urdf /path/to/so101_new_calib.urdf \
  --control-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd \
  --teleop-config configs/teleop/so101_droid.yaml \
  --orientation-mode fixed --no-record \
  --device cpu --livestream 2 --defer-images \
  --view-layout inset --inset-camera wrist \
  --output-dir output/so101_commissioning/banana
```

Add `--leader-calibration-dir` if needed. The collector starts and stops the USB helper automatically. Wait for READY before connecting the WebRTC viewer.

| Key on the server keyboard | Action |
| --- | --- |
| N | Start an attempt, or explicitly resume a paused attempt |
| H | Engage/release the clutch; releasing requests a fresh reference capture |
| R | Reject the active attempt and reset the scene |
| B | Hide/show the inset, if enabled |
| Esc | Stop collection cleanly |

The leader controls arm motion and gripper state; keyboard motion keys and K do not compete with it. The physical gripper must match the currently latched simulated command before start/resume. At the beginning of an attempt that means open.

Clutching pauses physics and recording while the viewport keeps rendering. Reposition the leader, match the gripper, and press H or N to capture new references and resume. Paused wall time does not advance the simulation timeout.

## Mapping and commissioning settings

[configs/teleop/so101_droid.yaml](../configs/teleop/so101_droid.yaml) contains the mapping parameters. The provided values are initial settings, not hardware-validated calibration.

- `alignment_wxyz` rotates leader-base axes into DROID robot-root axes.
- `translation_gain` is currently **2.0**: 1 cm of leader translation requests 2 cm of DROID translation, subject to filtering, workspace bounds, and per-step limits. Rotation remains 1:1 in `pose` mode. `filter_alpha` and deadbands suppress jitter. Restart the collector after changing the YAML; an existing session keeps its loaded configuration.
- `base_pan_gain` is **3.0**: base-joint left/right displacement is amplified threefold relative to the existing translation mapping (on top of `translation_gain: 2.0`). FK isolates the displacement due to shoulder pan with the other joints at their current angles; their own contribution retains its existing gain. Start/resume captures a new base-joint reference. Wrist-roll and pose orientation gains are unchanged. Workspace bounds and command/tracking limits still apply, so large movements may clamp or pause. The synthetic reader does not move its joints and therefore does not exercise this gain.
- Workspace limits are offsets from the pose captured at each start/resume.
- Translation and rotation limits apply to the desired physical increment before division by the IK action scale.
- Persistent excessive tracking error pauses control and requires explicit recentering/resume.
- `gripper_open` and `gripper_closed` must match measured normalized endpoints. The defaults assume 100=open and 0=closed. Hysteresis thresholds determine the binary latch.
- `sample_timeout_s` defaults to 0.2 seconds and includes USB read latency. Tune from measured sample ages.
- `--leader-read-hz` defaults to 60; simulation remains at the environment's 15 Hz action rate. Hardware read failures get a bounded number of retries.

The default `fixed` mode maps translation while holding the captured DROID orientation. `--orientation-mode pose` also maps leader orientation. Select the mode at launch. The five arm joints provide a constrained pose family, not six independently controllable Cartesian axes.

The extra `--orientation-mode wrist-roll` mode maps the calibrated leader wrist-roll joint directly to a 1:1 spin about the DROID gripper's local +X tool axis. It holds the captured tool direction fixed and ignores leader tilt for orientation. Position still uses the existing FK translation mapping and 2× gain; opening/closing is unchanged. Start/resume captures a fresh wrist-roll reference, so the initial joint angle causes no rotation jump. Filtering, angular step limits, and tracking-error pauses still apply. Rotation uses the shortest angular path across the joint's wrap boundary. Recorded orientation-mode codes are `0=fixed`, `1=pose`, and `2=wrist-roll`.

To select it on this server, stop the existing collector with Esc and run:

```bash
bash /tmp2/leocheng/forks/RoboLab/scripts/start_so101_commissioning.sh --orientation-mode wrist-roll
```

Commands target DROID's current `base_link` frame on robot-root axes. The controller verifies the action configuration and compensates for its scalar scale (currently 0.5). It uses shortest-path axis-angle rotation errors, not Euler-angle subtraction.

## Collect and materialize images

Once commissioning is satisfactory, remove `--no-record` from the command and add:

```text
--num-demos 10 --output-dir output/so101_demos/banana
```

Successful attempts go to `so101_demos.hdf5`; the collector selects a numbered filename if necessary. `--save-failures` retains rejected and interrupted attempts in a separate failure file. Otherwise only successful attempts are retained.

The shared keyboard collection options remain available: real-time pacing, compression, flush interval, camera scale, inset, live images, and deferred images. See [keyboard teleoperation](keyboard_teleoperation.md) for stream/performance settings.

For deferred images:

```bash
python scripts/materialize_demo_images.py \
  --input output/so101_demos/banana/so101_demos.hdf5 \
  --output output/so101_demos/banana/final/so101_demos.hdf5 \
  --device cuda:0 --hdf5-compression gzip
```

Keep the input HDF5 and its `env_cfg.json` together. Image frame zero uses the initial state; subsequent frames use the previous post-step state, preserving pre-step observation alignment.

## Recorded diagnostics and failures

The standard `actions` dataset contains the exact seven-dimensional commands passed to DROID. Additional `teleop/so101/*` datasets are aligned one-to-one with these actions and include joints, leader pose, gripper input, sample sequence/age, target pose, orientation mode, reference generation, pause count/duration, and wall time between actions.

Dataset metadata records the helper version, calibration/model checksums, mapping, joint units/order, quaternion convention, action scale, and control-device configuration. `so101_events.jsonl` logs collection events separately from action rows.

A stale sample pauses motion. Fresh samples alone do not resume it; press N or H with a fresh reading and matching gripper. A helper exit, disconnected USB bus after retries, invalid packet, changed session, or lost keyboard ends collection and marks the active attempt unsuccessful. Fix the connection and restart the collector; there is no automatic motion resumption.

If a target cannot be followed, clutch and reposition the leader before resuming. Check FK conventions, alignment, and gain during commissioning rather than increasing limits to suppress tracking errors.

## Recovering from a tracking-error pause

A tracking-error pause means the simulated gripper stayed more than 12 cm from the mapped target, or more than 0.8 radians (45.8 degrees) from the requested orientation, for 15 consecutive control steps. At the nominal 15 Hz rate that is about one second. This is separate from a USB/stale-input pause. The error message reports the measured position and orientation errors, their limits, and the consecutive step count.

The command is limited to a 3 cm translation increment and a 0.10 radian (5.7 degree) rotation increment per action. These are command limits, not guaranteed arm velocities. Fast or large leader movements can put the mapped target ahead of the simulated arm; collisions, joint limits, and unreachable targets can also prevent tracking. Fixed-orientation mode can restrict which positions are reachable.

While paused, reposition the leader into a comfortable central pose, match the simulated gripper's current open/closed state, and press N to capture a new reference. For a fresh scene, press R, fully open the leader gripper, then press N. Start with 2–3 cm movements over about one second, one direction at a time, and let the simulated arm settle. Use H to clutch before repositioning for a larger movement. A recurring pause on these small motions needs diagnosis from the reported errors and movement direction.

An independent CPU-physics check on 2026-09-17 held the starting pose and commanded 5 cm movements along each of the six axis directions at 2 cm/s, with fixed orientation and the original 1.0 translation gain. All seven cases completed 105 steps without pausing; moving-case peak tracking error was under 4.7 mm and settled error under 0.44 mm. Results are saved in `output/so101_setup/tracking_check_20260917.json`. This checks the simulator/controller path; it does not establish physical leader alignment, all-scene reachability, or operator video latency.

A second CPU-physics comparison on 2026-09-17 tested 2.0 translation gain and pose orientation with eleven movement cases per profile. Raising the translation increment from 1 to 3 cm and rotation increment from 0.05 to 0.10 radians reduced settling time for 12 cm axis movements from about 2.7 to 1.1 seconds, and for 0.8 radian roll/pitch movements from about 3.7 to 1.9 seconds. Settling was measured from movement start to within 5 mm and 0.035 radians of the final target. All eleven cases with the new settings completed without pausing; a 25 cm translation that paused with the old settings completed with the new settings. Tracking-error thresholds remain unchanged. The comparison is saved in `output/so101_setup/speed_tuning_20260917.json`. These simulator tests do not measure physical operator response or video latency.

## One-command terminal commissioning on RoboLab

The configured server setup can be started from any directory with:

```bash
bash /tmp2/leocheng/forks/RoboLab/scripts/start_so101_commissioning.sh
```

This launcher selects both Python environments, the connected leader, saved calibration ID, and local URDF. It opens Banana-in-Bowl with the wrist inset and recording disabled, using the temporary terminal controls selected for commissioning. Additional collector options can be appended. Keep the terminal focused for N/H/R/Esc; when run over SSH these episode keys cross the network, while leader motion remains on USB/IPC.

For multi-line commands, a backslash must be the final character on its line. Spaces after it break line continuation and cause errors such as `--task: command not found`. The launcher avoids the need to paste multi-line commands.

## RoboLab server setup (2026-09-17)

The connected USB adapter is `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68010085-if00`, currently resolving to `/dev/ttyACM0`. After granting serial access, a read-only probe confirmed all six STS3215 motors (model 777), IDs 1–6, at 1,000,000 baud. The saved calibration contains all six IDs with nonzero ranges; the physical joint-to-ID assignment and direction still need operator validation.

The separate reader environment is `.venv-so101`. The official model is available at `output/so101_setup/so101_new_calib.urdf`, pinned to upstream commit `385e8d7c68e24945df6c60d9bd68837a4b7411ae`. Its download URL and SHA-256 are recorded in `output/so101_setup/urdf_source.json`. These are local deployment files, outside version control. The installed dependency versions are saved in `output/so101_setup/requirements-installed.txt`. LeRobot imports, calibration CLI help, dependency consistency, model FK, the mock reader, and IPC between the simulator and reader environments passed. Calibrated joint and pose readings have also passed the hardware validation described below.

Serial access for `leocheng` is now working after the following command was run on the server:

```bash
sudo setfacl -m u:leocheng:rw /dev/ttyACM0
```

This ACL may need to be reapplied after reconnecting USB. Calibration is now saved under ID `robolab_so101_leader` at `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/robolab_so101_leader.json`, and the reader verified that it matches the motor registers.

The initial voltage fault is now resolved after replacing the 12 V adapter with the correct supply for the socket labeled 5 V DC. The latest read-only check found 5.3–5.4 V on all six motors, clear packet error flags and status registers, and torque disabled. Results are saved in `output/so101_setup/power_recheck_20260917T115048Z.json`.

For reference, the earlier check reported 12.6 V and voltage errors on every motor. Motor 2's configured maximum was 8.0 V; the others were 12.0 V. Those results are preserved in `output/so101_setup/connection_probe.json` and `output/so101_setup/voltage_probe.json`. Voltage feedback uses 0.1 V units per [Feetech's documentation](https://www.feetechrc.com/20210430-56680.html). The standard leader uses a **5 V supply** with its 7.4 V motors; see the [official assembly instructions](https://huggingface.co/docs/lerobot/main/assemble_so101#configure-motors).

The operator completed calibration directly in a server terminal. For future recalibration, use the command below: position the joints near the middle of their travel and press Enter, then move every prompted joint except wrist roll through its full range and press Enter to save. Recalibration is not needed merely to start another collection session.

```bash
.venv-so101/bin/lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68010085-if00 \
  --teleop.id=robolab_so101_leader
```

To inspect the calibrated readings again, run the standalone diagnostic:

```bash
.venv-so101/bin/python scripts/so101_leader_reader.py \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68010085-if00 \
  --id robolab_so101_leader \
  --urdf output/so101_setup/so101_new_calib.urdf \
  --duration 15
```

The hardware reader and actual local IPC path were checked for 15 seconds after calibration: 892 distinct samples at 59.4 Hz, no stale-data pauses or reader faults, median USB read duration 1.42 ms, p95 2.21 ms, and maximum received sample age 5.89 ms. Five intermediate sequence values were skipped by the latest-sample transport; no stale data was consumed. All five arm angles, the gripper reading, and the computed pose were finite and constant during this observation; no physical motion test was requested in that interval.

Results are saved in `output/so101_setup/hardware_reader_20260917T120227Z_summary.json`, with the sample stream beside it. Calibration SHA-256: `ae83b71ed56f2a9f1fc61a9ffbdbc86733b49e2f2500595cf3bfdaf97efe5dc6`. A checksum-named calibration snapshot is saved in `output/so101_setup/calibration/` for this validation record; normal launches continue using the standard LeRobot calibration path.

No physical USB keyboard was detected. The operator selected `--control-source terminal` temporarily for commissioning; when used over SSH, episode keys also cross the network while leader motion remains on USB/IPC. A server USB keyboard is still required for the original viewing-only Wi-Fi setup. Physical-to-simulated direction, joint assignment, gripper endpoints, and a complete task attempt remain to be validated with the operator. The diagnostic reader has stopped and released the USB port.

## Validation

Validated on 2026-09-17: 75 focused unit tests and 24 simulator/integration regressions passed. The SO-101 collector completed a 40-action headless mock run with no demonstration file. The preserved keyboard entry point completed a 20-action paced benchmark at approximately 15 Hz. These measurements used CPU physics and deferred images without a connected WebRTC viewer or physical leader.

Run the pure tests without starting Isaac Sim:

```bash
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_so101_retarget.py tests/unit/test_so101_source.py \
  tests/unit/test_so101_lifecycle.py tests/unit/test_so101_reader.py \
  tests/unit/test_so101_options.py -q
```

Run the simulator recording regressions:

```bash
python -m pytest tests/test_so101_collection.py \
  tests/test_keyboard_teleop.py tests/test_deferred_images.py -q
```

Before accepting hardware-collected data, verify physical direction/zero conventions, gripper endpoints, stationary stability, start/reset/clutch behavior, disconnected-device behavior, and a complete task demonstration. Synthetic and simulator checks do not establish physical calibration or end-to-end operator video latency.
