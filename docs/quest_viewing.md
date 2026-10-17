# Quest 3S: native XR viewing and L100 teleoperation

Branch: `quest`. Viewing entry point: `examples/view_xr.py`.
L100 control entry point: `examples/teleop_omy_xr.py`.

The viewing-only entry point displays the **frozen reset state** of a RoboLab task through
Isaac Sim's native VR renderer. Head tracking changes the viewpoint and native
XR supplies the per-eye projections. It does not read the L100, map controllers
to the robot, advance physics after reset, or record demonstrations. Controller
scene-manipulation tools are disabled as well. The existing OMY entry point and
calibration are unchanged.

The L100 entry point attaches the same native viewing layer to the existing
OMY joint-space runner. It advances physics, reads the calibrated leader, and
supports the existing HDF5 demo recorder. See [L100 control in XR](#l100-control-in-xr).

```
RoboLab scene → native Isaac Sim VR/OpenXR → SteamVR → ALVR → USB → Quest 3S
                                ← headset tracking ←
```

The viewing-only path has no CloudXR, browser viewer, JPEG stream, wrist inset,
or policy-image camera. Camera sensor configurations are removed using Isaac
Lab's XR helper, with a final check for inherited config defaults. Ordinary
Kit/XR viewing cameras are necessary and are not the additional sensor cameras
being removed. The L100 path also omits image sensors by default; it adds policy
cameras only when `--record-images` is requested.

## Host and headset setup

Run on the simulation host's **local, GPU-backed desktop as the same user** as
SteamVR. SSH X11 forwarding does not replace this session. `--headless` hides
Isaac's window; it does not remove SteamVR's desktop/runtime requirements.

1. Install Steam and its device rules if absent (ask the operator for sudo):

   ```bash
   sudo apt-get install --no-install-recommends steam-installer steam-devices
   ```

   These are minimal bootstrap packages; Steam may request more host libraries
   on first launch. Do not blindly downgrade CUDA/drivers to COLLAB-SIM's old
   Isaac Sim 4.5 recommendations. Do not launch Steam or Isaac as root.

2. Launch `steam`, sign in, install **SteamVR**, and launch it once.
3. Install matching ALVR server and headset client versions from the
   [official releases](https://github.com/alvr-org/ALVR/releases). The release
   checked for this implementation is 20.14.1. The Linux launcher can install
   the matching streamer and APK. Keep downloaded binaries outside source code.
4. Enable Quest developer mode. Connect a USB data cable and accept USB debugging
   authorization in the headset. In ALVR's Devices screen, enable **Wired
   Connection**. ALVR provides its own ADB. See the
   [official wired guide](https://github.com/alvr-org/ALVR/blob/master/wiki/ALVR-wired-setup-%28ALVR-over-USB%29.md).
5. In the ALVR launcher, select the installed streamer and start its dashboard.
   If SteamVR was already running before ALVR registered its driver, close
   SteamVR completely and start it again from ALVR. The SteamVR driver list
   (`vrpathreg show`) should include `alvr_server`. In ALVR's Devices tab,
   trust the headset when it appears.
6. Start SteamVR from ALVR and verify **SteamVR Home appears in the headset** and
   follows head rotation and translation. This is the transport acceptance test,
   before adding Isaac Sim.

Use ALVR's [Linux troubleshooting guide](https://github.com/alvr-org/ALVR/blob/master/wiki/Linux-Troubleshooting.md)
if the display is black despite tracking. In particular, its SteamVR launch
workaround uses the actual installation's `bin/vrmonitor.sh %command%`. Resolve
that path from Steam's storage settings; don't copy a different user's path or
apply downloaded patch scripts without inspection.

## Preflight and launch

From the RoboLab root, in that local desktop terminal:

```bash
.venv/bin/python examples/view_xr.py --check
```

The preflight does not start Isaac, install anything, or change the default
OpenXR runtime. It checks the active runtime manifest (or common Steam library
paths) and its library. Its generic warning about headset connectivity remains
even when ALVR is streaming; `errors: []` is a passing preflight. For a
nonstandard Steam library, specify it explicitly:

```bash
.venv/bin/python examples/view_xr.py --check \
  --runtime-json /absolute/path/to/SteamVR/steamxr_linux64.json
```

Once SteamVR Home is working:

```bash
.venv/bin/python examples/view_xr.py --task BananaInBowlTask
```

Pass the same `--runtime-json` if needed. Runtime selection is process-local;
the viewer does not rewrite `active_runtime.json` or your Steam settings.
Wait for **XR display active; initial view placed.** `Scene ready` alone is not
proof of headset rendering. Failure to activate within 60 seconds after scene
loading exits with an error; adjust `--xr-timeout` if needed. Ctrl+C stops.

Default rendering uses Kit GPU index 0 with multi-GPU disabled; initialization
uses CPU physics. Use `--render-gpu N` to match SteamVR/ALVR's GPU. This only
selects Isaac's renderer; it cannot change SteamVR or ALVR GPU selection. Do not
remap `CUDA_VISIBLE_DEVICES` and assume all three applications use the same GPU.

## Initial viewpoint

The initial virtual eye is `(-0.25, 0, 0.95)` looking at `(0.53, 0, 0.48)` in
scene-world meters, Z-up. Since this task's tabletop is approximately Z=0,
this places your view close behind the Franka with the work area about 0.95 m
below eye level. Both eye and target were raised by 30 cm to preserve the
previous viewing direction. Isaac's additional physical-height adjustment is disabled so
it cannot lift this stage-world eye position again. Stand upright and look
comfortably forward when the XR display activates. Change the view with:

```bash
.venv/bin/python examples/view_xr.py \
  --eye -0.25 0 0.95 --lookat 0.53 0 0.48
```

For a closer view, increase the eye X value toward zero (for example, `-0.15`).
For a lower view, reduce eye Z (for example, `0.85`). Increase look-at Z to
bring the arm higher in your forward gaze. Change one coordinate at a time and
stop before the camera intersects the robot geometry. These options change only
the viewer, not the robot/task initial pose or physical tracking scale.

On XR display activation, `XRCore.schedule_set_camera()` places the initial
view relative to the headset's current pose. Subsequent tracking belongs to the
runtime; the script does not repeatedly overwrite the head pose. A detected
display reactivation places the initial view again. No controller teleport or
recenter bindings are installed in this version. Stop and relaunch to choose a
different view. Only move within your safe physical headset boundary.

Geometry has view-dependent stereo/parallax. The HomeOffice background is a
dome-light environment image, not a reconstructed room: do not expect room-scale
parallax from the photographic background.

## L100 control in XR

Stop the viewing-only process and any other L100 reader before launching this
version. Keep SteamVR/ALVR running on the local desktop. Support the leader in
the saved READY pose, with the trigger released, while the scene loads:

```bash
.venv/bin/python examples/teleop_omy_xr.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib omy_calib_robolab.json --device cuda:0 --render-gpu 0
```

The existing `omy_calib_robolab.json` is also the default for this entry point.
For generated tasks, use `--task-dirs single_pnp --task SimplePnp064ComputerMouse02Task`.
The OMY launchers read that folder's `manifest.json` and import only the selected
task; no global folder changes or `python -c` wrapper are needed. The default
folder remains `benchmark`. Missing or ambiguous tasks fail before Sim starts.
It preserves the calibrated joint directions, READY offsets, corrected wrist
rotation and trigger endpoints. No new calibration is captured unless you
explicitly supply `--auto-zero`. The robot still starts at DROID's task pose.
The viewpoint, runtime selection and `--eye`, `--lookat`, `--render-gpu` and
`--runtime-json` options are shared with the working viewer. The default remains
CPU physics with GPU rendering and bounded workers; `--device cuda:N` changes
only the physics device.

For the current BananaInBowlTask/Robotiq asset, explicitly use `--device cuda:0`
as above. The September 30 GPU grasp completed successfully with tracked joint
anchor gaps below one micrometer; CPU diagnostic attempts showed centimeter-scale
separation and object launches. These were interactive tests, not identical-action
backend replays. Do not trade away the working GPU grasp behavior for the older
CPU throughput benchmark. Keep the tested 72 Hz ALVR/XR configuration unchanged.

Normal operation should omit `--diagnostics`: the full GPU diagnostic capture
measured about 45 ms per control step for state readback and trace serialization.
Lightweight performance summaries remain enabled without it.

OMY pose observations now share an articulation-based calculation within each
observation evaluation, using the configured `eef_frame` offset without refreshing
the frame-transform sensor's extra rigid-body view. Recording retains the same
observation keys, root-relative frames, units, and pre-action alignment. No values
are cached across evaluations or resets. `--reference-observations` restores the
original frame-sensor observation functions for A/B validation. The change applies
to both recorded and non-recorded runs; it does not remove dataset observations.
Leader commands are validated on CPU before their single transfer to the simulation
device. Physics, solver parameters, task predicates, and render cadence are unchanged.

Both XR launchers disable RTX texture streaming before scene loading to avoid
texture pop-in. This increases VRAM usage and takes effect on the next launch;
restart an existing session. Desktop mode leaves texture streaming unchanged.
This does not change ALVR encoding, render resolution, physics, or control settings.

Both `teleop_omy_xr.py` and `view_xr.py` accept
`--xr-resolution-multiplier 0.8` to override the native VR render-buffer scale.
The allowed range is 0.1–2.0. Omitting the option preserves Kit's existing
setting; it does not assume a default of 1.0. This is an absolute multiplier,
not a relative reduction: check the current XR Resolution Multiplier first.
For example, reduce 1.0 to 0.8, or 0.5 to 0.4. It is independent of desktop
viewer resolution, recorded camera dimensions, and SteamVR's resolution
percentage. Keep SteamVR at 100% for an isolated Kit-resolution experiment.
The override is applied before XR activation, restored on normal shutdown,
printed at startup, and recorded as `xr_resolution_multiplier_override` in
the run's XR/viewer metadata (null means unchanged, not a measured resolution).
The option is rejected with `--desktop`. GPU physics and control cadence are
unaffected. Keep the headset at 72 Hz when comparing runs.

Worker limits can be adjusted with `--cpu-workers` (Kit/TBB, default 8) and
`--physics-workers` (CPU physics, default 2). On the 20-logical-CPU host, append
`--cpu-workers 20 --physics-workers 20` to try larger worker pools. Compare the
reported physics time and control rate during the same grasp: larger pools can
increase scheduling overhead and compete with SteamVR/ALVR and the serial reader.

Add `--check` to the command to inspect the runtime, installed OMY dependency,
and serial-path permissions without opening the leader or starting Isaac.
This does not test exclusive port ownership or live serial communications.

After resetting the scene, the runner enables native XR and waits for an active
display **before opening the leader**. When it reports `XR display active; L100
control enabled.`, movement starts using the existing OMY mapper and rate
limiter. Head tracking controls only the viewpoint; Quest controllers are not
mapped to robot actions. Ctrl+C stops and closes the leader, disables the XR
profile, and closes the environment/recording.

Physics remains 120 Hz with two substeps per action (nominal 60 Hz control),
and XR renders every control step. No JPEG encoder or additional image sensors
run by default. Actual control/render rates depend on scene and stereo-rendering
cost; the console reports those rates. Headset refresh and ALVR latency remain
separate measurements.

If Kit reports the XR display inactive, the runner renders the frozen scene
without advancing the mapper, physics, or recorded actions. It resumes
automatically when the display returns, preserving the last command and rate
limiter; it does **not** recenter or reset the arm on reconnection. Return the
leader to its prior pose before resuming if you moved it during the pause.
An inactive display lasting `--xr-timeout` seconds (default 60) ends the run.
This detects Kit display state, not whether the headset is worn or whether
ALVR is delivering frames: SteamVR can keep its display active through a
transport interruption.

For demo collection, append:

```bash
--record --output-dir output/omy_xr_banana_001
```

Use a new/empty output directory for each run. `--record` saves the existing
absolute joint actions, robot/object states and proprioception in
`omy_demos.hdf5`; only successful attempts are kept by default. Add
`--save-failures` to retain interrupted/failed test attempts. Add
`--record-images` to enable and record the DROID policy cameras as well; these
are additional sensor images, not headset eye images, and add rendering cost.
`--num-episodes N` retains the existing task-completion/timeout reset behavior.
The console and `omy_performance.json` report timing, and `xr_status.json`
records runtime activation and pause counts. Dataset metadata includes the
viewpoint/runtime alongside calibration and control settings.

## Grasp and articulation diagnostics

For the lag and separating-finger investigation, keep the working visible-window
launch and enable `--diagnostics`. Start with the original 8 Kit/TBB and 2 physics
workers so the worker experiment does not confound the next capture:

```bash
.venv/bin/python examples/teleop_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib omy_calib_robolab.json --cpu-workers 8 --physics-workers 2 --diagnostics --stats-interval 1 2>&1 | tee /tmp/omy_xr_diagnostics.log
```

The console prints the run output directory. It contains:

- `omy_diagnostics.jsonl`: reset state, each completed physics substep, and each
  control step's timing. States include every arm and passive gripper joint,
  actual body-link poses/velocities, tracked joint-anchor separation, object
  states/speeds, normal contact forces, the exact consumed leader sample and
  its age, serial read duration/error counts, and the eight commanded actions.
- `omy_diagnostic_metadata.json`: state-array names/order, effective drive
  stiffness/damping/effort/velocity limits, CPU settings, composed USD joint
  anchors/mimic relationships, and robot collision settings.
- `omy_diagnostic_summary.json`: maxima and the first invalid-state event, if any.

The trace observes the standard control loop after `scene.update()` at every
physics substep. It stops before the next substep on nonfinite joint/body/object
states or a joint exceeding its hard limits by more than 0.2 rad, saves the
fault, pauses physics, then closes normally with exit code 2. The allowance can
be changed with `--diagnostic-joint-tolerance`. Large arm tracking error alone
does not trigger a stop. Anchor separation is diagnostic only and is omitted
for joints whose bodies are not both tracked physics links (for example after
fixed-link merging).

This capture uses the existing binary gripper and calibration. Contact-force
reads are normal forces, not tangential friction forces. Trace writes are
buffered and flushed at least once per second while steps run, and at shutdown
or a detected fault. Abrupt process termination can lose the latest buffer.
Diagnostic state reads may force lazy sensor updates or GPU transfers; their
cost appears as `diagnostic_state`, and serialization/writes as `diagnostic_io`.
Compare diagnostic runs separately from clean performance benchmarks.

First hold still, then lift/move with the trigger released. If that remains
stable, approach the banana with the gripper open before trying a close. Stop
with Ctrl+C after observing the issue; the failure guard cannot detect every
possible visual or constraint defect.

`--desktop` on **this control entry point** runs L100 teleoperation in a normal
viewport without SteamVR, so it still opens the leader and moves the simulated
robot. For a frozen, input-free preview, use `examples/view_xr.py` instead.

## Hardware-free checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --confcutdir=tests/unit \
  tests/unit/test_xr_view.py tests/unit/test_omy_xr.py -q
.venv/bin/python examples/view_xr.py --desktop --headless --frames 60
```

`--desktop` is explicitly **not XR validation**. It loads the same camera-free,
zero-action scene without requiring SteamVR. Omit `--headless` for an ordinary
desktop scene preview. Unit tests exercise manifest validation, launch options,
native lifecycle with mocks, timeout reporting, and the render-only loop.
L100/XR unit checks additionally exercise the real control runner with simulated
device/environment boundaries: activation before input, freeze/reconnect,
mapper reset counts, and resource cleanup. They do not replace a live combined
L100 + headset acceptance run.

Each run writes `env_cfg.json` and `viewer_status.json` under a new directory in
`output/quest_view/` (or a new/empty `--output-dir`). Host render-update rates are
not headset FPS or motion-to-photon latency; use SteamVR/ALVR statistics for those.
`xr_display_seen` is runtime activation, not proof that the operator saw correct
stereo, scale, or latency. Those remain headset acceptance checks.

## Implementation boundary

Validated against the local Isaac Lab 2.2 / Isaac Sim 5.0 APIs. The launcher uses
Isaac Lab's XR experience, adds `omni.kit.xr.profile.vr`, suppresses AR autostart,
and selects the SteamVR runtime through `omni.kit.xr.system.openxr`. It does not
use COLLAB-SIM's older direct SteamVR output plugin. Native APIs live in
`robolab/core/xr_view/native.py`; `robolab/core/omy_teleop/xr.py` attaches that
viewing layer to the existing L100 runner without duplicating its control math.

Reference: [COLLAB-SIM installation architecture](https://github.com/NVlabs/collab-sim/blob/release/docs/install_docs.md).
On 2026-09-25, a live run on this host reached SteamVR through ALVR: Kit
activated the VR/OpenXR display, allocated left and right eye render targets,
and completed 120 render updates. The first startup spent several minutes
compiling RTX shaders. The operator should still confirm that the scene is
visible, correctly scaled, and head-tracked in the headset; software activation
alone does not establish those perceptual checks.
