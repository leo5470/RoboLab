# L100 + Quest demonstration collection

Use `examples/collect_omy_xr.py` for recorded attempts. `teleop_omy_xr.py`
remains the direct teleoperation smoke test.

Both scripts accept `--task-dirs` (alias `--task_dirs`), using the same folder
names as policy evaluation. For example, select a generated task with
`--task-dirs single_pnp --task SimplePnp064ComputerMouse02Task`. Only its task
module is imported; its folder's `manifest.json` provides the lookup. Omitting
the option keeps the `benchmark` default. Recorded task-folder and source-file
provenance lets offline rendering select the same definition automatically.
See [task selection](omy_teleoperation.md#select-a-task-folder-without-importing-unrelated-tasks).

Run from the computer's GPU-backed desktop, with the Isaac Sim window able to
receive keyboard focus. F9 controls READY and retry in both collection modes:

1. After scene reset, wait for `SETTLING` to finish, then return the physical
   L100 to its calibrated READY pose and release the gripper trigger. The
   simulated robot stays at home while objects settle under physics.
2. Press F9 in the Isaac Sim window. The mapped arm target must be within
   `--ready-tolerance-rad` of the DROID home, with a fresh leader sample and a
   stable pose for `--ready-stable-s`.
3. Move the leader or trigger by `--start-motion-rad` to start the attempt.
   The first action is recorded and applied together.
4. Press F9 while recording to export an unsuccessful attempt and reset the
   scene. Successful task termination exports automatically and resets the
   scene. Return the L100 to READY and press F9 for the next attempt.

Holding F9 counts once. F9 while armed cancels arming. Object mode also uses
F12 to skip to the next layout, as described below. Space remains Isaac Sim's
Play/Pause shortcut and must not be used for episode control; pausing the
timeline can block physics stepping even while the XR view remains responsive.
XR loss or stale
leader input ends a running attempt and requires another READY confirmation.
The physical leader is never driven by scene reset. The same seed is used on
every reset for repeatable object placement; calibration is never changed.
New recordings store `reset_key: "F9"` in their collection metadata; existing
recordings are not modified.

With fixed direct-wrist calibration over serial, the reader resolves an initial
whole-revolution offset in L100 joint_6 when its equivalent Franka J7 target is
inside the joint limits. This handles an extended-position encoder reporting
the saved READY orientation plus 360 degrees. The correction is printed once,
recorded in the `armed` event, and held fixed across motion and episode resets.
It changes neither motor registers nor the calibration file. F9 rejection
messages report mapped joint offsets, a closed trigger, stale input, or the
required stability interval. A real pose offset still needs alignment or
recalibration; the reader only corrects whole revolutions.

Example:

```bash
cd /tmp2/leocheng/RoboLab && .venv/bin/python -u examples/collect_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib /tmp2/leocheng/RoboLab/omy_calib_robolab.json --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 0 --cuda-connections 1 --task-checks cpu --xr-resolution-multiplier 0.8 --vel-scale 2 --stats-interval 1
```

For the optimized CUDA profile described below, use the Bash launcher with a
task name from `single_pnp` or `benchmark`, an optional episode count, and an
optional console-log flag:

```bash
scripts/collect_omy_xr.sh SimplePnp034CannedMushrooms24Task --log
scripts/collect_omy_xr.sh BananaInBowlTask --num-episodes 25 --no-log
scripts/collect_omy_xr.sh --object wooden_bowl --postprocess-skip-failed --postprocess-preview-videos --log
```

Add `--stats-interval 0` to disable the entire periodic reporting block,
including statistics formatting, the joint-error GPU readback, and placement
diagnostics. This option also works with the Python entry points. Episode
timing files and completion messages remain enabled. The Bash launcher reports
every second by default; use a positive interval to change that frequency.

### Prepared starts for new collections

The collection state machine is `RESET → SETTLING → WAITING/READY → ARMED → RECORDING`.
Before each first attempt, retry, layout switch, and resume, the collector holds
the simulated robot at home with its gripper open and advances physics without
recording actions or evaluating success. Simple-PnP objects and the bin are
placed using their scaled, rotated geometry bottom and the table's actual top,
with 2 mm initial clearance. Other task arrangements are settled in place.

Preparation checks upward support forces, and explicit table support for flat
Simple-PnP layouts. It also requires low velocities and a stable pose window:

| Setting | Default | CLI override |
|---|---:|---|
| Minimum elapsed simulation time | 0.5 s | `--scene-settle-min-s` |
| Continuous stable interval | 0.3 s | `--scene-settle-stable-s` |
| Timeout | 5 s | `--scene-settle-timeout-s` |
| Maximum linear speed | 0.01 m/s | `--scene-settle-linear-m-s` |
| Maximum angular speed | 0.5 rad/s | `--scene-settle-angular-rad-s` |
| Maximum pose drift over the stable interval | 1 mm / 0.5° | Recorded preparation profile |

The angular ceiling was adjusted after real PhysX tests: small supported objects
can report residual angular velocity while their measured poses remain stable.
The independent position/rotation drift checks continue to reject actual motion.
All durations use simulated time. The XR display and keyboard continue updating.
If preparation fails, recording stays unarmed; F9 retries preparation and F12
can skip the layout in object mode. No manipulation attempt is exported or counted.

After preparation, the collector refreshes the recorder's `initial_state` and
first observations from the settled scene. Episode and success timers reset
without resetting the poses. The simulation clock may already be nonzero;
recorded commands still advance exactly 1/15 simulated second each.
Preparation settings are part of the session resume contract. The last report
is `PART/preparation_latest.json`, all outcomes are in `collection_events.jsonl`,
and each exported attempt has an `omy_preparation` HDF5 attribute.

This profile applies to newly created sessions. Sessions predating its complete
preparation contract require a new collection session. Existing datasets are
not migrated or trimmed. A scene-generation helper also supports geometry-aware
placement; the physical solver uses asset bounds in Kit or explicit
`local_bounds` metadata rather than assuming a centered asset origin.

### Automatic postprocessing and dataset location

The Bash launcher automatically validates native-15 timing, packages valid
attempts and renders both policy cameras after collection ends. This includes
reaching the success quota and pressing Ctrl+C. The launcher first waits for
the active attempt, recording files, XR/device resources and Isaac Sim to
finish closing. Rendering then runs in a persistent headless batch process, without
opening the leader port or starting XR. There is no fixed delay between phases.
The supervisor confirms collector exit before finalizing the handoff, including
when Isaac Sim's fast shutdown exits inside its app-close call.

Each invocation, including `--resume`, writes to its own absolute destination:

```text
SESSION/postprocess/RUN_ID/
  dataset.json               # completed camera episodes and relative paths
  summary.json               # collection status, processing status and failures
  processing_options.json    # frozen selection and preview policy for retries
  reports/RECORDING_KEY/index.json
  raw/RECORDING_KEY/ATTEMPT_ID/trajectory.hdf5
  rendered/RECORDING_KEY/ATTEMPT_ID/images.hdf5
  workers/isaac_*/            # worker log, requests and per-job acknowledgments
  verification_videos/       # optional CPU-encoded success clips and combined reel
```

The terminal prints `Postprocessing destination:` when processing starts and
`Postprocessed dataset:`, `Dataset manifest:` and `Processing report:` when it
finishes. These contain absolute paths. With `--log`, the same output and
shutdown messages are saved in the printed `/tmp/omy_collect_fast_*.log`.
The manifest lists only completed, verified outputs; inspect the summary for
failed or unprocessed attempts. Empty runs print `No recorded episodes;
postprocessing skipped` and the raw session location.

By default, processing retains successful, failed and interrupted valid attempts,
including their quality flags. `--postprocess-skip-failed` selects only explicit
successes before packaging/rendering; raw failures remain saved. Skipped failures
and unknown outcomes have separate report counts and do not become render errors.
An empty selection starts no Isaac worker. The default camera profile uses the configured resolution,
realtime rendering on `cuda:1`, and two renders per frame. Raw files remain
unchanged. These HDF5 camera datasets remain `training_ready: false`; numerical
mapping and LeRobot conversion still require a separate explicit recipe.

Collection continues to use GPU 0. Once the collector exits and postprocessing
starts on GPU 1, start the next collection in another terminal. Processing stays
in the original terminal so its progress and Ctrl+C cancellation remain available.
OMY renderers for the same user and GPU take turns, so a later collection's
postprocessing waits if an earlier renderer still occupies GPU 1. The GPU lock
is held for the worker's entire lifetime, including gaps between episodes.
Override the processing GPU with `--postprocess-device cuda:0` (for example, on
a single-GPU machine). The index uses the host's `nvidia-smi` ordering. Each
renderer runs in a private device namespace exposing only that GPU, so CUDA
and Vulkan both see it as local GPU 0. The host GPU identity is recorded in
`render.log` and the render manifest's `gpu_execution` field.

This requires Linux, `nvidia-smi`, bubblewrap (`bwrap`) and working user/mount
namespaces; these are already available on the collection host. The selected
NVIDIA device and shared control/UVM nodes are mounted into the child's private
`/dev`; other GPUs, DRM nodes and display-server connections are excluded.
This changes only the renderer process, not host device permissions or the
collection environment. Missing isolation support or an unavailable GPU fails
processing instead of falling back to an unisolated renderer.

The RTX 3080 (GPU 1, 10 GiB) passed a full-resolution 136-frame wooden-bowl
episode with both 1280×720 cameras. The implemented device-isolated launcher took 62 seconds
with warmed caches, including startup and verification. NVML compute and
graphics sampling recorded zero job memory/processes on host GPU 0 across
1,048 samples; GPU 1 peaked at 5,768 MiB. Earlier device-selection-only runs
allocated memory on GPU 0, even with a CUDA visibility mask. The private device
namespace removes that allocation. CPU and storage remain shared; concurrent
collection latency has not been benchmarked.

The first Ctrl+C during collection requests graceful shutdown and then starts
processing. Another Ctrl+C during cleanup cancels the handoff. Ctrl+C during
processing cancels the active renderer and retains completed outputs and
diagnostics. Startup or cleanup failures prevent automatic processing. Successful
processing preserves collection status `130` after Ctrl+C; partial/failed
processing returns `2`, and interrupted processing returns `130`.

Use `--no-postprocess` to collect without automatically processing. Every launched
invocation stores an explicit recording list and cleanup status in
`SESSION/collection_runs/RUN_ID.json`. A resumed invocation processes only its
new recording parts, preserving earlier datasets. To retry a finalized
invocation without restarting collection, use the command printed at processing
startup, or run:

```bash
.venv/bin/python -u scripts/postprocess_omy_session.py --handoff SESSION/collection_runs/RUN_ID.json --device cuda:1
```

For an older object-layout session without a handoff, process its recorded parts:

```bash
.venv/bin/python -u scripts/postprocess_omy_session.py --session SESSION --device cuda:1
```

The command prints its output and retry command, including `--run-id`. It briefly
holds the collection lock to snapshot the closed recording list, then releases
it before processing. Collection can resume into new parts while the old parts
are processed. `session_input.json` freezes the part list for retries;
the original session metadata and recordings remain unchanged. Historical
collector exit status is unknown and recorded as `null`.

Retries verify and reuse matching archives and camera caches. Conflicting or
damaged outputs are reported rather than overwritten. A second retry for the
same invocation is refused while processing is already running. See
[OMY postprocessing](omy_postprocessing.md) for formats and manual stages.

### Collect layouts for one object

Select the object without a layout-specific task name:

```bash
scripts/collect_omy_xr.sh --object canned_mushrooms --episodes-per-layout 1
```

This collects one successful demonstration for each retained CannedMushrooms
layout, in layout-number order, using one simulator, environment, XR session,
and leader connection. CannedMushrooms has 50 layouts (`00` through `49`), so
this command targets 50 successful demonstrations. Use
`--episodes-per-layout 10` for 500 successful demonstrations. Other objects may
have fewer retained layouts; collection uses the manifests rather than assuming
50 entries. Geometry, robot, task logic and reset compatibility are checked
before the session starts; only object/bin poses change between compatible layouts.

The object quota defaults to one success per layout. Select a catalog ID with
`--object-id 034`, or combine it with `--object canned_mushrooms` to disambiguate
a duplicate name. The existing per-task command remains available with its
10-success launcher default and collects only that task's layout:

```bash
scripts/collect_omy_xr.sh SimplePnp034CannedMushrooms24Task --num-episodes 10
scripts/collect_omy_xr.sh --object-id 034 --episodes-per-layout 10
```

To collect only 10 successes across different layouts and automatically
postprocess the successful attempts with review videos:

```bash
scripts/collect_omy_xr.sh --object milkjug_a01 --num-episodes 10 \
  --postprocess-skip-failed --postprocess-preview-videos --log
```

The catalog spelling is `milkjug_a01` (ID 230). With the default one success per
layout, this uses the first 10 successful layouts in retained order. Failed
attempts do not count; F12 skips to another layout. Reaching 10 stops collection
before opening another recording and starts postprocessing on GPU 1. The
terminal prints the processed dataset and video locations.

Choose one task, one object, or one resumed session. `--num-episodes` limits
total successful episodes in either task or object mode. In object mode, omit
it to collect every retained layout; `--episodes-per-layout` sets the per-layout
quota and defaults to 1. For example, combining `--num-episodes 10` with
`--episodes-per-layout 2` collects two successes on each of five layouts. The
total can end partway through a layout quota. A total larger than the sum of
all layout quotas is rejected; increase `--episodes-per-layout` if needed.
The Python entry point accepts the same object/resume flags; its single-task
interface continues to use `--task TASK`.

| Event | Behavior in object mode |
| --- | --- |
| F9 at READY | Validate the leader pose and arm; movement starts recording. |
| F9 while armed | Cancel arming. |
| F9 while recording | Save the unsuccessful attempt and reset the same layout. |
| Failure, timeout, stale input or XR loss | Save the active attempt and return to READY on the same layout. |
| Success below the layout quota | Save the success and reset the same layout. |
| Success reaches the layout quota | Mark the layout complete and load the next layout automatically. |
| Success reaches `--num-episodes` total | End collection and start automatic postprocessing. |
| F12 at READY or while armed | Skip to the next layout without creating an empty attempt. |
| F12 while recording | Save the attempt as manually interrupted and skip to the next layout. |

Every reset and layout change requires READY confirmation again. A held key
counts once. F12 affects object sessions only; F9 retains the original
single-task behavior. Override the bindings with `--reset-key KEY` and
`--next-layout-key KEY`. Supported keys are unmodified F3, F4, F5, F6, F9 and
F12; reset and next-layout keys must differ. Known Isaac Sim/Kit shortcuts are
rejected, and object mode checks the running Kit hotkey registry for additional
conflicts. F10 is Isaac Sim's screenshot key and should be left
available for that action. Check custom bindings in the focused simulator window.
Progress identifies the current task/layout, attempt number, success quota,
completed/skipped counts, and total progress when a total limit is set.
A skipped layout keeps its previous successes
and remaining quota. At the end of the layout list the session ends and reports
pending layouts; it does not silently loop back to skipped layouts. Reaching
the total success target completes the session even when other layouts remain
unvisited. Exhausting the layout list before reaching that target leaves work
for a later resume.

Resume the remaining work with:

```bash
scripts/collect_omy_xr.sh --resume output/omy_teleop/SESSION
```

Resume restores the stored collection profile, calibration path, and both quotas,
then collects incomplete, skipped, and unvisited layouts while retaining
previous successes. Those successes count toward the saved total target; a
session that has already reached it opens no new recording. For example, a session started with
`--solver-position-iterations 32` resumes with that cap using only `--resume`;
an explicitly supplied conflicting cap is rejected. Omit `--num-episodes`,
`--episodes-per-layout`, and `--output-dir` when resuming. The collector validates
the recorded sources and run contract before reusing a session, so use the same
collection profile, robot settings and task definitions. Completed recordings
are reconciled with saved progress after an interruption. A hard crash can lose
the active attempt still buffered in memory.

The Python entry point also restores saved TCP host/port settings. Explicit
transport overrides can select a moved endpoint without changing the saved
simulation and calibration contract.

Object sessions store `session.json` and `collection_events.jsonl` at their root.
Each task has its own recording/configuration pair; resuming creates a new part
and preserves prior recordings:

```text
SESSION/
  session.json
  collection_events.jsonl
  SimplePnp034CannedMushrooms00Task/
    part_000/
      omy_demos.hdf5
      env_cfg.json
      attempt_0000_timing.npz
    part_001/                 # created when resuming this layout
      omy_demos.hdf5
      env_cfg.json
  SimplePnp034CannedMushrooms01Task/
    part_000/
      ...
```

Image materialization can process each part's HDF5 file using the adjacent
`env_cfg.json`; the configuration identifies that part's actual layout. Events
record task identity and transition reasons, including skips and interruptions.
Each part also keeps an adjacent `collection_events.jsonl` so existing archival
tools retain the attempt start time and end reason without mixing layouts.
Use `--output-dir DIR` for an explicit new session directory. Launcher paths
passed to `--output-dir` or `--resume` are relative to the invoking directory.

### Validation status

The object-selection and compatibility checks passed for all 6,754 retained
layouts across 136 catalog entries on 2026-10-21, without importing the simulator.
CPU tests cover the production collection loop with simulated input, quotas,
retry/skip races, per-layout recording rotation, interrupted-export recovery,
resume, and legacy task commands. There are 519 unique passing CPU tests in
the [consolidated validation report](../output/omy_layout_cpu_validation_20261021/README.md).
The original prototype loading measurements
are documented in [the loading benchmark](omy_layout_loading_benchmark.md).

The integrated simulator/GPU check passed on 2026-10-21 on the RTX 3090, with all
50 CannedMushrooms layouts, one environment, 50 resets and 50 recordings. It
exercised the real collector, session and recording paths with three synthetic
commands per layout and automatic advancement. First-ready time was 24.8353 s;
the median recording-open-to-ready transition was 46.170 ms across 49 changes.
That transition metric excludes preceding export, session advancement and part
creation. The complete diagnostic took 58.685 s, including 150 commands and QC.
See the [integrated report](../output/omy_layout_collection_integration_20261021_r2/README.md),
[measurements](../output/omy_layout_collection_integration_20261021_r2/summary.json),
and [validation of all 50 recordings](../output/omy_layout_collection_integration_20261021_r2/all_layouts_validation.json).

All recordings passed structural/native-15 validation and remained explicitly
excluded from training selection and export. The added offscreen camera and
synthetic success consumer required boundary-update/query-sharing fallbacks;
held-target commands remained enabled. Real Carbonite provider-event checks
passed for F9 repeats, F12 delivery, Shift+F9 rejection and status-widget
creation/update. The collection loop used synthetic controls, READY confirmation,
leader actions and XR lifecycle, with success defined after three commands.
Physical keys, the leader, actual grasp success and live XR still need an
operator check. Reproduce the diagnostic without connecting a leader or XR:

```bash
.venv/bin/python -u scripts/check_omy_layout_collection.py \
  --output-dir DIR --render-camera
```

### Collection profile and recording contents

New collection sessions use held-target stepping, compact recording, 15 Hz
commands, render interval 4, **32 position iterations and two GPU solver
partitions**. Both the Python collector and Bash launcher default to **zero
PhysX CPU workers, one CUDA driver connection and CPU task snapshots**.
Kit/TBB retain eight workers. Zero PhysX workers runs the CPU portion on the
calling thread. CUDA connections are selected before simulator startup, with
copy-connection and launch-queue overrides cleared for the collector process.
The previous environment is restored when the Python entry point returns.

For audited single-environment placement tasks, `--task-checks cpu` packs poses,
velocities, forces, joints, origins and the episode clock into one CPU transfer
per control boundary. Placement checks, event tracking and scalar subtask queries
share it. The placement settling window and event transitions retain their
existing rules; resets/restores invalidate snapshots and timers. Unsupported
consumers, noncompact recording and CPU physics retain the native path with
explicit fallback reasons. Startup and HDF5 performance metadata report requested
and effective modes.

`--task-checks native` retains shared placement evaluation on the simulator's
device. `--placement-checks independent` also disables the CPU snapshot. To
compare the previous runtime settings in a **new** session:

```bash
scripts/collect_omy_xr.sh SimplePnp291Bowl00Task \
  --physics-workers 2 --cuda-connections 8 --task-checks native
```

Worker counts, CUDA connections and requested task-check mode are part of the
resume contract. Older sessions keep native checks, eight connections and the
previous two/eight PhysX/Kit worker defaults; conflicting explicit resume flags
are rejected. See the [implementation and verification](omy_collection_latency_implementation_20261028.md).

Identical placement checks remain shared between termination and subtask
evaluation at each audited control boundary.
The [2026-10-24 bowl investigation](omy_physics_investigation_20261024.md)
reproduced persistent finger penetration at 16 iterations. Solver, partition,
placement-check and collider settings are recorded in the session contract.
Resuming an older session retains its original settings, including independent
placement evaluation; conflicting explicit options are rejected.

`--finger-colliders hulls` enables experimental simpler finger/pad colliders.
It preserves the original finger masses, COM frames and full inertia tensors
using a versioned reference, and checks the robot asset hash and simulator
versions. This option requires one CUDA environment and the unscaled reference
robot. Its identity is recorded and checked on resume. The default is
`--finger-colliders original`; hollow object geometry is retained. For example:

```bash
./scripts/collect_omy_xr.sh SimplePnp291Bowl00Task --finger-colliders hulls --log
```

The earlier [implementation verification](omy_collection_optimization_implementation_20261024.md)
measured the production cache at 74.2 ms per headless command and the combined
candidate at 72.7 ms, versus 75.6 ms for the 32-iteration baseline. Live XR speed
for the subsequent worker/driver/snapshot combination is recorded in the
[live experiment report](omy_live_xr_latency_experiments_20261027.md). Independent
grasp/release qualification across object types remains pending.
OMY uses `velocity_limit_sim=5.0` rad/s for the driven gripper joint; the legacy
implicit-actuator `velocity_limit` field did not enforce a simulator limit.
The trigger still selects fully open or fully closed targets. The gripper limit
is specific to the OMY robot configuration and is recorded in `env_cfg.json`.
Attempts allow 120 simulated
seconds (1,800 commands). For a single task, the launcher defaults to 10 successful
episodes; override this with `--num-episodes N`. Failed, timed-out, stale-input, XR-loss,
and manually reset attempts are saved with separate attempt IDs but do not
count toward that target. Console progress shows successes against the target,
and `collection_events.jsonl` records `success_target_reached` when it is met.
Console logging is off by default; `--log` prints a
unique log path under `/tmp`. Demonstrations and collection metadata are saved
in both modes. The launcher locates the repository from its own path, so it can
also be invoked by absolute path from another directory. Use `--help` for details.

New single-task runs are named `output/omy_teleop/YYYYMMDD_HHMMSS_TASKNAME/` using the host's
local time, for example `20261020_143005_SimplePnp308SugarBox00Task`. A numeric
suffix distinguishes runs of the same task started in the same second. An
explicit `--output-dir` takes precedence over automatic naming.

The output directory is printed at startup. A single-task run contains `omy_demos.hdf5`,
`env_cfg.json`, `collection_events.jsonl`, per-attempt timing `.npz` files,
and performance/XR summaries. The
HDF5 file contains actions, non-image observations, and scene states. Successes,
manual resets, stale-input resets, and step-limited attempts are retained as
separate attempts. The event file records why an attempt ended.

New recordings also save action-aligned L100 inputs under
`data/demo_N/leader/` in `omy_demos.hdf5`, in both compact and full profiles:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `joint_position` | `(T, 6)` | Float64 source angles for `joint_1` through `joint_6`, in radians, before wrist correction, calibration, mapping, or limiting. |
| `gripper_position` | `(T, 1)` | Float64 continuous source trigger reading, before binary thresholding. Serial uses the SDK's legacy half-angle radians: `(present_ticks - goal_ticks) * pi / 4095`. |
| `wrist_turn_offset_rad` | `(T,)` | Subtract this value from raw `joint_6` to recover the mapper's wrist input. |
| `valid` | `(T,)` | Whether the row contains a fresh, complete, finite source reading. Ignore numeric position fills when false. |
| `source_wall_ns` | `(T,)` | Source-buffer Unix wall timestamp, rather than a hardware encoder timestamp. |
| `read_monotonic_ns` | `(T,)` | Host monotonic timestamp when the action consumed this sample. |
| `source_age_s` | `(T,)` | Source-sample age at consumption. |

Each row comes from the exact input consumed by `device.advance()`, without a
second serial read. Position fields use zero fills and timestamp/age fields use
`-1` when invalid. The `robolab_omy_teleop` HDF5 metadata includes
`leader_proprioception` with field order, units, conversions, and alignment.
Sampling follows the action rate (15 Hz with the Bash launcher); intermediate
100 Hz reader samples are not retained. Existing recordings are unchanged.

The timing files store action and step-completion monotonic timestamps,
simulation times, and the timestamp/age of the exact source sample consumed by
the upstream mapper. The source timestamp comes from the reader's wall clock;
`source_read_monotonic_ns` marks when that sample was consumed. They allow
wall-rate analysis without a disk write per step.

The live loop does not render dataset cameras or encode images. It buffers up
to `--max-attempt-steps` (default 1200) state-only control steps per attempt;
HDF5 writing happens when an attempt ends. This caps recorder memory but can
briefly pause the display at the episode boundary. The Python collector's
`--num-episodes` defaults to 1000 successful episodes; the Bash launcher defaults
to 10. Ctrl+C exits earlier, regardless of how many successes have been collected.

The default is 120 Hz physics, 60 Hz L100 commands, and 60 nominal XR renders
per simulated second. Add `--control-hz 30` or `--control-hz 15` to test a lower
command frequency: physics stays at 120 Hz and XR rendering stays configured
at every second physics step. At 15/30 Hz, the active loop spaces its renders
throughout each command instead of rendering a burst and sleeping at the end.
`--read-hz` still controls the independent L100 serial reader (default 100 Hz).
The rendered frame interval and overall rates appear in the periodic report
and `omy_performance.json`; actual headset smoothness also depends on SteamVR.

The default native-15 collection profile can also be specified explicitly:

```bash
cd /tmp2/leocheng/RoboLab
PYTHONPATH=/tmp2/leocheng/RoboLab .venv/bin/python -u examples/collect_omy_xr.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib /tmp2/leocheng/RoboLab/omy_calib_robolab.json \
  --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 0 \
  --cuda-connections 1 --task-checks cpu \
  --control-hz 15 --solver-position-iterations 32 --gpu-max-num-partitions 2 \
  --placement-checks shared --stepping-profile held-target \
  --recording-profile compact --render-interval 4 \
  --xr-resolution-multiplier 0.8 --vel-scale 2 --stats-interval 1
```

Run this from a desktop with SteamVR/ALVR and the calibrated leader available.
The scene cap and collider profile are applied before environment creation;
reference mass properties are restored after collider cooking and before use.
The direct collector and Bash launcher default to this physics, task-query,
stepping, recording and render cadence. Generic teleoperation/evaluation retains
its existing defaults.
Interval 4 renders at nominal 30 Hz and requires 15 or 30 Hz commands. It keeps
the desktop viewport enabled. Operator comfort and realtime throughput still
need validation with the headset worn.

Held-target admission is based on capabilities, not a task-name allowlist.
The SDK lifecycle remains audited for Isaac Lab 2.2.0, one CUDA DROID environment
and eight 120 Hz substeps. `capabilities.features` in the performance configuration
reports target holding, boundary-rate scene updates, compact observation reuse,
boundary queries and shared event queries separately. For example, history/air-time
sensors keep physics-rate scene updates without disabling compatible held targets
or observation reuse. External wrenches restore per-substep target submission and
updates; interval events or unknown recorder lifecycles use the upstream loop.
Unknown task consumers disable boundary-rate updates and shared queries, rather
than being assumed safe. Each disabled feature has its own fallback reasons.

Compact recording retains `initial_state`, `actions`, full thirteen-joint
`states`, arm/gripper pre-action observations, and task/event tracking. It adds
per-command simulation/source/monotonic timestamps and stepping/solver profile
fields. It omits derived EE and bounding-box channels, preallocates storage at
READY, and enforces `--episode-buffer-mb` (default 64 MiB). Exceeding the byte
budget stops admission. No recorded command may exceed the fixed attempt
capacity. The default native-15 buffer holds at most 300 commands.

Each admitted attempt also saves `attempt_NNNN_performance.json` with work and
command-interval p50/p95/p99, missed budgets, render gaps, and total export
duration. Export remains synchronous; its transfer and disk costs have not yet
been separated. First- and second-command terminations count as completed
attempts and are exported before returning to READY.

`--max-attempt-seconds` defaults to 20 simulated seconds at every command rate;
the step cap remains a separate memory bound. A 15 Hz attempt therefore ends
after at most 300 commands by default, while a 60 Hz attempt ends after at most
1200. A lower command rate evaluates BananaInBowlTask's placement and contact
success once per longer command interval, which may miss a brief contact. Check
successful grasp-and-release attempts at the selected rate before collecting a
large dataset. The robot command and measured state remain distinct in HDF5.

For timing comparisons, use `--contact-sensors full` as the sensor baseline and
`--contact-sensors required` (default) for dependency-based selection. The planner
unions termination predicates, every subtask stage, observations, recorders and
event consumers. It keeps all gripper sensors (including distractors/table and
individual batch-layout fallbacks), plus any required object-pair force sensors.
Known spatial/contact primitives are registered by callable identity and bound
parameters; an unknown consumer or missing dependency keeps the full sensor set.
Task names and `manifest.json` do not grant optimization eligibility.

For the current definitions, Banana and generated ComputerMouse retain 5/7
sensors, while Yogurt retains 21/211. Yogurt does not require yogurt–bowl contact
for its configured geometric placement predicate; Banana and ComputerMouse do.
Only observation sensor views are removed: colliders, gravity, robot geometry,
solver settings and the eight physics steps are unchanged by generalization.

Read-only query sharing expires every command/reset/restore. Single-env indexed
pose/velocity queries reuse the same batched boundary snapshot; validated gripper
contact masks can also be reused with a full sensor profile in `OmyEnv`. Force
vectors retain their original reduction/sign convention. Event and subtask state
machines still execute in their original order and frequency; their labels are
not reconstructed from images or skipped. The pre-action observation/action/post-
state alignment and 1/15 s simulation timestamps remain unchanged. Per-command
timing records both `held_target` and `boundary_scene_updates` for runtime fallbacks.

For a loop A/B, additionally compare `--stepping-profile ordinary` against
`held-target` with the **same** sensor profile, cap, task, seed and recording flags.
Automated CPU tests cover dependency planning/cadence/alignment; the GPU/headset
grasp–lift–release and F9 reset checks must still pass before bulk collection.

### Generalization validation (2026-10-05)

The full unit suite passed 655 tests; after the final changes, all 79 focused
dependency/capability/cadence tests and 21 headless GPU integration tests passed.
GPU checks used Isaac Sim 5.0 / Isaac Lab 2.2.0 on GPU 1 (RTX 3080), without XR,
dataset cameras or a leader connection. The existing GPU-0 XR session was left running.
The test launcher and raw final log are temporarily available at
`/tmp/robolab-omy-generalized-SNY8Qe/check.py` and `gpu_verified.log` in that directory.

Six additional benchmark files were sampled reproducibly with seed `20261005`,
excluding Banana and Yogurt, which already had dedicated integration coverage:

| Task | Original contact sensors | Retained sensors |
| --- | ---: | ---: |
| ElectronicsInBinTask | 106 | 15 |
| PickUpBluePitcherTask | 11 | 5 |
| FoodPacking1BoxesTask | 16 | 6 |
| GrabABagelTask | 211 | 21 |
| ButterAboveRaisinTask | 7 | 5 |
| BlockStackingOrderAgnosticTask | 16 | 12 |

All six admitted every fast-path capability, advanced eight physical steps per
command, and returned finite scene states. At identical physical boundaries,
independent event trackers with sharing disabled/enabled produced identical
labels, including a deliberately displaced distractor's out-of-workspace event.
The stacking task retains every possible block-neighbor contact, not just the
initial ordering; surface-support tasks retain the original force-vector query.

Banana, Yogurt and generated ComputerMouse also passed actual HDF5 checks for
pre-action observations, eight-dimensional actions, thirteen-joint post-states
and `1/15` simulated-second intervals. Terminal export, reset, same-counter
restore, and runtime-wrench fallback regressions passed. Standard evaluation does
not opt into these OMY caches or the OMY manager loop.

These are short correctness/admission tests, **not** completed manipulation tasks
or a measured live-XR speedup. An initial pitcher scene load reported a jerrican
mesh falling back to CPU collision processing; the optimization does not alter
that mesh. Restart the collector to load the new code, then test F9/READY,
grasp–lift–release and F9 reset with the headset worn before bulk collection.

### All-benchmark validation (2026-10-05)

All **120 benchmark tasks** passed the headless GPU suite and the production
postprocessor's recording validator. Each task ran two reset seeds with 30
synthetic arm/wrist/gripper commands per seed: 240 finalized HDF5 episodes,
7,200 commands and 57,600 recorded physics substeps. Physics stayed at 120 Hz,
control/data at native 15 Hz, and the scene position-iteration cap at 16.
Maximum native interval error was `2.36e-16` s; pre-action arm alignment error
was exactly zero. All twenty isolated pytest batches exited successfully.

118 tasks admitted every generalized fast-path capability. The two ordered
placement tasks, `BananaThenRubiksCubeTask` and `RubiksCubeThenBananaTask`, kept
full sensors, substep scene updates and unshared queries for their unregistered
stateful termination, while retaining safe held-target commands and compact
observations. Independent shared/unshared event, termination and subtask checks
matched at 7,080 recorded boundaries; all 118 admitted tasks also passed a
nonempty displaced-object event check outside their saved trajectories.

The full unit run passed 655 tests inside the sandbox. Two unrelated mock-only
IPC tests were denied local socket binding by the sandbox and both passed on
an unsandboxed rerun, giving 657 distinct passing unit tests. Production code,
task definitions, calibration and headset/system settings were not changed
for this validation; only test tooling and documentation were added/updated.

These are short synthetic correctness tests, not task completions, XR latency
measurements, camera-materialization tests or proof of grasp accuracy. Their
recordings are **test-only, not training demonstrations**. QC retained wall-time
gap warnings in 21 episodes despite their valid native-15 simulation timing.
Existing asset CPU collision-mesh fallback warnings are also preserved. Manual
F9/READY, grasp–lift–release and F9 reset checks are still required with L100/XR
before bulk collection.

The [full report and per-task table](../output/omy_pipeline_benchmark/20261005_213209_all_benchmark/README.md)
link to each batch log, configuration/dependency report, raw test HDF5 and
postprocessor QC. Reproduce the suite without opening a leader or headset:

```bash
cd /tmp2/leocheng/RoboLab && .venv/bin/python -u scripts/check_omy_benchmark.py --gpu 1 --batch-size 6 --steps 30 --seeds 2
```

### Live XR verification

Run the same task, seed, GPU, XR resolution, and `--steps` limit in separate
processes. Compare an idle-leader run first, then repeat a manual drop and a
grasp–lift–release sequence. Each run gets a new output directory. Compare
the saved windows with:

```bash
cd /tmp2/leocheng/RoboLab && .venv/bin/python -u examples/teleop_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib /tmp2/leocheng/RoboLab/omy_calib_robolab.json --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 2 --xr-resolution-multiplier 0.8 --vel-scale 2 --steps 600 --stats-interval 5 --contact-sensors full
```

Repeat the same command with `--contact-sensors required`. Keep the L100
still for the idle comparison; use another pair of runs for manipulation.
The final timing window contains at most 300 control steps, so keep each
activity comparable near the end of its run. Locate the two output paths
printed at startup, then compare them:

```bash
.venv/bin/python scripts/compare_omy_performance.py output/omy_teleop/<full-run> output/omy_teleop/<required-run>
```

The report shows control Hz, real-time factor, p95 step time, and physics,
render, sensor, and task-check time. The `contact_cache` entry in each required
run's `omy_performance.json` counts batch contact masks computed, reused, and
fallbacks to individual sensors. A high fallback count means the installed
PhysX filter layout did not match the task's one-object-per-filter mapping.
Lazy sensor updates can be charged to task-check time rather than sensor time;
compare both. Check task success and object contact behavior before accepting
a faster run. This profile changes contact-sensor observation overhead, not
collision response, gravity, physics dt, or the DROID arm reset pose.

Render policy-camera observations later, after collection, using the recorded
states and the OMY absolute-joint task registration:

```bash
cd /tmp2/leocheng/RoboLab && .venv/bin/python scripts/materialize_demo_images.py --input output/omy_teleop/<run>/omy_demos.hdf5 --output output/omy_teleop/<run>/omy_demos_images.hdf5 --device cuda:0
```

Use an output directory separate from the raw recording if the image-enriched
`env_cfg.json` sidecar is also needed for replay. Offline materialization
validates action/state counts and renders pre-action camera observations from
the stored scene states.
