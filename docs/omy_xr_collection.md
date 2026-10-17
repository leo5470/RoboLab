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
receive keyboard focus. F9 is the only episode key:

1. After scene reset, return the physical L100 to its calibrated READY pose and
   release the gripper trigger. The simulated robot stays at home while you do so.
2. Press F9 in the Isaac Sim window. The mapped arm target must be within
   `--ready-tolerance-rad` of the DROID home, with a fresh leader sample and a
   stable pose for `--ready-stable-s`.
3. Move the leader or trigger by `--start-motion-rad` to start the attempt.
   The first action is recorded and applied together.
4. Press F9 while recording to export an unsuccessful attempt and reset the
   scene. Successful task termination exports automatically and resets the
   scene. Return the L100 to READY and press F9 for the next attempt.

Holding F9 counts once. F9 while armed cancels arming. Space remains Isaac Sim's
Play/Pause shortcut and must not be used for episode control; pausing the
timeline can block physics stepping even while the XR view remains responsive.
XR loss or stale
leader input ends a running attempt and requires another READY confirmation.
The physical leader is never driven by scene reset. The same seed is used on
every reset for repeatable object placement; calibration is never changed.
New recordings store `reset_key: "F9"` in their collection metadata; existing
recordings are not modified.

Example:

```bash
cd /tmp2/leocheng/RoboLab && .venv/bin/python -u examples/collect_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib /tmp2/leocheng/RoboLab/omy_calib_robolab.json --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 2 --xr-resolution-multiplier 0.8 --vel-scale 2 --stats-interval 1
```

The output directory is printed at startup. It contains `omy_demos.hdf5`,
`env_cfg.json`, `collection_events.jsonl`, per-attempt timing `.npz` files,
and performance/XR summaries. The
HDF5 file contains actions, non-image observations, and scene states. Successes,
manual resets, stale-input resets, and step-limited attempts are retained as
separate attempts. The event file records why an attempt ended.

The timing files store action and step-completion monotonic timestamps,
simulation times, and the timestamp/age of the exact source sample consumed by
the upstream mapper. The source timestamp comes from the reader's wall clock;
`source_read_monotonic_ns` marks when that sample was consumed. They allow
wall-rate analysis without a disk write per step.

The live loop does not render dataset cameras or encode images. It buffers up
to `--max-attempt-steps` (default 1200) state-only control steps per attempt;
HDF5 writing happens when an attempt ends. This caps recorder memory but can
briefly pause the display at the episode boundary. `--num-episodes` defaults
to 1000; Ctrl+C exits earlier.

The default is 120 Hz physics, 60 Hz L100 commands, and 60 nominal XR renders
per simulated second. Add `--control-hz 30` or `--control-hz 15` to test a lower
command frequency: physics stays at 120 Hz and XR rendering stays configured
at every second physics step. At 15/30 Hz, the active loop spaces its renders
throughout each command instead of rendering a burst and sleeping at the end.
`--read-hz` still controls the independent L100 serial reader (default 100 Hz).
The rendered frame interval and overall rates appear in the periodic report
and `omy_performance.json`; actual headset smoothness also depends on SteamVR.

An experimental native-15 CUDA profile is available with explicit options:

```bash
cd /tmp2/leocheng/RoboLab
PYTHONPATH=/tmp2/leocheng/RoboLab .venv/bin/python -u examples/collect_omy_xr.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib /tmp2/leocheng/RoboLab/omy_calib_robolab.json \
  --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 2 \
  --control-hz 15 --solver-position-iterations 16 --stepping-profile held-target \
  --recording-profile compact --render-interval 4 \
  --xr-resolution-multiplier 0.8 --vel-scale 2 --stats-interval 1
```

Run this from a desktop with SteamVR/ALVR and the calibrated leader available.
The scene cap is applied before environment creation. The default remains the
32-iteration scene with ordinary stepping, full recording, and interval 2.
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
