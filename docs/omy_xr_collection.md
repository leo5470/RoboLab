# L100 + Quest demonstration collection

Use `examples/collect_omy_xr.py` for recorded attempts. `teleop_omy_xr.py`
remains the direct teleoperation smoke test.

Run from the computer's GPU-backed desktop, with the Isaac Sim window able to
receive keyboard focus. Space is the only episode key:

1. After scene reset, return the physical L100 to its calibrated READY pose and
   release the gripper trigger. The simulated robot stays at home while you do so.
2. Press Space in the Isaac Sim window. The mapped arm target must be within
   `--ready-tolerance-rad` of the DROID home, with a fresh leader sample and a
   stable pose for `--ready-stable-s`.
3. Move the leader or trigger by `--start-motion-rad` to start the attempt.
   The first action is recorded and applied together.
4. Press Space while recording to export an unsuccessful attempt and reset the
   scene. Successful task termination exports automatically and resets the
   scene. Return the L100 to READY and press Space for the next attempt.

Holding Space counts once. Space while armed cancels arming. XR loss or stale
leader input ends a running attempt and requires another READY confirmation.
The physical leader is never driven by scene reset. The same seed is used on
every reset for repeatable object placement; calibration is never changed.

Example:

```bash
cd /home/leocheng/RoboLab && .venv/bin/python -u examples/collect_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib /home/leocheng/RoboLab/omy_calib_robolab.json --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 2 --xr-resolution-multiplier 0.8 --vel-scale 2 --stats-interval 1
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
cd /home/leocheng/RoboLab
PYTHONPATH=/home/leocheng/RoboLab .venv/bin/python -u examples/collect_omy_xr.py \
  --task BananaInBowlTask --source serial --port /dev/robotis_left \
  --calib /home/leocheng/RoboLab/omy_calib_robolab.json \
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

Held-target stepping is currently admitted only for the audited BananaInBowl
compact contract on Isaac Lab 2.2.0: one CUDA environment, plain implicit
actuators, eight 120 Hz substeps, and current-only contact sensors. Unsupported
consumers, sensor histories, interval events, substep diagnostics, or external
wrenches select ordinary stepping with a logged reason. The effective profile
and solver settings appear in recording metadata and `env_cfg.json`.

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

For timing comparisons, use `--contact-sensors full` as the baseline and
`--contact-sensors required` for the audited OMY task variant. The latter is
the default for `BananaInBowlTask` and `AnimalsInBinTask`; other tasks keep all
contact sensors. Banana retains gripper sensors, banana–bowl contact, and
banana–table contact. Animals retains gripper sensors and table contacts for
the lizards and bin. Physical collision handling is unchanged.

Run the same task, seed, GPU, XR resolution, and `--steps` limit in separate
processes. Compare an idle-leader run first, then repeat a manual drop and a
grasp–lift–release sequence. Each run gets a new output directory. Compare
the saved windows with:

```bash
cd /home/leocheng/RoboLab && .venv/bin/python -u examples/teleop_omy_xr.py --task BananaInBowlTask --source serial --port /dev/robotis_left --calib /home/leocheng/RoboLab/omy_calib_robolab.json --device cuda:0 --render-gpu 0 --cpu-workers 8 --physics-workers 2 --xr-resolution-multiplier 0.8 --vel-scale 2 --steps 600 --stats-interval 5 --contact-sensors full
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
cd /home/leocheng/RoboLab && .venv/bin/python scripts/materialize_demo_images.py --input output/omy_teleop/<run>/omy_demos.hdf5 --output output/omy_teleop/<run>/omy_demos_images.hdf5 --device cuda:0
```

Use an output directory separate from the raw recording if the image-enriched
`env_cfg.json` sidecar is also needed for replay. Offline materialization
validates action/state counts and renders pre-action camera observations from
the stored scene states.
