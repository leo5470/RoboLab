# OMY optimization implementation — first milestone

The design reference is `/home/leocheng/RoboLab/docs/omy_pipeline_optimization.md`.
The solver-transfer evidence is preserved in
[`20261002_solver_transfer`](../output/omy_pipeline_benchmark/20261002_solver_transfer/README.md).
This implementation is available in `/tmp2/leocheng/RoboLab` and the SSD
checkout at `/home/leocheng/RoboLab`.

## Implemented

- Opt-in scene position caps 16/32, applied before environment creation.
  Physics remains CUDA at 120 Hz for the held-target candidate; velocity solver
  settings, gripper linkage, and calibration are preserved.
- Dedicated OMY environment injection. The supported compact BananaInBowl
  contract processes/submits a command once, performs eight physics steps,
  renders at their scheduled steps, and updates lazy scene caches at the
  control boundary. Unsupported configurations select ordinary stepping.
- Independent XR render intervals 2/4, with explicit render pacing and the
  desktop viewport enabled.
- Compact fixed-capacity raw recording. All thirteen joints and rigid object
  poses/velocities remain in the existing HDF5 paths. Arm/gripper pre-action
  observations retain the production normalized gripper convention. Initial
  state, actions, task/event tracking, timing, and physics provenance remain.
- Scoped read-only world-query sharing between termination/event recording;
  predicate parameterizations and state-machine transitions stay distinct.
  The redundant compact post-step observation pass is removed for admitted
  configurations.
- Exact mapper-consumed source sample timing, bounded attempt buffers,
  per-attempt work/interval percentiles, render gaps, and total export timing.
- Early OMY termination is exported once and counted as an attempt. The common
  environment's legacy startup-artifact reset now executes once. Explicit OMY
  resets finalize pending data and clear the next episode's buffer. State
  restoration invalidates OMY caches without requiring a new step counter.

The reference CLI defaults remain ordinary stepping, full recording, and the
existing 32-iteration scene cap. See
[`omy_xr_collection.md`](omy_xr_collection.md) for the candidate launch options.

Validation results are in
[`20261002_implementation`](/tmp2/leocheng/RoboLab/output/omy_pipeline_benchmark/20261002_implementation/README.md):
138 pure unit checks, ten focused GPU contract checks, the existing OMY/recorder
regressions, nine saved-grasp replays with equal settling horizons, and one
84-frame-per-camera offline materialization. The headless cap-16 held-target
path averaged 52.6 ms/command versus 62.5 ms for ordinary compact stepping.

## Remaining validation and work

Headless contract tests do not establish XR head-tracking correctness, comfort,
contact/substep stability, or realtime operator throughput. The combined
profile remains opt-in pending matched replay/settling comparisons and the
independent human-operated attempts specified in the design.

The shared query context is an initial reduction in duplicate work. EventTracker
still performs its Python lifecycle decisions and scalar transfers; a single
batched host decision transfer is not implemented. Export is synchronous with
total duration instrumentation; snapshot/transfer/write/close instrumentation
and a background writer remain follow-up work driven by boundary-gap evidence.

Policy-camera materialization still restores actual recorded states. PPS export
recipe validation, production reference π0.5 launch/dependency setup, and
training-transfer evaluation remain later milestones. Diagnostic files from
the solver-transfer suite are not training-ready exports.
