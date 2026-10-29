# Collection latency implementation — 2026-10-28

At the user's request, the tested worker/driver/CPU-snapshot combination is now
the default for new dedicated collection sessions. Both
`examples/collect_omy_xr.py` and `scripts/collect_omy_xr.sh` select:

| Setting | New collection default |
| --- | --- |
| PhysX CPU workers | 0, CPU work on the calling thread |
| Kit/TBB workers | 8 |
| CUDA driver connections | 1, configured before CUDA/Kit startup |
| Task/event evaluation | Shared CPU snapshot when the consumer audit passes |
| Physics / command cadence | 120 Hz / 15 Hz simulation time |
| Solver / partitions | Existing TGS, position cap 32 / two partitions |
| Recording / stepping | Compact state-only / held targets |
| Render interval | Four physics substeps |
| Finger and object models | Existing original geometry and properties |

The direct Python collector now uses the same native-15 stepping, recording and
attempt limits as the optimized Bash profile. Generic teleoperation and policy
evaluation retain their existing defaults. The worker validator accepts zero;
negative PhysX workers and nonpositive Kit workers are rejected.

## Runtime ownership and fallback

`task_snapshot.py` captures the required poses, velocities, robot joints, contact
matrices, environment origins and episode clock in one packed CPU transfer per
admitted task-evaluation boundary. It reuses the existing placement and event
implementations. It changes no action, physics substep, recorded state or
observation. Manager termination results return on the simulator device.

The consumer audit requires one CUDA environment with the existing audited
native-15 query/compact-observation capabilities, compact recording, shared
placement checks, one resting-in-container termination, and the supported
placement/grasp subtask primitives. Unknown consumers, unsupported sensors or
dependencies, different recording modes and CPU physics retain native queries
with recorded fallback reasons. Contact-filter expansion uses individual contact
matrices rather than assuming a one-to-one batch mapping.

CPU event-tracker state is instance-owned and migrates without clearing its
history. Subtask state machines still advance once per command. Snapshot/timer
state is invalidated on reset and restore, including restoring at the same
episode counter. Placement diagnostics read the current snapshot's actual timer;
the ordinary native world is used outside the audited boundary. Snapshot copy
time is included in the task-check performance region.

The collector process selects `CUDA_DEVICE_MAX_CONNECTIONS` before importing
CUDA/Kit, clears copy-connection and launch-queue overrides, and records the
applied environment. The Python entry point restores prior environment values on
exit, including exceptions. No global driver or persistent GPU setting changes.

## Overrides and resume

To collect with the previous runtime settings in a new session:

```bash
scripts/collect_omy_xr.sh SimplePnp291Bowl00Task \
  --physics-workers 2 --cuda-connections 8 --task-checks native
```

`--cpu-workers N` is also forwarded. `--cuda-connections` accepts 1–32. Requested
and effective task-check modes, fallback reasons and CUDA environment values
appear in the startup/HDF5 performance metadata. Preflight reports the selected
runtime without opening the simulator or leader.

New resume contracts include both worker counts, CUDA connections and requested
task-check mode. Stored values are restored and conflicting explicit flags are
rejected. Contracts predating these fields retain the previous launch defaults:
two PhysX workers, eight Kit/TBB workers, eight connections and native checks.
The older partition/placement/collider contract variants remain supported. The
Bash wrapper does not inject defaults that could override a resumed profile.

## Verification

The production implementation was compared with native task queries in ten
fresh GPU processes, two replays per process, on the bowl, both pail layouts,
milk carton and pudding-mix box fixtures. Both sides used the same original
models, solver, partition, worker and driver settings. These checks opened no
XR session or physical leader.

All **620 checks across ten matched replay pairs** pass exact dtype/shape/value
comparison for numeric states, actions, observations, subtask records and task
events, along with unchanged physical properties and solver configuration.
All twenty recorded episode timelines retain native 15 Hz simulation intervals.
Source recording hashes remain unchanged. Both successful and failed cold
replays are retained; matching results do not establish new grasp success rates.

Artifacts are under
`output/omy_diagnostics/20261027_collection_latency_integration/`; the directory
date uses the UTC experiment date. See its `verification.json`, exact launch
commands, per-case summaries and `integration_summary.json`. All **80 unit tests**
pass. A separate tensor-only CUDA probe also verifies CPU/CUDA device aliases,
manager result placement and independence from caller mutations. Unit tests
cover snapshot ownership, force
direction/contact aliases, settling timers, same-counter restore, placement
diagnostics, unsupported consumers, event history, environment cleanup, worker
validation, shell forwarding and old/new resume contracts.

The earlier [live XR/L100 measurements](omy_live_xr_latency_experiments_20261027.md)
tested the diagnostic implementation: a small repeated bowl timing improvement
and less perceived lag in the first manual candidate block. The integrated
implementation has not received another live XR round. Candidate release,
reverse-order manual trials, additional live scenes and sustained runs remain
unqualified; the pre-existing right-finger sticking issue is not fixed here.

Postprocessing continues to validate and preserve the recorded 15 Hz simulation
timeline. Slow wall-clock collection does not resample commands or change their
simulated duration, and image materialization continues to restore recorded
states without replaying physics.
