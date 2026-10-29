# One versus two GPU solver partitions — 2026-10-27

Keep the collection default at **two partitions** for now. This pilot found no
consistent speed or placement advantage from one partition. Both settings still
have replay failures and occasional contact-overlap spikes; neither is qualified
across the object library by these results.

## Controlled comparison

Ran **32 diagnostic replays in 16 fresh simulator processes** on isolated host
GPU 1 (RTX 3080), with GPU ownership serialized through the repository lock.
Fixed settings: 32 position-iteration cap, 120 Hz physics, 15 Hz recorded commands,
held-target stepping, two physics workers, shared placement checks, original
finger colliders, compact recording, no rendering or XR. Saved simulation configs
differ only in `gpu_max_num_partitions`; robot configs and runtime masses, COMs,
inertias, joint properties and collision offsets match between paired settings.

The three source trajectories were selected before running:

- Bowl layout 00, `demo_1`, from `20261024_141641_bowl`: the existing regression
  trajectory, originally recorded as unsuccessful. Replay the first 240 commands,
  enough to include grasp, transport, release and a subsequent observation window.
- Plastic pail layout 01, `demo_1`, 102 commands, and layout 02, `demo_7`, 121
  commands, from `20261024_134851_plasticpail_a01`: the first two successful
  trajectories in sorted task order. Replay their complete recorded commands.

Each trajectory/partition pair had two throughput replays and two replays with
contact separation sampled from both inner fingers every physics step: 24 runs.
Four additional pairs of pail replays allowed up to two seconds holding the final
recorded action, stopping early on task success: eight more runs. These are new
replays from the initial state, not continuations of the earlier GPU processes.

## Speed

Only the runs without contact instrumentation contribute to this table. Each mean
uses exactly source command rows 20–99, across two repeats, at both settings.
Startup, rendering and XR are excluded; differences are pilot measurements, not
statistically established performance guarantees.

| Trajectory | One partition | Two partitions | One versus two |
| --- | ---: | ---: | --- |
| Bowl 00 | 62.80 ms/command | 64.98 ms/command | 3.3% faster |
| Pail 01 | 64.24 ms/command | 67.06 ms/command | 4.2% faster |
| Pail 02 | 73.99 ms/command | 69.98 ms/command | 5.7% slower |

## Task success and settling

The first table reports success **within the specified recorded-command horizon**.
A failure here can include a trajectory that needs more time to settle.

| Trajectory | One partition | Two partitions |
| --- | ---: | ---: |
| Bowl 00 | 4/4 | 4/4 |
| Pail 01 | 0/4 | 0/4 |
| Pail 02 | 2/4 | 2/4 |

The separate pail replays with up to two additional seconds produced:

| Trajectory | One partition | Two partitions |
| --- | ---: | ---: |
| Pail 01 | 1/2 | 2/2 |
| Pail 02 | 2/2 | 1/2 |
| Total | 3/4 | 3/4 |

Three of these eight runs became successful during the extra hold: one pail-01
run at each setting, and one pail-02 run with one partition. Two runs remained
unsuccessful after the complete hold, one at each setting. This supports reporting
success at the recorded endpoint separately from success reached after additional
settling. It does not justify automatically treating a failed endpoint as passed.

These are repeated executions of three trajectories and two object types, not
32 independent demonstrations or estimates of collection-wide success rates.
Restored states omit hidden contact/solver caches, so fresh-process replays can
reach different trajectories even at the same settings. The old recorded pail
success labels therefore do not establish success under this collection profile.

## Contact quality

Neither setting showed persistent finger–target attachment after the final open
command in the observed contact-instrumented runs. All had no such contact in the
last 0.25 seconds. This checks both inner fingers, not every robot/scene collider.

Worst reported overlap across the baseline and continuation contact runs:

| Trajectory | One partition | Two partitions |
| --- | ---: | ---: |
| Bowl 00 | 0.741 mm | 0.645 mm |
| Pail 01 | 0.287 mm | 0.264 mm |
| Pail 02 | 4.982 mm | 2.030 mm |

Each pail-02 spike above 1 mm occupied one sampled physics step, at source command
48. Both runs ultimately satisfied task success. The one-partition spike occurred
in the continuation experiment, during the original grasp commands before the
extra hold. These are contact-separation reports, not exhaustive mesh-intersection
measurements, and a transient report can describe overlap resolved during that
step. They still show why a success flag alone cannot establish contact quality.

Bowl task successes also do not establish continuous grasp retention: some traces
lose inner-finger contact before the requested opening while still landing in the
bin. Keep task success, grasp retention and release/penetration diagnostics separate.

## Artifacts and verification

All 16 processes exited normally. Source recording hashes remained unchanged,
and the original benchmark script and collection defaults were not edited. The
continuation variant is an output-only diagnostic copy with its exact patch saved.
GPU 1 was released after completion; no simulator process remained.

- [Baseline analysis](../output/omy_diagnostics/20261027_partition1_compare/analysis.json)
- [Continuation analysis](../output/omy_diagnostics/20261027_partition1_compare/settling/analysis.json)
- [Commands, source hashes and GPU identity](../output/omy_diagnostics/20261027_partition1_compare/manifest.json)
- [Continuation commands](../output/omy_diagnostics/20261027_partition1_compare/settling/manifest.json)
- [Configuration/property and source-integrity verification](../output/omy_diagnostics/20261027_partition1_compare/verification.json)
- [Diagnostic continuation patch](../output/omy_diagnostics/20261027_partition1_compare/tail_harness.patch)

The artifact directory also contains per-run configurations, logs, trajectories,
contact samples, replay launchers and analysis scripts. No videos were generated.
