# Collection runtime latency experiments — 2026-10-27

This round measures further collection latency reductions with the current
physics and robot model. The strongest candidate combines one CUDA driver
connection with a CPU snapshot shared by placement checks, event tracking and
subtask checks. XR, rendering, cameras and the leader are disabled throughout.

The baseline already uses zero PhysX CPU workers, the small improvement from
the [earlier headless experiments](omy_headless_latency_experiments_20261027.md).
Savings here are additional to that earlier change. All implementations in this
round remain diagnostic prototypes; collection defaults are unchanged.

## Experiment design

Runs use isolated host GPU 0, an RTX 3090 with 24 GiB VRAM. Fixed settings are
TGS with a 32 position-iteration cap, two GPU solver partitions, 120 Hz physics,
15 Hz commands, held-target stepping, shared placement checks, compact recording,
and the original robot colliders, inertias, actuators and mimic constraints.

The latency measurement is full command work, including physics, task checks,
event tracking, recording and observations. It excludes XR, rendering, leader
communication and the collection rate limiter. Every timing window uses recorded
command rows 20–99; profiling and extra contact readback are disabled.

The five source trajectories cover the bowl, two plastic-pail layouts, a milk
carton and a pudding-mix box. Each replay restores the recorded initial state,
executes recorded commands and permits up to two seconds holding the final action.
Success may end a replay early. This is a paired runtime comparison, not a new
estimate of task success rates. The manifest retains paths, episode identifiers,
source hashes, effective settings and exact launch commands.

The confirmation compares the baseline, one CUDA connection alone, and the
combined prototype in two fresh-process rounds across all five trajectories.
Case order and configuration order alternate. Raw results retain every process.
The comparison also identifies pairs whose sampled GPU clocks, across both
80-command windows, span at most 30 MHz. Different boost clocks can change
command time by several milliseconds, so unmatched pairs do not establish a
configuration's speed benefit.

## CPU scheduling

The bowl screen used two replays per configuration, plus baseline runs at both
ends. Reducing Kit/TBB pools, limiting PyTorch threads, using physical cores only,
pinning the main thread away from worker threads, and restricting the process
to four physical cores produced no material gain.

| Setting | Mean command work, ms |
| --- | ---: |
| Baseline, Kit/TBB 8/8 | 71.53 |
| Kit/TBB 1/1 | 71.22 |
| Kit/TBB 2/2 | 71.64 |
| Kit/TBB 4/4 | 71.43 |
| Kit/TBB 2/2, PyTorch one thread | 71.30 |
| Physical cores only | 71.45 |
| Main thread isolated from other threads | 71.44 |
| Four physical cores | 71.48 |

All of these windows sampled 1695 MHz. Differences of approximately 0.3 ms or
less in this screen do not justify changing affinity or thread-pool settings.

## Shared CPU snapshot

The existing collection performs many tiny GPU operations for geometry checks,
event tracking and scalar subtask queries in a single environment. The prototype
packs the required poses, velocities, contact forces and robot joint positions
into one 268-byte device-to-host transfer per command boundary. It then runs the
existing placement predicate and event-tracker logic on CPU tensors. Scalar
subtask queries reuse those values within the same audited boundary.

Moving only placement evaluation to the CPU saved approximately 0.1–0.6 ms in
the initial bowl/pail/carton comparisons. Duplicated GPU work in event tracking
limited that version. Sharing the snapshot with event tracking saved about
1.8–2.1 ms at matched clocks; reusing it for scalar subtask checks removed another
small amount of recording overhead. Physics, sensor cadence, predicate thresholds,
event rules and the recorded command sequence are unchanged.

The prototype is specialized to the tested single-environment SimplePnP tasks.
It keeps predicate timers in a private CPU world and uses diagnostic wrappers.
Production integration needs explicit CPU snapshot ownership, reset/diagnostic
state handling and a fallback for unsupported task dependencies. These wrappers
are evidence for the optimization, not a general collection implementation.

## CUDA driver scheduling

`CUDA_DEVICE_MAX_CONNECTIONS` controls CUDA driver work queues. It is separate
from PhysX's GPU solver partition count, which remains two. NVIDIA documents
connections from 1 to 32, with a default of eight, and documents launch-queue
capacity scaling through `CUDA_SCALE_LAUNCH_QUEUES` at 2× and 4×.
See [NVIDIA's CUDA environment-variable documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

Each setting is passed in the fresh child process's environment before CUDA or
Kit initializes. Baseline processes clear these overrides. No global GPU clock,
power or driver setting is changed.

The initial same-clock bowl comparison saved 2.10 ms with one connection:
71.65 → 69.56 ms per command. The physics region accounted for the improvement.
Thirty-two connections did not improve this comparison: 71.75 ms.

Combining one connection with the shared CPU snapshot saved 4.36 ms in the first
bowl pair: 71.45 → 67.09 ms, with both runs at 1695 MHz. This setting is intended
for the single-scene workload under test. Stream scheduling can behave differently
with rendering or multiple environments.

<!-- CONFIRMATION_RESULTS -->

## Confirmation results

The table uses only stable, clock-comparable fresh-process pairs. Values are
means of the process-level measurements; p95 values are also averaged across
the included processes. All raw and qualified comparisons remain in the JSON.

| Trajectory | Baseline, ms | Combined prototype, ms | Saved, ms | p95: baseline → combined, ms | Clock-comparable pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bowl 00 | 71.36 | 67.04 | 4.32 | 78.62 → 73.74 | 2/2 |
| Pail 01 | 72.24 | 67.57 | 4.66 | 81.69 → 76.58 | 2/2 |
| Pail 02 | 74.42 | 70.09 | 4.34 | 84.45 → 79.62 | 2/2 |
| Milk carton 00 | 70.65 | 66.06 | 4.60 | 75.52 → 70.67 | 1/2 |
| Pudding-mix box 01 | 70.42 | 66.17 | 4.25 | 75.31 → 70.54 | 1/2 |

One CUDA connection alone saves **2.17–2.28 ms** across the clock-qualified case averages.
The combined prototype saves **4.25–4.66 ms**, or approximately 5.8–6.5%.
The CPU snapshot contributes roughly another two milliseconds beyond the
driver setting. Recording and task-check regions account for most of that
additional reduction.

Two second-round baseline/combined comparisons are excluded from this speed
table because their clocks change or differ between configurations. Their
recorded outputs still pass equivalence checks. They are retained in the raw
results; excluding them avoids attributing boost-frequency changes to this
optimization.

The combined prototype still exceeds the 66.7 ms command budget in several
case means, and every case has a p95 above that budget. This improves latency
without establishing reliable 15 Hz collection. XR performance is untested.

VRAM remains approximately 4.7 GiB. The gain does not depend on allocating
more of the available 24 GiB. The driver setting and CPU snapshot address
runtime coordination and bookkeeping in this workload.

The next implementation should expose the driver connection setting before
simulator launch, record its provenance and enforce resume compatibility.
The CPU snapshot should become an explicit collection capability with reset
and fallback handling. Both should be scoped to the qualified workload.

## Larger CUDA launch queues

The capacity follow-up brackets 2× and 4× launch queues with fresh baseline
processes. These settings keep the default eight connections. All timings
and sampled clock ranges are shown below; different clocks are not treated
as a capacity benefit.

| Process | Repeat | Command work, ms | Physics, ms | Sampled GPU clock, MHz |
| --- | ---: | ---: | ---: | ---: |
| queue_capacity / base | 0 | 71.95 | 62.91 | 1695–1695 |
| queue_capacity / base | 1 | 71.96 | 63.22 | 1695–1695 |
| queue_capacity / queues2 | 0 | 71.32 | 62.73 | 1695–1695 |
| queue_capacity / queues2 | 1 | 71.66 | 63.07 | 1695–1695 |
| queue_capacity / queues4 | 0 | 71.24 | 62.59 | 1695–1695 |
| queue_capacity / queues4 | 1 | 71.62 | 62.95 | 1695–1695 |
| queue_capacity_end / base | 0 | 71.63 | 62.84 | 1695–1695 |
| queue_capacity_end / base | 1 | 71.83 | 63.05 | 1695–1695 |

queues2: mean saving 0.35 ms across 2 clock-matched repeats; queues4: mean saving 0.41 ms across 2 clock-matched repeats.

This is a small signal from one fresh process per capacity setting, with
two replays inside each process. It is much smaller than the confirmed
connection-count and CPU-snapshot improvements. It does not yet establish
a useful additional capacity change for the combined prototype.

## Completed checks

Completed **70 simulator processes and 110 diagnostic replays**.
All 44 paired configuration comparisons pass exact-output verification,
covering 68 matched replay comparisons and 4152 checks.
Some baseline replays are reused for more than one configuration comparison.
In the two-round confirmation, each setting has the same 8/10 task outcomes;
the two pail-02 failures occur in every setting. Equivalence checks include
these failures as well as the successful replays.
All processes exit normally. Source recordings, the production benchmark,
robot asset and captured source-code snapshot are unchanged.

<!-- END_CONFIRMATION_RESULTS -->

## Behavior and source integrity

Verification compares matched repeats without filtering by task outcome. It
requires exact equality of actions, robot joint positions and velocities, object
poses, robot body positions, task events, success flags and termination command
counts. It also compares recorded HDF5 initial states, observations, state channels,
subtask results, simulation times and physics/stepping provenance. Actual host
timestamps and source-age timing fields are excluded.

The physical configuration and live mass, inertia and drive summaries are compared
for each pair. Source recording hashes, the production benchmark, robot USD and
the source-code snapshot are checked separately. Saved helper hashes identify
the diagnostic implementation used by each later process.

An identical failure remains an identical failure: initial-state replay omits
hidden contact/solver caches, and pail 02's first repeat already fails in the
baseline. Some bowl inputs also drop before commanded opening while still
reaching task success. This runtime work does not change those physical behaviors.

## Artifacts

- [Launch manifest, settings and hashes](../output/omy_diagnostics/20261027_collection_runtime_latency/manifest.json)
- [All command timings, clock samples and region summaries](../output/omy_diagnostics/20261027_collection_runtime_latency/analysis.json)
- [Fresh-process comparisons with clock qualification](../output/omy_diagnostics/20261027_collection_runtime_latency/comparison.json)
- [Exact-output and source-integrity verification](../output/omy_diagnostics/20261027_collection_runtime_latency/verification.json)
- [Shared CPU snapshot prototype](../output/omy_diagnostics/20261027_collection_runtime_latency/host_placement_fused.py)
- [Original event logic adapted to the CPU world](../output/omy_diagnostics/20261027_collection_runtime_latency/host_events.py)

The artifact directory also retains launch/analysis scripts, per-process logs,
configurations, diagnostic HDF5 files, state arrays and external NVML samples.
No videos were generated.
