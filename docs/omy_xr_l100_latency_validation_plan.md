# XR + L100 latency validation plan

Status: executing. The user confirmed hardware availability and authorized
continuing the experiments. See the [live experiment log](omy_live_xr_latency_experiments_20261027.md)
for completed measurements, the recorded release failure, and pending comparisons.
The repeated full-combination bowl screen and first actual serial L100 A/B are
complete. Reverse-order manual release/placement, other object layouts and
sustained qualification remain pending; production defaults are unchanged.

Objective: determine whether the headless latency improvements also improve
live collection response and headset frame delivery. The 4.25–4.66 ms headless
saving is a candidate result, not an XR acceptance result.

## Fixed conditions and references

Use the actual collection application with recording enabled. Capture the exact
launch command, effective configuration and enabled/fallback capabilities for
every run. Use fresh diagnostic sessions rather than resuming existing data.

Primary baseline is the current collection configuration: CUDA physics on GPU 0,
rendering on GPU 0, two PhysX CPU workers, default CUDA connections, existing GPU
task/event checks. Pin the CUDA connection baseline explicitly to eight and clear
the copy-connection and launch-queue overrides. Preserve TGS, position cap 32,
two PhysX GPU solver partitions, 120 Hz physics, 15 Hz commands, original robot
geometry/mimics, held-target stepping and compact recording.

Keep the actual XR render cadence/pacing, headset refresh, resolution, encoder,
bitrate, connection method, scene, recording mode and statistics interval fixed.
Record their values rather than assuming the requested rates are achieved.
Use the same CPU affinity and Kit/TBB pools. Record GPU UUIDs, clocks, temperature,
utilization and VRAM externally. Keep unrelated GPU workloads consistent;
postprocessing must not overlap timed collection runs.

Retrieve the last known comfortable live configuration if it differs from the
current baseline. Preserve it as a separately identified reference. If needed to
reproduce the reported CPU-versus-GPU reversal, add a CPU-physics/GPU-rendering
control with matched physical settings. Record stepping fallbacks: CPU physics
does not qualify for the current CUDA-only optimized stepping path. That control
compares complete supported configurations, not the physics backend in isolation.
Recheck grasp/release behavior before treating it as an acceptable alternative.

## Candidate matrix

PhysX worker count, CUDA connection count and CPU snapshot evaluation are three
independent factors. Test the complete matrix if the initial screen warrants it.
CUDA connection count is distinct from PhysX solver partition count.

| ID | PhysX CPU workers | CUDA connections | Task/event evaluation | Purpose |
| --- | ---: | ---: | --- | --- |
| A | 2 | 8 | Existing GPU path | Current baseline |
| W | 0 | 8 | Existing GPU path | Worker change alone |
| C | 2 | 1 | Existing GPU path | Connection change alone |
| S | 2 | 8 | Shared CPU snapshot | Snapshot change alone |
| WC | 0 | 1 | Existing GPU path | Worker/connection interaction |
| WS | 0 | 8 | Shared CPU snapshot | Worker/snapshot interaction |
| CS | 2 | 1 | Shared CPU snapshot | Candidate retaining two workers |
| WCS | 0 | 1 | Shared CPU snapshot | Headless winner |

Do not spend initial hardware time on larger buffers, launch-queue capacities or
CPU affinity: their headless gains were absent or much smaller. Do not change
solver iterations or robot models during this study.

## Measurements and preparation

Before timed runs, prepare opt-in diagnostic launch controls for zero workers and
the CPU snapshot. The current live launcher fixes two workers, and the snapshot
is an output-only prototype. Add explicit configuration/provenance and verify
that its predicate timers, resets and placement diagnostics remain correct.
Record effective CUDA environment settings before simulator initialization.
These are prerequisites; the eight configurations are not all launchable through
the current unmodified collection script.

Save full-run data, not only the rolling performance summary:

1. **Command responsiveness:** command interval and admitted command duration,
   p50/p95/p99, longest stall and fraction exceeding the 66.7 ms command budget.
   Include pacing in elapsed-time results; retain physics, render, input,
   task-check, recording and pacing regions separately. Existing `work` timing
   excludes pacing and is insufficient by itself.
2. **Leader freshness and delay:** source age at consumption, invalid/stale samples,
   and consumption-to-command-completion time. Estimate buffered-sample-to-step-
   return delay as `source_age_s + (step_end_monotonic_ns - read_monotonic_ns)/1e9`.
   The existing `read_monotonic_ns` is consumption time, not encoder acquisition
   time. This estimate is neither first-visible-response nor motion-to-photon
   latency; serial acquisition time and display delivery are not fully captured.
3. **XR delivery:** application render-start intervals and longest gaps, plus
   SteamVR/ALVR timing, dropped/reprojected frames and available encode/network/
   decode statistics. Record each metric's clock, meaning and availability.
   Simulator render-call counts are not headset FPS or presentation timestamps.
4. **Behavior:** arm tracking error, input mapping, grasp retention, opening/
   release, placement success, F9 reset and READY handling. Save task events and
   complete attempted episodes, including failures.
5. **Hardware:** GPU clock/temperature/utilization/memory and active workload
   identities. Retain all runs; label clock/load differences rather than silently
   discarding unfavorable samples.

Keep telemetry overhead the same for every configuration. Use profiling only in
separate diagnostic runs if a regression needs explanation. If a camera capable
of capturing both the physical leader movement and the actual headset view is
available, add a physical motion-to-photon check. Otherwise leave that measurement
explicitly unverified; a desktop mirror is not a substitute for headset delivery.

## Staged execution once hardware is available

### 1. Establish a stable live baseline

Confirm calibration, READY, stereo viewing and the actual GPU assignments. Warm
the application and headset, then run A twice in separate processes to measure
ordinary variation. Include stationary leader/head movement, leader movement with
a steady head, and simultaneous movement. Save the headset/network configuration.
Resolve an unstable baseline before ranking small improvements.

### 2. Screen individual changes with XR active

Use a fixed recorded bowl input through a diagnostic source feeding the actual
collection loop, with XR, recording and normal pacing active. This source adapter
must preserve the collection lifecycle and render scheduling; the previous
headless benchmark is not a replacement. Timestamp replayed commands as synthetic
and exclude them from physical L100 latency claims.

Compare A, W, C and S in fresh processes, twice each, with the second round in
reverse configuration order. After approximately 20 seconds of warmup, accumulate
60 seconds of active commands per process. Repeat episodes as needed; identify
reset/READY periods separately and keep the same segment/reset protocol in every
configuration. Keep headset movement comparable and record the prescribed pattern.

This stage isolates interactions with XR rendering while keeping command inputs
repeatable. It does not validate the serial leader path.

### 3. Test combinations and confirm on contact-heavy scenes

If the individual candidates remain promising, test WC, WS, CS and WCS using the
same two-round protocol and interleaved baseline references. A small individual
loss may justify one interaction check; stop pursuing a configuration with a
clear, repeatable live regression.

Compare A and the best surviving configuration on both pail layouts and one small
object (carton or box), again with fresh processes and reversed order. Compare
equivalent action/contact phases. Keep goal-failure and timing results separately;
never retain only successful runs. Investigate any new physical or event mismatch.

### 4. Validate the complete XR + physical L100 path

Use actual serial L100 control, the user's normal headset movement and the real
collection workflow. Compare A and the finalist in counterbalanced order, with at
least two attempts per configuration on bowl, pail and the selected small object.
Include free-space tracking, grasp/lift/hold, release and F9/READY reset.

Record the user's perceived lag, stutter and control quality after each block,
alongside the timing metrics. Human trajectories will differ; do not require
bitwise equality or infer latency solely from task duration/success. A visible
regression warrants returning to A immediately and retaining its diagnostics.

### 5. Repeat a winner under normal sustained use

Run A and the finalist for comparable 5–10 minute blocks, including repeated
episodes and resets, to check thermal/clock drift, queue buildup and occasional
stalls. Keep the same background workload and XR connection. This stage is needed
before selecting new live defaults.

## Acceptance and decision

Use the repeated A/A baseline to set noise tolerances before inspecting candidate
results. A candidate must improve live response reproducibly beyond that variation,
without a reproducible worsening of command p95/p99, headset frame gaps, dropped/
reprojected frames, input freshness or grasp/release/reset behavior. Report mean
and tail results, raw and clock-comparable comparisons, and repeat counts.

An improvement in physics time that is absorbed by pacing, offset by rendering,
or accompanied by worse displayed response does not qualify as a live latency
improvement. Mixed or small effects are inconclusive and need a targeted repeat,
not automatic promotion of the headless winner. If nothing qualifies, retain A.

Allow roughly 15–20 minutes for setup and the first screen, and approximately
60–90 minutes for a full session with combinations, manual attempts and sustained
checks. Stop after the early screen if it already shows a clear regression.
Hardware time is staged; the full matrix is not a prerequisite to ending a
failed candidate's evaluation.

Deliverables: run manifest, complete timing/telemetry, paired comparison tables,
behavior findings, and a recommendation distinguishing measured improvements,
regressions and unresolved measurements. Production changes follow the live
acceptance result rather than the headless ranking.

Related: [headless candidate results](omy_collection_runtime_latency_experiments_20261027.md).
