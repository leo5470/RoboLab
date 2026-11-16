# PPS evaluation performance investigation — 2026-11-06

The supplied single-environment, horizon-eight command has further speedup
opportunities. The strongest measured runtime candidate is disabling Kit's
wall-clock render rate limiter. The strongest startup candidate is resolving
task names without importing every task in `single_pnp_v2`.

Production evaluation code and saved results were not changed. Diagnostic
scripts, profiles, replay recordings and measurements are in
[`output/pps_eval_performance_20261106`](../output/pps_eval_performance_20261106/).

## Existing evaluation measurements

The saved [fork/pitcher worker](../output/pps_v2_eval_20261106_fork_pitcher/worker.sh)
uses the supplied configuration, including sensor video and horizon eight.
Its 200 steered episodes (gamma 0.4/0.8) contain 145,995 control steps:

| Region | Mean ms/control step | Share of measured loop |
| --- | ---: | ---: |
| Environment step | 167.74 | 81.23% |
| Policy inference, including client work | 34.27 | 16.60% |
| Sensor video writing | 4.49 | 2.18% |
| Total | 206.50 | 100% |

The 100 base episodes average 198.69 ms/step, with 86.30% in the environment.
These are step-weighted aggregates, not averages of rounded per-episode rates.
`TimingStats.wall_total_s` sums named regions; it excludes startup, environment
construction, resets, final video closure and result summarization. It is not
end-to-end process wall time.

## Local experiment

Replayed the first 128 saved actions of `SimplePnpV2303Pitcher00Task`, gamma 0.8,
on GPU 0 (RTX 4090), with one environment, eight-action chunks, the same two
initial resets, 120 Hz physics, decimation eight and sensor video. The diagnostic
timeout was shortened to 128 steps. No policy server was contacted. The initial
cProfile run is separate from the unprofiled comparisons.

The profile attributes about 68 ms/render to Kit rendering/app updates,
54 ms/control step to eight physics calls, and 22 ms/control step to scene and
sensor updates. The two observation passes together cost about 6.6 ms/step.
These are inclusive host timings, including waits; they are not GPU kernel
measurements and nested rows must not be added together.

**Disable wall-clock render limiting first.** The live setting was
`/app/runLoops/main/rateLimitEnabled=True`, with frequency 15 Hz. Setting it to
false reduced the measured 128-step replay loop from 21.614 s to 15.044 s
(30.4% less time, 1.44× throughput). Environment time fell from 162.0 to
110.8 ms/step. All 42 HDF5 numeric datasets matched exactly, including actions,
robot/object trajectories and subtask records. Requests remained at steps
0, 8, ..., 120. Physics dt, decimation and simulated render cadence were retained.

This is a replay improvement, not a measured closed-loop PPS improvement. The
roughly 51 ms/step saving would imply about 25% less measured loop time against
the saved steered runs if inference latency and other costs stayed unchanged.
Rendered policy images and full manipulation success equivalence were not
validated by this short replay.

Append this to the supplied pinned-launcher command:

```bash
--kit_args "--/app/runLoops/main/rateLimitEnabled=false"
```

The exact CLI path was tested in a fresh process: the effective setting was
false and the 128-step loop took 14.843 s (109.1 ms/step inside the environment).
All 42 numeric datasets again matched the normal baseline exactly. The wrapper
prepends its GPU-pinning settings to the supplied Kit arguments, so GPU pinning
is retained. This flag disables wall-clock pacing, not sensor rendering.

The distinction is between real time and simulation time: Kit may wait to
pace its main/render loop in real time, while physics advances by fixed
simulation timesteps. Disabling the limiter removes those pacing waits and
lets updates finish as fast as the hardware permits. Here, physics still runs
at 120 Hz in simulation time, with eight physics steps per action and one
render per control step. Camera observations, sensor video, and the eight-action
policy request horizon remain enabled and unchanged. See
[Faster Offline Evaluation](environment_run.md#faster-offline-evaluation) for
the usage guidance.

**Disable unused contact history for eligible evaluation tasks.** The default
contact sensors have `history_length=6`. Isaac Lab consequently updates them
on every physics substep despite `lazy_sensor_update=True`. The profile recorded
11,264 contact-buffer updates over 128 control steps: 11 sensors × 8 substeps.
The current RoboLab predicates use current forces, not these histories.

Setting contact history to zero in the diagnostic saved about another
15 ms/control step. Both changes together reduced loop time by 38.9%
(1.64× replay throughput). A separate ordinary-baseline repeat returned to
21.640 s, supporting that the large savings are configuration-dependent.

| Configuration | 128-step measured loop | Environment ms/step |
| --- | ---: | ---: |
| Baseline | 21.614 s | 162.0 |
| Disable Kit rate limiter | 15.044 s | 110.8 |
| Contact history zero | 19.734 s | 147.2 |
| Both | 13.198 s | 96.1 |
| Baseline repeat | 21.640 s | 162.0 |

All five cases had exactly equal values in all 42 numeric HDF5 datasets and
the same request steps. The timeout was intentionally 128 steps, so these are
not successful completed manipulation trials. Contact-history removal needs
guards for history/air-time consumers and longer success/failure/contact
coverage before becoming an evaluation default. Existing teleoperation code
already applies this optimization to eligible sensors, but PPS evaluation
does not. The combined savings would project to roughly 32% less measured
steered-loop time if they carry over; this is not a live PPS measurement.

**Remove the eager task-name scan.**
`EnvFactory.batch_create_env_cfgs()` builds a name map by calling
`get_task_class_name_from_file()` on every Python file under the selected task
directories. That helper actually imports the module. Each generated task
imports its USD scene at class definition time. Thus requesting even one task
imports all 6,550 task modules in `single_pnp_v2`.

The diagnostic's first physics-context log appeared about 163 s after Kit start
with normal class-name registration, versus about 10 s when the diagnostic
registered just the selected task file with `task_dirs=None`. These timestamps
include imports/registration and are not an isolated registration timer. A
read-only AST index of all 6,550 task classes took 0.84 s and matched every
manifest class/file pair. Production should use a validated manifest or AST
index with fallback for dynamic task definitions, then import only requested
tasks. This is a per-process saving, amortized over the user's 50-task batches.
Passing file paths alone to the existing runner does not fix its eager scan
while `task_dirs` remains set.

## Smaller candidates and protocol constraints

The first unprofiled comparison produced these measured loop times:

| Configuration | 128-step loop |
| --- | ---: |
| Baseline | 21.859 s |
| Remove unused viewport camera | 21.637 s |
| Skip duplicate pre-recorder observations | 21.703 s |
| `--rendering-type performance` | 21.800 s |
| Baseline repeat | 21.799 s |

These are small changes from single sequential trials, not robust large
speedups. All shared numeric HDF5 datasets matched exactly; viewport removal
omitted its two initial camera-pose datasets. Policy image equivalence was not
checked. The performance rendering preset can change policy inputs.

The redundant observation pass exists in Isaac Lab's `ManagerBasedRLEnv.step()`:
it computes observations before `record_post_step()` whenever any recorder is
active, then computes them again at the end. The current default post-step
recorders read simulator state directly. A production optimization needs an
explicit recorder compatibility guard, preserving final observations/history
updates, resets and interval events.

Current code already skips cached-step Pi0/PPS observation extraction, unused
headless visualization and legacy per-step subtask deep copies. The older
[PnP replay report](../output/pnp_perf_validation_CbaEDW/REPORT.md) measured those
changes; they are not additional savings available now.

Turning video off only removes about 2.2% of the measured production loop; it
does not remove policy camera rendering. Increasing horizon eight, changing
physics dt/decimation, or disabling scoring would change the evaluation
protocol. `PpsClient.infer_batch()` still uses serial requests: increasing
`num_envs` is not a batched PPS inference solution and changes episode/env seed
identities. Two initial resets also affect initial-state sampling and should
not be removed as a superficial cleanup.

No full closed-loop PPS A/B run or long-duration memory test was performed.

## Reproduction

The archived scripts are diagnostic only and hard-code the source task/actions
above. They create new output directories and refuse to overwrite existing ones.
For example, from the repository root:

```bash
.venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 \
  output/pps_eval_performance_20261106/compare_pps_eval_extra.py \
  --headless --output /tmp/pps_eval_reproduction \
  --modes baseline no_rate_limit no_history combined baseline_repeat
```

The `baseline` mode uses normal simulation configuration; explicit task-file
registration only avoids the expensive all-task import scan. `no_rate_limit`
sets the Kit boolean after environment creation. `no_history` changes sensor
history length before creation. `combined` applies both. The replay client
retains image extraction/request packing but replaces remote inference with
saved actions. The profiling script uses ordinary task-name registration and
writes `profile.pstats` and `profile.txt`.
