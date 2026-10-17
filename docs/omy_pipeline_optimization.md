# OMY + XR collection optimization plan

**Current status (2026-10-05):** The initial opt-in fast collection path has since
been implemented and generalized by consumer dependencies/capabilities. The
active checkout is `/tmp2/leocheng/RoboLab`; F9 replaces Space for episode control.
See [the current collection guide and validation](omy_xr_collection.md#generalization-validation-2026-10-05)
for launch options, per-task sensor selection and automated results. The dated
investigation below is retained as historical design/evidence, not a statement
that those initial options are still unimplemented. Full manipulation/XR comfort
and training-transfer acceptance gates remain open.

Investigation: 2026-10-01–02; reviewed against the 2026-10-02 solver-transfer and live π0.5 report. Scope: the active `/home/leocheng/RoboLab` checkout, branch `quest`, OMY-L100 joint-position control, native SteamVR/ALVR XR, and state-only demonstration collection. This is a design proposal supported by benchmark prototypes and limited regression tests; the production changes below remain unimplemented. Existing calibration, leader mapping, Space/READY behavior, other collectors, and standard DROID task configurations remain unchanged.

## Recommendation

Build an OMY-specific fast collection path with native 15 Hz commands, fixed 120 Hz CUDA physics, an **opt-in 16-position-iteration scene cap supported by limited regression tests**, held-target submission once per command, boundary-rate scene bookkeeping, and compact state-only recording. Retain the 32-iteration reference path while validating the candidate. Make XR rendering independently configurable at nominal 60 or 30 Hz, with 30 Hz as the first candidate for achieving realtime collection on this host. Keep the desktop viewport enabled: disabling it stalled the XR test.

The experiments reached approximately 15 wall-clock commands/s during light motion and approximately 12–14 commands/s during a recorded grasp/placement replay. Therefore this is a demonstrated direction, not a claim that the complete live pipeline is already realtime. The remaining objective is to recover several milliseconds of steady-state headroom and reduce render stalls, then validate actual operator input with the headset worn.

The latest solver-transfer results show measurable trajectory differences between caps 16 and 32. Preserve each demonstration's actual reached states and physics profile, generate offline images from those recorded states, and use fresh observations in the reference 32-iteration environment for live π0.5 evaluation. The existing-checkpoint tests support this evaluation workflow; transfer from models trained on the proposed collection data remains a later experiment.

Do not reduce the physics rate, change gravity, remove the passive gripper linkage, or increase apparent throughput by relabeling timestamps. Twelve and eight solver iterations did not satisfy the grasp regression used here and should not be the initial collection defaults.

## Evidence and limitations

The investigation scripts, raw profiles, logs, HDF5 recordings, and validation reports are preserved in [`output/omy_pipeline_benchmark/20261001_optimization`](../output/omy_pipeline_benchmark/20261001_optimization/README.md). Useful entry points are [`summary.json`](../output/omy_pipeline_benchmark/20261001_optimization/summary.json), [`export_validation.json`](../output/omy_pipeline_benchmark/20261001_optimization/export_validation.json), and [`grasp_validation.json`](../output/omy_pipeline_benchmark/20261001_optimization/grasp_validation.json).

The follow-up [`solver-transfer and live π0.5 report`](../output/omy_pipeline_benchmark/20261002_solver_transfer/README.md) preserves 63 replay/stress trials, six full live-policy episodes, one short inference-delay probe, and 66 pure OMY unit checks. Its [`summary.json`](../output/omy_pipeline_benchmark/20261002_solver_transfer/summary.json) and [`live_policy_summary.json`](../output/omy_pipeline_benchmark/20261002_solver_transfer/live_policy_summary.json) document trajectory differences and inference behavior. These headless tests used ordinary manager stepping with XR disabled; they extend solver/evaluation evidence rather than validate the combined fast collection path.

Tests used the active Isaac Sim 5.0 / Isaac Lab 2.2 environment, GPU 0 (RTX 3090), CUDA physics, performance rendering, eight Kit workers, and two physics workers. They used generated joint commands or an existing recorded trajectory; **the leader serial port was never opened**. No persistent SteamVR/ALVR, GPU, CPU-power, or system settings were intentionally changed.

The most recent human-operated 15 Hz run, `output/omy_teleop/20261001_222734_101521`, achieved 5.27 wall-clock commands/s, 21.08 render calls/s, and a realtime factor of 0.351. Mean work per command was 189.3 ms, including 87.8 ms physics, 52.1 ms rendering, 15.9 ms scene writes, and 11.9 ms scene updates. Leader input was only 0.36 ms; active recording was not enabled in that run. Its logs do not establish HDF5 or HDD writing as the live bottleneck.

The controlled results below are separate experiments, not a matched before/after comparison with that human-operated run. They differ in motion, contact load, instrumentation, headset activity, and runtime conditions.

### Controlled throughput results

Times are per native 15 Hz command, whose realtime budget is 66.67 ms. Wall rates and realtime factor include pacing when enabled; work times exclude pacing sleep.

| Test | Mean work | Wall commands/s | Realtime factor | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Physics only, scene cap 32, ordinary stepping | 85.0 ms | 11.75 | 0.785 | Physics alone cannot sustain realtime in this test |
| Physics only, cap 32, submit/update once | 76.6 ms | 13.05 | 0.870 | Python/scene cleanup helps, but the solver still exceeds budget |
| Physics only, cap 16, ordinary stepping | 55.5–56.0 ms | 17.8–18.0 | 1.19–1.20 | Reducing actual solver work is the largest measured gain |
| Physics only, cap 16, submit/update once | 46.2 ms | 21.64 | 1.442 | Creates room for XR and recording |
| XR, cap 16, full state recording, ordinary stepping, nominal 60 Hz render | 89.0–89.2 ms | 11.2 | 0.747–0.749 | Recording enabled and episodes really exported |
| Same, submit/update once | 79.0 ms | 12.65 | 0.843 | Repeated writes and scene updates cost about 10 ms here |
| Same, nominal 30 Hz render | 66.2–66.4 ms | 15.06–15.11 | 1.004–1.007 | Light-motion tests reach the mean target with little margin |
| XR, cap 16, compact recording, submit/update once, nominal 30 Hz render, realtime pacing | 64.2–64.5 ms | 15.00 | approximately 1.000 | Two repeated light-motion tests; p95 work 67.6–68.1 ms |
| Same compact/paced configuration, saved grasp/placement sequence | 68.1–81.0 ms | 12.05–14.15 | 0.803–0.944 | Contact load and render stalls still exceed the target; p95 work 82–84 ms |

The contact test spends approximately 44.5–44.6 ms in physics, 4.0–4.3 ms in recording, 1.7–1.8 ms in task checks, and 14.8 ms rendering in its better run. A repeat had 27.3 ms average rendering time. The candidate is close to the steady-state budget, but is not robust against workload spikes yet.

### Actual GPU physics work

CUDA profiling covered 80 physics steps under each solver configuration:

| Scene cap | GPU kernel time across those steps | Kernel launches | Tendon/mimic kernel launches |
| --- | ---: | ---: | ---: |
| 32 | 605.4 ms | 61,030 | 2,640 |
| 16 | 333.1 ms | 39,270 | 1,360 |

At cap 32, the tendon/mimic constraint kernel accounted for 28.7% of GPU kernel time. Unified constraint solving accounted for 20.8%, internal articulation constraints 18.0%, articulation stepping 8.3%, and static constraint solving 6.4%. These five kernels accounted for about 83% of the measured kernel time. The CPU profile also spent most physics time waiting in `PhysX.fetch_results`.

This is genuine solver work and synchronization, not evidence that Torch arithmetic or disk throughput dominates. A single articulated robot can run long chains of small dependent kernels without saturating the GPU's aggregate utilization. Adding CPU workers or another GPU does not remove these dependencies.

The robot requests 64 position iterations, while RoboLab's scene currently caps them at 32. Isaac Lab documents that actor iteration requests are clamped to the scene's minimum/maximum settings. PhysX also documents that an island's highest actor request determines its iteration budget, so changing only an individual object's request is not necessarily enough. Sources: [Isaac Lab 2.2 PhysxCfg](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg), [PhysX island management](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Simulation.html#island-management).

### Initial grasp and data regression results

The saved 60 Hz trajectory was first replayed at its original rate. A separate stress test selected every fourth command and executed it as a native 15 Hz command held for eight physics steps. This constructed sequence is a regression input, not a recommendation for exporting or downsampling training demonstrations.

- At sixteen iterations, native 15 Hz ordinary stepping and the submit/update-once candidate reproduced the lift with about 4.4 mm maximum banana-path difference. A repeated ordinary run varied by about 4.6 mm. Peak banana height was approximately 18.5 cm in those runs. This compares stepping methods at the same solver cap; it does not establish equivalence between caps 16 and 32.
- With sixteen iterations, compact recording and native XR, all three 84-command replay episodes received the environment's success flag. Grasp trajectories matched the corresponding headless replay results. This is repeated replay of one trajectory, not three independent human attempts or comprehensive stability validation.
- The twelve-iteration compact-recording test failed all three replay episodes and reached only approximately 5.8–8.4 cm peak banana height. Do not promote that configuration.
- At eight iterations, banana-path variation was approximately 16 cm, and one run reached 1.30 m/s while the close command remained active. The finger also reached its full closed angle where the sixteen-iteration runs remained blocked by the held object. That is not sufficient grasp consistency for a default, despite the faster physics.
- Actual exported episodes retained eight-dimensional commands and full thirteen-joint position/velocity state. All numeric values were finite. Measured pre-action arm positions matched the previous post-action state exactly, and the first pre-action arm position matched the recorded initial state exactly.
- Full short-episode exports took about 15–17 ms; compact exports about 10–11 ms. Live full recorder work was about 6 ms/command and compact recorder work about 4–5 ms/command. These are short-episode measurements, not long-run durability or sustained archival benchmarks.
- Eighty-four selected existing pure unit tests passed. New production stepping, recorder, and writer changes still require their own tests.

### Solver transfer, repeat variability, and settling

The follow-up suite used the same 84-command BananaInBowl regression, three synthetic hold variants, and a bundled 648-command native-15-Hz cube-and-banana policy demonstration. The 84-command input still derives from every fourth command of one human 60 Hz trace. The hold variants and repeated replays add stress coverage, but no independently human-operated native-15-Hz attempts.

Across nine cap-16 versus cap-32 pairings of the original regression, maximum banana-position differences were **17.9–44.9 mm** over the common command prefix. In the original-duration repeat groups, within-cap runs varied by up to 10.7 mm at cap 16 and 28.8 mm at cap 32. The earlier 4.4 mm stepping comparison is therefore not a tolerance for solver transfer. The bundled open-loop demonstration obtained 0/3, 0/3, and 1/3 success flags within 648 commands at caps 16, 24, and 32; repeated cap-32 runs differed by up to 262.7 mm despite identical restored initial state arrays. Reference repeat variability prevents attributing every discrepancy to the solver cap.

With the reported settling allowance, the original regression succeeded in all three repeats at each cap, but cap 32 needed 86, 86, and 87 commands, including 2–3 commands holding the final source target. The suite allowed up to two additional simulated seconds for caps 24/32 while cap 16 used the original test duration. Report these horizons explicitly; a nonterminal source endpoint alone does not establish placement failure. Future comparisons must use the same settling allowance at every cap and distinguish success within the source sequence from success during settling.

The two-second grasp hold succeeded in all repeats at every cap. The five-second peak-lift hold succeeded in 3/3, 2/3, and 2/3 repeats at caps 16, 24, and 32 under the respective horizons above. The five-second pre-release hold succeeded in only 1/3, 0/3, and 2/3 repeats within its original duration; it also involves placement/contact. These results leave grasp/release robustness open. Finite boundary states and connected joint anchors do not measure contact penetration, friction forces, or peaks inside physics substeps.

### Live π0.5 evaluation evidence

The existing `pi05_droid_jointpos` checkpoint used fresh external/wrist observations, normal `env.step`, and controlled per-query RNG seeds:

| Position-iteration cap | BananaInBowl successes, seeds 0/1/2 | Mean command work | Wall commands/s | Realtime factor |
| --- | --- | ---: | ---: | ---: |
| 32 | 3/3 | 167.1 ms | 5.86 | 0.391 |
| 16 | 3/3 | 133.2 ms | 7.31 | 0.487 |

Cap 16 reduced mean command work by approximately **20.3%**. These timings include camera rendering, model calls, and diagnostic readbacks; keep them separate from XR collection measurements. Six successful episodes are a smoke sample for an existing checkpoint, not evidence of training transfer from the proposed faster collection data or established success rates.

All 65 full-episode inference requests and six delayed requests advanced zero simulator substeps and showed zero measured joint/banana-transform change in direct PhysX tensor reads. Live inputs were nonblack `224×224×3` images, model chunks were `15×8`, and executed actions were finite. The delay probe added two seconds to each request and stopped after 90 commands; count it only as a pause/inference check. This supports evaluation below wall-clock realtime in the tested synchronous path while retaining 15 simulated commands/s, but does not establish identical policy trajectories across separately rendered runs.

### What the remaining recording cost actually is

A CPU call profile of 120 compact-recording commands spent 0.402 seconds in `SubtaskCompletionRecorderTerm.record_post_step` (approximately **3.35 ms/command**), including 0.317 seconds in `EventTracker.check_events` (approximately **2.64 ms/command**). In contrast, the full scene-state recorder's `get_state` cost was 0.060 seconds (approximately **0.50 ms/command**). Termination computation took about **1.72 ms/command**. These are instrumented cumulative call times; the nested costs must not be added twice.

Thus the remaining “recording” hotspot is primarily task/event bookkeeping, not copying the raw joint/object arrays or HDF5 writing. This makes a coherent shared predicate evaluation—or moving nonessential detailed event tracking offline—a more important next optimization than rewriting the small state buffer. The profile is preserved under `physics_record16_compact_profile/cpu_profile.txt`.

## Proposed code changes, in implementation order

### 1. Make the effective configuration explicit and measurable

Files: `robolab/core/omy_teleop/options.py`, `config.py`, `performance.py`, `runner.py`, `collection.py`, and `xr.py`.

Add an OMY-only solver-budget option and apply it to the copied environment configuration **before** `create_env`. Start with an opt-in sixteen-iteration candidate; retain the existing thirty-two-iteration reference path. Log the articulation's authored request, scene min/max, unchanged velocity-iteration settings, physics device, timestep, decimation, actual rendering interval, and active recording/stepping profile. Persist them with every recording.

Do not change shared `DroidCfg`, the global base environment, upstream package files, or the SO-101/keyboard collectors to enable this profile. Editing the articulation USD attribute at runtime did not reduce solver work in the test; a flag that changes only the displayed value would be misleading.

Add bounded per-attempt timing summaries: p50/p95/p99 work and command intervals, missed budgets, longest render gap, episode-buffer bytes, and snapshot/transfer/write/close durations. Separate startup, shader compilation, READY, reset, and export from active-control statistics. Keep detailed physics-substep diagnostics opt-in because their readbacks and JSON output change the workload.

For input latency, capture the timestamp and age of the **same** serial sample used by `device.advance`, rather than measuring one sample in `ReadyGate` and advancing from another. Preserve the upstream mapping and limiter. Do not read the serial device from the simulation loop or queue old commands for later replay.

### 2. Introduce an audited OMY implicit-joint stepping path

Proposed new module: `robolab/core/omy_teleop/stepping.py`. Add a narrowly scoped environment/stepping-class injection to `robolab/core/environments/runtime.py` if required; keep `RobolabEnv` as the default for all existing callers.

For this path, process an absolute joint command once, submit it once, then hold it for all eight 120 Hz physics substeps. PhysX's implicit drives continue controlling the joints throughout those substeps. Keep all physics steps, rendering opportunities, episode counters, termination checks, recorder hooks, and reset behavior.

The eligibility checks must include:

- One environment with the audited seven absolute Franka joint targets plus binary gripper action.
- All actuators are plain supported implicit actuators; no explicit feedback actuator needs a new effort calculation each physics step.
- No external force/wrench application or interval event requires scene writes at substep frequency.
- No sensor history, air-time tracking, reward, controller, or diagnostic consumer requires substep scene updates, joint acceleration, or effort telemetry.
- The supported Isaac Lab version and manager lifecycle match the tested stepping contract.

If any condition is unmet, use the ordinary path and explain why. Do not ship the benchmark's temporary monkeypatches as the architecture.

Advance scene cache timestamps once at the control boundary with accumulated dt, using Isaac Lab's scene update API. Existing renders must still receive fresh PhysX/Fabric transforms at their scheduled physics step; verify this visually with head tracking and a moving gripper. Reset/teleport must invalidate state and observation caches even when a step counter has not advanced.

Before declaring the stepping option validated, require the following automated contract checks, with simulator integration tests where manager or PhysX behavior is involved:

- Exercise every eligibility rejection and verify ordinary stepping is selected with a recorded reason, without leaving a partially enabled fast path.
- Verify one action-processing call, one target submission, all eight physics substeps, scheduled render opportunities, correct accumulated scene dt, episode counters, and recorder-hook ordering for each admitted command.
- Exercise initial reset, F9 reset, early-termination reset, and direct state restoration/teleport without advancing the step counter. Verify refreshed scene, contact/predicate, and observation caches against the restored state, and isolate the next episode from the previous buffer.
- Exercise success, failure, and timeout. Verify that the terminal pre-action observation, action, and post-action state are retained before automatic export, that finalization occurs exactly once, and that no later command is admitted to the frozen episode. Check manual reset and interruption finalization as well.
- Compare ordinary and optimized stepping at the same solver cap on controlled task/event fixtures and replay inputs. Require identical predicate/state-machine outcomes for identical sampled inputs, no duplicate state-machine transitions, and matching recorder/outcome semantics. Assess physical replay differences against repeated ordinary runs; do not require bitwise equality of contact trajectories.

This optimization changes the cadence of approximate actuator effort telemetry and finite-difference acceleration. Neither is used by the current PPS arm-position/gripper observations. Keep full-substep semantics for consumers that need them, and do not silently export control-interval acceleration as a 120 Hz quantity. Isaac Lab confirms that implicit drives are solved by the physics engine while the actuator wrapper computes approximate effort for telemetry: [version-matched actuator implementation](https://isaac-sim.github.io/IsaacLab/v2.2.0/_modules/isaaclab/actuators/actuator_pd.html#ImplicitActuator.compute).

### 3. Make XR cadence independent of action cadence

Files: `robolab/core/omy_teleop/xr.py`, `performance.py`, `collection.py`, and `runner.py`.

Replace XR's unconditional `render_interval == 2` validation with explicit supported cadence choices. At 120 Hz physics, interval 2 means nominal 60 Hz application rendering and interval 4 means nominal 30 Hz. Keep the headset refresh rate and ALVR encoding configuration separate; a 72 Hz headset does not make the application render 72 new scene frames/s.

Retain interval 2 as the reference option and expose interval 4 as the first realtime-collection candidate. Continue rendering *within* the eight-physics-step command block rather than presenting a burst only after the block. Use explicit realtime pacing and disable competing implicit Kit waits. Recover deadlines after reset, XR pause, or a long stall without a catch-up burst or skipped physics steps.

Keep the Kit window and desktop viewport working so F9 reset remains usable. The attempted viewport-update disable stalled XR and is excluded. A smaller desktop render product may be investigated through the supported viewport API, with independent eye-resolution and keyboard/head-tracking validation; it is not an assumed safe substitute.

The application should prioritize fresh state and bounded command intervals over unnecessary duplicate renders, but any best-effort frame-drop policy needs a maximum presentation gap and an input-event pump. Do not add an unsynchronized second render process or claim that scheduling decoupling makes rendering free.

The existing XR resolution multiplier is a secondary option. A 0.7 test was contaminated by another simulator started by pytest and establishes no speed gain. Use an isolated pixel-count sweep with the headset worn before recommending another reduction. The HomeOffice room is an HDR dome background, not a large furniture-mesh hierarchy that can simply be stripped for a guaranteed gain.

### 4. Replace generic live recording with a compact raw contract

Proposed module: `robolab/core/omy_teleop/recording.py`; integrate through the dedicated collector and recorder manager. For predicate/event consolidation, inspect `robolab/core/events/subtask_recorder.py`, `robolab/core/task/event_tracker.py`, `predicate_logic.py`, `conditionals.py`, and `robolab/core/world/world_state.py`, keeping any new fast context opt-in for OMY. Adapt the materializer/exporter only where they depend on removed derived channels.

The tested compact profile already removes redundant robot-root, EE-pose, bounding-box channels, and EE observations while preserving initial state, actions, full post states, arm/gripper pre-observations, and subtask tracking. It saved approximately 3 ms/command across recorder and observation work. Use it as the starting point, not as the final minimum-overhead implementation.

Preserve this raw contract:

| Field | Timing and purpose |
| --- | --- |
| Initial scene state and resolved configuration | Before the first admitted action; full robot/object state, joint order, assets, units, calibration, scene seed, software/physics profile, and policy camera definitions |
| Measured arm/gripper observation | Immediately before action `a[t]`; not the command and not post-step state |
| Actual mapped/limited command `a[t]` | Seven absolute arm targets plus the binary gripper command actually sent; store encoding and open/closed meanings |
| Full scene state after `a[t]` | All thirteen robot joint positions/velocities, robot root pose/velocity, and every rigid object's root pose/velocity |
| Simulation and monotonic wall timestamps | Simulation time defines training cadence; wall time and exact source-sample age diagnose latency |
| Outcome and sparse events | Success/failure/reset reason and task progress; include compact contact/predicate data if needed for exact offline event reconstruction |

Keep the existing required `initial_state`, `actions`, `states`, and arm/gripper observation paths, or provide an explicit versioned compatibility adapter. Do not shrink the thirteen-joint robot state to the eight commanded DOFs: the passive gripper joints are required for faithful rendering and debugging.

Record the measured gripper field's joint identity, units, and any normalization separately from command encoding. The follow-up suite's diagnostic HDF5 files put raw finger radians in `obs/proprio_obs/gripper_pos`, while the tested production π0.5 observation uses normalized 0–1 closure. Replay actions use OMY ±1 encoding and live-policy actions use 0/1. Those files require an explicit compatibility adapter and validation before training export; their matching field names alone do not establish compatibility.

Preallocate buffers to the admitted attempt length, with an explicit byte budget. The current default twenty simulated seconds at 15 Hz is 300 actions, even though the independent step cap is 1,200. Use packed state storage or a small number of contiguous copies instead of many tiny per-leaf launches and reallocations. Snapshots must remain immutable after the next physics step or reset.

Avoid the upstream step's duplicate observation computation when the OMY compact post-step recorder does not need policy observations. Compute the next pre-action observation once after the state is refreshed; handle reset/history semantics explicitly. Derive EE poses, boxes, and policy images offline from recorded states and model metadata.

Prioritize the profiled subtask/event cost before undertaking buffer micro-optimizations. The existing code repeatedly materializes active/frozen masks with `nonzero().tolist()`, checks GPU scalars, and evaluates Python task state machines. Share one coherent control-boundary state/predicate evaluation between termination, event tracking, and recording, then perform one small batched host transfer for the required lifecycle decisions. Cache shared poses/contact/geometry inputs; distinct predicate options still need distinct results. Preserve success semantics and do not evaluate a state machine twice accidentally.

If detailed progress moves offline, preserve the live success/reset decision and any contact predicates required to reconstruct progress. Restoring poses for offline rendering does not reproduce the contact forces that existed during collection. Do not drop that information without deciding what both proxy-training datasets and quality filters need.

### 5. Isolate finalization and disk writing without overengineering the hot loop

Follow the bounded SSD staging/HDD archive design in [`postprocess.md`](postprocess.md#proposed-collection-io-isolation). Raw state files are small in these tests; disk I/O is not the first steady-state bottleneck. Boundary writes can still interrupt XR and need separate measurement.

Use one CPU-side writer if export gaps justify it. At a boundary, detach an immutable episode snapshot, transfer it to owned CPU memory, and hand it to the writer. If using asynchronous pinned transfers, keep source buffers alive until the copy event completes. A background HDF5 writer does not remove the GPU synchronization/copy cost.

Route **all** finalization paths through the same ownership model: success inside `RobolabEnv._reset_idx`, F9 reset, limits, stale leader/XR loss, interruption, and shutdown. Moving only the collector's manual-reset save would leave automatic success export blocking inside `env.step`.

Bound pending bytes and episode count, acknowledge durable saves, and apply backpressure before arming another attempt. Keep XR serviced in READY while waiting. Preserve recoverable failures; never silently discard a demonstration. Archive closed verified files to HDD after the collection session, and run policy-camera rendering/video encoding/PPS export offline.

### 6. Preserve the native 15 Hz PPS training contract

Keep leader sampling at approximately 100 wall-clock Hz and choose the freshest sample at each command boundary. Issue one command per 1/15 simulated second, hold it for eight fixed 1/120-second physics steps, and record one aligned raw sample for that command. When workload fits, pace the same sequence to approximately 15 wall-clock commands/s so gravity and arm speed appear normal.

For dataset row `t`, pair the pre-action state/image `x[t]` with `a[t]`. `x[0]` comes from the initial scene state; later pre-action states come from the preceding post-step state. Render external/wrist cameras offline by restoring recorded full scene states, not by replaying actions through physics.

Higher-quality offline rendering must retain the recorded geometry/state at each sample. Replaying cap-16 commands through cap-32 physics produces a new rollout with different reached states; preserve its own states, images, outcome, and provenance if used for validation or augmentation. Do not pair those new images with the original demonstration's state observations or overwrite the original episode.

Use the same validated physical/action/observation contract for reference and task proxy exports, with distinct versioned episode-selection manifests as needed. Convert OMY commands with `pps_gripper_command = (1 - omy_gripper_command) / 2` for the inspected 0=open/1=closed PPS action contract. Choose measured gripper units separately for the selected training/evaluation recipe; retain raw finger radians and normalized closure internally. The tested π0.5 client's normalized observation does not settle every PPS recipe's state units. Keep the sampling/alignment decisions in [`postprocess.md`](postprocess.md#alignment-and-units-to-settle) authoritative.

If the runtime cannot meet wall-clock realtime, record that honestly in the timing/QC manifest. Do not declare a 5 Hz human interaction to be realtime because the HDF5 simulation timestamps are spaced at 1/15 second. Downsampling faster-control recordings remains a separate, explicitly validated export mode, not the initial native-15 solution.

### 7. Retain a reference live π0.5 evaluation path

Use the reference 32-iteration environment, ordinary manager stepping, fresh pre-action external/wrist camera and proprioceptive observations, and the selected checkpoint's 15-action chunks. Each executed action still spans eight fixed 1/120-second physics substeps. Inference waits may lengthen wall time without advancing simulation in the tested synchronous path. Keep this pause behavior explicit when changing evaluation scheduling, and recheck it using direct physics state reads.

Configure the optional `openpi_client` dependency and select a complete verified checkpoint before a production launch. The active venv lacked that dependency, and the original checkpoint cache contained truncated blobs; the temporary driver used existing OpenPI client source and a separately verified checkpoint. The verified weights are archived at `/tmp2/leocheng/RoboLab/output/omy_pipeline_benchmark/20261002_solver_transfer/checkpoint/`. Capture checkpoint identity, client/server versions, environment profile, and model RNG seeds with evaluation results rather than relying on the temporary harness's host-specific imports.

After validating native-15 collection and exports, train the intended reference/task proxies and evaluate the resulting models with live observations at cap 32. Compare them using a shared evaluation protocol and recorded physics profiles. The existing-checkpoint smoke tests do not complete this training-transfer experiment.

## Validation and rollout gates

Implement one stage at a time, with a reference switch and recorded configuration. Before making the combined profile the default:

1. Run repeatable A/B/A light-motion and contact tests, one simulator at a time, after warmup. Keep CPU policy, GPU, SteamVR/ALVR settings, rendering resolution, and headset wearing state consistent. Include automatic success exports in latency reports.
2. Replay multiple saved grasps, releases, long holds, and table-approach trajectories. Use the same source-command horizon and up to two additional simulated seconds (30 native-15 commands) holding the final target at every tested cap, stopping on termination. Record success within the source sequence, success during settling, and nonterminal outcomes at the limit separately; retain settling commands as validation provenance rather than appending them to the source demonstration. Compare trajectories over their common source-command prefix against repeated runs at each cap. Require no new link separation, table penetration, object ejection, invalid states, or unexpected gripper opening, with targeted contact/substep diagnostics where boundary samples are insufficient. Sixteen iterations passing one trajectory is not a complete stability certificate.
3. Perform at least twenty independent human-operated native-15-Hz BananaInBowl attempts per profile (forty total) with the headset worn, alternating reference and candidate under matched starting conditions. Predeclare the final sample size, operator allocation, success horizon, exclusions, and confidence-interval method. Exercise grasps, long holds, releases, table approaches, Space/READY, successful placement, failures, disconnect/reconnect, and reset from a displaced leader. Count unsuccessful admitted task attempts, including timeouts, in the success denominator; report deliberate fault-injection checks separately. Synthetic holds and repeated replays do not count toward these attempts.

    For default promotion, propose an allowable success-rate loss of 10 percentage points: require the one-sided 95% lower confidence bound for candidate minus reference success rate to be at least −10 points, no new motion/contact failure mechanism, and candidate p95 source-sample-to-first-physical-response latency no greater than the reference. Also require the throughput/comfort gate below. Report success counts, rate differences, uncertainty, and response distributions. If the corresponding upper confidence bound is below −10 points, reject promotion; if uncertainty spans the margin, keep the profile opt-in and plan a larger fixed-size follow-up. Failed latency/comfort or a new failure mechanism also rejects promotion. These are proposed acceptance choices, not conclusions of the existing tests. Twenty attempts per profile is a minimum, not a guarantee of adequate precision; do not collect until a favorable threshold is crossed.

4. Target at least 14.5 wall-clock commands/s during representative active attempts, realtime factor at least 0.97, and p95 command intervals no greater than approximately 75 ms as an initial usability gate. Aim for p95 active work below 60 ms to retain headroom; report p99/maximum render stalls separately. These are proposed acceptance targets, not achieved guarantees. Lower application render cadence also needs the operator's comfort approval.
5. Validate every episode's sample counts, pre/action/post alignment, timestamps, finite values, thirteen-joint state, command/gripper units, terminal frame, and reset isolation. Check raw finger radians, normalized closure, command open/closed endpoints, and transformed actions against the selected consumer contracts, including values during an object-blocked grasp. Match the current materializer's restored states and camera frames on a small verified dataset. Verify camera states against recorded states rather than an action replay, and preserve the resolved solver/stepping profile with each episode.
6. Render and export three independently human-collected successful native-15-Hz episodes and load a batch through both intended PPS proxy training configurations. Validate the chosen state units and normalization for each recipe. The follow-up diagnostic HDF5 files are excluded until explicitly adapted and validated. This gate checks compatibility; evaluating models trained on the new data remains a later training experiment.
7. Exercise writer queue saturation, low SSD space, writer failure, interrupted saves, and shutdown. No acknowledged episode may disappear; XR must continue servicing the waiting state.
8. Repeat the existing-checkpoint live-policy smoke test through the production evaluation path at cap 32 after relevant environment/evaluation changes, including a delayed inference request. Confirm fresh nonblack camera inputs, expected action chunks, finite actions, and zero physics advancement during synchronous requests using direct physics reads. Report the truncated delay probe separately from completed task episodes.

Measure command latency separately from throughput: source sample → processed target → first physical joint response → scene render submission. Include the upstream velocity limiter and gripper drive response in the analysis. Host timing cannot establish headset motion-to-photon latency; use synchronized visible motion/video or suitable runtime telemetry for that last stage. Even at realtime, native 15 Hz action sampling adds up to one command period of waiting, which cannot be eliminated by a faster recorder.

## What not to prioritize

More CPU workers, further HDD-to-SSD migration, multiple disk writer threads, higher serial sampling, and moving GPU assignments are not the leading changes supported by this profile. The runtime is already on SSD and the leader path is cheap. A generic CUDA-graph or Torch rewrite also does not remove the PhysX/Kit stepping and rendering dependencies.

Keep the real Robotiq linkage and collision model, required task contact sensors, Fabric updates, current gravity, and fixed physics dt. If the validated optimizations still cannot supply enough headroom, investigate rendering presentation stalls and optional visual-only quality reductions next. Any later collision/model simplification is a separately versioned physics change that must be validated for collection and policy evaluation, not an invisible performance shortcut.

The first implementation milestone should combine stages 1–4 behind explicit opt-in options, preserve the ordinary stepping/recording and 32-iteration reference paths, and run the validation suite. The latest headless solver tests do not close the combined-path, independent operator, contact/substep, or training-export checks. Adopt background writing only with boundary-gap evidence, then complete offline materialization/PPS export and the production reference evaluation path. Evaluate models trained on the resulting collection data as a subsequent experiment. No production launch command for the new solver/stepping/recorder options is provided here because those options have not been implemented yet.
