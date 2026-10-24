# OMY bowl contact and physics performance investigation

Investigated 2026-10-24 on the active `/tmp2/leocheng/RoboLab` checkout, Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 / Omni Physics 107.3.18, RTX 3090. The source is the failed bowl collection [`20261024_141641_bowl`](../output/omy_teleop/20261024_141641_bowl).

## Decision

**The reduction from the normal scene cap of 32 to 16 contributes to this failure. Increasing iterations alone is not a sufficient general fix.** The strongest immediate candidate is **32 position iterations with two GPU solver partitions**, retaining the existing timestep, control rate, collision geometry, materials, actuators and optimized stepping. It improves the measured cost of the 32-iteration configuration by about 21%, while the tested bowl replays release cleanly.

This is a candidate for a new, supervised collection run, not a claim of collection-wide qualification or measured live-XR speed. No leader or XR session was opened in this investigation. The measured full collection baseline was about **10 actual commands/s**, despite its nominal 15 Hz command rate.

The deliverables are:

- [Recording forensics and evidence images](omy_bowl_recording_forensics.md).
- [33 primary resources and their version/implementation caveats](omy_contact_physics_sources.md).
- [All completed trial summaries, release metrics and plots](../output/omy_diagnostics/20261024_bowl_contacts/analysis_results.md), with [machine-readable results](../output/omy_diagnostics/20261024_bowl_contacts/analysis_results.json).
- [A reusable replay/contact benchmark](../scripts/benchmark_omy_contacts.py), plus sequential experiment drivers and the analysis script under [`harness/`](../output/omy_diagnostics/20261024_bowl_contacts/harness).

## What the recording establishes

The failed episode is `demo_1` in `SimplePnp291Bowl00Task/part_000/omy_demos.hdf5`. The actual settings were CUDA physics, TGS, 120 Hz physics, 15 Hz commands, eight physics steps per command, and a **scene position cap of 16**. The robot requests 64 but is clamped by the scene; the original repository scene cap was 32. The recording's `performance.profile="cpu"` label does not identify its backend: `physics_device="cuda:0"` and the saved environment configuration establish that it ran on CUDA.

The first grasp closes at source row 53. Opening starts at row 100, and the master gripper joint reaches its fully open position by the end of row 103, about 6.93 simulated seconds. The bowl nonetheless hangs from one finger through the 120-second timeout. This is different from the earlier milk-jug jam, where the master joint itself stayed closed.

The bowl's source shell is roughly **0.7–1.5 mm** thick. The bowl and finger pads use convex decomposition and friction coefficients of 2; the finger material combines friction with `max`. CCD is already enabled on the scene, bowl and inner fingers. The gripper speed correction to `velocity_limit_sim=5` was already present.

Two further model concerns matter:

- Angular USD stiffness 100 resolves to **5729.58 N m/rad** at runtime; damping 0.0002 resolves to **0.011459 N m s/rad**. NVIDIA's [Robotiq 2F-85 tuning example](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html) discusses almost this exact high-stiffness configuration, thin bowls, and the USD/radian conversion. Copying a few example gains did not fix this replay.
- Four follower joints use compliant mimics with natural frequency 1000 and damping ratio 0.05. Passive joints in the source recording reach about 9.6 rad/s even though the master limit is 5. Runtime tensors also confirm unusually small arm-link inertias, down to approximately `6e-8 kg m²` for a 0.405 kg wrist link. These warrant coherent asset validation; recomputing inertia alone did not fix the grasp.

The logical gripper contact sensor currently covers **only the left inner finger**. Placement logs can therefore report detachment while the bowl remains on the right finger. This episode still correctly failed because it lacked bin support. Future release qualification should cover both fingers through separate single-body views, rather than an unsupported many-body filtered sensor.

## Solver accuracy: measured contacts, not only linkage error

The new benchmark restores the recorded initial state and replays the original native-15-Hz commands. Accuracy runs read both inner fingers' filtered contact separation after every physics step. Negative reported separation measures contact overlap; it is not an exhaustive volumetric mesh-intersection test, and a contact report can describe a transient overlap resolved during that step.

| Configuration, original geometry unless specified | Diagnostic replays | Placement successes | Largest per-repeat penetration p95 | Peak reported penetration | Release behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| 16 iterations, 8 partitions | 2 | 0 | 5.08 mm | 6.49 mm | Contact persists through all 9.3 s of observed open commands |
| 32 iterations, 8 partitions | 2 | 2 | 0.23 mm | 15.51 mm, one transient sample | Contact ends within about 0.05 s of opening |
| 64 iterations, 8 partitions | 2 | 0 | 0.14 mm | 0.36 mm | Misses/loses the grasp before commanded release |
| **32 iterations, 2 partitions** | **4** | **4** | **0.34 mm** | **0.67 mm** | **All release; no persistent open contact** |
| 32 iterations, 1 partition | 4 | 3 | 0.36 mm | 13.58 mm, transient in the failed grasp | No persistent open contact; one grasp fails |
| 16 iterations, simpler finger hulls | 4 | 4 | 0.88 mm | 1.89 mm | All release; collision representation changed |

These are repeated executions of **one recorded attempt**, not independent operator trials or population success-rate estimates. Two earlier simpler-hull replays also succeeded, but repeated prefixes across fresh processes should not be counted as additional independent evidence. Higher counts change the reached trajectory; they are not monotonically better at completing the same open-loop grasp.

![Contact separation across baseline solver caps](../output/omy_diagnostics/20261024_bowl_contacts/penetration_vs_time.png)

The persistent multi-millimetre contact at 16, versus release and much smaller usual overlaps at 32, supports insufficient convergence as a contributor. The transient at 32 and the missed grasp at 64 show why neither a success flag nor a high iteration count establishes penetration-free physics.

## Speed: the useful optimization is GPU partition count

Throughput runs have **no per-substep contact readbacks**. The following comparison uses the same source rows **20–99**, including the entire closing interval and ending before any trial's success termination. Timings include task checks and compact recording, exclude startup, rendering and XR, and use separate fresh simulator processes.

| Configuration | Mean command work | p95 | Replay outcome |
| --- | ---: | ---: | --- |
| 16 iterations, default 8 partitions | 61.4 ms | 74.3 ms | Both fail with persistent attachment |
| 32 iterations, default 8 partitions | 95.5 ms | 115.0 ms | Both place successfully |
| **32 iterations, 2 partitions** | **75.7 ms** | **83.2 ms** | **Four place successfully** |
| 32 iterations, 1 partition | 72.2 ms | 79.3 ms | Three of four place successfully |
| Return to 32 iterations / 8 partitions | 96.0 ms | 119.7 ms | Both place successfully |
| CPU / 64 iterations, original model | 25.2 ms | 40.4 ms | Both fail to complete the grasp |
| 16 iterations, simpler finger hulls / 8 partitions | 59.0 ms | 68.4 ms | Four place successfully; different collision geometry |

Returning to eight partitions restored the larger cost, supporting the partition change rather than a simple timing drift explanation. Two partitions recover roughly **20 ms per command**, or 21% of the original 32-iteration work. One partition is slightly faster but showed worse consistency in this sample. The limit changes GPU solver scheduling and can change contact trajectories, so its accuracy must be checked as well as its speed. It is applied before PhysX creation.

![Contact accuracy and timing comparison](../output/omy_diagnostics/20261024_bowl_contacts/solver_partition_comparison.png)

This is not yet proof of matching live collection speed. The original XR run averaged **98.19 ms**, p95 **112.72 ms**, at **10.09 commands/s**, with physics 56.73 ms, rendering 22.20 ms, recording 8.01 ms and task checks 7.00 ms. The two-partition/32-iteration configuration still costs more than the original 16-iteration configuration in matched headless work. Rendering, synchronization and presentation must be remeasured together; simply adding or subtracting these separate timings is not an XR benchmark.

### Candidate launch

The Bash launcher and Python collector now accept `--gpu-max-num-partitions`. Defaults remain eight partitions and the existing solver selection. Run a **new session** to compare profiles:

```bash
./scripts/collect_omy_xr.sh SimplePnp291Bowl00Task \
  --num-episodes 5 \
  --solver-position-iterations 32 \
  --gpu-max-num-partitions 2 \
  --log
```

Compare the rolling mean/p95, actual command rate, rendering gaps and complete grasp–hold–release behavior against the existing 98/113 ms baseline, with the headset worn and presentation conditions matched. A nominal 15 Hz setting is not a measured 15 Hz wall rate. The new partition value is stored in the session contract and recording metadata; a conflicting resume is rejected. Older sessions retain their original effective partition value of eight and their exact stored contract.

## Other experiments and their implications

**CPU physics has the largest potential headroom, but is not a validated replacement for this model.** Original-model CPU 32/64 runs cost about 18–22 ms across their full measured horizons, yet missed, dropped or ejected the bowl and showed larger linkage errors. Hard mimics, stronger mimic damping, collider-derived arm inertia, smaller steps, changed drive gains, armature and simpler finger hulls were tested. Some improved particular metrics; none of the tested CPU variants established reliable complete grasp/release. One combined inertia/hard-mimic case placed once in two trials. The current CPU path also honestly falls back to ordinary stepping; its fast-path guard was not removed. The launcher now exposes `--device cpu` for further explicit experiments. NVIDIA's [performance guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/guides/physics-performance.html) supports CPU for small scenes, but that is performance guidance rather than a fidelity guarantee.

**Simpler finger colliders are a second useful direction.** Per-mesh convex hulls replaced the four finger/pad decomposition sources in memory; the articulation's collision-offset array decreased from 144 shape slots to 20. Six contact-instrumented 16-iteration replays and four throughput replays placed successfully, with lower persistent penetration and lower cost. At 32 iterations, however, only one of four replays placed successfully. A hull fills concavities and changes the grasp, so a carefully authored smooth pad/finger proxy must be qualified on bowls, handles and other object shapes before collection-wide use. The original asset was not edited.

**Smaller steps can improve contacts but were slower here.** The matched ordinary-stepping comparison of 240 Hz / 16 iterations against 120 Hz / 32 kept the action rate at 15 Hz. Reported overlaps were small, but 240 Hz completed only one of two placements and took about 126 ms on the common horizon, versus about 108 ms for ordinary 120 Hz / 32 with contact diagnostics. It did not recover speed on this stack. The existing held-target optimization is audited only for 120 Hz, so the ordinary reference matters.

**Uniform friction reduction and larger offsets did not solve this trace.** Lowering both robot and object friction to 0.5 placed once in two trials; increasing both shapes' contact offset to 4 mm placed neither and increased work. Do not lower only the bowl material: the finger's `max` combine mode can keep pair friction at two. Runtime offsets were already around 1.36–1.50 mm on relevant shapes and zero rest offset, despite apparent scene-level offset values in the base configuration. Those nonexistent `PhysxCfg` fields are skipped; actual shape properties must be checked.

**Adaptive live cap editing was not established.** A stationary 16→32→64→32→16 USD scene-cap probe stayed near 51 ms per command instead of acquiring the expected cost changes. A changed USD number is not proof that the live solver consumed it. Keep cap changes before environment creation unless an effective runtime path is independently demonstrated. Similarly, raising only a finger/body's request cannot promise localized work: the robot is one articulation and solver islands use their highest request.

**Already-exhausted or weak avenues:** all held-target/boundary/compact/query capabilities were active in the source run, and it retained all seven required contact sensors. Existing same-host tests found no useful gain from more physics workers or smaller XR/desktop pixel dimensions. State recording is buffered; disk writes and serial sampling are not the main active-step bottleneck. Further CPU/GPU readback batching may recover some of the task/recording cost, but it should follow the measured partition improvement.

## Follow-up priorities

The [further optimization follow-up](omy_further_optimization_20261024.md) adds 13 replays: a measured 2.8% headless improvement with zero physics workers, unsuccessful 24/28-iteration grasp replays, and a Python profile locating 82% of replay time inside PhysX completion. It also distinguishes task/event work from the much smaller state-copy cost.

1. Validate the **32 iterations / 2 partitions** candidate with matched live XR and independent human grasps on the bowl, milk-jug handle, a small object and a slippery/heavy object. Retain the measured baseline rather than judging by the configured rate.
2. Extend release/contact diagnostics to **both fingers**, and retain substep separation logging in dedicated qualification runs. Track overlap duration, grasp retention and release as well as peaks and mimic residuals.
3. Inspect cooked colliders with OmniPVD and author smooth, deliberate finger-pad/object proxies. Keep hollow bowls and handles hollow. A dynamic SDF may be useful on a compatible GPU stack, but this version's GPU-only SDF collision path is not a drop-in CPU optimization.
4. Repair and verify mass/COM/full-inertia units as a coherent asset model, then tune drives and mimic damping against that model. The vendor-hosted identified Franka inertias are a plausibility reference with different masses, not numbers to paste into this asset. Recomputed inertia was a diagnostic intervention, not a production repair.
5. Revisit CPU physics after model qualification. Its speed leaves room for higher accuracy, while the present replays show why a backend switch alone is insufficient.
6. In an isolated upgraded stack, evaluate `solveArticulationContactLast`, introduced after bundled patch 107.3.18, plus relevant contact/mimic fixes. Do not author an unsupported attribute into the current stack and assume it took effect. Version details, limitations and authoritative links are in the [source review](omy_contact_physics_sources.md).

## Validation and scope

The aggregate contains **97 completed diagnostic runs across 42 configurations**, including the dedicated one-command runtime-cap probe; startup/API failures and the nonfinite CPU-gain trial are listed separately. These are one source trajectory's diagnostic outcomes, not training demonstrations or a collection success-rate study. Detailed contact runs are separated from throughput runs. No collision assets, shared DROID/evaluation defaults, calibration, serial settings, XR preferences or existing recordings were changed.

The production additions are opt-in partition/backend launcher controls, OMY-only partition configuration and provenance, and resume validation with compatibility for older sessions. The benchmark supports backend/solver/timestep comparisons, state-restored opening probes, contact distances, drive/mimic/inertia/collider interventions and timing. **All 682 OMY unit checks passed**, including source snapshot ownership, parser/launcher forwarding, resume contracts, evaluation isolation and existing collection/postprocessing coverage. Shell syntax, Python compilation and whitespace checks also passed. The [verification record](../output/omy_diagnostics/20261024_bowl_contacts/verification.json) preserves the commands and scope.
