# CUDA-only measurements: shared task checks and collision/model optimization

The [implementation follow-up](omy_collection_optimization_implementation_20261024.md)
documents the subsequently enabled collection defaults and optional collider profile.
The measurements below describe the experiments before that implementation.

## Scope and result

At the user's request, this follow-up measures **only (1) shared task-condition evaluation and (2) collision geometry / robot-model changes**. Physics stays on **CUDA**, TGS, 120 Hz, 32 position iterations, two GPU solver partitions and two physics workers. No CPU physics, worker tuning, XR changes, source-asset edits or production-default changes were made.

**Sharing placement checks is a small, repeatable improvement. Simplifying finger colliders while explicitly preserving the original mass properties is a promising additional candidate. Together they measured about 3.3% less headless command work in the main comparison.** These savings do not recover the full cost difference between the earlier 16- and 32-iteration configurations.

The work comprises **61 completed replays across 19 configurations** of the same recorded bowl attempt. It does not estimate collection-wide success rates. All speed comparisons use exactly source rows **20–99**, with 15 Hz commands, compact recording and normal task checks. Contact-instrumented runs and the Python profile are excluded from speed comparisons. Headset presentation and rendering are not included.

| Configuration | Timing repeats | Mean command work | p95 |
| --- | ---: | ---: | ---: |
| Original geometry, cache off | 4 | 75.405 ms | 83.432 ms |
| Original geometry, shared placement check | 4 | **74.122 ms** | **81.498 ms** |
| Simpler finger colliders, original mass properties, cache off | 4 | 74.574 ms | 80.464 ms |
| Simpler finger colliders, original mass properties, shared placement check | 4 | **72.942 ms** | **79.039 ms** |

Cache on/off is interleaved within each simulator process in **ABBA BAAB** order, with A = off and B = on. The geometry comparison uses separate processes because collision geometry must be cooked before simulation. Additional unchanged-geometry controls averaged **74.424–75.768 ms**; an explicit mass-restoration control averaged **76.418 ms** with the cache off. Geometry's roughly 1–2 ms effect is therefore modest and subject to process/time variation. Do not present it as a large or precisely established speedup. The combined 75.405→72.942 ms comparison is **2.463 ms / 3.27%**, not a live XR measurement.

## 1. Share placement-condition evaluation

The termination manager and subtask state machine both evaluate `object_resting_in_container` at each boundary. The diagnostic wrapper shares a single batched evaluation for the one-environment collector and returns the corresponding scalar result to the second consumer. Every parameter participates in the cache key; the existing boundary context controls invalidation. Calls outside that context and multi-environment calls use the original evaluator. State-machine transitions remain separate.

The same-process comparison measured:

- Original geometry: **1.283 ms / 1.70%** lower mean command work.
- Simplified fingers: **1.632 ms / 2.19%** lower mean command work.
- Mass-restoration control: **1.241 ms / 1.62%** lower mean command work.

For the original-geometry interleave, recording-region time fell **5.974→4.102 ms**, task checks changed **3.330→3.572 ms**, and physics changed **63.902→64.247 ms**. Thus combined recording/task work fell approximately **1.63 ms**; the measured end-to-end saving was smaller because the physics timing varied. This is a reduction in duplicate evaluation, not in the amount of recorded information.

A separate Python profile confirms **112 actual placement evaluations for 112 commands**, compared with 224 in the previous uncached profile. Cache counters show one computation and one reuse per command, with no fallback in eligible replay boundaries. Its placement/timer calls fell from two per command to one. Profile timings include instrumentation overhead and are not the speed evidence.

Correctness evidence:

- In the four matched standalone baseline/cache replays, robot joints, body trajectories, bowl poses, task events and termination rows match exactly.
- HDF5 actions, observations, states and subtask channels match exactly: **16 dataset leaves per episode across four episodes**. Wall-clock timing channels are intentionally excluded from that equality check.
- Tests execute the real placement predicate, settling timer and subtask state-machine bodies. They cover scalar/batched call order, new boundaries, reset, lost support, renewed gripper contact, different thresholds, return-value ownership, invalid parameters and unsupported scopes.
- The longer original-geometry eight-repeat interleave includes one failed grasp in an uncached repeat. The four cached repeats place successfully. This is not evidence that caching improves physics: repeated restores omit contact caches, and the four matched replay comparisons establish unchanged trajectories. One source trajectory is insufficient for a general reliability claim.

The prototype lives in [omy_optimization_probes.py](../scripts/omy_optimization_probes.py), installed only by the benchmark before task registration. It is not enabled in collection. The cache design avoids repeated scalar decisions and evaluations, consistent with PyTorch's [synchronization guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).

## 2. Collision geometry and model changes

### Collider changes also changed inferred mass properties

The first hull experiments were not pure collision-geometry changes. With the asset's existing mass-property authoring, changing the collider approximation also changed the runtime finger masses and inertias:

| Each inner-finger body's mass | Runtime value | Change |
| --- | ---: | ---: |
| Original decomposition | 9.513 g | Reference |
| Pad meshes changed to convex hulls | 17.974 g | +88.9% |
| Finger structure and pads changed to convex hulls | 20.964 g | +120.4% |

Both unconstrained finger variants failed both tested grasps. Restoring the **complete original body masses, COM transforms and full inertia tensors** changed the interpretation: the simpler finger-collider variant then grasped and released successfully. This is why a collider flag alone must not be treated as a physics-neutral optimization.

The controlled variant uses one convex hull per original finger/pad mesh, retains the bowl's original decomposition, and reapplies the reference runtime mass properties to both robot and bowl before warm-up. Body identities and tensor shapes are checked. Readback confirms identical masses and COM transforms, with only numerical round-trip differences in inertia (maximum approximately `5.82e-11` on the robot and `4.66e-10` on the bowl). Shape-offset storage decreases from **144 to 20 robot shape slots**; this is a runtime array measurement, not an inference from an authored hull limit.

### Mass-preserving finger hulls: clean release, small speed gain

The four-repeat contact qualification produced **4/4 placements**:

| Repeat | Peak reported overlap | p95 overlap | Last contact after open command |
| --- | ---: | ---: | ---: |
| 0 | 0.676 mm | 0.276 mm | 0.025 simulated seconds |
| 1 | 0.614 mm | 0.233 mm | 0.025 simulated seconds |
| 2 | 0.591 mm | 0.280 mm | Before opening |
| 3 | 0.600 mm | 0.252 mm | 0.025 simulated seconds |

Both earlier screening repeats also placed, but they reproduce the first two prefixes and should not be counted as independent grasps. All eight uninstrumented repeats with interleaved cache on/off placed successfully. There was no persistent attachment after opening in the four contact qualification runs. A separate original-geometry mass-restoration control also placed in both contact runs, with peaks 0.679 and 0.549 mm.

The speed improvement is small despite the large shape-slot reduction. At fixed 32 iterations, solver/completion cost remains dominant. Convex hulls also fill concavities, so success on this bowl does not qualify the new contact surfaces for handles, thin objects or the whole collection. Original mass properties are a behavioral reference, not a claim that those properties have been physically calibrated. NVIDIA's [Robotiq guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html) calls for multiple grasp widths, difficult shapes, load retention and release checks for precisely this reason.

### Other measured variants did not establish a better candidate

All of the following retain 32 iterations / two GPU partitions / 120 Hz. Contact results come from instrumented runs, not timing comparisons.

| Intervention | Result | Interpretation |
| --- | --- | --- |
| Bowl decomposition limit 256→64 | 0/2 placements | Low usual overlap did not preserve the grasp |
| Bowl limit 256→128, inferred mass properties | 2/2 contact placements; 4/4 timing placements | Uninstrumented mean **76.093 ms**: no demonstrated speed benefit |
| Bowl limit 128, original mass properties restored | 1/2 placements; one 13.79 mm transient contact overlap | Geometry-only variant not qualified |
| Pad-only finger hulls, inferred mass properties | 0/2 placements | Changed both contact geometry and dynamics |
| All finger hulls, inferred mass properties | 0/2 placements | Restoring original mass properties was material to the later result |
| Mimics at 1000 rad/s, damping ratio 1 | 0/2 placements | More damping did not solve this trace |
| Gripper stiffness 275 N m/rad, damping 0.06 N m s/rad | 0/2 placements; persistent overlap about 3.8–4.3 mm at p95 | Copied gains worsened the sticking case |
| Collider-derived arm COM/inertia, existing arm masses | 1/2 placements; peaks 1.87 / 3.10 mm | Not a validated dynamics repair or optimization |

The bowl's authored mass remains 0.5 kg in the hull-cap tests, but its inferred COM/principal frame and inertia change. This is another reason to retain the mass-controlled comparison. None of these model experiments establishes physically calibrated replacement parameters. The [articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html) explains why inertia, drives and mimic constraints must be assessed together; the local measurements determine whether a particular choice actually helps this collector.

## Recommendation from these measurements

Prioritize the shared placement check as the smaller behavioral change: its benefit repeats across all three on/off comparisons and its matched recorded outputs are identical. Keep the mass-preserving finger colliders as an experimental candidate for broader contact qualification. Their incremental speed benefit is small, so changing the collision asset solely for throughput is not yet strongly justified. Neither the reduced bowl decomposition nor the tested drive/mimic/inertia interventions should become a collection default on this evidence.

The combined candidate saves about **2.5 ms per command** against the main reference. This is useful but does not establish matching the original live collection speed with 32 iterations. No render–physics latency claim follows from these headless measurements.

## Artifacts and checks

- [Full result table](../output/omy_diagnostics/20261024_cuda_optimizations/analysis.md) and [machine-readable results](../output/omy_diagnostics/20261024_cuda_optimizations/analysis.json).
- [Matched recording equivalence](../output/omy_diagnostics/20261024_cuda_optimizations/placement_cache_equivalence.json) and [cache Python profile](../output/omy_diagnostics/20261024_cuda_optimizations/placement_cache_profile.txt).
- Each configuration directory retains its summary, effective environment, collider/drive audit, diagnostic HDF5, per-repeat state/contact arrays and per-command region timings.
- Sequential drivers: [screen](../output/omy_diagnostics/20261024_cuda_optimizations/run_screen.py), [model probes](../output/omy_diagnostics/20261024_cuda_optimizations/run_models.py), [mass-controlled geometry](../output/omy_diagnostics/20261024_cuda_optimizations/run_fixed_geometry.py), [interleaved comparisons](../output/omy_diagnostics/20261024_cuda_optimizations/run_final.py), and [mass-restoration controls](../output/omy_diagnostics/20261024_cuda_optimizations/run_mass_control.py).
- [Benchmark](../scripts/benchmark_omy_contacts.py) and [probe helper](../scripts/omy_optimization_probes.py) contain the opt-in experiments. The collector and source assets are unchanged.
- **57 focused checks passed**, covering the cache's real placement/timer behavior and existing replay-source/placement checks. Python compilation and whitespace checks passed. See the [verification record](../output/omy_diagnostics/20261024_cuda_optimizations/verification.json).
