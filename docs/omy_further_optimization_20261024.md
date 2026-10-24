# Further OMY collection optimization

Follow-up to [the bowl investigation](omy_physics_investigation_20261024.md), using the same recorded `demo_1`. The additional work comprises **13 replays in six configurations**. All are headless, with no XR, leader device or dataset camera rendering. Production defaults and assets are unchanged.

The subsequent user-directed investigation keeps physics on CUDA and measures only shared task checks and collision/model changes. Its [61-replay measurement report](omy_cuda_optimization_measurements_20261024.md) supersedes the unmeasured estimates for those two directions below. CPU physics is outside the requested follow-up scope.

## What the additional measurements establish

**Zero physics worker threads provides a small additional improvement at 32 iterations / two GPU partitions.** Zero means that PhysX runs its CPU work on the calling thread, not that physics work is disabled. Carbonite and TBB remain at eight workers; these are separate settings. NVIDIA specifically recommends trying zero physics workers for small scenes in its [performance guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/guides/physics-performance.html).

| Configuration | Throughput repeats | Mean command work | p95 | Placement outcomes |
| --- | ---: | ---: | ---: | --- |
| 32 iterations / 2 partitions / 2 physics workers, fresh control | 2 | 75.30 ms | 82.81 ms | 2/2 |
| 32 iterations / 2 partitions / 0 physics workers | 4 | 73.17 ms | 79.98 ms | 4/4 |

The comparison uses exactly source rows **20–99** in every repeat, with no contact readback or Python profiling. The mean improvement is **2.13 ms / 2.8%**. Joint and bowl pose arrays are numerically identical between the first two zero-worker runs and their two-worker counterparts. Two separate zero-worker contact runs also place successfully, with worst reported penetration **0.645 mm**, and release when commanded. This is a small sequential experiment on one source trace; it is not a collection-wide reliability estimate or a measured live XR speedup.

The diagnostic script already accepts `--physics-threads 0`. The production Bash launcher currently fixes two workers, and the shared XR argument validator rejects zero. Thus this result is **benchmark-only**, not a flag that can already be passed to `collect_omy_xr.sh`. Promoting it should add explicit forwarding and metadata, allow the documented zero value, preserve existing defaults/resume behavior, and repeat the full XR comparison.

**Intermediate iteration counts do not yet provide a qualified faster configuration.** At two partitions, both 24-iteration replays and both 28-iteration replays fail the recorded grasp/placement. The 28-iteration contact maxima remain below 0.91 mm, illustrating that low overlap alone does not establish grasp retention. At 24, one run includes a very large transient negative separation; reported contact separation is not an exhaustive geometric intersection measurement. Neither count is promoted to the production CLI. These failures concern this fixed open-loop trace; they do not prove that every operator grasp at those counts fails.

## Where the remaining time goes

A separate `cProfile` run at 32 iterations / two partitions / two workers executes 112 commands and succeeds. It spends **6.83 of 8.33 seconds (82%) inside `omni.physx.fetch_results`**, called eight times per command. This call waits for completion of the simulation, including solver/collision work and associated synchronization. It does **not** mean that 82% is a removable data-copy overhead, and cProfile cannot separate GPU kernel time from CPU scheduling or synchronization inside PhysX. A Tracy/CUDA timeline is needed for that distinction.

Selected inclusive costs from that instrumented run, averaged over all 112 commands:

| Work | Time per command | Interpretation |
| --- | ---: | --- |
| PhysX `fetch_results` | 61.0 ms | Dominant completion wait; includes actual simulation work |
| `object_resting_in_container`, two evaluations | 5.18 ms total | Termination and subtask consumers both evaluate it |
| Event tracker | 2.52 ms | Contact, movement and error-event evaluation |
| Subtask recorder | 4.84 ms | Includes the event tracker and one placement evaluation |
| Post-step state capture | 0.385 ms | Actual scene snapshot, before storage bookkeeping |
| Recorder active-environment selection | 0.142 ms | Repeated CUDA `nonzero` / host-list conversion |

These rows overlap and **must not be added**. Profiling has overhead; the separate uninstrumented worker comparison is the speed evidence. The result also refines the interpretation of the earlier broad recording timer: most of that region is task/event evaluation, not disk I/O or copying the saved scene state.

## Prioritized engineering work

1. **Reduce duplicate task evaluation while keeping the existing physics.** The same placement predicate is evaluated twice per boundary. Cache identical read-only predicate inputs/results with complete parameter keys; retain correct settling windows, reset invalidation, final-step events and per-consumer state transitions. For this single-environment collector, test one compact boundary snapshot for CPU predicate/event processing instead of many tiny GPU operations and scalar decisions. Keep the general batched evaluator for other workloads. The measured task/event budget supports a modest millisecond-scale opportunity, not a promised large speedup. Relevant code: `conditionals.object_resting_in_container`, `EventTracker.check_events`, the subtask recorder and `boundary_queries`. PyTorch documents [the synchronization caused by scalar extraction, `nonzero`, transfers and CUDA-dependent Python control flow](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization). Merely eliminating explicit `.item()` calls is insufficient: those calls account for only about 0.07 ms per command in this profile.

2. **Optimize actual collision geometry and then requalify accuracy.** Inspect the cooked finger pads and bowl rim, and replace unnecessary decomposition pieces with deliberate smooth proxies that preserve the opening and contact surfaces. Previous per-mesh finger-hull trials show that geometry can reduce both sticking and cost, but they also change which grasps succeed; a convex hull is not a universally valid replacement. Validate thin bowls, handles, small objects and heavy/slippery objects before adoption. Profile narrowphase, contact count and solver separately to establish which part improves. NVIDIA's [asset/performance guidance](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html#robot-asset-optimizations) supports simpler colliders. Turning off gripper/object collisions or filling hollow objects would invalidate the task.

3. **Repair the robot dynamics coherently to unlock the larger CPU opportunity.** Verify mass, COM, principal axes and full inertia together; then calibrate drive effort/stiffness/damping and mimic compliance. The existing combination of tiny arm inertias, approximately 5730 N m/rad gripper stiffness, and lightly damped followers makes contact convergence demanding. CPU already measured about 21–25 ms per command on the earlier matched horizon, versus about 75 ms for CUDA/32/two partitions, but currently fails reliable grasping. That is potential headroom, not an available safe speedup. Geometry-derived inertia alone and copied gripper gains already failed to establish a solution. Follow the [articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html) and [Robotiq grasp/release test sequence](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html), using independently measured or validated parameters.

4. **Profile live XR presentation independently.** The original recording spends 22.2 ms per command in rendering. Separate drawing, application update, headset submission and presentation waiting before choosing an intervention. Previous resolution reductions did not materially help, so a pixel-count optimization has weak evidence. Preserve fresh frames and measure input-to-display latency if testing a different rendering schedule or separating simulation from presentation. The [5.1 handbook's asynchronous-rendering setting](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html#asynchronous-rendering) concerns stopped/paused operation and is not evidence of a drop-in active-collection optimization on installed Isaac Sim 5.0. Extra GPUs also do not split a single PhysX GPU simulation across devices; assigning presentation elsewhere requires measuring transfer and synchronization costs.

5. **Test an upgraded physics stack in isolation after preserving a baseline.** Newer articulation/contact solver ordering may allow the same fidelity at lower cost, but the relevant `solveArticulationContactLast` support postdates the installed patch. Repeat grasp–hold–release, task event and throughput checks rather than assuming that authoring a newer schema property changes the current engine. The [existing source review](omy_contact_physics_sources.md) contains the exact version evidence.

For immediate work, keep the **32 / two-partition** candidate as the fidelity reference, qualify the small zero-worker gain, and target the duplicated task computation. The largest longer-term performance opportunity remains making the model reliable on the much faster CPU path. Even eliminating the roughly 10 ms headless task/recording budget would not remove the approximately 61 ms PhysX completion cost, so Python work alone cannot deliver a large end-to-end improvement.

## Already enabled or unsupported shortcuts

- Fabric is enabled; scene-query support and robot self-collisions are already disabled in the measured configuration. Recommending these switches again does not create additional savings.
- Held targets, boundary scene updates, compact recording and boundary query reuse are already active. Disk export is outside the active replay loop.
- Lowering to 16, 24 or 28 iterations, selecting one partition, and changing collision shapes all require accuracy qualification; the available experiments do not justify adopting them globally.
- 240 Hz / 16 iterations was slower than the 120 Hz / 32 reference. Dynamic scene-cap edits were not shown to affect the active solver. Per-finger requests do not establish isolated solver work within an articulation/contact island.
- Do not skip required `fetch_results` calls or physics steps to improve a timer. Preserve simulation completion, callback order, fresh state and fixed simulated time.

## Artifacts and verification

- [Machine-readable analysis](../output/omy_diagnostics/20261024_further_optimization/analysis.json), with profile arguments, outcomes, common-horizon timings and diagnostic exclusions.
- [Python profile](../output/omy_diagnostics/20261024_further_optimization/python_profile.txt); raw `.pstats`, contact arrays and trajectories remain in the corresponding profile directories.
- [Sequential screening driver](../output/omy_diagnostics/20261024_further_optimization/run_screen.py), [worker comparison driver](../output/omy_diagnostics/20261024_further_optimization/run_workers.py), and [analysis script](../output/omy_diagnostics/20261024_further_optimization/analyze.py).
- [Diagnostic replay](../scripts/benchmark_omy_contacts.py) now accepts intermediate caps from 16 through 64 and optional `--profile-python`, explicitly excluding profiled runs from throughput evidence. Intermediate values are assigned before creating PhysX; production choices are unchanged.
- Replay source-loading checks: **5 passed**. All 13 diagnostic replays completed. Source recordings and assets were not modified.
