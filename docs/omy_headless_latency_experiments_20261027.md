# Headless collection latency experiments — 2026-10-27

**Zero physics workers is the best small improvement measured in this round:**
about 2–3 ms less work per command in comparisons with stable, similar GPU clocks.
Doubling the already-large PhysX buffers increased VRAM use by about 1 GiB and
produced essentially unchanged command latency at comparable clocks. The traces
point to articulation constraint solving and many kernel launches as the larger
remaining opportunity. These are headless results; live XR was excluded.

## Design and scope

Completed **66 diagnostic replays in 50 fresh simulator processes**, on isolated
host GPU 0, an RTX 3090 with 24 GiB VRAM:

- 30 unprofiled timing processes: five trajectories, two worker settings, three
  fresh processes per setting. Setting order and case order were alternated.
- 20 contact-instrumented replays in ten processes: two repeats per case/setting.
- Four CPU/CUDA timeline profiles: bowl and pail 02, at zero and two workers.
- Twelve buffer-capacity control replays in six processes: bowl and pail 02,
  including a reversed-order pail repeat after GPU clock differences were found.

The trajectories cover four object types: a bowl, two plastic-pail layouts, a
milk carton and a small pudding-mix box. The latter two broaden the earlier bowl
and pail comparison. Full paths, episode identities, source hashes, launch commands
and GPU identity are in the manifest.

Fixed settings: CUDA physics, 32 position-iteration cap, two GPU partitions,
120 Hz physics, 15 Hz recorded commands, held-target stepping, shared placement
checks, original finger colliders, two Kit/TBB thread pools of eight workers,
and compact recording. The experiments vary PhysX's worker count, not whether
physics runs on CUDA. Saved effective settings confirm the requested worker count.

Replays restore the recorded initial state and execute recorded commands, followed
by up to two seconds holding the final action. Bowl uses the existing first-240-
command regression horizon; the other cases use their full recorded commands.
Task success can end a replay early. Success reached during the extra hold is
reported separately from success reached by the original command endpoint.

Timing uses exactly source command rows **20–99** in every unprofiled comparison.
Contact readback and CPU/CUDA profiling are excluded from speed statistics.
All measured render counts are zero. No leader, headset or camera was opened.

## Worker count

The first two timing columns retain **all three fresh processes** per setting,
including GPU frequency variation. The saved NVML samples revealed different
boost frequencies in some runs, typically 1695 versus 1920 MHz. The saving column
is a secondary comparison using only process pairs whose sampled replay clocks
stay within a common 30 MHz band. No runs are removed from the raw results.

| Trajectory | Two workers, mean ms/command | Zero workers, mean ms/command | Saved at comparable stable clocks, ms | Clock-comparable pairs |
| --- | ---: | ---: | ---: | ---: |
| Bowl 00 | 71.49 | 69.17 | 2.29 | 2/3 |
| Pail 01 | 73.55 | 71.74 | 2.01 | 2/3 |
| Pail 02 | 74.22 | 74.11 | 2.62 | 2/3 |
| Milk carton 00 | 72.70 | 70.40 | 2.30 | 3/3 |
| Pudding-mix box 01 | 72.81 | 70.45 | 2.35 | 3/3 |

The recorded joint positions/velocities, object poses, robot body positions,
actions, task events and task-success outcomes match exactly across all **22
paired worker comparisons**: 15 unprofiled pairs, five contact pairs and two
profiled pairs. Some pairs contain two replays. These comparisons establish
behavior preservation for these inputs, not a collection-wide success estimate.

Tail timing also improves in this pilot. Values below are averages of the three
process-level percentiles; the JSON retains every process's median, p95, p99,
physics time and deadline misses. An 80-command window makes p99 particularly
sensitive to individual samples.

| Trajectory | Two workers: p95 / p99, ms | Zero workers: p95 / p99, ms |
| --- | ---: | ---: |
| Bowl 00 | 79.81 / 83.00 | 76.11 / 77.18 |
| Pail 01 | 84.05 / 87.96 | 81.45 / 82.15 |
| Pail 02 | 85.46 / 87.43 | 84.10 / 84.58 |
| Milk carton 00 | 78.93 / 80.54 | 75.65 / 76.69 |
| Pudding-mix box 01 | 79.34 / 81.47 | 75.45 / 75.80 |

This does not establish reliable 15 Hz wall-clock collection. Mean command work
at zero workers is still approximately 69–74 ms across the raw case averages,
above the 66.7 ms command budget, and many individual commands miss that budget.

The unprofiled task outcomes are 12/15 at each setting. Contact-instrumented
outcomes are 9/10 at each setting. Pail 02's first replay fails at both settings;
its second replay succeeds in each contact process. Initial-state restore omits
hidden contact/solver caches, so a prior recorded success is not proof that a
fresh action replay will succeed. Neither worker setting fixes that existing
replay limitation. Some bowl traces also lose the grasp before commanded opening
while still reaching task success; retain grasp-retention diagnostics separately.

## Contact validation

Both inner fingers were sampled every physics step. Worker pairs have identical
contact and motion behavior. The worst reported overlap is **0.645 mm** across
these worker-validation cases; none has persistent finger–target contact after
opening in the final observed 0.25 seconds. Contact separation is not an exhaustive
mesh-intersection measurement. This result is limited to the tested trajectories.

## GPU execution

CUPTI traces capture CPU command/substep annotations, CUDA runtime/driver launch
activities, native GPU kernels, copies and memsets. Analysis selects the 80 source
commands and verifies exactly **640 CPU physics-substep annotations**. GPU-projected
annotations are excluded from the CPU step/command counts.

| Profile | GPU kernels per command | Largest kernel's share of summed GPU kernel time | Captured GPU-active fraction of command window |
| --- | ---: | ---: | ---: |
| bowl_w0_profile | 4487 | 31.1% | 70.4% |
| bowl_w2_profile | 4487 | 31.1% | 68.4% |
| pail02_w0_profile | 4497 | 29.6% | 71.0% |
| pail02_w2_profile | 4497 | 29.6% | 69.0% |

The largest kernel is `artiSolveInternalTendonAndMimicJointConstraints1T`, followed
by `artiSolveInternalConstraintsTGS1T`. The scene's articulation constraints are a
substantial cost even during approach, before object grasp contact. The four
profiles show essentially the same GPU kernel counts and kernel execution costs
at zero and two workers. Zero workers therefore improves CPU coordination without
removing solver work. In the Python profiles, the long native call moves from
`fetch_results` at two workers to `simulate` at zero workers; comparing either call
alone would misrepresent the total physics cost.

GPU-active fraction is a union of captured kernel/copy/memset intervals, not SM
occupancy or proof of a compute-versus-bandwidth bottleneck. CUDA API durations
include host waits and can overlap GPU execution; do not add those durations to
kernel time. Profiling adds substantial overhead, so its command times are not
used as speed evidence.

No allocation API calls appear in the selected command windows. The captured
PhysX driver API activity is largely kernel launches; this is not an exhaustive
audit of its internal driver allocations. That coverage limitation motivated the
separate capacity control below.

## Memory-capacity control

Compared the existing **1 GiB** heap, temporary-buffer and collision-stack
capacities against **2 GiB each**, holding zero workers and all other physics
settings fixed. These capacity fields do not all represent dedicated VRAM
allocations. Measured replay VRAM rises from about **4.7 GiB to 5.7 GiB**.

The first pail comparison ran at different boost clocks. It was retained, and a
second process pair ran in reverse order. The following table uses only replays
with sampled clocks remaining within a 30 MHz band throughout replay, and matches
the frequency across capacity settings. Mixed-clock runs remain in the JSON.

| Trajectory | GPU clock | Existing capacities, ms/command | Doubled capacities, ms/command | Existing / doubled replay counts |
| --- | ---: | ---: | ---: | ---: |
| Bowl 00 | 1695 MHz | 71.15 | 71.21 | 2 / 2 |
| Pail 02 | 1920 MHz | 67.69 | 67.70 | 1 / 3 |

There is no established latency benefit from the larger buffers in this pilot.
All three buffer comparisons preserve trajectories exactly for both repeats.
No capacity-overflow or allocation-failure warning was found. NVML sampling runs
outside the simulator at 5 Hz; reported peaks are device-wide and can miss brief
spikes. The pail's matched-clock baseline has only one replay, so it is a limited
capacity check rather than a precise performance estimate.

## Next engineering direction

Qualify zero workers as the small collection optimization, with explicit launcher
forwarding, provenance and resume compatibility if it is promoted. The larger
experimental branch should target the articulation/mimic solver and CPU/GPU launch
coordination, including an isolated newer-engine comparison using the same asset
and trajectory baseline. Model changes require grasp-retention, release and
penetration validation. The earlier mass-preserving collider experiment saved
only about 1–2 ms; the new traces make articulation work the higher priority.

## Artifacts and checks

All 50 simulator processes exited normally. Source recordings, assets, the
production benchmark and collection defaults are unchanged. The diagnostic
harness is an output-only copy with its exact patch retained. The timelines are
losslessly compressed with verified decompression hashes. No videos were generated.

- [Run manifest and source hashes](../output/omy_diagnostics/20261027_headless_latency/manifest.json)
- [Timing, contact, clock and memory analysis](../output/omy_diagnostics/20261027_headless_latency/analysis.json)
- [CUDA timeline analysis](../output/omy_diagnostics/20261027_headless_latency/trace_analysis.json)
- [Configuration, trajectory and source-integrity checks](../output/omy_diagnostics/20261027_headless_latency/verification.json)
- [Diagnostic harness patch](../output/omy_diagnostics/20261027_headless_latency/harness.patch)
- [Timeline compression integrity](../output/omy_diagnostics/20261027_headless_latency/trace_compression.json)

Per-process logs, configurations, diagnostic recordings, state/contact arrays,
NVML memory/clock samples and compressed Chrome timelines remain in the artifact
directory. Analysis and launch scripts are included for reproduction.
