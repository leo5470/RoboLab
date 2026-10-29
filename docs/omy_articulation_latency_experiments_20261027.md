# Articulation latency experiments — 2026-10-27

**No larger general collection speedup qualified in this round.** The supported
headless candidate remains TGS, a 32 position-iteration cap, two GPU partitions,
original gripper coupling and zero physics workers. The previous worker study
measured about 2–3 ms less command work without changing the tested trajectories.

The new experiments found promising bowl-only results with hard mimic coupling
at 28 and 30 iterations. Broader checks exposed a carton settling regression at
28 and an early box drop at 30, even though the latter still satisfied terminal
task success. Neither candidate was promoted. This is evidence about the tested
open-loop replays, not proof that all human grasps under either profile would fail.

## Completed work

Completed **76 replays in 38 fresh simulator processes**, all on isolated RTX 3090
host GPU 0, with XR, cameras and rendering disabled:

- Four target-removed causal diagnostic replays in two processes.
- Thirty bowl screening replays in the first 15-process parameter cohort.
- Eighteen bowl screening replays in a nine-process refinement cohort.
- Twenty-four broader qualification replays on both pails, the carton and box.

All use CUDA, 120 Hz physics, 15 Hz commands, two GPU partitions, zero physics
workers, held-target stepping, shared placement checks, original colliders,
unchanged robot masses/COMs/inertias, and unchanged live actuator drive settings.
The variable factors are solver algorithm, position-iteration cap, and mimic
compliance. Experimental overrides are applied in memory before creating PhysX.
The source robot USD, source recordings and production defaults were not changed.

Each process performs two repeats. Source commands are followed by up to two
seconds holding the final recorded action; no opening command is injected to
rescue a trial. Task success stops a replay early. Initial-state restore omits
hidden contact/solver caches. Repeated replays are not independent demonstrations.

## Screening rules

Screened task success together with at least 5 cm lift, maximum reported contact
overlap of 1 mm, closed-linkage error p95 at most 2 degrees and peak at most
5 degrees, an observed open-command window of at least 0.25 seconds, and no
finger–target contact in its final quarter-second. The bowl screen additionally
requires two-inner-finger contact coverage of at least 65% after the initial
closing interval. These are declared screening rules for this investigation,
not universal physical limits. Contact counts alone do not establish load-bearing
force or replace object-motion checks.

Broader qualification requires at least the baseline's per-case task-success
count, no material regression in overlap/linkage error, no retained open contact,
and no substantial loss of both-finger contact coverage. All raw results and
rejection reasons are retained. Aggregate success counts are insufficient.

## What the parameter screen established

PGS was tested at 32 and 64 iterations. Both configurations failed both bowl
grasps, with peak reported overlaps approximately 39 mm and 21 mm respectively.
The higher count was also substantially slower in the instrumented screen. This
branch did not establish a useful replacement for TGS.

Soft mimic settings used natural-frequency values 120, 500, 1000 or 2000 s^-1,
with critical damping where applicable. Lower frequencies sometimes produced a
passing terminal task at lower solver counts but allowed excessive linkage error
or overlap. Keeping the outer left–right coupling rigid and tuning only the four
inner followers did not produce a qualifying 16- or 24-iteration candidate.

Hard mimic constraints improved linkage tracking. At 32 iterations, the bowl
replays passed with submillimetre contact overlap but no demonstrated speed gain.
Hard coupling at 16 or 24 iterations failed. The bounded 28/30 refinement passed
the bowl screen and proceeded to other objects.

| Profile | Task successes | Full screen passes | Peak overlap, mm | Worst linkage p95, degrees |
| --- | ---: | ---: | ---: | ---: |
| base | 2/2 | 2/2 | 0.645 | 1.31 |
| hard16 | 0/2 | 0/2 | 6.623 | 0.82 |
| hard24 | 0/2 | 0/2 | 16.530 | 0.50 |
| hard32 | 2/2 | 2/2 | 0.723 | 0.40 |
| passive16 | 0/2 | 0/2 | 6.786 | 2.47 |
| passive24 | 0/2 | 0/2 | 1.939 | 1.40 |
| passive32 | 2/2 | 2/2 | 0.645 | 1.31 |
| pgs32 | 0/2 | 0/2 | 39.297 | 1.30 |
| pgs64 | 0/2 | 0/2 | 20.679 | 1.27 |
| soft120_16 | 0/2 | 0/2 | 0.556 | 31.95 |
| soft120_24 | 0/2 | 0/2 | 0.344 | 32.15 |
| soft120_32 | 0/2 | 0/2 | 37.996 | 32.35 |
| soft500_16 | 2/2 | 0/2 | 8.855 | 9.82 |
| soft500_24 | 2/2 | 0/2 | 2.089 | 12.25 |
| soft500_32 | 1/2 | 0/2 | 86.600 | 12.12 |
| all2000_24 | 0/2 | 0/2 | 50.287 | 0.54 |
| base | 2/2 | 2/2 | 0.645 | 1.31 |
| hard28 | 2/2 | 2/2 | 0.616 | 0.21 |
| hard30 | 2/2 | 2/2 | 0.647 | 0.23 |
| inner1000_16 | 0/2 | 0/2 | 6.687 | 2.28 |
| inner1000_24 | 0/2 | 0/2 | 2.069 | 1.14 |
| inner2000_16 | 0/2 | 0/2 | 5.864 | 0.98 |
| inner2000_24 | 0/2 | 0/2 | 2.489 | 0.55 |
| inner500_24 | 0/2 | 0/2 | 6.095 | 15.74 |

`base` uses the original mixed mimic configuration. `passive` clears authored
follower drive gains. `softF_I` uses frequency F and I iterations on all mimic
joints; `innerF_I` tunes only inner followers. `hardI` uses hard coupling at I
iterations. The baseline appears in both fresh cohorts.

The authored right-outer-joint drive initially looked suspicious: damping 5000
with a 5 N m force cap. Runtime readback showed PhysX already suppresses that
drive for the mimic follower: live damping and effort cap are zero. Clearing the
authored gains yields exactly matching trajectories, contacts and task events.
It therefore provides a control, not a new optimization. Authored USD values
alone do not establish the live actuator behavior.

## Cross-object results

| Trajectory | Original coupling / 32 | Hard coupling / 28 | Hard coupling / 30 |
| --- | ---: | ---: | ---: |
| Bowl 00 | 2/2 | 2/2 | 2/2 |
| Pail 01 | 2/2 | 2/2 | 2/2 |
| Pail 02 | 1/2 | 2/2 | 1/2 |
| Milk carton | 2/2 | 1/2 | 2/2 |
| Pudding-mix box | 2/2 | 2/2 | 2/2 |
| Total | 9/10 | 9/10 | 9/10 |

The identical aggregate totals conceal different problems:

**Hard 28, carton repeat 1:** the object is released and its sampled pose barely
changes, yet the original settling predicate never passes during the two-second
hold. Over the last 0.533 seconds, translation changes by only **0.072 mm** and
orientation by **0.035 degrees**. Reported angular speed still spikes, ending at
**0.725 rad/s**, above the task's **0.5 rad/s** threshold. This is a measured
predicate failure with a pose/velocity inconsistency, not proof of visibly large
object motion. The threshold was not relaxed to turn it into a pass.

**Hard 30, box repeat 0:** task success is true, peak reported overlap is only
0.409 mm, and linkage p95 is below 0.4 degrees. Nevertheless, the object loses all
sampled inner-finger contact **0.458 seconds before the requested opening**.
Its distance from the jaw midpoint changes by **49.8 mm while airborne**, compared
with less than 0.8 mm for the baseline repeats. The event log records grasp loss
at step 90; the first open command is zero-based row 97. The box falls into the
bin and subsequently receives terminal success. This is a real grasp-retention
regression that terminal-success, penetration and linkage checks alone would miss.

Hard 28 improves the second pail attempt in this small sample, while hard 30
matches its aggregate success count with a different failed repeat. These
observations reinforce that solver counts and changed coupling are not monotonic
success improvements, and that replay repeat order/contact caches matter.

## Latency interpretation

The target-removed cost diagnostic compares original mimic coupling against
removing all five mimic schemas. At the same sampled 1695 MHz GPU clock, mean
command work over source rows 20–99 changes from **74.58 to 57.80 ms**, and the
physics region from **66.10 to 49.34 ms**. This supports the earlier GPU profile's
finding that mimic constraints are expensive. The uncoupled variant does not
represent a functioning gripper and is not a valid task solution or collection
profile.

All grasp-screening and broader qualification runs include per-substep contact
readback. Their timings remain diagnostic; they are not uninstrumented throughput
claims. Neither faster candidate passed the broader gates, so the planned fresh-
process throughput phase was not run. A lower count's attractive timing does not
compensate for a new failure or a weaker grasp.

The next larger performance experiment should preserve coupling and contact
quality while changing engine execution or a coherently calibrated robot model.
Removing coupling, arbitrarily softening the fingers, or accepting a terminal
success after an unrequested drop would not establish that improvement.

## Postprocessing implication

The proposed [success-validation plan](postprocess_success_validation_plan.md)
now distinguishes terminal continuation from complete action replay under an
explicit target profile. It also records trajectory-quality flags independently
from the terminal goal. The box case demonstrates why endpoint continuation
cannot verify earlier grasp retention. The carton case motivates retaining both
pose drift and reported velocities in the evidence. The checker remains proposed;
this investigation did not add success-validation behavior to postprocessing.

## Provenance and artifacts

All 38 processes exited normally. Source recording and production benchmark
hashes match their starting values. The source robot USD hash remains
`f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41`.
Runtime robot mass/COM/inertia and live actuator drive properties match the
baseline; the experiments deliberately vary mimic compliance in memory. The
inner-only scope is checked against the original outer coupling identity.
State arrays are finite and all measured render counts are zero. No videos
were generated.

- [First-cohort commands and source hashes](../output/omy_diagnostics/20261027_articulation_latency/manifest.json)
- [First-cohort measurements](../output/omy_diagnostics/20261027_articulation_latency/analysis.json)
- [Refinement and cross-object commands](../output/omy_diagnostics/20261027_articulation_refinement/manifest.json)
- [Refinement and cross-object measurements](../output/omy_diagnostics/20261027_articulation_refinement/analysis.json)
- [Qualification decisions](../output/omy_diagnostics/20261027_articulation_refinement/decision.json)
- [Carton and box failure analysis](../output/omy_diagnostics/20261027_articulation_refinement/failure_analysis.json)
- [Configuration, source and state verification](../output/omy_diagnostics/20261027_articulation_latency/verification.json)

Both artifact directories retain executable launch/analysis scripts, exact harness
patches, configurations, diagnostic HDF5 recordings, joint/object/contact arrays,
NVML memory/clock samples and logs. The refinement also records each mimic's
gearing, compliance and reference-joint relationships.

## Primary references

NVIDIA documents mimic coupling, compliant frequency/damping tuning and the
competition between drives, contacts and mimic constraints in its
[articulation guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance)
and [stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html).
The [PhysX API](https://nvidia-omniverse.github.io/PhysX/physx/latest/_api_build/classPxArticulationReducedCoordinate.html)
defines zero natural frequency/damping as a hard mimic constraint. Online docs
can include features newer than the installed patch; only settings supported by
this installation were used here. The measured outcomes above determine local
qualification, rather than the existence of a documented tuning option.
