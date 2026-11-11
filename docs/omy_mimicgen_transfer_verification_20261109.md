**MimicGen transfer feedback verification — 2026-11-09**

The new feedback identifies a real omission: raising the phase boundary leaves both a potentially long fixed-duration connection and the source's remaining bin-relative carry. The proposed over-bin start cut and adaptive timing fit RoboLab's existing loop. They should be pilot switches before production generation. CPU evidence supports the design, but does not certify the transformed motion's clearance, IK or tracking.

No GPU, simulator, policy rollout or training was used. The generator and installed Isaac Lab were not modified. The [plan](omy_mimicgen_diversity_plan_20261108.md) was revised, and the [independent CPU audit](../output/omy_diversity_verification_20261109/transfer_audit.py) writes [per-source/per-target results](../output/omy_diversity_verification_20261109/transfer_audit.json). The supplied scripts were inspected, run and [archived unchanged](../output/omy_diversity_verification_20261109/feedback_scripts/).

**Speed claim: reproduced, with a timing correction**

`bridge_speed.py` reproduces the nominal table. For each human it computes the 95th percentile of object-root XY speed while root lift exceeds 2 cm; it then takes the median or maximum across humans. Its bridge lengths come from the previous CPU FK audit, minimizing over qualified sources at each target with zero boundary offset.

| Quantity | Fork | Pitcher |
|---|---:|---:|
| Human per-demo p95, median / maximum | 0.230 / 0.271 m/s | 0.342 / 0.430 m/s |
| Bridge length divided by nominal 1 s, median / p90 / maximum | 0.185 / 0.371 / 0.505 m/s | 0.598 / 0.900 / 0.984 m/s |
| Nominal bridge exceeds maximum human p95 | 15/50 | 29/50 |
| Length / median human p95, median / maximum | 0.81 / 2.20 s | 1.75 / 2.88 s |

The code has an endpoint-count detail: `mimicgen.py:257` requests 15 interpolation points at 15 Hz. Installed `isaaclab/utils/math.py:1860` increments that count, appends the endpoint, and `waypoint.py:252` removes only the initial pose. The executed bridge therefore has **16 commands, about 1.067 seconds**, for a full, uninterrupted bridge. Commanded XY speed is length × 15/16, not length / 1 s.

| Accounting for emitted commands | Fork | Pitcher |
|---|---:|---:|
| Bridge speed, median / p90 / maximum | 0.174 / 0.348 / 0.474 m/s | 0.560 / 0.844 / 0.922 m/s |
| Exceeds maximum human p95 | 11/50 | 29/50 |

The 29 nominally excessive pitcher cases are exactly the 29 negative-y targets. The timing correction does not change their count. This supports retiming; it does not establish a physical failure threshold. “Maximum human p95” is not the fastest instantaneous human motion or a validated safe speed.

The supplied speed mask does not check gripper closure. An independent pre-action-aligned, closed-gripper mask changes the human median p95 values only to 0.230/0.342 m/s. However, CPU FK of measured end-effector XYZ gives median p95 values of 0.263/0.306 m/s. The reference point and dimensions matter, especially when the held object rotates. Use phase-specific end-effector translation and angular limits, acceleration/jerk checks and the actual new path when retiming. The reported 1.75–2.88 s estimates are not validated durations for the redesigned transfer.

**Boundary claim: all 18 have candidate cuts**

`boundaries.py` reproduces the candidate lift and over-bin frames. Repeating the calculation using `mimicgen.source_arrays`, with the generator's pre-action alignment, confirms that all ten fork and eight pitcher humans have:

- A closed-gripper root lift of at least 13.5 cm.
- A later closed-gripper root center inside the bin footprint shrunk by 3 cm.
- Root lift at least 13.5 cm and closed gripper throughout that intervening recorded interval.

The candidate lift occurs 0.53–0.87 s after the 2 cm trigger for fork and 0.67–0.87 s for pitcher. Before that point, qualified-source lateral motion is 2.6–12.0 cm for fork and 6.3–12.6 cm for pitcher. Across all ten fork humans the minimum is 2.2 cm; the 2.6 cm minimum describes qualified sources. The supplied script uses post-action object states; its frame numbers cannot be copied directly into the generator's pre-action boundaries.

These findings remove a missing-recorded-lift-height prerequisite for these source sets. They do not prove a geometry-safe hand-off: root-height increase is not the minimum height of a rotating object's full shape, and root-center entry does not imply the complete footprint clears the opening. The source's early lateral motion can also become unsuitable after transformation to a new layout. Retain geometry-based rejection or an independently validated vertical extension for such cases; the generic missing-height fallback remains relevant to other objects.

**Over-bin cut: useful independently of raising the boundary**

The independent audit reconstructs full transformed end-effector command poses from the robot USD joint chain. It compares the following proposed cuts using currently qualified sources and authored v2 target poses:

1. Raise the first segment's end to the first 13.5 cm root-lift candidate and retain the adjacent source start for the bin segment.
2. Use the same raised end, but start the bin segment at the later closed-gripper over-bin candidate, omitting the source's intervening carry.

For each target, minimizing XY path length over eligible sources gives:

| Remaining commanded XY path from the raised cut to the over-bin waypoint | Fork median / p90 | Pitcher median / p90 |
|---|---:|---:|
| Raised hand-off, original bin-relative carry retained | 0.436 / 0.631 m | 0.760 / 1.064 m |
| Raised hand-off, direct connection to over-bin segment | 0.348 / 0.553 m | 0.435 / 0.651 m |

This measures bridge length plus the retained source carry to the same over-bin endpoint, not only bridge length. A direct connection can be longer than the old bridge alone because it replaces that remaining carry. Each aggregate independently chooses its best source and ignores IK, collisions, contact and tracking. The simulator pilot must compare the same source/target/seed to isolate the cut's effect.

The redesigned bridge still needs retiming: its maximum XY length is about 0.656 m for fork and 0.760 m for pitcher. It also needs angular checks. Choosing the source with the shortest direct XY bridge yields median wrist-orientation changes of about 46°/75° and p90 of 137°/132° respectively. A translation-only timing rule or center-point clearance check is insufficient. These values illustrate the tradeoff in choosing sources solely by distance; they do not prescribe those sources for execution.

The cut removes the requirement to replay the human's long carry direction. Thus reverse-direction pitcher source recovery need not block trying milestone 2 with the existing five qualified sources. Recovery still matters for grasp, approach and orientation coverage, and the early source lift prefix remains direction-dependent. Successful execution of reversed targets is still a pilot criterion.

**Implementation hooks: confirmed locally**

- RoboLab computes boundaries and calls `generate_trajectory` for each segment in `mimicgen.py:481–489`. The library slices source end-effector targets, gripper actions and reference poses using the supplied intervals. A gap between the two retained source segments can be represented locally. Add explicit validation instead of relying on the absence of later assertions.
- Source-specific pre-grasp starts can be supplied in that array. Changing the start also changes the source object-reference pose used by the transform; check the anchor and pre-action indexing.
- A synthetic lift can extend the waypoint list. It must execute before the next call uses its last commanded waypoint as the interpolation start.
- `data_generator.py:158` stores a reference to `env.cfg`; it does not reload the configuration there on every call. The interpolation count is read from that referenced object's subtask configuration when generating a segment. Mutating the nested count per segment works; replacing `env.cfg` wholesale is not equivalent. Restore temporary changes, and freeze the source before deriving its timing.
- The installed start-offset sampler does use the lower range value for both integer bounds, yielding a constant for every source. Avoid that option for per-source approach cuts.

No Isaac Lab fork is needed for these changes. The local SkillGen survey recommends a planner when collision avoidance between segments becomes limiting. Trying a geometrically checked direct connection first is consistent with that recommendation; it does not establish that straight connections avoid robot-link collisions on every target.

**Changes to the plan**

Step 2 now explicitly includes independently switchable approach cropping, higher lift cut, over-bin cut and translation/rotation-aware retiming. The reversed-carry pilot compares the over-bin cut on/off under matched timing; long-bridge retiming and approach cropping have separate checks. Absolute 0.75/1.0/1.25 s diversity variants have been removed.

The first PPS comparison is reduced to **two 50-demo datasets for granola on v2 scenes**: the local mixture and a profile with 50% of generated slots on explicit targets, holding generator, sources, control, scene builder and export fixed. The report's granola results are 40/50 base versus 30/50 at gamma 0.4 and 7/50 at gamma 0.8. Fork remains a positive-steering control (0→8/50); pitcher remains a generator stress case. Additional scene builders, target shares and larger datasets follow useful validation improvement rather than being initial prerequisites.

To reproduce the independent audit without a simulator:

```bash
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  tools/simple_pnp_v2/usd_python.sh output/omy_diversity_verification_20261109/transfer_audit.py
```

The two supplied-script outputs are saved as [bridge speeds](../output/omy_diversity_verification_20261109/bridge_speed_reproduced.txt) and [boundaries](../output/omy_diversity_verification_20261109/boundaries_reproduced.txt). The independent JSON records input-manifest hashes and all per-source path candidates, making its optimistic source selection explicit.
