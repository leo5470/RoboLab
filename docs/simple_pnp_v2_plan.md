# Simple-PnP v2 generation plan

Status: planning reference; no v2 scenes or tasks have been generated or qualified by this plan.

## Scope and target count

Generate **6,800 scenes and 6,800 matching tasks**: 50 new, distinct layouts for each of the **136 retained catalog entries** in the v1 scene manifest.

- Keep excluded objects excluded. Use the retained manifest as the object allowlist, identifying assets by catalog index and path rather than display name alone.
- Preserve repaired asset substitutions and intended scales from the retained scenes.
- Regenerate layouts for all retained objects. The 46 launch-filtered layouts are replacement slots, not additional object exclusions.
- Use a bin identical to v1: prim name `grey_bin`, payload `assets/fixtures/grey_bin.usd`, and uniform root scale `(0.007, 0.007, 0.007)`. Preserve its geometry, appearance/materials, collision geometry, and physical properties; do not substitute, resize, or modify the shared bin asset.
- Only the bin's placement pose may change to satisfy novelty and valid table support. Measure its actual composed geometry for placement; the incorrect v1 solver dimensions are not the bin specification.
- Retain 50-second episodes.
- Preserve v1 files and place v2 artifacts in separate directories.

The current v1 library contains 6,754 layouts. Restoring 50 layouts for each retained entry gives 6,800 requested layouts. **This is a generation target, not a guaranteed qualified count.** Unresolved slots must remain visible.

## Known v1 issues and prevention

| Known issue | V2 prevention |
| --- | --- |
| The solver assumed a 23 × 17 × 13 cm bin; the composed bin at scale `0.007` measures approximately 42 × 28 × 10.5 cm. | Compose and measure actual payloads at their intended scales and orientations before placement. Preserve the bin scale. |
| Root heights assumed centered pivots, leaving some objects airborne or embedded. | Compute supported root height from transformed geometry bottoms and actual support height. Start at 2 mm clearance. |
| Target/bin footprints overlapped or extended too close to table edges. | Audit complete transformed footprints with at least 20 mm pair clearance and 10 mm edge clearance. Increase spacing for the selected gripper. |
| Solver success did not establish valid final scene geometry. | Check solver results, then independently audit the authored USD, including complete poses, payload resolution, and inherited relative paths. |
| Falls, launches, or tipping could be hidden by eventual settling. | Observe contacts and motion throughout qualification; check total displacement from the intended start as well as final stability. |
| Stable scenes could still lack grasp access or reachability. | Qualify pregrasp, grasp, lift, and hold for a named robot/gripper separately from geometry and physics. |
| Historical launch filtering retained geometry warnings. | Require applicable geometry checks to pass before physics qualification; do not waive failures to reach the target count. |
| Task success conditions were not evidence of a valid starting scene. | Validate initial scenes independently and test task success/reset semantics against the exact scene and robot profile. |

The documented static audit flagged **3,771 of 6,754 retained v1 scenes** with at least one issue. These are screening findings, not a measured physics failure rate. They motivate regenerating the full layout set.

The 46 layouts removed solely for `fell_below_table_in_launch_audit` involved 17 retained entries: 20 computer-mouse layouts, seven tomato-soup-can layouts, three Gregory's coffee-cup layouts, three utility-jug-a01 layouts, and one layout each for 13 other entries. Exclusion records establish the observed falls; the original launch logs were unavailable during planning, so per-layout causes and detection thresholds are not established here.

## Workflow

### 1. Freeze inputs and acceptance profile

- Snapshot the retained object allowlist, actual payload paths, scales, and asset hashes.
- Pin the v1 bin payload and dependency hashes and verify v2 uses the same asset/configuration and scale, allowing only placement-pose differences. Treat any unexpected bin asset/configuration change as an input mismatch requiring resolution before qualification.
- Snapshot v1 placements for each retained catalog entry, including archived excluded layouts where recoverable. Record reference scene hashes and missing archive coverage for the novelty gate.
- Read the current base scene and catalog; verify asset identities and repaired substitutions against the retained scene payloads.
- Measure target, bin, table, mount, and relevant obstacle geometry from composed USD, including pivot offsets.
- Preserve the existing released-and-resting placement semantics and episode duration.
- Select and record the intended robot/gripper, starting pose, workspace, simulator version, and physics settings before dynamic qualification.
- Check the available project environment for USD inspection and Isaac execution. Unavailable dependencies leave their gates pending.

**Open input:** the intended robot/gripper and execution profile. Geometry work can proceed independently, but robot-specific readiness cannot be claimed without this choice and its checks.

### 2. Pilot all retained entries

Generate one candidate layout per retained catalog entry before expanding to 50 layouts each.

Prioritize the computer mouse, tomato soup can, coffee cup, and utility jug associated with historical falls, plus the pail and large objects implicated in setup problems. Establish whether each entry admits a valid arrangement with the actual bin geometry and selected robot profile. Carry unresolved entries forward explicitly.

Test the novelty thresholds below during the pilot and assess whether the remaining placement space can support 50 distinct layouts per entry. One passing pilot layout does not establish capacity for all 50. Report shortages rather than silently relaxing thresholds.

### 3. Generate distinct layouts with bounded retries

- Vary target/bin XY placement and permitted yaw using recorded deterministic seeds.
- Preserve requested objects, scales, and clearances.
- Compute heights using transformed bounds and `supported_root_height`; do not substitute half the catalog height for the asset's actual pivot.
- Check both solver success results when solvers are used. Account for pivot offsets and independently validate final transforms.
- Try at most 20 deterministic candidates per requested layout by default.
- Record rejected candidates and measured reasons. After exhaustion, mark the slot unresolved and continue other layouts.
- Do not fill the quota by shrinking assets, reducing clearances, duplicating layouts, or silently dropping objects.

#### Placement novelty relative to v1

A different random seed does not guarantee different placements. Make novelty an explicit acceptance gate against **every recorded v1 layout for the same catalog entry**, including recovered archived layouts.

Compare target XY, bin XY, target-to-bin XY offset, and orientations. Use a common table coordinate frame and consistent geometry-based reference points, such as transformed footprint centers, so pivot or payload changes cannot create artificial novelty. Record yaw differences with object symmetries accounted for; rotation alone does not qualify a placement as new.

The proposed strict gate requires all three distances below to be **at least 3 cm for every matching v1 layout**:

- Euclidean distance between v2 and v1 target XY positions.
- Euclidean distance between v2 and v1 bin XY positions.
- Euclidean distance between their target-to-bin XY offset vectors.

Reject a candidate if any distance is below 3 cm for any reference layout. This intentionally requires both target and bin placements to change, as well as their relative arrangement. Evaluate this gate alongside geometry before expensive physics/grasp checks, and rerun it on the final saved poses after any settling or pose edits.

Apply a separate diversity check among accepted v2 layouts for the same entry. As a proposed minimum, reject a candidate when both its target and bin XY positions are within 3 cm of those in an already accepted v2 layout; rotation alone must not defeat this duplicate check.

Record novelty status, thresholds, reference-set hash and coverage, nearest reference layout IDs, and minimum measured distances. Freeze thresholds after pilot review. If the strict rule leaves too little usable space, report the constraint conflict and unresolved slots; do not weaken it automatically to reach 6,800.

The novelty guarantee applies only to the v1 placements represented in the reference set. Claiming separation from every originally generated v1 placement requires recovering all relevant archived poses. Missing archives must be reported as incomplete historical coverage, even when all available-reference comparisons pass.

### 4. Qualify scenes

Keep geometry, physics, and grasp statuses independent: `passed`, `failed`, or `pending`.

**Geometry:** use the shared flat-table audit with both target and bin, checking resolved composition, full footprints, separation, edge clearance, and authored bottom gaps. The default accepted bottom-gap interval is −1 to +5 mm; generation targets +2 mm. Bounding-envelope checks do not establish cavity validity, exact mesh penetration, or grasp feasibility.

**Physics:** apply the scene-generation skill's proposed ordinary-tabletop protocol to every candidate being qualified:

- Three reset seeds and at least five simulated seconds per reset.
- Stationary arm and open gripper at the intended starting pose; check initial robot collisions.
- Observe contacts and poses per physics step where possible.
- Require support on the intended table, no unintended contacts, an upright destination, and a retained target pose suitable for pickup.
- Require a final continuous 0.5-second stable window.
- Proposed limits: at most 20 mm total translation, 15° change in local-up direction, 0.01 m/s linear speed, 0.5 rad/s angular speed, and stable-window drift within 1 mm / 0.5°.

Freeze these limits before qualification. Record measured maxima and failures; do not weaken limits after failures merely to qualify layouts. A fixed-duration settling export or an existing preparation `ready` result does not establish this full protocol.

**Grasp:** for the selected robot/gripper, verify reachable pregrasp/grasp/lift poses, finger access and opening, approach/closing clearance, and an actual short lift/hold trial. Record tested poses, finger contact mappings, sampling cadence, and outcomes. Leave grasp pending if the robot is unspecified or trials cannot run.

After any saved pose, asset, scale, or layer change, invalidate affected evidence and re-audit/requalify the exact resulting scene. Bound simulator runtime and teardown, and persist partial results.

### 5. Generate and validate tasks

- Generate one uniquely named task/class for each matching scene, using exact scene prim names and contact bodies.
- Use `object_resting_in_container` for success and the same function/parameters for `pick_and_place` full credit.
- Preserve the existing 0.5-second sustained resting requirement, 0.05 m/s linear threshold, 0.5 rad/s angular threshold, 1 mm position tolerance, and 0.5° rotation tolerance, with table contact excluded.
- Require release and bin support. Protrusion and stable rim-spanning placement remain accepted; this is not full-enclosure semantics.
- Retain `time_out`, 50-second episodes, and consistent default/vague/specific instructions.
- Compile generated source; in the intended Isaac environment, verify runtime imports, actual conditional signatures, scene references, and contact mappings.
- Check success false at the intended start, true after valid release and the stability interval, and false for representative invalid outcomes such as continued gripper contact, table-supported placement, or an unstable/tipped bin. Verify timer/reset behavior.
- Run relevant existing tests and verify discovery for the actual v2 folder. Keep pending drafts outside automatic task discovery.

Initial-scene physics thresholds and task-goal thresholds serve different purposes; record both rather than treating one as validation of the other.

### 6. Publish artifacts and qualification counts

| Artifact | Proposed location |
| --- | --- |
| V2 scenes and manifest | `assets/scenes/simple_pnp_v2/` |
| Qualified task library | `robolab/tasks/single_pnp_v2/` |
| Task drafts with required gates pending | `output/task_drafts/simple_pnp_v2/` |
| Validation reports and generation records | `output/simple_pnp_v2_validation/` |

For every requested slot, record catalog identity, payload paths/scales, generation seed, scene hash, composed asset/layer hashes, gate statuses, limits, reasons, and report links. Include v1 novelty and within-v2 diversity results and historical reference coverage. Dynamic reports must include the robot and simulator/physics profile and reset seeds.

Report requested, authored, passed, failed, and pending counts per object and for the whole library. Keep scene geometry/physics/grasp results separate from task syntax/runtime/goal checks. Register only tasks meeting the declared qualification requirements. Never report 6,800 ready tasks solely because 6,800 files exist.

## Existing safeguards and remaining implementation

The updated skills already specify actual-geometry measurement, pivot-aware height placement, bounded candidate retries, independent geometry/physics/grasp evidence, exact-scene evidence handoff, task-goal semantics, and isolation of pending task drafts. Apply those requirements through the batch workflow rather than treating the skill text itself as executed validation.

Reuse the shared geometry checker and placement helpers. The existing task generator is hardcoded to v1 paths and preflights geometry only; v2 needs a reproducible batch scene generator, parameterized task generation, and qualification orchestration with resumable reports. The original batch scene-generation program is not present under `tools/`, according to the investigation.

## References

- [Scene-generation skill](../.agents/skills/robolab-scenegen/SKILL.md)
- [Scene validation and evidence contract](../.agents/skills/robolab-scenegen/references/validation.md)
- [Task-generation skill](../.agents/skills/robolab-taskgen/SKILL.md)
- [V1 setup-failure investigation](scene_generation_investigation_20261025.md)
- [V1 scene manifest](../assets/scenes/simple_pnp/manifest.json)
- [V1 task exclusions](../robolab/tasks/single_pnp/exclusions.json)
- [Existing task generator](../tools/generate_simple_pnp_tasks.py)
- [Shared scene audit](../tools/audit_simple_pnp_scenes.py)
