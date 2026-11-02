# Simple-PnP v2 generation plan

Status: planning reference; no v2 scenes or tasks have been generated or qualified by this plan.

## Scope and target count

Generate **6,800 scenes and 6,800 matching tasks**: 50 new, distinct layouts for each of the **136 retained catalog entries** in the v1 scene manifest.

- Keep excluded objects excluded. Use the retained manifest as the object allowlist, identifying assets by catalog index and path rather than display name alone.
- The exclusion list is `excluded_targets` in the v1 scene manifest (176 entries): 151 marked `does not fit safely beside the grey-bin destination`, 12 bin targets (the bin is reserved as the destination), 7 user-excluded targets, 5 fixtures, and 1 oversized asset whose scale was replaced. `robolab/tasks/single_pnp/exclusions.json` records the excluded v1 tasks and target patterns. The fit exclusions were recorded alongside the incorrect 23 × 17 × 13 cm solver bin; the original program is unavailable, so their exact basis is not established. The actual bin is larger, which does not make re-admission more plausible. Re-admitting any excluded entry is a separate, explicit decision outside this plan.
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
| Root heights assumed centered pivots, leaving some objects airborne or embedded (for example, the milk jug in `pnp_230_milkjug_a01_00` is authored 88 mm above the table). The `dims[2] / 2 + 0.002` formula comes from the upstream scene-generation skill (`28b9313b`). The v1 batch applied it without checking asset pivots. | Compute supported root height from transformed geometry bottoms and actual support height. Start objects and the bin in contact with the tabletop (0 mm clearance). |
| v1 scenes replaced each payload root's transform-op order with `[translate, orient, scale]` and a yaw-only `orient`, discarding rotations the asset root authors. `computer_mouse.usd` authors `rotateXYZ = (90, 0, 0)` to lie flat. In v1 it therefore starts on its side, about 11 cm tall instead of 4.2 cm, and topples. This likely explains all 20 of its `fell_below_table_in_launch_audit` exclusions, which are 20 of the 46. Of the 137 retained payloads (including the bin), only the mouse has an up-tilting root rotation. `spatula_08` and `spatula_12` lose 4.1° and 1.7° of yaw. | Compose the placement yaw with the payload root's existing rotation: world orientation = yaw ⊗ root rotation. Do not overwrite it. After authoring, check each object's composed up-axis and transformed dimensions against its intended resting pose. Measure handle directions in the composed frame the scene actually uses. |
| Upstream `assets/scenes/base_empty.usda` places `GroundPlane` at z = −0.697, while the dynamic 0.70 m table spans z −0.70 to 0. At the first physics step the table rises 3 mm out of the ground and carries everything on it. Measured: table root z −0.350 → −0.347. | In each v2 scene layer, author `GroundPlane` at the measured table bottom (−0.700). Do not edit `base_empty.usda`, which is shared with v1 and all other tasks. |
| Target/bin footprints overlapped or extended too close to table edges. | Audit complete transformed footprints with at least 20 mm pair clearance and 10 mm edge clearance. Increase spacing for the selected gripper. |
| Solver success did not establish valid final scene geometry. | Check solver results, then independently audit the authored USD, including complete poses, payload resolution, and inherited relative paths. |
| Falls, launches, or tipping could be hidden by eventual settling. | Observe contacts and motion throughout qualification; check total displacement from the intended start as well as final stability. |
| Stable scenes could still lack grasp access or reachability. | Qualify pregrasp, grasp, lift, and hold for a named robot/gripper separately from geometry and physics. |
| Historical launch filtering retained geometry warnings. | Require applicable geometry checks to pass before physics qualification; do not waive failures to reach the target count. |
| Task success conditions were not evidence of a valid starting scene. | Validate initial scenes independently and test task success/reset semantics against the exact scene and robot profile. |

The documented static audit flagged **3,771 of 6,754 retained v1 scenes** with at least one issue. These are screening findings, not a measured physics failure rate. They motivate regenerating the full layout set.

The 46 layouts removed solely for `fell_below_table_in_launch_audit` involved 17 retained entries: 20 computer-mouse layouts, seven tomato-soup-can layouts, three Gregory's coffee-cup layouts, three utility-jug-a01 layouts, and one layout each for 13 other entries. Exclusion records establish the observed falls; the original launch logs were unavailable during planning, so detection thresholds are not established here. The computer-mouse falls very likely come from the discarded root rotation described in the table above. That has not been confirmed by re-running those layouts. The other 26 falls have no established cause.

## Workflow

### 1. Freeze inputs and acceptance profile

- Snapshot the retained object allowlist, actual payload paths, scales, and asset hashes.
- Pin the v1 bin payload and dependency hashes and verify v2 uses the same asset/configuration and scale, allowing only placement-pose differences. Treat any unexpected bin asset/configuration change as an input mismatch requiring resolution before qualification.
- Snapshot v1 placements for each retained catalog entry, including archived excluded layouts. `output/backups/simple_pnp_before_origin_20261017_152537/` holds all 7,200 original v1 layouts: the retained 6,754, the 46 launch-filtered layouts, and the excluded targets. Its manifest coordinates matched the current manifest in the scenes checked. The backup scenes' relative payload paths do not resolve from the backup directory. To measure their footprint centers, compose them from a copy at the original directory depth. Record reference scene hashes and any remaining coverage gaps for the novelty gate.
- Read the current base scene and catalog; verify asset identities and repaired substitutions against the retained scene payloads.
- Measure target, bin, table, mount, and relevant obstacle geometry from composed USD, including pivot offsets.
- **Usable area (decided 2026-11-02):** the measured tabletop is 0.70 × 1.00 m (X 0.197–0.897, Y −0.50 to 0.50). Use it with the 10 mm edge clearance, plus reach constraints for the selected robot. The v1 `table_bounds` [0.30, 0.80] × [−0.40, 0.40] is not the support boundary. Reach constraints:
  - The target's grasp point must be reachable top-down at grasp height and at lift height (0.20 m tool height).
  - The bin center must be reachable at release height: 0.20 m, or 0.35 m for targets taller than 18 cm.
  - Use a conservative reach map, requiring every neighboring grid cell to be reachable.
  - A kinematic reach map was computed with `tools/simple_pnp_v2/reach_map.py` from `franka_robotiq_2f_85_flattened.usd`, with the tool point 15 mm inside the fingertip. It covered the full table width up to X ≈ 0.60 m, narrowed to about |Y| ≤ 0.10 m at X = 0.80 m near table height, and reached nothing at X ≥ 0.85 m. In v1, 232 of 6,754 layouts placed the target or bin center outside this map.
  - The map ignores arm/table/bin collisions and covers top-down approaches only. It is a placement prefilter; the grasp gate decides feasibility. Freeze the map, its tool offset, and its hash with the other inputs.
- Preserve the existing released-and-resting placement semantics and episode duration.
- The project `.venv` used for the 2026-11-02 tests has Isaac Sim 5.0.0 and Isaac Lab 2.2.0. Record the versions actually used for qualification.
- Select and record the intended robot/gripper, starting pose, workspace, simulator version, and physics settings before dynamic qualification.
- Check the available project environment for USD inspection and Isaac execution. Unavailable dependencies leave their gates pending.
- For handle-visibility entries (see [Handle visibility](#handle-visibility-overshoulderleft)), freeze the `OverShoulderLeftCameraCfg` pose, intrinsics, and convention from `robolab/variations/camera.py`, the robot start pose, and per-entry handle records measured from the actual retained payloads.

**Robot profile (decided 2026-11-02):** OMY collection's `OmyDroidCfg`. This is the DROID Franka with a Robotiq 2F-85 (`assets/robots/franka_robotiq_2f_85_flattened.usd`), based at the world origin, with the DROID reset joint pose and the OMY execution profile. The camera is `OverShoulderLeftCameraCfg`. Record the simulator version and physics settings when dynamic qualification runs.

### 2. Pilot all retained entries

Generate one candidate layout per retained catalog entry before expanding to 50 layouts each.

Prioritize the computer mouse (with its root rotation preserved, to confirm the fix), then the tomato soup can, coffee cup, and utility jug associated with historical falls whose causes are unknown, plus the pail and large objects implicated in setup problems. Establish whether each entry admits a valid arrangement with the actual bin geometry and selected robot profile. Carry unresolved entries forward explicitly.

Test the novelty thresholds below during the pilot and assess whether the remaining placement space can support 50 distinct layouts per entry. One passing pilot layout does not establish capacity for all 50. Report shortages rather than silently relaxing thresholds.

### 3. Generate distinct layouts with bounded retries

- Vary target/bin XY placement and target yaw using recorded deterministic seeds. Keep the bin at yaw 0, as in every v1 layout (long 42 cm side along X). Only the bin's XY position and support height change.
- Sample from computed feasible sets rather than uniformly over the table. First compute the bin-center region allowed by table support, edge clearance, and v1 novelty, and sample the bin from it. Then compute the target region allowed by the sampled bin, footprint clearance, novelty, and handle rules, and sample the target from it. A rough estimate with uniform sampling put the median per-try acceptance near 0.1 and left some entries unlikely to resolve within 20 tries. With feasible-set sampling, an unresolved slot indicates a real shortage, not sampler inefficiency.
- Preserve requested objects, scales, and clearances.
- Compute heights using transformed bounds and `supported_root_height`; do not substitute half the catalog height for the asset's actual pivot.
- **Start in contact (decided 2026-11-02):** place each target's and the bin's transformed bottom at the measured tabletop height (clearance 0). Author the scene's `GroundPlane` at the measured table bottom so the table does not move at startup. Evidence:
  - Across all 136 retained assets and the bin, no authored collision shape extends below its visual bottom; the largest difference is 0.83 mm above it.
  - In OMY-profile tests on copies of five v1 layouts with the ground plane at −0.700 (milk jug, blue block, utility jug, Gregory's coffee cup, white packer bottle), starting at 0.01 mm left the table motionless. Objects moved 0.01–0.22 mm in total, peak speed was ≤ 0.01 m/s, and all reached `ready` in 0.533 s.
  - A 2 mm start produced a ~2 mm drop. With the upstream ground plane, the table rose 3 mm instead.
  - The authored collision check does not cover hulls PhysX cooks at load time (`convexDecomposition`). The physics gate must still measure start-up motion for every layout.
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

Apply a separate diversity check among accepted v2 layouts for the same entry. As a proposed minimum, reject a candidate when both its target and bin XY positions are within 3 cm of those in an already accepted v2 layout; rotation alone must not defeat this duplicate check. This rule is weaker than the v1 gate: a layout that moves only its bin by 3 cm counts as distinct. Decide during the pilot whether to require the target and offset distances as well. Record the choice either way.

Record novelty status, thresholds, reference-set hash and coverage, nearest reference layout IDs, and minimum measured distances. Freeze thresholds after pilot review. If the strict rule leaves too little usable space, report the constraint conflict and unresolved slots; do not weaken it automatically to reach 6,800.

The novelty guarantee applies only to the v1 placements represented in the reference set. Claiming separation from every originally generated v1 placement requires recovering all relevant archived poses. Missing archives must be reported as incomplete historical coverage, even when all available-reference comparisons pass.

#### Handle visibility (OverShoulderLeft)

For retained entries with a handle, the handle must be visible in the initial image from `OverShoulderLeftCameraCfg` (position `(0.05, 0.57, 0.66)`), with the robot at its frozen start pose. No other camera is a requirement. Across the planning area (X 0.30–0.80 m, Y −0.35 to 0.35 m), the horizontal bearing from an object to this camera ranges from about 105° to 164°, so the required yaw depends on each layout's XY position.

**Handle records.** For each entry, measure the handle direction in the asset frame (degrees, 0° = +X, counter-clockwise, Z up) and extract a handle point set from the actual retained payload. The directions below are preliminary mesh-outline estimates from `tools/simple_pnp_v2/handle_directions.py`. Confirm each with a render before freezing it. Each record holds the payload path and hash, the method, the direction, the point set, and the confirmation image.

| Class | Entries (catalog index, preliminary handle direction) | Generation rule |
| --- | --- | --- |
| A: side handle | measuring_cups (23, ≈0°), measuring_cups_1 (24, ≈0°), ceramic_mug (61, ≈97°), coffee_pot (63, ≈185°), HOT3D mug (73, ≈185°), HOT3D pitcher (74, ≈190°), milkjug_a01/a02/a03 (230/231/232, ≈270°), utilityjug_a01/a02 (283/284, ≈270°), YCB mug (301, ≈180°), YCB pitcher (303, ≈222°) | Yaw targeting plus occlusion check |
| B: top, bail, or upright handle | potato_masher (75), lunchbag (91), plasticjerrican_a/a01/a02 (237/238/239), plasticpail_a01/a02 (242/243), screwtoppail_a01 (251), squarepail_a01 (279) | Occlusion check. If the render shows a handle resting to one side, move the entry to class A. The jerrican direction estimates were unreliable and need inspection. |
| C: long utensil handle lying flat | spoon/spoon_1/spoon_2 (29/30/31), measuring_spoon (25), fork_big/fork_small (220/221), spatula_01 and spatula_03–spatula_19 (258, 260–276), spoon_big/spoon_small (277/278), scissors (304) | Occlusion check only |

These 49 entries (650 + 450 + 1,350 = 2,450 requested slots) are the scope. gregorys_coffee_cup (90) showed no handle lobe; confirm visually before leaving it out. Bottles and the remaining retained entries have no handle. Handled assets in the exclusion list stay excluded: the hammers (13–21), ladle (22), salad_tongs (26), serving_spoon(s) (27–28), megaphone (72), spatula (80), wooden_spoons (83), case_a01 (124), closedheadpail_a01 (140), liquidscrewtoppail_a01/a02 (227–228), metalfstylecan_a01 (229), moppingbucket_b01 (233), plasticjerrican_a03/a04 (240–241), plasticpail_a03/a04 (244–245), screwtoppail_a02–a04 (252–254), spatula_02 (259), squarepail_a02 (280), utilitybucket_a01/a02 (281–282), and utilityjug_a03 (285).

**Class A yaw.** Choose the target XY first, then set

```
yaw = bearing(target footprint center → camera) − handle_direction_local + δ,   δ ∈ [−θ_gen, +θ_gen]
```

with δ drawn from the recorded seed. For example, a VOMP jug at (0.55, 0) gives yaw ≈ 131° + 90° = 221°. Run the footprint audit after setting yaw, because pivots are off-center. The VOMP jug bodies sit 2–4 cm from their pivots.

**Pre-physics screen (all classes).** Reject a candidate when:

- for class A, the angle between the world handle direction and the camera bearing exceeds θ_screen;
- a ray from any of a sampled subset of handle points to the camera intersects the transformed bin envelope or the target's own body. This matters because the 10.5 cm bin can hide the handle of a 17–28 cm jug.

Proposed values are θ_gen = 30° and θ_screen = 60°. Freeze them after the pilot. This narrows where the bin can sit, so the pilot must check whether 50 distinct layouts per entry still fit. Report shortages instead of relaxing the limits. Novelty stays XY-based; a yaw change alone still never counts as novel.

### 4. Qualify scenes

Keep geometry, physics, and grasp statuses independent: `passed`, `failed`, or `pending`.

**Geometry:** use the shared flat-table audit with both target and bin, checking resolved composition, full footprints, separation, edge clearance, and authored bottom gaps. The default accepted bottom-gap interval is −1 to +5 mm; generation targets 0 mm (contact). Also check that the authored `GroundPlane` height equals the measured table bottom. Bounding-envelope checks do not establish cavity validity, exact mesh penetration, or grasp feasibility.

**Physics:** apply the scene-generation skill's proposed ordinary-tabletop protocol to every candidate being qualified:

- One nominal run in a fresh environment plus perturbed runs, each observed for at least five simulated seconds. Perturbed runs apply a seeded perturbation to the authored start pose; proposed: four runs with ±2 mm XY and ±2° yaw, frozen after the pilot. A layout passes only if every run passes. Plain reset seeds are not evidence. In a 2026-11-02 test with the OMY profile, seeds 0, 1, and 2 produced identical post-reset states. Separate processes reproduced positions exactly. Successive resets within one process drifted 0.2–0.5 mm, even with the same seed, and this grew to 3–4 mm / 7–8° for an unstable layout. Run each qualification case in its own environment, or record its position in the reset sequence.
- Stationary arm and open gripper at the intended starting pose; check initial robot collisions.
- Observe contacts and poses per physics step where possible.
- Require support on the intended table, no unintended contacts, an upright destination, and a retained target pose suitable for pickup.
- Require a final continuous 0.5-second stable window.
- Proposed limits: at most 20 mm total translation, 15° change in local-up direction, 0.01 m/s linear speed, 0.5 rad/s angular speed, and stable-window drift within 1 mm / 0.5°.

Freeze these limits before qualification. Record measured maxima and failures; do not weaken limits after failures merely to qualify layouts. A fixed-duration settling export or an existing preparation `ready` result does not establish this full protocol.

**Grasp:** for the selected robot/gripper, verify reachable pregrasp/grasp/lift poses, finger access and opening, approach/closing clearance, and an actual short lift/hold trial. Record tested poses, finger contact mappings, sampling cadence, and outcomes. Leave grasp pending if the robot is unspecified or trials cannot run.

**Visibility (handle entries only):** use the simulated state after preparation and settling (see Saved pose below), with the robot at the frozen start pose. Render depth from a qualification copy of `OverShoulderLeftCameraCfg` with identical pose and intrinsics plus a depth output. The shared configuration produces RGB only and must not be edited. project the handle point set, and count the points whose rendered depth matches within tolerance. Require the handle inside the image and at least 50% of handle points visible, with a minimum pixel area to be set in the pilot. Also add a yaw-drift limit to the physics protocol for these entries, proposed ≤ 10° about the vertical axis, so that settling cannot turn the handle away. Recompute the handle direction from the final pose. Record the camera configuration, renderer, visible fraction, pixel area, and image paths. Visibility is a fourth status alongside geometry, physics, and grasp. The handle-toward-camera rule changes approach directions, so grasp evidence must come from these exact poses.

**Saved pose (decided 2026-11-02):** v2 scene files store the generated pose with objects in contact. Settled poses are not written back to the scenes. Run the physics, yaw-drift, and visibility gates on the simulated state after settling, which is the state OMY collection records as the episode start. Exported settled poses would depend on solver settings and require every gate to be rerun on the exported file. Do not use `settle_scenes.py` to produce v2 scenes.

After any saved pose, asset, scale, or layer change, invalidate affected evidence and re-audit/requalify the exact resulting scene. Bound simulator runtime and teardown, and persist partial results. In the 2026-11-02 tests, every Isaac process crashed at shutdown (exit 139) after writing its results. The qualification runner must write and flush each case's report before teardown. It must judge results from those reports, not from the process exit code, and still record teardown crashes.

### 5. Generate and validate tasks

- Generate one uniquely named task/class for each matching scene, using exact scene prim names and contact bodies.
- Use `object_resting_in_container` for success and the same function/parameters for `pick_and_place` full credit.
- Preserve the existing 0.5-second sustained resting requirement, 0.05 m/s linear threshold, 0.5 rad/s angular threshold, 1 mm position tolerance, and 0.5° rotation tolerance, with table contact excluded. Also write out `support_force_threshold = 0.01` and `max_container_tilt_deg = 15.0` in the v2 parameters. v1 inherits these from `object_resting_in_container`'s code defaults, so a later default change would silently alter v2 goals.
- Require release and bin support. Protrusion and stable rim-spanning placement remain accepted; this is not full-enclosure semantics.
- Retain `time_out`, 50-second episodes, and consistent default/vague/specific instructions.
- Compile generated source; in the intended Isaac environment, verify runtime imports, actual conditional signatures, scene references, and contact mappings.
- Check success false at the intended start, true after valid release and the stability interval, and false for representative invalid outcomes such as continued gripper contact, table-supported placement, or an unstable/tipped bin. Verify timer/reset behavior.
- Run relevant existing tests and verify discovery for the actual v2 folder. Keep pending drafts outside automatic task discovery. Automatic DROID registration scans only `DEFAULT_TASK_SUBFOLDERS = ['benchmark']`. Drafts in `robolab/tasks/single_pnp_v2_drafts/` therefore load only when named explicitly. They must live under `robolab/tasks/`, because `register_omy_envs` resolves task files only there. A draft under `output/` could not run in the OMY environment used for qualification. Move a task to `robolab/tasks/single_pnp_v2/` only after its required gates pass.

Initial-scene physics thresholds and task-goal thresholds serve different purposes; record both rather than treating one as validation of the other.

### 6. Publish artifacts and qualification counts

| Artifact | Proposed location |
| --- | --- |
| V2 scenes and manifest | `assets/scenes/simple_pnp_v2/` |
| Qualified task library | `robolab/tasks/single_pnp_v2/` |
| Task drafts with required gates pending | `robolab/tasks/single_pnp_v2_drafts/` |
| Validation reports and generation records | `output/simple_pnp_v2_validation/` |

`output/` is gitignored, so reports and drafts there are local only. Links from the committed v2 manifest would not resolve elsewhere. Before publishing, commit compact summaries (gate statuses, measured maxima, hashes) next to the manifest, or record where the full reports are archived.

For every requested slot, record catalog identity, payload paths/scales, generation seed, scene hash, composed asset/layer hashes, gate statuses, limits, reasons, and report links. Include v1 novelty and within-v2 diversity results and historical reference coverage. For handle entries, also record the handle class, handle-record hash, θ values, δ, predicted and final handle directions, and visibility status and measurements. A task's `specific` instruction may mention the handle only once visibility has passed. Dynamic reports must include the robot and simulator/physics profile and reset seeds.

Report requested, authored, passed, failed, and pending counts per object and for the whole library. Keep scene geometry/physics/grasp results separate from task syntax/runtime/goal checks. Register only tasks meeting the declared qualification requirements. Never report 6,800 ready tasks solely because 6,800 files exist.

## Execution machine

Implementation, qualification, and collection move to a machine with two RTX 4090 GPUs. It already has an environment identical to the planning machine's: the project `.venv` with Isaac Sim 5.0.0 and Isaac Lab 2.2.0. The 2026-11-02 measurements were taken on the planning machine's RTX 3090.

**Before the pilot,** reproduce the start-up physics findings on each 4090 with `tools/simple_pnp_v2/start_probe.py` (commands in `tools/simple_pnp_v2/README.md`):
- post-reset states identical across seeds and positions identical across processes;
- the ~0.2–0.5 mm drift between successive same-process resets;
- the 3 mm table rise with the upstream ground plane, and no table motion with the corrected ground plane;
- 0.01–0.22 mm of object motion from contact starts.

Run the same cases on both GPUs and compare them with `analyze_start_probe.py`. Record differences instead of assuming identical results. Every dynamic report records host, GPU model, and driver version. Because qualification and collection share this machine, evidence and use match. Evidence from another GPU model still needs a spot-check before reuse.

**GPU sharing:** qualification and collection compete for the same two GPUs. Schedule them apart, or assign each a GPU through `isolated_gpu_launch`. The untracked `robolab/core/omy_teleop/gpu_pool.py` maps `CUDA_VISIBLE_DEVICES` to host GPU indices, so it may be reusable for scheduling; it has not been reviewed.

**Compute budget:** on the RTX 3090, one physics run (preparation plus a 5.5 s hold) took 6–7 s of wall time, plus a few seconds per environment setup, with one environment per process. With 6,800 layouts and five runs each, that is roughly 60+ GPU-hours run sequentially, before grasp trials and visibility renders. Two GPUs roughly halve the wall time. The pilot measures 4090 throughput, how many headless OMY processes fit per 24 GB GPU, and how often nudged runs fail when the nominal run passes; the run count is set from that rate.

**Carry-over checklist:**
- Commit and push this plan, the scene-generation skill updates, and `tools/simple_pnp_v2/`.
- Commit the `.claude/skills/robolab-*` symlink retargeting to `.agents/skills/`. The committed links point to the removed `skills/` directory.
- Run `git lfs pull` (1,648 LFS files, including `assets/scenes/base_empty.usda`).
- Copy `output/backups/simple_pnp_before_origin_20261017_152537/` (143 MB). It is gitignored, and it is the novelty reference set.
- Check `bwrap` (required by `isolated_gpu_launch`), the NVIDIA driver (the planning machine runs 580.178.04), and `tools/simple_pnp_v2/usd_python.sh`. The wrapper locates Isaac's bundled USD libraries by path.

## Existing safeguards and remaining implementation

The updated skills already specify actual-geometry measurement, pivot-aware height placement, bounded candidate retries, independent geometry/physics/grasp evidence, exact-scene evidence handoff, task-goal semantics, and isolation of pending task drafts. Apply those requirements through the batch workflow rather than treating the skill text itself as executed validation.

Reuse the shared geometry checker and placement helpers. The existing task generator is hardcoded to v1 paths and preflights geometry only; v2 needs a reproducible batch scene generator, parameterized task generation, and qualification orchestration with resumable reports. The original batch scene-generation program is not present under `tools/`, according to the investigation.

**Environment reuse.** Qualification reuses the OMY collection environment, run through `isolated_gpu_launch`: `register_omy_envs`, `OmyDroidCfg`, the OMY performance profile, and `ScenePreparation`. The 2026-11-02 probes already used it. Using the consumer's own configuration is intended. Required before qualification:

1. `ScenePreparation` folder check and clearance (the two paragraphs below). Both block qualification, not only collection.
2. Add an option to `register_omy_envs` to supply the qualification camera with depth (see Visibility). It currently adds `OverShoulderLeftCameraCfg` as RGB only, with no override.
3. Keep v2 scenes in `assets/scenes/simple_pnp_v2/` and reference them as `simple_pnp_v2/<file>.usda`. `find_scene_file` matches by path suffix and returns the first depth-first match. Nesting v2 scenes under a directory named `simple_pnp` to satisfy item 1 could make v1 tasks load v2 files; fix item 1 instead.
4. Run standalone USD tools, including `tools/audit_simple_pnp_scenes.py`, with `tools/simple_pnp_v2/usd_python.sh`. Plain `.venv/bin/python` cannot import `pxr`. This avoids adding `usd-core` to `uv.lock`.
5. Grasp trials need a scripted pregrasp/grasp/lift driver. OMY's action space is joint targets. Whether the existing IK (`DroidIKActionCfg`) or MimicGen code can be reused has not been checked.

**Deferred: v1-only consumers.** OMY layout and collection code reads `robolab/tasks/single_pnp/manifest.json` and `assets/scenes/simple_pnp/manifest.json` (`robolab/core/omy_teleop/layouts.py`). Task directories are hardcoded to `single_pnp` in `collection_options.py`, `scripts/check_omy_preparation.py`, `scripts/check_resting_pnp.py`, and `scripts/benchmark_omy_layout_loading.py`. Before v2 is used for collection, add a v2 task manifest in the v1 format and make these paths configurable. Keep task filenames matching `pnp_<index>_<name>_<layout>_task.py`, and keep class names distinct from v1's.

`ScenePreparation` (`robolab/core/omy_teleop/preparation_runtime.py`) applies its flat-layout height correction, clearance preflight, and tilt rejection only when the scene's parent directory is named exactly `simple_pnp`. Scenes in `assets/scenes/simple_pnp_v2/` would silently skip all three checks. Fix this before qualification (environment-reuse item 1).

`ScenePreparation` also re-seats every object at a hard-coded `PreparationSettings.clearance_m = 0.002`, and `PreparationSettings` rejects a clearance of 0. Before qualification, make the clearance configurable and allow 0 (contact) for v2. Keep v1 collection behavior unchanged.

Upstream's `rest_offset = 0.01` and `contact_offset = 0.02` in `robolab/core/environments/base.py` are silently ignored. `PhysxCfg` has no such fields, and the `hasattr` guard skips them. Treat the effective offsets as the PhysX defaults, and do not rely on those lines.

## References

- [Scene-generation skill](../.agents/skills/robolab-scenegen/SKILL.md)
- [Scene validation and evidence contract](../.agents/skills/robolab-scenegen/references/validation.md)
- [Task-generation skill](../.agents/skills/robolab-taskgen/SKILL.md)
- [V1 setup-failure investigation](scene_generation_investigation_20261025.md)
- [V1 scene manifest](../assets/scenes/simple_pnp/manifest.json)
- [V1 task exclusions](../robolab/tasks/single_pnp/exclusions.json)
- [Existing task generator](../tools/generate_simple_pnp_tasks.py)
- [Shared scene audit](../tools/audit_simple_pnp_scenes.py)
- [V2 diagnostics (handle directions, collision bottoms, reach map, capacity estimate, start-up probe)](../tools/simple_pnp_v2/README.md)
