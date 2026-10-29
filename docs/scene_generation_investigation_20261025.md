# Simple-PnP setup failures: 2026-10-25

The retained scene library still contains unsafe starts. The clearest cause is
that generation allocated space for a 0.23 × 0.17 m bin, while the composed
`grey_bin` payload at scale 0.007 measures **0.42 × 0.28 × 0.105 m**. Correcting
object height during collection cannot repair overlapping XY placements.

## Recorded examples

| Case | Geometry and recorded outcome |
| --- | --- |
| `SimplePnp242PlasticpailA0115Task` | The pail and bin's projected bounding envelopes overlap by a minimum separating-axis depth of 32.3 mm. Preparation timed out at 5 s with both bodies unsupported by the table; the pail was still moving at 0.115 m/s and 1.064 rad/s. |
| `SimplePnp242PlasticpailA0138Task` | A second retained pail layout has overlapping bounding envelopes (40.8 mm). No corresponding collection outcome was established in this investigation. |
| `SimplePnp037ChocolatePuddingMix00Task` | Initial envelopes overlap (56.4 mm). The stored preparation report nevertheless says `ready` at 0.8 s after the box changed its local-up direction by about 90.1°. Its final root was near the table edge at x=0.239 m, y=0.404 m. |
| `SimplePnp294CheezIt10Task` | Initial envelopes overlap (101.1 mm); preparation timed out with the target unsupported and rotating at 2.185 rad/s. |

Envelope overlap is a conservative geometry warning, not an exact measurement
of mesh penetration. The recorded pail and Cheez-It preparation failures provide
independent evidence of bad starts; they do not establish that every flagged
layout fails physics.

Source reports:

- [Pail 15 preparation](../output/omy_teleop/20261024_134851_plasticpail_a01/SimplePnp242PlasticpailA0115Task/part_000/preparation_latest.json)
- [Pudding 00 preparation](../output/omy_teleop/20261023_212243_chocolate_pudding_mix/SimplePnp037ChocolatePuddingMix00Task/part_000/preparation_latest.json)
- [Cheez-It 10 preparation](../output/omy_teleop/20261023_230626_cheez_it/SimplePnp294CheezIt10Task/part_000/preparation_latest.json)

## How the problems escaped

1. **Assumed geometry.** `assets/scenes/simple_pnp/manifest.json` explicitly records
   the smaller `assumed_solver_dims`. Planning and the layout viewer can look
   safe while the loaded payload occupies much more space.
2. **Wrong pivot assumption.** The scene-generation skill's example writes
   `dims[2] / 2 + 0.002` as root height, discards solved Z, and never checks
   `phys_success`. That height only works for centered assets. The pail's origin
   is at its bottom: its authored bottom is 63.9 mm above the table. OMY's existing
   preparation corrects it to 2 mm, but the authored scene remains airborne for
   consumers that do not use this preparation path.
3. **Incomplete spatial checks.** The solver used `max(width, depth)/2` as a
   collision radius, which does not enclose rectangular corners. It also clamped
   positions to table bounds after checking collisions and then returned success
   without checking the changed positions. Its bounds check always returned true.
4. **Settling was treated as validation.** `settle_scenes.py` exports after 300
   updates without verifying support, tipping, table bounds, or grasp access.
   The collector's previous gate was better, but checked only support and recent
   motion. A box resting on its side could pass. Its preflight tested the object
   center against table bounds, not the full footprint or object/bin clearance.
5. **Historical filtering deliberately retained geometry warnings.** The retained
   manifest states that geometry warnings alone did not exclude tasks. Filtering
   observed falls in one run does not certify the remaining layouts for another
   robot, physics profile, or corrected starting height. Task success predicates
   concern the final placement; they do not validate the starting scene.

The original batch scene-generation program is not present under `tools/`; the
committed task generator consumed the scene manifest without geometry checks.
The findings above distinguish direct evidence in the current manifests/scenes
from defects in the available generation guidance and solver.

## Full static audit

[Machine-readable results](../output/scene_generation_audit_20261025/static_audit.json)
contain every layout's composed bounds, bottom gaps, footprint separation, scene
hash, and shared USD asset hashes. All 6,754 retained scenes were read successfully.

| Check | Scenes flagged |
| --- | ---: |
| Overlapping target/bin bounding footprints | 989 |
| Less than 20 mm footprint clearance, including overlaps | 1,584 |
| Less than 10 mm table-edge clearance / overhang | 49 |
| Authored geometry bottom outside −1 to +5 mm of table top | 2,721 |
| Of the height flags: below table | 129 |
| Of the height flags: above table | 2,592 |
| Any static issue (union) | 3,771 |

Categories overlap. These counts are repair/review candidates, not a failure
rate, and the 2,983 unflagged scenes have not been certified graspable.

Reproduce in a Python environment with `pxr` and numpy available:

```bash
python tools/audit_simple_pnp_scenes.py \
  --report output/scene_generation_audit_20261025/static_audit.json
```

Exit 1 means issues were found; no scene or task is modified. Use
`--pattern '*plasticpail_a01*'` for a focused audit.

## Safeguards implemented

- Shared geometry checks use transformed local bounds including pivot, scale,
  and orientation. Separate Simple-PnP objects require 20 mm projected clearance
  and complete footprints at least 10 mm inside the actual table.
- OMY preflights the entire layout before teleporting any object or advancing
  preparation physics. It rechecks geometry during preparation and rejects
  changes in local-up direction above 15° from the authored orientation, even
  if the final pose is stationary. Existing support and stability checks remain.
  Intentional non-Simple-PnP stacking/containment is outside this flat-layout rule.
- The preparation contract is version 2; the existing session compatibility
  checks distinguish it from older collection sessions. New checks apply to new
  collections, not retroactively to recorded data or policy evaluation paths.
- The spatial solver encloses corners with a conservative circumscribed circle,
  checks collisions after clamping, and refuses impossible/fixed out-of-bounds
  layouts. Accurate asset geometry is still required; the solver cannot repair
  incorrect dimensions supplied by a caller.
- New task generation audits composed geometry for every input scene before
  writing any task. `--update-existing` remains a migration of task success
  criteria and does not claim to recertify existing scenes.

The generated scenes, manifests, existing tasks, and recorded datasets have not
been bulk rewritten. Rejecting a bad setup prevents new collection attempts;
repairing the library is a separate regeneration and qualification step.

## Required generation/qualification process

1. Resolve the actual payloads and scales. Keep local min/max bounds and pivot
   offsets, not dimensions alone; hash the assets used. Plan with the actual bin.
2. Generate or resample XY layouts with full rotated footprint and approach
   clearance inside the robot's reachable workspace. Compute root Z as
   `old_root_z + table_top_z + clearance - transformed_bottom_z`.
   Check both solver return values and preserve solved poses.
3. Run the static audit on every layout. Reject or resample failed layouts before
   physics. Never rely on physics to push overlapping objects apart.
4. In the intended robot/physics configuration, check robot/start collisions,
   then settle and hold without actions for several seconds across repeated
   resets. Require table support, upright bin, bounded movement from the intended
   start, no tipping, and sustained low velocity. Save accepted settled poses;
   reload those saved scenes and repeat the checks.
5. Check actual grasp feasibility for the selected gripper: available opening,
   collision-free approach and closing, reachable pregrasp/grasp/lift poses, and
   a short lift trial. A 20 mm geometry margin is a screening threshold, not a
   guarantee that fingers fit. Keep this result separate from policy success.
6. Publish only qualified layouts, with task/scene/asset hashes, simulator and
   robot profile, reset seeds, checks, and failure reasons. Requalify when assets,
   scale, robot, physics settings, or preparation logic change.

The scene-generation skill's former centered-origin example and the fixed-duration
settler must not be used as acceptance criteria. The revised
[scene skill](../.agents/skills/robolab-scenegen/SKILL.md) and
[task skill](../.agents/skills/robolab-taskgen/SKILL.md) carry the replacement workflow;
their [validation reference](../.agents/skills/robolab-scenegen/references/validation.md)
distinguishes static, physics, and robot-specific evidence.

Validation completed initially: 105 unit/collection regressions and 3 composed-USD
integration tests passed. GPU queries failed inside the restricted sandbox at
that stage. Subsequent host checks found usable GPUs; the
[2026-10-26 simulation trials](scene_generation_trials_20261026.md) now provide
physical starts, saved-scene reloads, grasp, task-behavior, and rendering evidence
for the repaired pail and two-block candidates, with explicit physics profiles.
