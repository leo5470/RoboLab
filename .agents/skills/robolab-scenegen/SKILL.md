---
name: robolab-scenegen
description: >
  Create or repair RoboLab USD scenes for tabletop robot manipulation from
  natural language, with geometry checks and explicit simulation validation status.
license: CC-BY-NC-4.0
metadata:
  author: nvidia
  version: "2.0.0"
---

# Scene generation

Create a USD arrangement that matches the requested task and distinguish what
has been authored from what has been validated. Scenes contain objects and
fixtures; reachability and grasp qualification belong to a named robot profile.

## Resolve the request

- Reuse the requested scene name, objects, output directory, and prior choices.
  Ask only for missing information that changes the arrangement or acceptance
  criteria. If no location is specified or established, use
  `assets/scenes/generated/<descriptive_name>.usda` and state the choice.
- Preserve existing files unless the user requested their repair/update. For a
  new scene, choose an unused name. Use isolated candidates while iterating.
- Read `assets/scenes/base_empty.usda` and `assets/objects/object_catalog.json`.
  Match assets by path as well as name; catalog names need not be unique. Count
  the current catalog instead of quoting a fixed inventory. Fixtures and user
  assets can be outside the object catalog.
- Read [scene_format.md](references/scene_format.md) before authoring and
  [validation.md](references/validation.md) before validating. Consult
  [object_catalog_guide.md](references/object_catalog_guide.md) for asset identity,
  and [predicates.md](references/predicates.md) when using the predicate solvers.

## Establish geometry before placement

Use a Python environment that can import `pxr.Usd`, `pxr.UsdGeom`, and numpy for
USD inspection. Isaac Sim is needed for physics, not static geometry. Check the
available project environment; do not assume Docker, a `CLAUDE.md`, or a globally
installed simulator. If a dependency is unavailable, keep its validation stage
pending and continue the independent work that can be verified.

Compose actual payloads with the intended root scales and orientations. Include
containers, supports, and other obstacles. Resolve missing payloads/composition
errors before accepting geometry. Record local min/max bounds and transforms;
catalog dimensions alone do not describe pivots or the final scene footprint.
Preserve asset scales unless a size change was requested, and remeasure after any
scale or payload change. Do not shrink objects or destinations just to pass checks.

## Generate candidates

1. Identify intended relationships: separate objects on one table, stacking,
   containment, shelf levels, or another support arrangement. Use the requested
   relationships to decide which contacts are expected. Do not apply a blanket
   separation rule to an intended stack or filled container.
2. For separate tabletop objects, place their complete transformed footprints
   inside the usable support area and robot workspace if specified. Use actual
   surface-to-surface spacing. Start with the shared validator's 20 mm pair and
   10 mm edge clearances; these are flat-table screening defaults, not gripper
   dimensions or guarantees of reachability. Increase clearance for the selected
   gripper's approach; do not reduce it merely to get a pass.
3. If using `SpatialSolver`, supply dimensions from the actual scaled assets.
   Its centered circumscribed-circle model is conservative only for correctly
   centered geometry: account for pivot offsets or use direct geometry-based
   placement. Validate the final composed transforms regardless of solver success.
   Check both solvers' `(success, message)` results; do not continue a failed solve.
4. Compute support height from geometry. For a separate tabletop object use
   `new_root_z = old_root_z + support_top_z + clearance - world_bottom_z`, with
   2 mm initial clearance by default. Use
   `robolab.core.scenes.placement.supported_root_height` and transformed bounds.
   For stacking, preserve the physical solver's solved Z and verify the intended
   support. A container rim is not its interior floor. Do not replace solved Z
   with `dims[2] / 2`, and do not assume `PhysicalSolver` fills table-only Z values.
5. Write complete poses with degree-to-radian conversion for yaw. Rebase **all**
   inherited relative asset paths when moving a copied base layer into a different
   directory, including the table, mount, and other referenced assets.

The physical solver and grammar feedback are planning aids. Inspect their actual
outputs: finite complete poses, all requested objects, and satisfied relationships.
An instance created without a simulator does not perform a physics validation.

## Validate and iterate

Run the appropriate geometry gate on the authored candidate before any settling.
For separate tabletop objects, the repository audit accepts a single scene:

```bash
python tools/audit_simple_pnp_scenes.py \
  --scene assets/scenes/generated/pail_pickup.usda \
  --objects plasticpail_a01 grey_bin \
  --report output/pail_pickup.geometry.json
```

Use the actual path and **all** tabletop object/container prim names. This command
requires an axis-aligned rectangular support, named `table` unless `--support`
is provided. It does not validate stacking, cavities, robot collisions, or grasps.
For those arrangements follow the relation-specific checks in the validation reference.

On failure, use the measured reason to change XY placement, yaw, or an invalid
height. Re-audit after every change. Keep the requested object set, relationships,
scales, and clearances. Use a bounded search (by default up to 20 deterministic
candidates per requested layout), retaining the seed and reasons. If no candidate
fits, report the constraint conflict rather than removing requested objects or
exporting the last failed candidate as ready. Continue other independent layouts.

After a geometry pass, run available physics and robot-specific checks described
in [validation.md](references/validation.md). Keep simulation qualification pending
if it cannot run. Preserve original scenes during repair; install a repaired
candidate only to the validation level that was actually achieved and report it.

## Deliver

Provide the scene path, object/asset summary, and linked validation reports. State
**geometry**, **physics**, and **grasp** results separately as passed, failed, or
pending, with the checked scope/profile and reasons. A screenshot is useful for
review, but is not a substitute for measurements. Do not call an untested scene
stable, graspable, or ready for collection.

For a task definition, use `robolab-taskgen` and pass along the exact scene and
validation evidence. Do not create an unrelated task or launch a policy merely
because a scene was requested.
