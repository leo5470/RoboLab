# USD scene authoring

Scenes normally extend `assets/scenes/base_empty.usda` and use a `world` default
prim, Z up, and meter units. Inspect the actual base rather than reconstructing
its table, materials, ground, or mount from an old example. A robot-specific
environment can replace the mount; qualify with the intended environment.

## Frames and usable support

In the standard base, the table center is approximately (0.55, 0) and its top is
near Z=0. Front is +X, left is +Y, and right is −Y. The usual X=[0.30, 0.80],
Y=[−0.40, 0.40] region is a planning convention, not proof of support or reach.
Measure the composed tabletop and check the full footprint, not just the root
position. Robot reach depends on the robot configuration.

## Paths and layers

Use `os.path.relpath(asset_path, scene_path.parent)` for each asset path authored
in that layer. The scene can be nested arbitrarily; paths are relative to the
layer that owns them. Copying base text to a deeper directory requires rebasing
its existing payloads, references, and other relative asset paths as well as the
new objects' paths. Prefer USD layer APIs over global string replacement.

New object prims need unique scene names; repeat instances can use suffixes while
retaining the same payload. Task object names must match those scene prim names.

Author translation, quaternion orientation `(w, x, y, z)`, and intended scale in
`xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]`.
Payload roots often already author transform operations. Reuse their existing
attributes with the declared precision and set the operation order once; adding
a duplicate translate/orient/scale operation raises an authoring error.
For yaw in **degrees**, use `angle = math.radians(yaw) / 2` then quaternion
`(cos(angle), 0, 0, sin(angle))`. Identity orientation preserves the asset's
authored axes; it does not establish that its opening faces upward.

## Bounds and root height

`UsdGeom.BBoxCache.ComputeUntransformedBound(prim)` excludes the root transform
while retaining descendants. Transform its eight min/max corners with
`UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(...)` to inspect the composed
placement. Do not apply the root scale twice. Do not combine a bound that already
includes the root transform with a second root pose.

For an object intended directly on a horizontal support:

```python
from robolab.core.scenes.placement import supported_root_height

# All three coordinates are in the same world frame, in meters.
new_root_z = supported_root_height(
    old_root_z, transformed_geometry_bottom_z, actual_support_top_z,
    clearance=0.002,
)
```

This equals `old_root_z + actual_support_top_z + 0.002 - bottom_z`. It works for
bottom-origin and centered-origin assets. Apply the world displacement in the
appropriate parent frame if the object has a transformed parent. Recompose and
recheck the result. For a stack, use the intended support's surface. For a hollow
container, inspect its interior floor and walls; its outer bounding-box top is
the rim, and its volume does not prove a usable cavity.

Dimensions and a root position cannot establish stability. Preserve an intended
lying orientation when appropriate for an asset, and test that the object retains
that orientation while settling. An upright-container requirement is separate.

## Planning with the solvers

Read `robolab/scene_gen/llm_scene_gen/` for supported predicates. Supply measured
geometry/`local_bounds` metadata where supported, plus the actual payload paths.
Check return flags from both `SpatialSolver.solve` and `PhysicalSolver.solve`.
The spatial solver does not model pivot offsets; the physical solver includes
heuristic stacking/containment paths. Neither is an acceptance gate.

`place-on-base` determines XY/yaw; compute its support-aware Z explicitly.
Preserve physical-predicate Z when supplied and check it against geometry.
`FeedbackSystem.generate_grammar_feedback` skips completeness checks for physical
predicates, so independently require finite x/y/z and a valid orientation for
every object before export.

## Before simulation

Reopen the written file and check default prim, units, resolved payloads and
composition errors, complete transforms, support relationships, and clearance.
Visual/render bounds are conservative envelopes; unusual collision proxies can
extend farther. Inspect collision geometry before physics qualification.
Follow [validation.md](validation.md) for what each gate does and does not prove.
