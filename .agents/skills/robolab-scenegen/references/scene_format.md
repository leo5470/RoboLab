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

The standard base's table is a dynamic rigid body, 0.70 m tall (Z −0.70 to 0).
Its `GroundPlane` is authored at Z = −0.697, so the table starts 3 mm inside the
ground. PhysX pushes it up 3 mm at the first step, together with everything on
it. Measured in simulation: table root Z −0.350 → −0.347. In a new scene layer,
author `GroundPlane`'s translate at the measured table bottom (−0.700 for the
standard table). Leave `base_empty.usda` unchanged unless the user asks, because
existing scenes and their evidence depend on it. Recheck if the table or its
height changes.

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
attributes with the declared precision; adding a duplicate translate/orient/scale
operation raises an authoring error.

**Preserve any rotation the payload root authors.** Assets can fix their up-axis
with a root op such as `rotateXYZ`, `rotateX`, `rotateZYX`, `orient`, or `transform`.
Writing `xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]`
with a yaw-only `orient` silently drops that op from evaluation. For example,
`assets/objects/hot3d/computer_mouse.usd` authors `rotateXYZ = (90, 0, 0)` to lie
flat. With it discarded, every v1 Simple-PnP mouse started on its side (11 cm tall
instead of 4.2 cm) and toppled. Before writing the pose, read the root's
`GetOrderedXformOps()`. Then either:

- keep the existing rotation op in the order after the new yaw, for example
  `["xformOp:translate", "xformOp:orient", "xformOp:rotateXYZ", "xformOp:scale"]`
  with `orient` as the yaw only; or
- author a single `orient` equal to yaw ⊗ (the root's existing rotation), and
  remove the old rotation op from the order.

The first form keeps the asset's correction visible. In both forms, yaw is about
world Z, applied after the asset's own rotation.

For yaw in **degrees**, use `angle = math.radians(yaw) / 2` then quaternion
`(cos(angle), 0, 0, sin(angle))`. A yaw-only orientation keeps the asset's resting
orientation only when the asset's own root rotation is preserved as above. It
does not establish that an opening faces upward. After writing, check each object's
composed local up-axis and transformed dimensions against its intended resting
pose. A height that matches a horizontal catalog dimension indicates a dropped
rotation.

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
    clearance=0.0,  # start in contact; the helper's default is 0.002
)
```

This equals `old_root_z + actual_support_top_z - bottom_z`, which puts the
transformed bottom on the support. It works for bottom-origin and
centered-origin assets.

Start in contact only after checking the collision geometry. Collision prims
must not extend below the visual bottom. `convexDecomposition` hulls are cooked
at load time, so the authored check cannot cover them; measure start-up motion in
simulation. In a 2026-11-02 test with the corrected ground plane, contact starts
left the table motionless and moved objects 0.01–0.22 mm in total. A 2 mm start
dropped them about 2 mm. Use a small positive clearance only when a measured
collision bottom requires it, and record the reason. Apply the world displacement in the
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
Check that every root rotation the payload authored is still in the evaluated op
order, and that each object's composed up-axis matches its intended resting pose.
Check that `GroundPlane` sits at the table's measured bottom.
Visual/render bounds are conservative envelopes; unusual collision proxies can
extend farther. Inspect collision geometry before physics qualification.
Follow [validation.md](validation.md) for what each gate does and does not prove.
