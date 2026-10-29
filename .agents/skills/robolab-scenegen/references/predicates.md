# Predicate Reference

Predicates are constraints that define how objects should be placed in a scene. Solvers propose (x, y, z, yaw) positions; verify complete outputs and the composed scene independently.

## Predicate JSON Format

Most predicates use `type` and `object`; `place-in` uses an `objects` list and `container`:

```json
{"type": "place-on-base", "object": "bowl", "x": 0.55, "y": 0.0}
```

## Spatial Predicates (2D)

These determine x, y, and yaw on the table surface.

| Type | Description | Required params | Optional params |
|------|-------------|-----------------|-----------------|
| `place-on-base` | Place on table at given position | `object` | `x`, `y`, `yaw` |
| `left-of` | Place to the left (+Y) of reference | `object`, `reference` | `distance` (default 0.1m) |
| `right-of` | Place to the right (-Y) of reference | `object`, `reference` | `distance` |
| `front-of` | Place in front (+X) of reference | `object`, `reference` | `distance` |
| `back-of` | Place behind (-X) reference | `object`, `reference` | `distance` |
| `random-rot` | Apply random yaw rotation | `object` | |
| `facing-left` | Orient facing +Y | `object` | |
| `facing-right` | Orient facing -Y | `object` | |
| `facing-front` | Orient facing +X | `object` | |
| `facing-back` | Orient facing -X | `object` | |
| `align-left` | Align left edge with reference | `object`, `reference` | |
| `align-right` | Align right edge with reference | `object`, `reference` | |
| `align-front` | Align front edge with reference | `object`, `reference` | |
| `align-back` | Align back edge with reference | `object`, `reference` | |
| `align-center-lr` | Center left-right with reference | `object`, `reference` | |
| `align-center-fb` | Center front-back with reference | `object`, `reference` | |

## Physical Predicates (3D)

These handle stacking and containment.

| Type | Description | Required params | Optional params |
|------|-------------|-----------------|-----------------|
| `place-on` | Stack on top of another object | `object`, `support` | `support_ratio`, `stability`, `position` |
| `place-in` | Place inside a container | `objects` (list), `container` | |
| `place-anywhere` | Place anywhere on table (solver decides) | `object` | |

## Common Patterns

### Simple table arrangement
```json
{
  "objects": [{"name": "bowl"}, {"name": "banana"}, {"name": "mug"}],
  "predicates": [
    {"type": "place-on-base", "object": "bowl", "x": 0.55, "y": 0.0},
    {"type": "random-rot", "object": "bowl"},
    {"type": "place-on-base", "object": "banana", "x": 0.40, "y": -0.20},
    {"type": "random-rot", "object": "banana"},
    {"type": "place-on-base", "object": "mug", "x": 0.70, "y": 0.20},
    {"type": "random-rot", "object": "mug"}
  ]
}
```

### Relative positioning
```json
{
  "objects": [{"name": "plate"}, {"name": "fork"}, {"name": "knife"}],
  "predicates": [
    {"type": "place-on-base", "object": "plate", "x": 0.55, "y": 0.0},
    {"type": "left-of", "object": "fork", "reference": "plate", "distance": 0.15},
    {"type": "right-of", "object": "knife", "reference": "plate", "distance": 0.15}
  ]
}
```

### Stacking
```json
{
  "objects": [{"name": "red_block"}, {"name": "blue_block"}],
  "predicates": [
    {"type": "place-on-base", "object": "red_block", "x": 0.55, "y": 0.0},
    {"type": "place-on", "object": "blue_block", "support": "red_block"}
  ]
}
```

### Object in container
```json
{
  "objects": [{"name": "bowl"}, {"name": "banana"}, {"name": "apple"}],
  "predicates": [
    {"type": "place-on-base", "object": "bowl", "x": 0.55, "y": 0.0},
    {"type": "place-in", "objects": ["banana", "apple"], "container": "bowl"}
  ]
}
```

## Solver Pipeline

1. **Parse**: `parse_predicates_from_dict()` converts JSON dicts into typed `Predicate` objects
2. **Spatial solve**: `SpatialSolver.solve()` resolves 2D positions (x, y, yaw) from spatial predicates, with collision avoidance
3. **Physical solve**: `PhysicalSolver.solve()` proposes stacking/containment poses from physical predicates; it does not fill Z for every `place-on-base` object
4. **Grammar feedback**: `FeedbackSystem.generate_grammar_feedback()` checks spatial completeness but skips physical predicates; separately check every object has finite x/y/z and a valid orientation

These planning modules live in `robolab/scene_gen/llm_scene_gen/` and can run with numpy/scipy without IsaacSim. That mode is not a physics validation. Actual USD geometry inspection additionally requires `pxr`; dynamic qualification requires the simulator.

Check success flags from both solver calls. Supply measured scaled asset dimensions, plus `local_bounds` metadata where supported. Account for root pivot offsets; the spatial solver assumes centered geometry. Compute table-only Z using actual transformed bottoms, preserve physically solved Z, and validate final requested relations and unintended intersections. Relative/alignment predicates should be checked against the final poses because collision resolution can move objects. Follow [validation.md](validation.md) before accepting a scene.
