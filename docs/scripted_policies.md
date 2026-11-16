# Scripted absolute-IK demonstration policy

`examples/collect_scripted_demos.py` is one collector for supported RoboLab
pick-and-place tasks. It uses privileged `WorldState` geometry and contact state
only to generate expert actions; recorded policy observations remain the configured
camera observations.

## Architecture and repository contracts

The data flow is:

```text
registered task cfg -> predicate extractor -> ScriptedTaskSpec
                    -> generic FSM -> grasp/placement strategies
                    -> robot-root absolute IK -> RoboLab recorder
```

Task modules are discovered under `robolab.constants.DEFAULT_TASK_SUBFOLDERS` by
`auto_register_droid_abs_ik_envs()`. The environment factory dynamically combines
the task scene, DROID robot, observations, `DroidIKActionCfg`, termination config,
and the task's list of `Subtask` objects, then registers the generated class as a
Gymnasium environment. A `Subtask` contains per-object condition ladders; elements
of `env_cfg.subtasks` are sequential stages, while each stage uses `all`, `any`, or
`choose` across its object groups.

The authoritative extractor mappings are:

| RoboLab predicate | Normalized relation | Multiplicity |
| --- | --- | --- |
| `object_in_container` | `inside` | `all`, `any`, `choose` |
| `object_on_top` | `on_surface` | `all`, `any`, `choose` |
| `object_outside_of_and_on_surface` | `outside_container_and_on_surface` | `all`, `any`, `choose` |
| `object_groups_in_containers` | grouped `inside` targets | all groups |
| `objects_placed_in_container_in_order` | staged `inside` targets | ordered all |

Unknown or ambiguous success predicates fail before environment execution and show
the discovered termination predicates/subtasks plus the `--task-spec` escape hatch.
Instruction text is never parsed. Explicit specs are checked first against the task's
contact entity list, then against live `WorldState.entities` after simulator creation.

## Action and frame convention

`DroidIKActionCfg` expects 8 floats per environment:

```text
[robot_x, robot_y, robot_z, qw, qx, qy, qz, gripper]
```

The pose is absolute in the robot-root frame and uses `wxyz` quaternions. The IK
body is the Robotiq `base_link`, not the fingertip and not the published
`eef_frame`. DROID's `frames` transformer publishes `gripper_base` and `eef_frame`;
`eef_frame` has `EEF_OFFSET_POS=(0,0,0)` and
`EEF_OFFSET_ROT=(0.5,-0.5,0.5,-0.5)` relative to `base_link`. Candidate poses are
expressed as `eef_frame` poses, converted by full SE(3) from env-local/world axes to
robot root, then converted to the IK body with
`T_robot_base = T_robot_eef * inverse(T_base_eef)`.

`WorldState.get_pose()` defaults to env-local position (world position minus the
environment origin), but retains world orientation. The collector never assumes
that this is robot-root coordinates: robot translation and rotation are both
inverted. Long pose changes are Cartesian-rate-limited.

The final scalar follows DROID's binary convention: `0` opens and `1` closes.
`PreStepActionsRecorder` records the submitted absolute-IK action tensor. It does
not record IK-produced joint targets. Post-step scene state and the DROID
`base_link` pose are recorded separately.

`env.step()` returns `(obs, reward, terminated, truncated, info)` tensors/dicts.
`RobolabEnv` freezes each finished environment, records non-timeout termination as
success, and exports it. The collector sets `EXPORT_SUCCEEDED_ONLY` by default;
`--save-failures` selects `EXPORT_ALL`.

## Active task coverage

- `BananaInBowlTask`: object-to-container extraction, container-fit generation,
  and `object_in_container` verification (including its configured contact rule).
- `BananaOnPlateTask`: surface-footprint generation, occupied-region checks, stable
  support, and `object_on_top` verification.
- `KeyboardOutOfBinTask`: source extraction, a packaged thin-plate profile that corrects
  the asset's broad authored bounds to its physical 429 x 129 x 20 mm envelope, edge candidates, full-OBB rim clearance, vertical extraction, table free-space
  placement, contact/lift grasp verification, and alternate-candidate retry.

The keyboard name appears in task data/tests only. There is no keyboard task class
check or keyboard execution branch in the FSM.

Dry-run the actual registered configs without creating an environment:

```bash
python examples/collect_scripted_demos.py --task BananaInBowlTask \
  --num-episodes 20 --output-dir output/scripted/banana_bowl --headless --dry-run-spec
python examples/collect_scripted_demos.py --task BananaOnPlateTask \
  --num-episodes 20 --output-dir output/scripted/banana_plate --headless --dry-run-spec
python examples/collect_scripted_demos.py --task KeyboardOutOfBinTask \
  --num-episodes 20 --output-dir output/scripted/keyboard --headless --dry-run-spec
```

Remove `--dry-run-spec` to collect. Optional explicit data examples are in
`configs/scripted/example_task_spec.yaml` and
`configs/scripted/example_grasp_profiles.yaml`; pass them with `--task-spec` and
`--grasp-profiles` respectively.

## Motion and verification

The FSM uses warmup, target/grasp selection, pregrasp, approach, close, verification
lift, transport lift, source clearance, placement selection, preplace, lower,
release, stable-placement verification, retreat, retry, success, and failure states.
Motion states advance on live pose error and have timeouts. A grasp requires contact,
measurable object motion during a small lift, and a stable EEF-object relative
transform. Source transfer remains vertical until the object's current complete OBB
is above the source's current top bound. Placement requires detachment, low linear
and angular velocity, and the same task predicate semantics used for termination.

Packaged profiles are loaded by default and may be overridden with `--grasp-profiles`.
Geometry fallback recognizes thin plates, elongated objects, box-like objects,
approximately round/cylindrical objects, small graspable objects, and objects too
large for an external 85 mm pinch. This is shape classification only: handles, rims,
and other semantic parts require asset/category/USD/exact-name profile data.

## Main tuning parameters

Defaults are in `ScriptedPolicyConfig`, `StateMachineConfig`,
`GraspGeneratorConfig`, and `PlacementGeneratorConfig`:

| Parameter | Default | Physical meaning |
| --- | ---: | --- |
| `position_tolerance` | 0.012 m | EEF waypoint acceptance radius |
| `orientation_tolerance_rad` | 8 deg | EEF waypoint angular tolerance |
| `max_translation_per_step` | 0.012 m | Cartesian command rate limit |
| `max_rotation_per_step_rad` | 8 deg | angular command rate limit |
| `verification_lift` | 0.04 m | lift used to prove object attachment |
| `transport_lift` | 0.10 m | normal post-grasp vertical lift |
| `source_clearance_margin` | 0.035 m | object-bottom clearance over a source rim |
| `max_aperture` | 0.085 m | Robotiq external-pinch opening limit |
| `tool_reach` | 0.1628 m | base-flange to fingertip distance estimate |
| `tilted_approach_downward_fraction` | 0.45 | downward/lateral ratio for oblique edge approaches |
| `surface_edge_margin` | 0.025 m | footprint margin inside a surface boundary |
| `container_wall_margin` | 0.025 m | conservative container-wall allowance |
| `preplace_height` | 0.12 m | vertical staging height above release |
| `minimum_reach` / `maximum_reach` | 0.35 / 0.72 m | preferred robot-root radial range used for ranking |
| `hard_minimum_reach` / `hard_maximum_reach` | 0.20 / 0.90 m | hard robot-root radial candidate limits |
| `motion_timeout_steps` | 180 | feedback steps allowed per normal motion |
| `max_grasp_attempts` | 4 | distinct candidates tried per target/episode |
| `max_placement_attempts` | 3 | placement candidates tried per target |

These are conservative geometry heuristics, not a global collision planner. Concave
container interior geometry, articulated destinations, semantic-part grasps,
in-hand reorientation, stacking, deformables, and task predicates outside the table
above remain unsupported. Straight segments are checked against inflated nearby
object bounds, but narrow clutter may need a profile or explicit spec and tuning.

