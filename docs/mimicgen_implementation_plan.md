**Implementation plan: teleop demonstrations → Isaac Lab Mimic → broader task coverage → SkillGen**

Written 2026-09-14. Implementation started 2026-09-16; see the [progress and review gates](mimicgen_progress.md). Synthetic generation is not implemented yet. This plan builds on the [MimicGen/SkillGen survey](mimicgen_skillgen_survey.md) and the subsequent inspection of benchmark task declarations. Proposed paths and schemas below are design targets, not existing interfaces.

The intended outcome is a reusable DROID integration that expands successful RoboLab teleop demonstrations into validated synthetic demonstrations. Add task coverage through explicit conversion rules and annotations, reusing the robot adapter, generation runner, and export pipeline. Establish Isaac Lab Mimic first; add SkillGen when measured transit failures justify its planning and modeling cost.

**Starting evidence and constraints**

| Inspected item | Finding | Implementation consequence |
|---|---|---|
| Simulator stack | Installed Isaac Lab 2.2.0; example keyboard recording reports Isaac Sim 5.0.0.0 | Establish the Mimic baseline on the collection stack. Pin code and dependencies. |
| Keyboard recording | `output/keyboard_demos/banana_wifi/keyboard_demos.hdf5` contains one successful 579-action episode; three suffixed files were empty | Validate episodes and success flags; do not count files as demonstrations. |
| Recording payload | Initial/per-step states, relative IK actions, EE/root poses, pre-step proprioception; no Mimic metadata | Prepare annotations and controller targets from existing demos. Images can be deferred. |
| Benchmark task census | All 120 directly defined benchmark Task subclasses declare subtasks; 95 use only standard pick/place helpers, 25 include explicit Subtask objects | Potential conversion coverage, not 120 verified generation-ready tasks. Regenerate this census during implementation. |
| Task logic | Seven tasks have multiple scoring stages; ten explicitly use `any`, five `choose` | Resolve selections and execution order before constructing a linear Mimic specification. Counts overlap. |
| Local environment | RobolabEnv freezes completed environments and exports episodes | Design generation-specific reset, success, and recording behavior. |
| SkillGen | Present in inspected upstream main, absent from installed 2.2.0 source | Use a separate compatible stack and validate old recordings there. |

The survey inspected upstream main at `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`. Do not mix those scripts with installed 2.2.0 APIs without compatibility work. Read upstream scripts from the selected implementation revision and record that revision in every run manifest.

**Architecture and ownership of semantics**

```text
Task definition + source HDF5 + saved environment configuration
    → task/data audit
    → normalized task intent + verified success predicate
    → demonstrated object selection and execution sequence
    → object-relative segments + reviewed boundary signals
    → version-specific Isaac Lab Mimic configuration
    → DROID generation environment and runner
    → successful episodes + provenance + failure report
    → optional image materialization and learner-specific export
```

Reuse these components:

| Existing code | Reuse |
|---|---|
| [Task definitions](../robolab/core/task/task.py), [conditionals](../robolab/core/task/conditionals.py) | Object relationships, completion constraints, candidate boundary predicates. |
| [Scripted task specification](../robolab/policies/scripted/task_spec.py), [predicate extractors](../robolab/policies/scripted/task_spec_extractors.py) | Predicate-to-target mappings, entity validation, serialization patterns, unsupported-task diagnostics. |
| [DROID configuration](../robolab/robots/droid.py) | Articulation, controlled link, action scale, gripper commands, camera calibration. |
| [Replay configuration](../robolab/core/replay/env_config.py), [state restoration](../robolab/core/replay/scene_state.py) | Source-scene reconstruction and replay drift measurement. |
| [Recorder terms](../robolab/core/events/basic_recorders.py), [recorder manager](../robolab/core/logging/recorder_manager.py) | HDF5 schema and bounded-memory recording where compatible with generation lifecycle. |
| [Image materialization](../robolab/core/replay/materialize.py) | Render accepted generated trajectories after state-only generation. |

Start by adapting the existing extractor's supported mappings. It already treats termination predicates as authoritative and rejects unsupported or ambiguous cases. Its ScriptedTaskSpec does not describe source trajectories, segment frames, annotations, or arbitrary task graphs. Keep a separate Mimic representation; extract shared predicate normalization into a neutral task module only when both consumers need it, preserving scripted-policy behavior.

Use actual success predicates to determine successful episodes. Treat scoring subtasks as candidate decomposition information. When instructions, subtasks, and success predicates disagree, report a conflict and exclude the task from automatic generation until corrected or explicitly resolved in a reviewed task override. Do not silently rewrite benchmark semantics.

**Proposed implementation layout**

| Proposed path | Responsibility |
|---|---|
| `robolab/datagen/spec.py` | Serializable task intent, resolved sequence, segment and provenance schemas; simulator-free. |
| `robolab/datagen/task_conversion.py` | Predicate mappings, scoring/success checks, custom overrides. |
| `robolab/datagen/annotation.py` | Source validation, event extraction, sequence resolution, boundary review, Mimic payload creation. |
| `robolab/datagen/envs/droid_mimic_env.py` | DROID Mimic API and generation lifecycle. |
| `robolab/datagen/envs/registration.py` | Separate Mimic registrations and config construction. |
| `robolab/datagen/runner.py` | Version-specific upstream integration, reset/export ownership, metrics. |
| `robolab/datagen/export.py` | RoboLab dataset contract, provenance, image and policy-export compatibility. |
| `scripts/datagen/{audit_tasks,prepare_demos,generate_demos}.py` | Thin launchers; import Isaac Sim only when needed. |
| `configs/datagen/` | Reviewed task overrides and generation presets. |
| `tests/unit/test_datagen_*.py` and simulator integration tests | Semantics, timing, frames, recording, and lifecycle. |

Paths may change to fit repository conventions. Keep Mimic/cuRobo imports lazy so auditing and normal RoboLab use do not require planner installation.

**Specification contract**

Keep three representations:

1. **Task intent:** entities, object/destination relations, completion logic (`all`, `any`, `choose`), cardinality, required ordering, success predicate/options, and randomization limits.
2. **Resolved sequence:** selected objects and concrete action order demonstrated in a compatible source cohort. Include repeated visits and intermediate placements. An unordered task still needs a concrete sequence for each generation run.
3. **Segment specification:** unique segment ID, manipulated object, reference frame, boundary signals, allowed source cohort, offsets, interpolation/noise settings, and optional SkillGen attachment/start-signal information.

Illustrative grasp segment:

```json
{
  "segment_id": "grasp_banana_0",
  "object": "banana",
  "object_ref": "banana",
  "term_signal": "banana_grasped_0",
  "boundary_rule": "verified_grasp_and_lift",
  "source_cohort": "banana_then_cube",
  "boundary_offset_steps": [0, 0],
  "action_noise": 0.0
}
```

The placement segment uses `object_ref="bowl"`. IDs must distinguish repeated grasps of the same object. Validate references, EEF names, signal shapes, segment order, finite transforms, and source availability before generation. Convert this representation to the pinned upstream SubTaskConfig through one compatibility layer.

**Phase 0 — audit source data, task semantics, and dependencies**

Deliver an audit command writing JSON/CSV reports without modifying source tasks or datasets. Use static parsing for the initial census; runtime normalization can occur separately when task imports need the simulator.

For each task, report recognized predicates, objects/destinations, scoring stages, completion logic, conflicts, missing entities, and one of: convertible candidate, override required, semantic conflict, or unsupported. Conversion readiness, annotation readiness, and execution validation are separate statuses.

For each source dataset, verify nonempty successful episodes, consistent action/state lengths, finite values, state schema, saved config/assets, simulator versions, and action/frame conventions. Preserve the original and write preparation results to a new file.

Known audit cases:

- [FoodPackingByColorTask](../robolab/tasks/benchmark/food_packing_by_color_task.py): inspected subtasks reverse mustard/coffee destinations relative to termination and omit sugar_box required by termination. Report the exact conflicting pairs.
- [BlockStackingOrderAgnosticTask](../robolab/tasks/benchmark/block_stacking_order_agnostic_task.py): scoring stages describe any two, then three, then four stacked blocks, not a required object order.
- [object_grabbed](../robolab/core/task/conditionals.py): gripper contact does not establish a secure grasp.
- `choose(K)`: [predicate logic](../robolab/core/task/predicate_logic.py) checks exactly K true results, while [subtask tracking](../robolab/core/task/conditionals_state_machine.py) accepts at least K completed groups. Preserve authoritative success semantics, including objects already satisfying the goal initially, and test this difference.
- Multiple non-timeout termination terms: require a verified success aggregation rule; not every non-timeout termination necessarily means success.

Pin the baseline Mimic stack, locate matching scripts, and document optional dependency installation. Keep the collection environment intact while evaluating another stack.

Exit condition: machine-readable audit, reproducible dependency/config manifests, and a verified BananaInBowl source episode selected for the first end-to-end run.

**Phase 1 — implement the shared DROID Mimic environment**

Build an environment derived from ManagerBasedRLMimicEnv using the source task's scene and DROID configuration. Register it separately from evaluation environments. The launcher must import those registrations and load the recording's resolved scene configuration.

Implement `get_robot_eef_pose`, `action_to_target_eef_pose`, `target_eef_pose_to_action`, `actions_to_gripper_actions`, `get_object_poses`, and boundary access using the pinned API. Expose an explicit generation success callable. Reuse world-state/contact facilities and reset their caches/history correctly.

Resolve these contracts:

- Current relative IK controls Robotiq `base_link`, uses scale 0.5, and accepts root-axis translations and rotation-vector deltas. Read the configured scale rather than assuming it for every dataset.
- The published `eef_frame` has a rotation offset inactive in the current relative-IK action config. Validate against the saved action config.
- Express EEF and objects in one documented frame; cover translated/rotated roots and multiple environment origins.
- Preserve gripper 0=open, 1=closed using Robotiq joints and geometry.
- Give generation ownership of resets and export. Do not inherit freeze-on-completion unchanged or let evaluation export finalize episodes prematurely.
- Per-environment reset must clear predicates, recorder state, counters, and annotations without disturbing other environments.

Exit condition: action-target round trips, correct physical response, successful source replay, and two sequential single-environment trials with independent recordings. Use float32-appropriate tolerances for pure pose conversion; establish separate physical tracking/replay tolerances from measured baseline behavior.

**Phase 2 — prepare and annotate existing demonstrations**

Start with action replay on the original stack. Restore initial state and replay actions while collecting Mimic metadata and measuring drift against recorded object/EEF states. Verify gripper behavior and final success; report failures per episode.

If replay does not reproduce a usable trajectory, add state-restoration preparation using the replay helpers. This mode recovers geometry but cannot assume reconstructed contact forces or predicate history. Use manual annotation or validated geometry-based rules when contact events cannot be recovered.

Honor temporal alignment:

```text
pre-action state at t=0: initial_state
pre-action state at t>0: states[t-1]
action at t:            actions[t]
post-action state:      states[t]
post-action EE pose:    ee_pose[t]
```

Infer commanded targets from pre-action poses and the actual controller action. Measured post-action poses are not commanded targets. Obtain the initial EEF pose from the restored articulation or forward kinematics. Verify the last action/state survives terminal handling.

Write `obs/datagen_info` with object poses, EEF poses, target EEF poses, and named termination signals according to the pinned schema. Preserve source actions and required metadata. Validate boundary transitions, nonempty ordered segments, equal lengths, and final-segment handling. Convert repeatable event detections into per-segment monotone signals, distinguishing repeated object interactions.

For automatic grasp annotation, combine contact with closure, persistence, lift or object-following evidence appropriate to the task. Review proposed boundaries in synchronized playback and save overrides. Record per-object events: aggregate `subtask/status`, `completed`, and `score` cannot reliably identify the selected object or all segment boundaries.

Exit condition: reviewed BananaInBowl data accepted by the pinned Mimic parser, with verified alignment/frames and no unresolved boundaries. Manual correction is acceptable for this milestone.

**Phase 3 — generate a small BananaInBowl dataset**

Use two segments: grasp/lift relative to banana; transport/place/release relative to bowl. Begin with original object identities, narrow pose changes, zero action noise, conservative interpolation, and one environment. Match source control rate and controller settings.

Use approximately 5–10 clean successful source demos as an initial experiment target, not a sample-efficiency guarantee. Generate 10 accepted episodes with explicit attempt/failure limits. Report attempts and successes separately because upstream `generation_guarantee` changes count semantics.

Separate failed-trial diagnostics from training data. Classify source replay, annotation, unreachable target, grasp loss, transit collision, placement, lifecycle/export, and timeout failures. Save source segment choices and randomization seeds.

Exit condition: 10 independently verified successful episodes with correct lengths/provenance and reliable repeated resets. If the budget is exhausted, diagnose failures before increasing scale.

**Phase 4 — expand standard task conversion and resolve branches**

Expand across the 95 standard-helper candidates incrementally; helper usage alone is insufficient. Use a predicate registry and typed overrides rather than parsing instructions.

| Pattern | Conversion policy | Validation target |
|---|---|---|
| Single pick/place | Object-relative grasp and destination-relative placement | BananaInBowl, YogurtInBowl. |
| Surface placement | Preserve support, orientation, release requirements | A supported pick_and_place_on_surface task. |
| Ordered objects | Preserve required order and matching source cohorts | BananaThenRubiksCube and reverse order. |
| Unordered `all` | Recover demonstrated order; partition compatible cohorts | RubiksCubeAndBanana. |
| `any` | Resolve the successful object and generate per selected branch | RubiksCubeOrBanana. |
| `choose(K)` | Resolve chosen set, initial satisfaction, order; enforce final cardinality | PutTwoMugsOnShelf. |
| Multiple destinations | Preserve object/destination bindings and check termination | Supported grouped-container predicates; conflicting tasks excluded. |

Initially generate only observed compatible cohorts. Do not enumerate all permutations or assume source coverage for unobserved branches. Preserve interleaving, temporary placements, and repeated grasps explicitly, or mark them unsupported by the simple template. Per-stage source selection still requires compatible held-object and scene state.

Allow overrides for frame choice, segments, source cohorts, orientation constraints, and randomization. Version/hash overrides in outputs. A converted task without matching source motions remains awaiting data.

Exit condition: verified generation on representative single-target, ordered, unordered, alternative, and cardinality tasks. Produce coverage reports separating conversion, annotation, generation, and policy-evaluation status.

**Phase 5 — add custom task families**

Add explicit templates for stacking, reorientation, spatial placement, and other interactions among the 25 custom-condition candidates. A final-state predicate such as upright does not specify a grasp/trajectory; an order-agnostic stack predicate does not name the support block for each segment.

Require demonstrated decomposition, reference frames, and boundary checks. For multi-object spatial relations, choose a justified object or constructed task frame and test coupled randomization. Independently perturbing every object may violate the skill's assumptions. Articulated targets require link poses rather than only rigid-object roots.

Exit condition per family: reviewed conversion, annotated source data, small successful generation batch, and held-out evaluation. Keep unsupported interactions explicit.

**Phase 6 — export, throughput, and learning evaluation**

Preserve RoboLab action/state and `env_cfg.json` contracts where practical. Add a documented conversion layer when upstream export differs. Record source file/demo/segment IDs, selected objects/order, seeds, annotation/spec versions, action frames/scales, timestep, config/asset fingerprints, and simulator/generator versions.

Keep failed trials out of training data. Render images from generated pre-action states through deferred materialization only when its schema/metadata contract is satisfied. Explicitly map observation keys, cameras, instructions, and actions to the learner. Joint-target policies need appropriate controller outputs; relative IK commands cannot be relabeled as joint targets.

After lifecycle validation, test two environments with staggered resets/success times, then scale concurrency based on throughput/memory. Expand to 100 and potentially 1,000 accepted episodes when yield and diversity justify it.

Compare teleop-only and teleop-plus-generated policies using identical held-out reset distributions. Keep source lineages and generated derivatives in the same train/validation partition. Report nominal and expanded-randomization success separately. Distinguish generation yield from policy success; include per-task attempts, accepted counts, failures, wall time, and dataset size.

Exit condition: reproducible datasets and evaluation reports establishing whether generated data benefits the intended learner for each task family.

**Phase 7 — add SkillGen when transit failures justify it**

Reuse the adapter, task specs, source cohorts, export, and metrics. Establish a separately pinned Isaac Lab/Isaac Sim/cuRobo stack and revalidate source replay. Inspected main documentation and source differ on automatic start signals and annotation naming; implement the selected revision's parser contract.

Add localized interaction start boundaries with sufficient grasp, lift, release, and retreat context. Start with manual review; automate validated predicates later. Respect the chosen parser's final-skill naming and episode-length endpoint conventions.

Build a DROID planner config with Franka/Robotiq kinematics, joint order/limits, controlled link, collision spheres, gripper configurations, and attached-object geometry. Compare planner and simulator forward kinematics at sampled joint states. Include table, receptacles, and clutter; validate intended-contact exclusions and attachment/detachment updates.

Provide explicit planner selection: inspected `CuroboPlannerCfg.from_task_name` falls back to Panda for unknown tasks. A DROID task must select its actual model. Retiming must match RoboLab control timing and physical tracking.

Exit condition: model/world alignment plus a matched Mimic-versus-SkillGen experiment demonstrating reduced transit failures or useful wider coverage. Planned collision-free connections do not guarantee successful contact interactions. Training the paper's hybrid skill policy is separate from using generated data for ordinary behavior cloning.

**Validation and delivery checkpoints**

| Checkpoint | Required evidence |
|---|---|
| Task conversion | Correct destinations/order/cardinality; actionable conflict and unsupported-predicate diagnostics. |
| Pose/action conversion | Round trips with zero rotations, nonidentity roots, multiple origins, gripper values; separate physical tracking checks. |
| Source preparation | Correct timing, finite equal-length arrays, valid boundaries, replay/restoration provenance. |
| Lifecycle | Sequential and staggered multi-env resets without leakage, duplicate export, or lost terminal data. |
| Dataset | Authoritative success checks, separated failures, source lineage, image/action alignment. |
| Coverage | Separate counts for recognized semantics, annotated sources, verified generation, evaluated policies. |
| SkillGen | FK/collision/attachment checks and matched generation comparisons. |

Implement focused behavior tests with each phase. Use synthetic predicate fixtures for simulator-free semantic tests and small real simulator tests for timing, physics, and lifecycle. Documentation alone does not require simulator execution.

Suggested reviewable change sequence: (1) audit/schema/extractor reuse; (2) DROID adapter/lifecycle; (3) preparation/annotation; (4) BananaInBowl generation/export; (5) standard conversion/branch cohorts; (6) throughput, dataset/policy evaluation, and custom families; (7) SkillGen stack/model/planner. Phases 0–3 are the first end-to-end milestone; broad coverage and SkillGen should not delay validating it.

**Upstream references**

The [Isaac Lab Mimic workflow](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html) documents recording, annotation, generation, and training. The [MimicGen custom-task guide](https://mimicgen.github.io/docs/tutorials/datagen_custom.html) describes object-relative segmentation and environment interfaces. The [SkillGen guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html) covers planner connections and annotation. The [survey](mimicgen_skillgen_survey.md) contains inspected source links and version-specific findings.
