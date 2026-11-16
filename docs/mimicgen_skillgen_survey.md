Surveyed on 2026-09-14 against this working tree, the installed Isaac Lab 2.2.0 source, and upstream Isaac Lab main at `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`.

**Recommendation: integrate Isaac Lab Mimic with RoboLab first; add Isaac Lab SkillGen when collision avoidance between manipulation segments becomes a limiting factor.** Existing keyboard demonstrations contain enough state and action information to make this feasible. This requires a DROID environment adapter and task annotations, rather than just selecting a different robot config. This survey does not implement or run generation.

Here, DROID means RoboLab's simulated Franka arm with a Robotiq 2F-85 gripper. The conclusions about recording reuse concern this repository's simulated demonstrations. Real-world DROID recordings would additionally need scene reconstruction, object pose estimation, and compatible robot/environment execution.

**Available approaches**

| Approach | How it expands demonstrations | Fit for RoboLab |
|---|---|---|
| Original MimicGen | Splits demonstrations into object-relative segments, relocates them to new object poses, connects and executes them, then checks success. | Feasible, but requires adapting its simulator interface and execution/data plumbing. |
| Isaac Lab Mimic | Implements the Mimic workflow within Isaac Lab, including annotation, generation, and example policy training. | Best first implementation: shares the existing simulator and manager-based environment infrastructure. |
| Isaac Lab SkillGen | Uses localized manipulation segments and cuRobo motion planning for connecting motions. | Useful for larger placement changes and clutter, after the Mimic adapter works; adds robot collision modeling and planner dependencies. |

The original MimicGen supports custom simulator interfaces, including action/target-pose conversion and object-pose access. Its published robot-transfer examples do not establish drop-in support for RoboLab's embodiment. [MimicGen interfaces](https://mimicgen.github.io/docs/modules/env_interfaces.html), [MimicGen project](https://mimicgen.github.io/).

Isaac Lab now documents both workflows. Neither the installed Mimic registrations nor the inspected upstream Mimic registrations provide a DROID task. The upstream tree also had no path containing `droid`. The closest starting point is the Franka relative-IK Mimic environment. [Mimic workflow](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html), [upstream registrations](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py).

Distinguish SkillGen data generation from the original paper's full learning/deployment system. The paper trains Hybrid Skill Policies that alternate learned interaction skills with planning during deployment. Isaac Lab's tutorial demonstrates training ordinary Robomimic BC on SkillGen-generated trajectories. Using that generator alone does not implement the paper's hybrid policy. [SkillGen project](https://skillgen.github.io/), [Isaac Lab SkillGen tutorial](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html).

**What the current recordings already provide**

I inspected [the keyboard collector](../examples/collect_keyboard_demos.py), [recorder terms](../robolab/core/events/basic_recorders.py), and the existing file `output/keyboard_demos/banana_wifi/keyboard_demos.hdf5` read-only. Its `demo_0` has `success=True`, seed 6, and 579 samples. The three suffixed HDF5 files in that directory contain no episodes; they should not be counted as source demonstrations.

| Recorded channel | Observed content | Use in generation preparation |
|---|---|---|
| `actions` | `(579, 7)` relative IK commands; gripper values 0 and 1 | Recover commanded Cartesian targets and gripper actions. |
| `initial_state` | Articulation and rigid-object state | Reconstruct the initial scene. |
| `states` | Post-step articulation and rigid-object states, including banana, bowl, and table | Recover object poses and restore the demonstrated trajectory. |
| `ee_pose` | Position and quaternion after each step | Measured end-effector trajectory; account for timing and coordinate frame. |
| `robot_root_pose` | Root pose after each step | Convert between robot-root and environment-local frames. |
| `obs/proprio_obs` | Pre-step proprioception | Policy inputs and additional consistency checks. |
| `obs/datagen_info` | Absent | Must be added by annotation/preparation. |

The file is state-only, consistent with deferred-image collection. Images are unnecessary for geometrical trajectory generation. Keep the HDF5, its `env_cfg.json`, seeds, matching assets, and collection-version metadata together. [Deferred recording and alignment](keyboard_teleoperation.md), [replay support](replay.md).

The required annotation payload includes `obs/datagen_info/object_pose`, `eef_pose`, `target_eef_pose`, and `subtask_term_signals`; SkillGen also consumes `subtask_start_signals`. HDF5 compatibility alone does not supply these semantics. Upstream annotation reconstructs them by replaying actions in a Mimic environment. [Annotation source](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py).

Two preparation paths are reasonable:

1. **Try upstream-style action replay first.** Restore the initial state, replay with the original controller/timestep, and record the Mimic payload. Measure drift against recorded states and check final success.
2. **If replay drifts, prepare from recorded states.** Restore each pre-action state, derive the measured pose and action target, and attach manually reviewed boundaries. RoboLab already has [state restoration helpers](../robolab/core/replay/scene_state.py). This needs a custom preparation script; restoring states does not automatically reproduce contact sensor histories or stateful predicates.

The indexing rule is essential: action `t` acts on `initial_state` when `t=0`, otherwise on `states[t-1]`. `ee_pose[t]` is post-action. Do not pair it directly with action `t` to infer that action's target. Reconstruct the initial end-effector pose through the restored articulation or forward kinematics.

**DROID adapter work**

Implement a generation environment derived from `ManagerBasedRLMimicEnv`, with RoboLab's task scene and DROID configuration. The upstream scripts check that environment type. Implement the following interface, using the signatures from the pinned version:

| Method | RoboLab implementation |
|---|---|
| `get_robot_eef_pose` | Return the actual IK-controlled Robotiq `base_link` pose as a 4×4 transform. |
| `action_to_target_eef_pose` | Apply the configured action scale and relative rotation convention to the pre-action pose. |
| `target_eef_pose_to_action` | Compute relative pose error and undo the action scale. |
| `actions_to_gripper_actions` | Extract the last action channel, preserving 0=open and 1=closed. |
| `get_object_poses` | Return task object frames in the same coordinates as end-effector poses. The default handles rigid objects; articulated targets need link-specific logic. |
| `get_subtask_term_signals` | Expose named boundaries for automatic Mimic annotation, or use manual annotations. |

These methods are defined by the [Mimic environment API](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py). The [Franka adapter](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py) provides the structural starting point.

Specific local issues to handle:

- **Action scale and orientation.** [DroidRelIKActionCfg](../robolab/robots/droid.py) uses scale `0.5`, root-axis translation deltas, and axis-angle rotation deltas. Thus `p_target = p_current + 0.5 * action[:3]`; rotation uses the controller's scaled rotation-vector update. The inverse conversion must divide by the configured scale. Use Isaac Lab pose utilities for quaternion/order handling. Round-trip this conversion before attempting generation.
- **Controlled frame.** The executable relative-IK config currently has no rotational body offset; its commented rotation is inactive. The separately published `eef_frame` has a nonidentity rotation. Some nearby comments disagree with the current executable config. Use the actual action term and recorded config as authority, and verify the frame on each source dataset.
- **Coordinate conversion.** Recorded end-effector poses are robot-root-relative; rigid-object states are environment-local. Transform one into the other's frame. Numerical agreement when the root happens to be at the origin is insufficient validation.
- **Gripper.** Use Robotiq `finger_joint` and its local binary convention. Panda finger names, aperture thresholds, and grasp heuristics cannot be copied directly.
- **Environment lifecycle.** [RobolabEnv](../robolab/core/environments/env.py) freezes terminated environments and exports recordings. Generation requires repeated per-environment reset and generator-controlled export/success evaluation. A dedicated generation environment should reuse RoboLab scene/task logic and deliberately implement reset and recorder behavior. Subclassing the evaluation environment unchanged is insufficient.
- **Registration and configuration.** [Local registration](../robolab/core/environments/config.py) selects `RobolabEnv`, and [runtime construction](../robolab/core/environments/runtime.py) can instantiate it directly. Provide separate Mimic registrations and a launcher that imports them. Upstream scripts import upstream tasks; they do not discover RoboLab automatically. Restore the recording's scene configuration through RoboLab's config loader rather than assuming upstream CLI defaults reproduce it.
- **Success signal.** Upstream annotation expects a termination term named `success`. BananaInBowlTask has one; tasks using several `success_<object>` terms need an aggregate generation success function.

**First task: BananaInBowlTask**

Use the existing [banana task](../robolab/tasks/benchmark/banana_in_bowl_task.py) as the first integration target. My proposed Mimic decomposition is:

| Segment | Reference frame | Boundary |
|---|---|---|
| Approach, grasp, and lift banana | Banana | Stable grasp/lift achieved. |
| Carry, lower, release, and withdraw | Bowl | End of successful demonstration. |

RoboLab's `pick_and_place` evaluation subtask and `subtask/{status,completed,score}` recording are not already this segmentation. Reuse predicates where suitable, but add named boundaries and review them visually. Gripper closure alone does not establish a successful grasp.

For SkillGen, narrow these to local grasp and placement interactions and let the planner handle approach/transit between them. Preserve enough approach, lift, release, and retreat context for execution to remain reliable. The held banana must become an attached collision object during transport.

The underlying relocation is approximately `T_new_ee(t) = T_new_object * inverse(T_source_object) * T_source_ee(t)` within a segment. Generation then executes the adapted trajectory in physics and checks success. It is not simply transforming recorded images. Object-relative decomposition and manual or automatic boundary annotation are part of MimicGen's custom-task workflow. [Custom-task guide](https://mimicgen.github.io/docs/tutorials/datagen_custom.html).

Start with fixed object identities and small pose changes. Expand to the existing [10 cm](../robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_10cm.py), [20 cm](../robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_20cm.py), and [30 cm](../robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_30cm.py) variants after measuring generation yield. Changing geometry or task ordering needs additional validation and potentially new source skills.

**Additional SkillGen work**

The installed `.venv` reports Isaac Lab 2.2.0 and contains Mimic source, but no SkillGen/cuRobo planner modules. [Project dependencies](../pyproject.toml) also offer an Isaac Lab 2.3.2.post1 stack; that alone does not establish SkillGen availability. Use an isolated, pinned compatible Isaac Lab/Isaac Sim/cuRobo environment for SkillGen, and validate old demo replay there before migrating collection. The linked `main` documentation is newer than the installed package.

SkillGen's documented workflow uses manual start/end annotation and `--annotate_subtask_start_signals`, then generation with `--use_skillgen`. Start with that workflow. There is a version-sensitive discrepancy: the inspected main source also supports automatic start signals through `get_subtask_start_signals`. Its manual annotation and parser use the subtask's `subtask_term_signal` name as the key in both signal dictionaries, and require a name for the final skill; the parser takes the final segment end from episode length. Implement against source from the chosen revision. [SkillGen guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html), [annotation implementation](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py), [segment parser](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_mimic/isaaclab_mimic/datagen/datagen_info_pool.py).

The planner needs a DROID kinematic model matching the simulated Franka/Robotiq assembly, joint ordering and limits, correct end-effector link, collision spheres including the gripper, open/closed joint geometry, and attached-object support. It also needs the actual table, bowl, and other obstacles in its world model. Validate planner and simulator forward kinematics at sampled joint states. Match execution timing to RoboLab's 15 Hz control rate.

An explicit planner selection hook is necessary: upstream `generate_dataset.py` calls `CuroboPlannerCfg.from_task_name`, whose unknown-task branch falls back to the Panda configuration. That default uses Panda finger joints and collision geometry. A custom DROID task name alone will not select an appropriate planner. [Planner configuration](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner_cfg.py), [generation launcher](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py).

My assessment is that this planner/model work makes SkillGen materially more expensive to integrate. Collision-free planning also does not guarantee successful contact interactions or perfect controller tracking; retain simulation success checks.

**Proposed delivery and evaluation sequence**

1. **Prepare one existing demo.** Add the adapter, separate task registration, source-config loading, and an annotation/preparation launcher. Verify action-target round trips, pre/post-step alignment, object/EEF frames, gripper behavior, and replay success.
2. **Establish a small Mimic run.** Collect approximately 5–10 clean successful demos as an initial engineering target, annotate grasp boundaries, and generate 10 successful trials with modest object perturbations. Begin with zero action noise, conservative connections, and one environment. These counts are a proposed experiment, not a general sample-efficiency guarantee.
3. **Measure before scaling.** Record attempts, successful exports, failure causes, runtime, reachability, and collision frequency. Separate source/replay failures from grasp, transit, and placement failures. Check multiple sequential resets before enabling parallel environments. Bound retries: upstream `generation_guarantee` controls whether the trial target counts successes, and `max_num_failures` limits failures. [Generation configuration](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/envs/mimic_env_cfg.py).
4. **Export for the intended learner.** Generate 100, then potentially 1,000 accepted episodes. Retain raw actions, initial/per-step states, source-demo provenance, annotations, task instruction, seeds, and config/version metadata. RoboLab's deferred renderer can be reused if the generated files preserve its state schema and sidecar contract. Render new generated states; do not reuse source images for relocated trajectories. Adapt observation keys and action representation explicitly for Robomimic or the desired policy trainer.
5. **Compare policy outcomes.** Evaluate teleop-only versus teleop-plus-generated data on identical held-out reset seeds, separately for nominal and wider distributions. Keep source-demo lineages and generated derivatives from leaking into validation. For joint-action policies, obtain executed joint targets through the action/controller pipeline; relative-IK vectors cannot simply be relabeled as joint targets.
6. **Add SkillGen if indicated.** Implement DROID planner selection, model/collision validation, attached-object handling, and start-boundary annotation. Compare it against Mimic with the same source demos and randomizations. A measured reduction in transit failures is the clearest reason to adopt it.

The first useful deliverable is therefore a DROID BananaInBowl Mimic adapter plus preparation/generation launchers, using the existing teleop collector. Evidence here is limited to source inspection, upstream documentation, and read-only HDF5 inspection; no simulator replay, planner execution, synthetic generation, or policy training was performed.
