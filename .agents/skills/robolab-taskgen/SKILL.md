---
name: robolab-taskgen
description: >
  Create, repair, or validate RoboLab manipulation task definitions, including
  scene references, instructions, success conditions, and subtasks.
license: CC-BY-NC-4.0
metadata:
  author: nvidia
  version: "2.0.0"
---

# Task generation

A task binds a scene to an instruction and success/subtask conditions. Its goal
predicate does not validate the starting scene, and robot-independent task syntax
does not imply every robot can execute it.

## Resolve scope and scene evidence

- Reuse the requested scene, instruction, output location, episode length, and
  earlier choices. Ask only for missing information that changes the task. If no
  task directory is specified or established, use `robolab/tasks/generated/` for
  qualified tasks or `output/task_drafts/` while required checks remain pending;
  state that choice. Use a unique file/class name for a new task; preserve existing
  files unless repair/update was requested.
- Read the actual scene and list its prim names and support/containment relations.
  Resolve the path unambiguously; names inside the task must match scene prims,
  not catalog display names. Include all required contact bodies and surfaces.
- Read [scene validation](../robolab-scenegen/references/validation.md) for the
  geometry/physics/grasp evidence contract. Check that reports match this scene
  and its current asset hashes. A missing/stale report is pending, never a pass.
  Run applicable available checks before describing a task as ready to use.
- A known-invalid scene needs repair and revalidation; do not publish its task
  as qualified. If the user requested task authoring while simulation is unavailable,
  complete the source as an explicitly unvalidated draft with pending gates.
  Place drafts outside automatically discovered task libraries (for example
  `output/task_drafts/`) unless the user explicitly chose a destination there.
  Do not silently rewrite a supplied scene or import another layout's evidence.

## Define task behavior

State the chosen success behavior and why it matches the user's goal. Inspect the
current signatures in `robolab/core/task/conditionals.py`; use
[conditionals.md](references/conditionals.md) to find candidates and
[examples.md](references/examples.md) for source patterns.

- For Simple-PnP with a released object resting in the bin, use
  `object_resting_in_container`. This checks sustained support, release, bin
  uprightness, and pose stability; `exclude_contact_with="table"` excludes a
  tabletop-supported target. It permits protrusion/stable rim-spanning placement,
  so do not describe it as full enclosure. Match the same function and parameters
  in `pick_and_place(placement_func=..., placement_params=...)` for full credit.
- Use `object_in_container` when the requested semantics are geometric containment
  plus its chosen contact/release requirements. It does not enforce sustained
  resting. Use other conditions for full enclosure, surface placement, stacking,
  ordering, or removal as the goal requires.
- Include `time_out` and the appropriate `success` termination. Require release
  for placement goals using the selected function's supported parameters;
  `object_resting_in_container` already includes release and does not accept
  `require_gripper_detached`. Do not pass generic arguments to every conditional.
- Default `gripper_name="gripper"` uses the robot's declared gripper group.
  For specified hands use their labels; a list means all listed grippers for
  contact, and no listed gripper for detachment. Verify the robot contact mapping
  when reporting release qualification.
- Preserve the user's instruction as `default`; supply `vague` and `specific`
  variants without inventing colors, positions, or affordances. Omit the type
  annotation on class-level instruction dicts to avoid dataclass mutable defaults.
- Use meaningful subtasks when requested/useful. Align their target/destination
  mapping and final placement check with termination. Scores should sum to 1.0;
  composites belong in `subtasks`, not `DoneTerm.func`. Do not turn unordered
  sorting into a forced sequence unless that order is part of the goal.

Reuse an appropriate local task template and verify each import, keyword, and
scene reference. A typical simple task lasts 50 seconds; preserve a supplied
duration. Tags should describe the actual task, using repository conventions.

## Validation and registration

1. Compile the exact new/changed task source. In an Isaac-enabled environment,
   import it and validate its scene/contact list and actual condition signatures.
   A syntax pass alone does not check runtime APIs or scene physics.
2. Run relevant tests that exist and cover the generated file/directory. Do not
   recommend a missing `tests/test_tasks_valid.py` or assume a benchmark-only
   suite covers a different task folder. Check the repository's current commands.
3. Check goal semantics in the intended environment: success false at the intended
   start, true after the requested action and release/stability interval, and false
   for representative invalid outcomes. Test timer/reset behavior for temporal
   conditions. A deliberately pre-satisfied task requires explicit user intent.
4. Report task syntax/runtime/goal-check results separately from scene geometry,
   physics, and robot-specific grasp results. Preserve failed/pending statuses.
5. When registration/metadata is in scope, verify task discovery includes the
   requested folder. `robolab/tasks/_utils/generate_task_metadata.py` supports
   `--tasks-folder` and `--output-folder`; use the actual library paths. Review
   generated registry changes and do not register unqualified drafts by accident.

`examples/run_empty.py --task <TaskClassName> --headless` is an environment smoke
test for registered tasks, not a scene stability/grasp qualification. Verify its
discovery path for custom folders. Use the project's available Isaac environment;
do not assume Docker or a `CLAUDE.md` exists.

## Deliver

Link the task, referenced scene, and check reports. Explain the success criterion,
episode duration, and any failed/pending validation. Include only commands that
apply to the actual output folder and installed tools. A generated file can be a
completed authoring deliverable while simulation/grasp qualification remains pending;
state that boundary without claiming it is ready for collection/evaluation.

Further repository references: `docs/task.md`, `docs/subtask.md`,
`docs/task_conditionals.md`, `docs/task_libraries.md`, and
`docs/environment_registration.md`.
