**OMY versus standard MimicGen: subtask randomization — 2026-11-09**

Our generator uses MimicGen's object-relative segmentation, transforms and interpolation, but a restricted source-selection mode. Boundary randomization already runs. Selecting a new source for each subtask is disabled, and production jobs load one source only. The proposed diversity plan also adds custom motion adaptations; it should not be described as a reproduction of the paper's Pick Place configuration.

This check used the official MimicGen paper/docs/code, the installed Isaac Lab package reporting version 2.2.0, and local source inspection. CPU checks executed only extracted boundary-sampling and selection-guard code with synthetic inputs. No simulator, GPU, generation or training ran; no generator code was changed.

| Meaning of “random subtask” | Upstream behavior | Current OMY behavior |
|---|---|---|
| Randomize where adjacent segments meet | Sample offsets around annotated boundaries | Active: grasp boundary offset 0–3 frames; final endpoint fixed |
| Choose a source once per generated episode | Supported: random or nearest-neighbor selection from a source pool | Low-level API supports it; production schedules one source per job |
| Choose a source again for each segment | Supported when per-subtask selection is enabled; selecting the same source again is possible | Disabled explicitly; grasp and placement retain one parent |
| Randomize subtask order | Standard task specification is an ordered sequence | Fixed grasp/lift then transfer/place, as expected |
| Add execution noise | Supported separately from boundary/source selection | Default noise is zero; the OMY adapter's implemented noise perturbs target position |

The official [boundary guide](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) explains offset randomization and shifting a boundary later than the first grasp signal. The [selection guide](https://mimicgen.github.io/docs/modules/datagen.html) treats selection frequency and selection strategy as separate settings. With per-subtask selection off, only the first segment's selection strategy chooses the source; that source supplies the remaining segments.

The [paper, Appendices N.3 and N.6](https://arxiv.org/html/2310.17596v1), uses one source per episode for many tasks because different grasps can require incompatible placement motions. However, Stack, Stack Three and Pick Place use per-subtask selection. The paper reports benefits primarily in generation yield, with smaller or inconsistent policy effects. Thus keeping one source is a supported MimicGen setting, but it is not enough to claim we match the paper's pick-and-place settings.

The [official experiment generator](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/scripts/generate_core_configs.py) enables per-subtask selection for those tasks. Its [Pick Place template](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/exps/templates/robosuite/pick_place.json) uses nearest-object selection for grasp segments and random selection for placement segments, whose reference object is `null`. The experiment generator overrides the template's default `select_src_per_subtask=false`. That benchmark moves four objects into fixed destinations; it is not identical to our one-object/movable-bin setup.

Local evidence:

- [mimicgen.py](../robolab/core/omy_teleop/mimicgen.py), `mimic_cfg`: `generation_select_src_per_subtask=False`, two reference frames, and grasp offset `(0, 3)`. The attempt loop calls `randomize_subtask_boundaries()` before generating both segments. At 15 Hz the offset contributes 0–0.2 seconds. This mechanism exists; its current window does not enforce the clearance needed for distant transfers.
- [pipeline.py](../robolab/core/omy_teleop/pipeline.py), `process_job`: rejects a supplied source pool and passes one episode to the generator. Quality checks and `verify_lineage` also require one loaded source. Consequently `--selection-strategy random` has only one choice in these production jobs. Diversity across scheduled human parents is different from mixing segments within an episode.
- [OMY documentation](omy_mimicgen.md) explicitly deprecates pooled generation for production. This is a local pipeline policy, not a MimicGen requirement. Choosing among candidate parents outside the worker, then loading the chosen parent, can still preserve the current one-parent contract.
- The attempt summary and episode provenance record one final `source_record(selected[EEF], sources)`. If segments used different sources, this would lose the earlier parent. Per-segment provenance, qualification evidence, selection quotas and split lineage need corresponding support before publishing such datasets.

There is also an installed-version control-flow issue. In `DataGenerator.generate_trajectory`, lines 349–354, per-subtask selection is subsequently gated on `selected_src_demo_inds[eef_name] is None` when `generation_select_src_per_arm=False`, the inherited default. OMY carries the selected index across segments. Simply changing the per-subtask flag to true therefore does not reselect at placement. The [pinned upstream v2.2.0 implementation](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py) contains the same gate.

The [CPU check](../output/omy_diversity_verification_20261109/subtask_randomization_audit.py) and [hashed results](../output/omy_diversity_verification_20261109/subtask_randomization_audit.json) reproduce this:

| Configuration in the extracted guard | Select at grasp? | Select at placement? |
|---|---|---|
| Current flags, carry selected index | Yes | No |
| Only set per-subtask true | Yes | No |
| Per-subtask true, explicitly clear index before placement | Yes | Yes |
| Per-subtask true and per-arm true | Yes | Yes |

These are selection-call decisions, not successful robot trials or a validated implementation prescription. The same script runs the installed boundary sampler over 64 seeds on synthetic nine-source/two-segment inputs, observes offsets `{0,1,2,3}`, and checks adjacency. Reproduce with `CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void .venv/bin/python -B output/omy_diversity_verification_20261109/subtask_randomization_audit.py`.

For the plan, the distinction between normal task configuration and custom adaptation should stay explicit:

- Defining object/bin references, annotating a later lift boundary, randomizing within suitable boundary windows, choosing source-selection settings, and sampling new scene arrangements are standard MimicGen configuration/integration work. A fixed interpolation-step setting is also standard; our one-second value is local.
- Cropping the approach, leaving a gap between lift and over-bin intervals, replacing source transit with a direct connection, computing duration per connection, and canonicalizing pail yaw are proposed local extensions. They may be useful, but upstream compatibility does not establish their physical validity. Stock MimicGen splits demonstrations into adjacent intervals; its transform/connect design alone does not validate our deliberately omitted interval. [Original generator](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/datagen/data_generator.py).
- The six-joint IK constraint, fixed posture interface, qualification/quality rules and dataset quotas belong to OMY. They are not restrictions imposed by the original method.

Before freezing the repaired generator, add a small source-selection comparison to the existing pilot: current scheduled parent; random/nearest-neighbor parent chosen once per attempt; and actual per-subtask reselection. Hold target layouts, scene seeds, boundary settings, timing and quality rules fixed within each comparison. Verify the selection calls and per-segment parent IDs rather than trusting configuration flags. Keep diagnostic outputs isolated from production; this need not add another policy-training arm.

For mixed-source trials, check the held object's transform relative to the gripper, including grasp height, tilt and offset. Pail yaw symmetry removes one orientation constraint, not all grasp incompatibility. A successful grasp from source A does not ensure that source B's placement poses carry the object to the intended location. Evaluate generation yield, accepted layout coverage and motion diversity; do not assume more combinations improve PPS.

Keep boundary jitter explicit when repairing cuts: calling the upstream sampler and then overwriting its endpoints could erase the randomness. Sample only within validated lift/closed-gripper/over-bin intervals, log realized cuts, and record a zero-width interval as a deliberate setting where no safe variation exists.

A separate standard feature to revisit is action noise. The [paper's action-noise ablation](https://arxiv.org/html/2310.17596v1) found improved learned policies despite reduced generation yield. Our initial zero-noise diagnostics are useful for isolating motion defects; production noise deserves a controlled comparison after feasibility. Its scale cannot be copied directly from another controller's action units into OMY's positional noise in meters.

**Follow-up verification, 2026-11-10**

The [selection-guard audit](omy_mimicgen_selection_guard_audit_20261110.md) now exercises the installed full `generate_trajectory()` and standard asynchronous `generate()` lifecycle using a synthetic pool and fake environment. I inspected and reran its script on CPU, writing [a separate result](../output/omy_selection_guard_audit_20261110/selection_guard_repro_verified.json); the complete JSON equals the original result, and all eight installed-module hashes match. No simulator or GPU ran, and no production code or installed package changed.

The central verdict is confirmed: per-subtask true with per-arm false selects only at grasp, including in stock `generate()`. Clearing the retained index under the opt-in mode restores placement selection (300/300 calls, 190 different parents in the random three-source test). Setting per-arm true instead violated a forced parent in 271/300 mixed-subtask trials. These are synthetic selection results, not physical success rates. The observed one-source trajectory identity is likewise limited to the synthetic test.

Upstream inspection confirms that the guard was added in the [multi-end-effector change](https://github.com/isaac-sim/IsaacLab/commit/464631fac18bf14ac26c4a6ba615f8d8ed0e1823); the preceding source lacks it, and the report's pinned main revision retains it. Classifying this as a single-arm regression is supported. Its intended multi-arm synchronization semantics remain an inference. Current production explicitly disables per-subtask selection and loads one parent, so this defect does not establish corruption of existing datasets.

Use the audit's proposed local opt-in clear, with fixed/scheduled-source combinations rejected and per-segment provenance implemented before exporting experimental mixed-parent episodes. Retain explicit schema/mode checks: null legacy parent fields alone do not guarantee rejection by every consumer (`parent_key()` is only invoked on relevant parent-accounting paths).

One correction to section 8 of that report: do not add parent identity to the content fingerprint or use parent IDs as geometric diversity features. [Recording validation](../robolab/core/omy_postprocess/validation.py) fingerprints motion/state content, and [dataset selection](../robolab/core/omy_teleop/final_dataset.py) uses it to reject duplicate trajectories. Different parent tuples should remain distinct lineage metadata, but identical trajectory content should still deduplicate. Parent quotas and lineage-disjoint splits can use the separate parent list without manufacturing additional motion diversity.
