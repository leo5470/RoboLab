# OMY MimicGen source-selection guard audit (2026-11-10)

Scope: CPU-only. Static inspection plus execution of the **unmodified installed** Isaac Lab Mimic
`DataGenerator.generate_trajectory` and `DataGenerator.generate` against a synthetic source pool and a fake
environment. No Isaac Sim, GPU, demonstration generation or training. No production code, installed
package or existing report was changed.

Artifacts: `output/omy_selection_guard_audit_20261110/`
- `selection_guard_repro.py`: reproduction script (sha256 `e13a0573…5e1ab5`)
- `selection_guard_repro.json`: full results, including hashes of every loaded module
- `selection_guard_repro_summary.txt`: results tables
- `upstream_comparison.txt`: hashes and guard excerpts of upstream revisions

## 1. Verdict

**Yes.** With `generation_select_src_per_subtask=True` and `generation_select_src_per_arm=False`, the installed
guard prevents placement-subtask reselection for a single-arm robot. This happens in RoboLab's loop and in
Isaac Lab's own `generate()` lifecycle.

- **Classification: an upstream defect, a regression in single-arm behavior.** It is not caused by
  RoboLab's custom loop. The guard first appeared in Isaac Lab commit
  [`464631f`](https://github.com/isaac-sim/IsaacLab/commit/464631fac18bf14ac26c4a6ba615f8d8ed0e1823), "Updates
  mimic to support multi-eef (DexMimicGen) data generation". The guard is probably meant to stop a second
  arm from overwriting the source that the first arm fanned out to it (lines 404–406). Its test is "this
  arm's index is `None`", though, and that test cannot tell "already chosen by another arm for this
  subtask" apart from "chosen by this arm at an earlier subtask". The per-subtask flag is therefore inert
  whenever `per_arm=False`.
- **Evidence that this is not intended:**
  1. The flag's docstring says "Whether to select source data per subtask" (`mimic_env_cfg.py:59-65`).
  2. Original MimicGen reselects unconditionally
     ([`mimicgen/datagen/data_generator.py#L274-L281`](https://github.com/NVlabs/mimicgen/blob/bcc9261a03ad4ffba68c46e4b2fc3d23f4e8921b/mimicgen/datagen/data_generator.py#L274-L281)).
  3. The Isaac Lab revision just before `464631f` also reselects unconditionally
     ([`8a58a23` L290-L296](https://github.com/isaac-sim/IsaacLab/blob/8a58a23cdad4dfdb279090306e0762d4f0c98381/source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py#L290-L296)).
  4. Isaac Lab ships single-arm Franka stack configs that set `generation_select_src_per_subtask = True` and
     leave `per_arm` at its default `False`. Examples are
     [`franka_stack_ik_rel_mimic_env_cfg.py#L30`](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py#L30),
     plus the abs, visuomotor, cosmos and blueprint variants. In those configs the flag silently does nothing.
- **Version status:** the installed `data_generator.py` and `mimic_env_cfg.py` are byte-identical to official
  **v2.2.0**, and v2.2.1 is identical too. **v2.3.0** and **main** (`2742c2d5`, fetched 2026-11-10) still
  contain the same guard at
  [main L391-L394](https://github.com/isaac-sim/IsaacLab/blob/2742c2d5e6b950d0150781e384bcd8388c29e3ee/source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py#L391-L394).
  No newer upstream behavior changes the conclusion. I found no upstream issue on this; a GitHub issue
  search for `generation_select_src_per_subtask` returned 0 results.
- **The integration issue in RoboLab is latent, not active.** RoboLab currently sets `per_subtask=False`
  (`mimicgen.py:250`), so nothing is mis-generated today. The latent risks are:
  - Flipping that flag alone would silently change nothing.
  - Provenance records one index per episode (`mimicgen.py:509`). That is correct only because the guard
    currently guarantees a single parent.

## 2. Versions and hashes

| Item | Value |
|---|---|
| Isaac Lab dist | `isaaclab-2.2.0`; `isaaclab` extension 0.44.9; `isaaclab_mimic` 1.0.12 (CHANGELOG 2025-07-31) |
| installed `datagen/data_generator.py` | `8d353a3969897459e09fce42315d7e71fe47d29fe6459d6505bc7c657c470e96`, identical to v2.2.0 and v2.2.1 |
| installed `envs/mimic_env_cfg.py` | `8dbd961654c6300b67124af1168ae0d89f90515bc7525be576ca388cc013aaaa`, identical to v2.2.0 |
| installed `datagen/selection_strategy.py` | `26862e8379a2ccb84db696661a3ff801095a8f8ff05ad7f8b49f0d7403b82339` |
| upstream v2.3.0 / main `data_generator.py` | `f54d7694…886b8` / `8e3156c1…c8b5` (same guard) |
| `robolab/core/omy_teleop/mimicgen.py` | `9a841716c31ec38243f6aa0e901aa313c97d5a1acff7f2274fceae2a0ffdfb2f` |
| `robolab/core/omy_teleop/mimic_sources.py` | `e36bd67078ab41a72e40367d910b83920d15ff0b24ec707ca050f3fb0a880ee0` |
| `robolab/core/omy_teleop/pipeline.py` | `71d3cba57b36011071a8333a670dfe09a66cc7d8837a1bf4a36405a8f876fd49` |
| Repro runtime | Python 3.11.15, torch 2.7.0+cu128 (CPU only, `CUDA_VISIBLE_DEVICES=`), numpy 1.26.0 |

The hashes of all eight real modules loaded by the repro are in `selection_guard_repro.json` under `loaded_real_modules`.

## 3. Control-flow trace

Installed file: `.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py`.

### 3.1 `generate_trajectory` (lines 315–550)

| Line | Behavior |
|---|---|
| 349 | `is_first_subtask = subtask_ind == 0` |
| 351 | `need = is_first_subtask or per_subtask` |
| 353–354 | if `not per_arm`: `need = need and selected_src_demo_inds[eef] is None` (**the guard**) |
| 359–385 | COORDINATION constraints can adopt a concurrent arm's source and force `need = False`. Only entries in `runtime_subtask_constraints_dict` trigger this. RoboLab passes `{}` (`mimicgen.py:489`) and sets `task_constraint_configs = []` (`mimicgen.py:264`), so this path is unreachable for OMY. |
| 390–399 | if `need`: `selected[eef] = select_source_demo(...)` |
| 401 | `assert selected[eef] is not None`. If you clear the index while `need` is otherwise False, this assertion fails (reproduced below). |
| 404–406 | if `not per_arm and need`: the chosen index is copied to **every** arm. This is the multi-arm sharing the guard protects. |
| 409 | `boundaries[eef][selected, subtask]` uses the selected source's randomized row. |

For a single arm with `per_arm=False`, `selected[eef]` is non-`None` after subtask 0, so line 354 forces
`need=False` at every later subtask, whatever `per_subtask` is set to.

### 3.2 Isaac Lab's standard `generate()` lifecycle (lines 552–783)

- Line 603/609: `current_eef_selected_src_demo_indices[eef] = None`, once per `generate()` call (one episode).
- Line 630–640: `generate_trajectory(...)` is called whenever an arm starts a subtask, passing the same dict.
- Line 643: a commented-out reassignment is the only other reference.
- Line 760: the subtask index advances.

Nothing resets the dict between subtasks. The stock lifecycle therefore behaves exactly like RoboLab's loop.

### 3.3 RoboLab's loop (`robolab/core/omy_teleop/mimicgen.py`)

- `:250` sets `generation_select_src_per_subtask=False`. `per_arm` is not set, so it inherits `False`.
- `:480-481` calls `randomize_subtask_boundaries()` per trial inside `trajectory_rng(selection_seed + trial)`.
- `:482` sets `fixed_source` from `trial["source_index"]` under `--follow-source-schedule`, otherwise from `--fixed-source-index`.
- `:485` creates `selected, previous = {EEF: fixed_source}, {EEF: None}` **fresh per trial**. Indices do not leak across trials; they persist only across the two subtasks of one trial, as in upstream.
- `:487-489` calls `generate_trajectory(0, EEF, subtask, boundaries, {}, selected, previous)` per subtask, each inside `trajectory_rng(... + subtask*100000)`.
- `:509` and `:518` record and print only the final `selected[EEF]`. `:540-545` writes that single index to the HDF5 `mimicgen_source` attr.

Pre-populating `selected` with `fixed_source` is what makes `--fixed-source-index` and
`--follow-source-schedule` work. With `per_arm=False`, the line-354 guard skips selection at subtask 0 as
well, so the forced index is used without any draw. **RoboLab's forced-source features depend on the same
guard that blocks per-subtask reselection.**

### 3.4 Production (`robolab/core/omy_teleop/pipeline.py`)

- `:99` rejects multisource jobs.
- `:113-121` runs every job as `--input X --target-input X --target-episode N --episodes N`, so the pool size is 1. Validation jobs also pass `--no-randomize --keep-failed`. Production never passes `--fixed-source-index` or `--follow-source-schedule`.
- `:166-167` and `verify_lineage` at `:242-265` require exactly one loaded source per job and one qualified parent per row.
- `final_dataset.py:40-45` (`parent_key`) and `:48-80` (balanced per-parent caps) assume one parent per row.

## 4. Intended meaning of the flags

- `generation_select_src_per_subtask`: choose a source at each subtask, not once per episode. This matches
  MimicGen's `select_src_per_subtask` ("select a different source demonstration for each subtask … else keep
  the same one for the entire episode").
- `generation_select_src_per_arm`: whether each arm chooses independently (`True`) or all arms share the first
  chosen source (`False`, lines 404–406). It was added with multi-eef support and has no MimicGen counterpart.
- The intended interaction for multi-arm with `per_subtask=True, per_arm=False` is presumably "one shared
  source per subtask across arms". It is ambiguous upstream because arms advance through subtasks
  asynchronously. For a **single arm**, per-arm sharing has nothing to share, so the guard should never fire.
  That it does fire is the defect.

## 5. CPU reproduction

`CUDA_VISIBLE_DEVICES= .venv/bin/python -B output/omy_selection_guard_audit_20261110/selection_guard_repro.py`

### Method

- The script loads the real installed modules by path. It stubs only Isaac Sim-bound imports: `omni.log`, the
  type names `ManagerBasedRLMimicEnv` and `TerminationTermCfg`, and `isaaclab.utils.datasets`.
- Real `math.py`, `configclass`, `mimic_env_cfg`, `datagen_info`, `selection_strategy`, `waypoint`,
  `datagen_info_pool` and `data_generator` all execute.
- The config mirrors `mimic_cfg`: two subtasks with object then container references, offset `(0,3)`/`(0,0)`, and `random` selection.
- RoboLab's real `trajectory_rng` is used.
- The pool has 3 sources (1 in the qualification-shaped cases). Each source carries its identity in its gripper
  channel, which `generate_trajectory` copies verbatim into waypoints. This independently confirms which
  source's data each segment actually used: there were 0 data/index mismatches in every case.

### Measurements

The script separates three things:
- **"invoked"**: whether `select_source_demo` was called, measured with a spy.
- **"differs"**: whether the placement parent differs from the grasp parent.
- A deterministic override that makes consecutive calls always return different indices. This separates
  "selection was invoked" from "random selection happened to return the same source".

Clear policies:
- `never` is the current loop.
- `always` sets `selected[EEF]=None` before subtask 1.
- `unless_fixed` clears only when no fixed or scheduled source is set (the proposed fix).

### Results: RoboLab loop pattern, N=3, 300 trials

| per_subtask | per_arm | fixed | clear | invoked s0/s1 | placement parent differs | fixed violated |
|---|---|---|---|---|---|---|
| F | F | – | never | 300/0 | 0 | – |
| **T** | **F** | – | **never** | **300/0** | **0** | – |
| T | F | – | unless_fixed | 300/300 | 190 (≈2/3; random can repeat) | – |
| T | F | – | unless_fixed + always-differ override (50) | 50/50 | **50/50** | – |
| T | F | – | never + always-differ override (50) | 50/0 | 0/50 | – |
| T | F | 2 | never | 0/0 | 0 | 0 |
| T | F | 2 | unless_fixed | 0/0 | 0 | **0** |
| T | F | 2 | always | 0/300 | 212 | **212** |
| F | T | 2 | never | 300/0 | 0 | **212** |
| T | T | – | never | 300/300 | 190 | – |
| T | T | 2 | never | 300/300 | 190 | **271** |
| F | F/T | any | always / unless_fixed (with clear) | AssertionError `data_generator.py:401` | | |
| T | F | – | unless_fixed, `nearest_neighbor_object`, nn_k=3 | 300/300 | 190 | – |
| T | F | – | unless_fixed, **N=1** | 50/50 | 0 | – |

- In every row where the parents differ, the recorded index (`mimicgen.py:509` semantics) misses the grasp
  parent. For example, 190/300 trials in the reselect rows.
- With N=1, the generated trajectories from the baseline (`per_subtask=False`, no clear) and from the proposed
  fix (`per_subtask=True`, clear) are **bit-identical** over 20 trials
  (`single_source_pool_trajectories_identical_with_clear_fix: true`).

### Results: stock `DataGenerator.generate()`, unmodified async lifecycle

| per_subtask | per_arm | selection calls / episode | episodes whose executed actions switch source |
|---|---|---|---|
| F | F | 1 (100/100) | 0/100 |
| **T** | **F** | **1 (100/100)** | **0/100** |
| T | F + always-differ override | 1 (30/30) | 0/30 |
| T | T | 2 (100/100) | 67/100 |
| T | T + always-differ override | 2 (30/30) | 30/30 |

This confirms the earlier extracted-guard probe
(`output/omy_diversity_verification_20261109/subtask_randomization_audit.json`), this time with the complete
function and the complete upstream lifecycle. It also adds two results the probe did not cover: the
forced-source interactions and the assertion when the index is cleared without `per_subtask`.

## 6. Three separate capabilities

1. **Choosing among sources once per episode: works now.** With default flags and a pool larger than one,
   subtask 0 always invokes selection. The guard does not block this, because the index starts as `None`.
   `--fixed-source-index` and `--follow-source-schedule` replace the draw deterministically. Production
   does not use this capability because `pipeline.py` restricts each job to one source, not because of
   Isaac Lab.
2. **Changing sources between subtasks: currently impossible**, in RoboLab and in stock `generate()`, whenever
   `per_arm=False`. It needs either clearing the index before later subtasks or setting `per_arm=True`.
   Section 7 explains why the latter is unsuitable.
3. **Randomizing subtask boundaries: independent of both flags, and active now.**
   - `randomize_subtask_boundaries()` (lines 190–237) samples an independent end offset per source row.
     For OMY that is `(0,3)` on the grasp subtask, 0–0.2 s at 15 Hz.
   - It runs once per trial under `trajectory_rng`.
   - When parents mix, subtask 0 ends at parent A's sampled row and subtask 1 starts at parent B's row, as in
     MimicGen.
   - Side note, an upstream issue that does not affect OMY: lines 204–206 sample the first-subtask *start*
     offset with `high=range[0]+1`, ignoring `range[1]`. This is still present on main. OMY uses the default
     `(0,0)`.

## 7. Fix options for the single-arm integration (proposed, not applied)

### Recommended (A): explicit opt-in clear inside RoboLab's loop

Four small edits in `mimicgen.py`. Defaults leave production unchanged.

```python
# build_parser
parser.add_argument("--select-source-per-subtask", action="store_true",
                    help="Reselect the parent at each subtask (experimental; incompatible with fixed/scheduled sources)")
# validate_args
if args.select_source_per_subtask and (args.fixed_source_index is not None or args.follow_source_schedule):
    parser.error("--select-source-per-subtask cannot be combined with a fixed or scheduled source")
# mimic_cfg (line 250); per_arm stays False
generation_select_src_per_subtask=args.select_source_per_subtask,
# loop (lines 485-489)
parents = []
for subtask in range(len(cfg.subtask_configs[EEF])):
    if subtask and args.select_source_per_subtask:
        selected[EEF] = None   # Isaac Lab 2.2.0 data_generator.py:353-354 otherwise suppresses reselection
    ...generate_trajectory(...)...
    parents.append(int(selected[EEF]))
```

- Both halves are required. Clearing with `per_subtask=False` hits the assertion at line 401. Setting
  `per_subtask=True` without clearing changes nothing.
- Rejecting the flag together with fixed or scheduled sources is preferred over silently ignoring it.
  `unless_fixed` (tested above) is the equivalent permissive variant: forced sources were preserved in
  300/300 trials.
- A future schedule could carry a per-subtask `source_indices` list and pre-populate the index before each subtask.

### Alternatives rejected or deferred

- **(B) `generation_select_src_per_arm=True`.** It removes the guard entirely, so subtask 0 also draws and
  overwrites the pre-populated index. **This breaks `--fixed-source-index` and `--follow-source-schedule`.**
  Fixed source 2 was violated in 212/300 trials with `per_subtask=False` and in 271/300 with
  `per_subtask=True`. Qualification (N=1) would still be correct only by coincidence.
- **(C) Patch or monkeypatch the installed guard**, for example applying it only when `len(subtask_configs) > 1`.
  This changes behavior for every Isaac Lab consumer, is lost on reinstall, and is out of scope here. It is the
  right *upstream* fix direction. A correct multi-arm fix would track the shared selection per subtask index
  instead of testing for `None`. File an upstream issue citing §5, which reproduces the defect with stock
  `generate()` and shipped Franka configs.
- **(D) RoboLab calls `generator.select_source_demo` itself before each subtask** and passes a pre-populated
  index. This gives full control, for example forcing a different placement parent or excluding self-pairs,
  but duplicates the argument plumbing at lines 391–399. Consider it only if (A) proves too coarse.

### Required regression checks for (A)

Add CPU tests in `tests/`, modeled on the repro:

1. **Defaults.** Without the flag: 1 invocation per trial, parents equal, and selection RNG draws unchanged.
   Compare the recorded per-trial parents against a pre-change run with the same `--selection-seed`.
2. **Flag on, no forced source.** 2 invocations. With the always-differ spy, parents differ in 100% of trials.
   With `random`, the distribution is non-degenerate.
3. **Forced sources.** The flag combined with `--fixed-source-index` or `--follow-source-schedule` causes a
   parser error, or 0 invocations and all parents equal to the forced index under the permissive variant.
4. **Qualification shape (N=1, `--no-randomize`).** Trajectories are bit-identical with and without the flag
   (verified here on CPU).
5. **Provenance.** The per-subtask parent list has length 2 and matches the decoded data. Legacy fields behave
   as described in §8.
6. **Pipeline guard.** `pipeline.process_job` must not forward the flag until §8 is implemented, and
   `verify_lineage` must reject any mixed-parent row it encounters.

## 8. Provenance and validation changes needed for mixed-parent episodes (not implemented)

- **Per-attempt summary (`mimicgen.py:509`):** add `sources_by_subtask: [i0, i1]`,
  `source_references_by_subtask`, `selection_invoked_by_subtask`, the per-subtask boundary actually used
  (`boundaries[EEF][parent_s, s]`) and `mixed_parent`. Set the legacy `source`/`source_reference` only when
  single-parent, otherwise `null`. `final_dataset.py:214-224` then falls through to `parent_key` raising
  "no single source parent", which fails loudly instead of mis-attributing the grasp parent.
- **HDF5 `mimicgen_source` attr (`:540-545`):** add a `parents` list. **Dataset attr `robolab_omy_mimicgen`
  (`:434-447`):** record `select_source_per_subtask` and bump `schema_version`.
- **Pipeline:**
  - `process_job` (`:99`, `:166-167`) and `verify_lineage` (`:242-265`) need a mixed-parent mode in which
    every parent is loaded, from this session, and qualified.
  - `generation_mode` labels (`:340`, `:352`, `:496`) must distinguish the modes.
  - `final_dataset.parent_key` and `parent_cap` water-filling (`:40-80`) need an explicit accounting rule.
    Options are a (grasp, place) pair as the lineage key, credit to both parents, or a fractional share.
    Without one, balancing becomes ambiguous.
- **Qualification:** today each human is qualified alone (`--no-randomize`). That does not cover cross-parent
  seams. Add either a pairwise seam qualification or a per-episode seam check:
  - the length and IK error of the interpolation bridge from A's last target to B's first transformed target
  - consistency of the gripper command at the seam
  - in-hand object pose at A's grasp versus the pose assumed by B's placement segment
- **Diversity and dedupe:** fingerprints and descriptors should include the parent tuple, so that A→B and B→A
  are counted as distinct lineages.

## 9. Verified versus unverified

**Verified on CPU** (control flow, unmodified installed code):
- the guard semantics and their effect in RoboLab's call pattern and in stock `generate()`
- the interaction with fixed and scheduled sources under each option
- the assertion at line 401
- the provenance gap
- N=1 trajectory identity under fix (A)
- upstream identity with v2.2.0/v2.2.1, and persistence of the guard in v2.3.0 and main

**Not verified, needs simulation:**
- Physical feasibility and success rate of mixed-parent episodes: the in-hand pose mismatch between parents,
  the placement transform relative to the container while holding A's grasp, and bridge and IK feasibility
  within `--interpolation-s`.
- Whether `nearest_neighbor_*` at placement (container reference) selects usefully on real data. In the
  synthetic pool, `nn_k=3` with N=3 degenerates to uniform random.
- Downstream effect on dataset diversity and policy performance.
- The repro's fake environment and synthetic poses exercise selection, slicing and waypoint assembly. They do
  not validate pose-transform correctness, IK or physics.
- DexMimicGen's original generator source is not in the public `NVlabs/dexmimicgen` repository, so the exact
  multi-arm intent is inferred from the Isaac Lab port and was not confirmed against it.
