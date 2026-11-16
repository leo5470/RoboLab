**MimicGen PPS object audit: granola and plastic pail — 2026-11-09**

The [plan](omy_mimicgen_diversity_plan_20261108.md) makes granola bars the primary object for its two-arm PPS comparison: a local mixture (L) against 50% of generated slots on explicit targets (T). Granola's raw recordings, and those of the plastic pail a01, were copied from the collection machine and audited here.

Both objects pass the generator's CPU prerequisites. Granola is a weak test of the L/T hypothesis, however: T adds little spatial coverage, and its γ = 0.4 losses do not grow with distance from the training layouts. The pail is the stronger object on both counts. Its main generator weakness, large wrist turns, comes from rotating each source to follow a rotationally symmetric object, and it can be removed in RoboLab.

Suggested plan changes:

1. Make plastic pail a01 the primary L/T object. Keep granola as a secondary arm, or drop it.
2. Add a switchable symmetric-object option: translate sources onto upright, rotationally symmetric objects instead of rotating them. Pilot it like the other step 2 switches.
3. Choose generation sources by predicted connection time (translation and rotation), after the six-joint screen, not by XY distance.
4. Expect the current layout preflight to reject pail 08 and granola 24 and 25 before generation.
5. Make simulator qualification of the pail's 9 sources the first GPU job.

No simulator, GPU, policy rollout or training was used. No repo code or the installed Isaac Lab was modified. The raw sessions are at `output/omy_teleop/20261023_215652_granola_bars/` and `output/omy_teleop/20261024_134851_plasticpail_a01/`, without `postprocess/`. Scripts and results are in [`output/omy_object_audit_20261109/`](../output/omy_object_audit_20261109/), which is gitignored.

**Raw data and eligibility**

The copies match every recording their PPS exports pin by SHA-256: 11 for granola and 10 for the pail.

| | Granola | Pail |
|---|---:|---:|
| Task folders | 27 | 20 |
| Tasks with a successful demo | 13 | 10 |
| Tasks with only failed attempts | 3 (12, 17, 22) | 3 (00, 06, 10) |
| Empty recording files (`no_episodes`) | 11 | 7 |
| Pass the pipeline's eligibility gate | 13 | 10 |
| Used as humans by the old export | 10 | 10 |

Eligibility uses the pipeline's own `discover_recordings` and `gather_candidates` (`eligibility.py`). Granola gains tasks 11, 15 and 25 over the old export; the pail's eligible set is exactly the old one. Every pail demo rejected by a quality gate (`command_velocity`, `joint_acceleration`) was also unsuccessful.

These October recordings use the current HDF5 layout. `mimicgen.source_arrays` loads every eligible demo and finds the 2 cm grasp trigger, at 2.0–2.8 cm of lift. Joint 3 commands never change. Recorded starts match the v1 manifest to within 1 mm for granola and 3 mm for the pail. The recordings predate the current preparation schema (version 1, without footprint or tilt checks).

**Layout preflight: predicted rejections**

Qualification and generation run `preparation_runtime._correct_placement`, which applies `flat_layout_issues`: a 20 mm object–bin footprint gap and 10 mm of table-edge clearance. For v1 scenes it uses the legacy `BBoxCache` untransformed bound, which is more conservative than `tools/audit_simple_pnp_scenes.py`'s exact bounds. `preflight.py` reproduces that live computation from the scene USD.

Validation: on all 18 fork and pitcher humans it rejects exactly pitcher 02 and 06, at 9.9 mm and 12.2 mm (saved records: 9.1 mm and 10.8 mm), and passes the rest. It reads gaps 0.8–1.4 mm larger than the live check.

| Object | Rejected | Gap | Remaining generator sources |
|---|---|---:|---:|
| Granola | 24 | 2.8 mm | 11 |
| Granola | 25 | −23.3 mm (bounding envelopes overlap) | |
| Pail | 08 | 1.5 mm | 9 |

Granola 11 (23.3 mm) and 19 (25.3 mm) pass narrowly. Rejected tasks remain valid human rows: human demos are not re-prepared. The pail's 9 sources equal the old export's 9 generator sources. The pail's exact box is 12.9 × 12.9 × 12.4 cm, against the bin's 42 × 28 × 10.5 cm.

**Source behaviour and transfer geometry**

`source_audit.py` applies the formulas of [`transfer_audit.py`](../output/omy_diversity_verification_20261109/transfer_audit.py) unchanged (rim-clearance height 13.5 cm, bin footprint ±0.21 × ±0.14 m less 3 cm, 16 bridge commands at 15 Hz), plus the rotation and time choice used in the previous follow-up. Without qualification records, all preflight-passing sources are candidates. That is optimistic: fork kept 7/10 sources, pitcher 5/8 and snickers 1/10.

| | Pail (9) | Granola (11) | Fork (qualified) | Pitcher (qualified) |
|---|---:|---:|---:|---:|
| Sources with rim-clear and over-bin cuts, continuous closed clearance | 10/10 | 13/13 | 10/10 | 8/8 |
| Rim clearance after the trigger | 0.47–0.73 s | 0.53–1.27 s | 0.53–0.87 s | 0.67–0.87 s |
| Lateral motion before rim clearance | 0.6–17.4 cm | 6.2–24.5 cm | 2.6–12.0 cm | 6.3–12.6 cm |
| Human lifted-object XY p95, median / max | 0.323 / 0.690 m/s | 0.266 / 0.476 | 0.230 / 0.271 | 0.342 / 0.430 |
| Held-object angular p95, median / max | 28 / 44 °/s | 24 / 32 | 17 / 24 | 45 / 133 |
| Sources carrying toward −y (targets needing −y) | 5/9 (25/50) | 5/11 (29/50) | | 0/5 (29/50) |
| Legacy bridge speed, median / max | 0.174 / 0.499 m/s | 0.121 / 0.355 | 0.174 / 0.474 | 0.560 / 0.922 |
| Targets above the fastest human p95 | 0/50 | 0/50 | 11/50 | 29/50 |
| Targets above the median human p95 | 4/50 | 2/50 | | |
| Remaining XY path, raised cut → over-bin cut, median | 0.410 → 0.258 m | 0.366 → 0.302 | 0.436 → 0.348 | 0.760 → 0.435 |
| Wrist turn, XY-choice → rotation-choice, median | 79° → 14° | 75° → 13° | 46° → 14° | 75° → 19° |
| Targets over 45° with the rotation choice | 14/50 | 1/50 | | 8/50 |
| Predicted time, XY-choice → time-choice, median | 2.79 → 1.60 s | 3.10 → 1.62 | 2.7 → 1.7 | 1.8 → 1.6 |

Fork and pitcher values come from the two earlier reports. Neither object shows pitcher's blockers: both carry directions are present, the best source's carry heading is a median 15° (pail) and 6° (granola) from each target's, and the legacy bridge is within human speeds. Granola task 21 is an outlier: 24.5 cm of lateral motion before clearing the rim, with only 15.1 cm of peak lift. As before, these are root-proxy and commanded-pose quantities, not clearance, IK or tracking certificates.

**Pail symmetry: translate instead of rotate**

The pail mesh (`SM_PlasticPail_A01_01`, the only mesh) is rotationally symmetric about its root z-axis. Its maximum radius per 10° sector is 64.5–64.7 mm, and every height band has the same radius in all twelve 30° sectors (48.5 mm at the base, 64.7 mm at the rim). There is no handle. Every source and v2 target pail is upright within 0.02°.

The generator transforms each source by the target object's full pose, so it rotates the grasp with the pail's arbitrary yaw. Keeping each source's own object orientation (`SYMMETRIC_YAW=1 source_audit.py`) makes the object-relative transform a pure translation:

| Pail, median / p90 / max | Rotate with pail (current) | Translate only |
|---|---|---|
| Wrist turn, XY-choice | 79 / 162 / 180° | 14 / 21 / 21° |
| Wrist turn, rotation-choice | 14 / 78 / 90° | 3 / 3 / 3° |
| Targets over 45°, any choice | 14/50 | 0/50 |
| Predicted time, time-choice | 1.60 / 2.85 / 3.17 s | 0.73 / 1.18 / 1.71 s |
| Remaining path with over-bin cut | 0.26 / 0.47 / 0.66 m | 0.22 / 0.36 / 0.52 m |

The remaining turn is the source's own wrist change during its carry, because v2 bins have near-zero yaw. Implementation needs no Isaac Lab fork. Give a declared symmetric object the same fixed upright orientation in the source poses loaded into the pool (`mimicgen.py:312`) and in the live poses. `data_generator.py:344` reads the live poses through `get_object_poses`, which `OmyMimicEnv` (`mimicgen.py:159`) can override.

This is valid only for upright objects whose collision geometry is symmetric. Granola's box is only 2-fold symmetric; choosing between the target yaw and the target yaw + 180° might help it, but that is untested.

**Why the pail is the stronger L/T test**

`object_screen.py` uses the previous split protocol: 300 random 25/25 v2 splits, with targets drawn from one half and scored on the other. Distance is the minimum over demos of the larger object/bin XY offset. L allocates the 40 generated slots 50/25/25 near/moderate/wide; T moves 20 of them to explicit targets. Positions only: no yaw, generation failures or selection.

| Object | Base / γ 0.4 / γ 0.8 | Humans only | L | T20 | L − T | γ 0.4 losses near / mid / far | p | γ 0.8 losses near / mid / far | p |
|---|---|---:|---:|---:|---:|---|---:|---|---:|
| Plastic pail a01 | 38 / 28 / 13 | 0.181 | 0.162 | 0.114 | 0.048 m | 1 / 5 / 10 | < 0.001 | 4 / 12 / 12 | < 0.001 |
| Chocolate pudding mix | 30 / 20 / 6 | 0.172 | 0.159 | 0.115 | 0.044 | 5 / 6 / 7 | 0.027 | 9 / 9 / 8 | 0.156 |
| Milk carton | 30 / 28 / 13 | 0.193 | 0.175 | 0.127 | 0.048 | 4 / 5 / 4 | 0.122 | 4 / 6 / 9 | 0.001 |
| Granola bars | 40 / 30 / 7 | 0.151 | 0.136 | 0.115 | 0.021 | 3 / 7 / 3 | 0.219 | 10 / 13 / 11 | 0.011 |
| Cheez_it | 11 / 30 / 8 | 0.150 | 0.128 | 0.098 | 0.031 | 2 / 0 / 3 | 0.337 | 3 / 2 / 5 | 0.084 |

Distance terciles use each v2 scene's distance to the nearest layout in the object's PPS export. Losses are base successes that fail when steered. p is a one-sided permutation test that lost scenes lie farther than kept successes. Positions come from the v1 manifest, and parents are the export's humans. Fork and pitcher, under the same protocol with their qualified parents, give L − T of 0.086 m (0.186 → 0.100) and 0.074 m (0.188 → 0.114).

With the audited sources, the pail gives L 0.163 → T20 0.114 m with 9 parents (0.049 m), and 0.046–0.050 m if only 5–7 random parents qualify. Granola gives 0.136 → 0.115 m with 11 parents (0.021 m), and 0.024–0.027 m with 5–7.

The pail matches granola's base level and γ = 0.4 loss, but its losses concentrate on distant layouts at both strengths, and T moves its coverage about twice as much. A null L/T result on granola would be weak evidence against the diversity hypothesis. Granola could serve as a control where position coverage is not the evident cause of the γ = 0.4 harm. Milk jug a01 is unsuitable: its pair already fails on its own training layouts (10/10 → 2/10).

**Remaining uncertainty**

The decisive test is simulator qualification under current gates. Snickers kept 1/10 sources, mostly to table contact. The pail's 9 preflight-passing sources would take 2 non-randomized attempts each, as for fork and pitcher. The L/T estimates assume qualified sources generate successfully. The distance–harm association is per-object and observational (50 scenes, positions only); yaw and scene-construction differences remain unmeasured.

**Reproduction**

From `output/omy_object_audit_20261109/`, with `R=/tmp2/leocheng/forks/RoboLab` and `D=$R/output/omy_object_audit_20261109`:

```bash
cd $R
.venv/bin/python $D/eligibility.py 20261024_134851_plasticpail_a01 $D/pail_eligibility.json
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void tools/simple_pnp_v2/usd_python.sh $D/layout_preflight.py   # fork/pitcher validation + granola
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void tools/simple_pnp_v2/usd_python.sh $D/preflight.py \
  pnp_242_plasticpail_a01 plasticpail_a01 $D/pail_eligibility.json $D/pail_preflight.json
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void tools/simple_pnp_v2/usd_python.sh $D/source_audit.py \
  20261024_134851_plasticpail_a01 plasticpail_a01 pnp_242_plasticpail_a01 $D/pail_eligibility.json $D/pail_preflight.json $D/pail_audit.json
# translate-only variant: prefix the previous command with SYMMETRIC_YAW=1 and write $D/pail_audit_symmetric.json
.venv/bin/python $D/object_screen.py
```

Replace the pail arguments with `20261023_215652_granola_bars granola_bars pnp_040_granola_bars` and granola's files for granola. `source_audit.py` reproduces the granola results exactly. `granola_pull.sftp` and `pail_pull.sftp` are the transfer batches used; the other PPS exports' raw sessions are on the collection machine under `/tmp2/leocheng/RoboLab/output/omy_teleop/`.
