# PPS proxy-pair evaluation results — 2026-11-04 to 2026-11-06

This report collects the PPS (proxy-pair steering) evaluations run on the
pi05-DROID joint-position base model during the Simple-PnP v2 work, with the
follow-up analyses. All success counts, paired comparisons, failure stages and
timings below were **recomputed from the saved `episode_results.jsonl` files**
on 2026-11-06. Where a number comes only from an analysis run interactively
during the sessions (no saved script), it is marked *(session analysis)*.

Campaigns:

| ID | Proxy pairs | Scenes | γ arms | Episodes | Status |
| --- | --- | --- | --- | ---: | --- |
| A | 9 pairs (`proxy_pairs_20261104`) | 50 v2 layouts per object | 0, 0.4, 0.8 | 1,350 | complete |
| B | same 9 pairs | each pair's own ~10 v1 **training** layouts | 0, 0.4, 0.8 | 276 | complete |
| B′ | same 9 pairs | same v1 training layouts | 0, with rate-limit flag | 92 | complete |
| C | fork_big, pitcher (`proxy_pairs_20261106_fork_big_pitcher`) | 50 v2 layouts per object | 0, 0.4, 0.8 | 300 | complete |

## Main results — all 11 pairs on v2

Successes out of 50 v2 layouts per object, one episode each. The three arms
share noise keys per scene. Pairs are grouped by batch and, within a batch,
sorted by success without steering. Batches A (the five non-jug objects) and
B (the four jugs) are both in `proxy_pairs_20261104` and were evaluated
together in campaign A; batch C is `proxy_pairs_20261106_fork_big_pitcher`.
The batch letters are not the campaign IDs.

| Object | Batch | v2, γ = 0 | v2, γ = 0.4 | v2, γ = 0.8 | Change at γ = 0.4 | Change at γ = 0.8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Plastic pail a01 |  | 38 | 28 | 13 | −10 | −25 |
| Granola bars | A | 40 | 30 | 7 | −10 | −33 |
| Chocolate pudding mix | A | 30 | 20 | 6 | −10 | −24 |
| Milk carton | A | 30 | 28 | 13 | −2 | −17 |
| **Cheez_it** | A | 11 | **30** | 8 | **+19** | −3 |
| Milk jug a01 | B | 45 | 1 | 0 | −44 | −45 |
| Utility jug a01 | B | 30 | 3 | 3 | −27 | −27 |
| Utility jug a02 | B | 26 | 3 | 1 | −23 | −25 |
| Milk jug a02 | B | 23 | 11 | 1 | −12 | −22 |
| Pitcher | C | 1 | 0 | 0 | −1 | −1 |
| **Fork_big** | C | 0 | **8** | 3 | **+8** | +3 |
| **Total, all 11 pairs (of 550)** | | **274 (50%)** | **162 (29%)** | **55 (10%)** | −112 | −219 |
| 95% CI, all 11 pairs | | 46–54% | 26–33% | 8–13% | | |
| Lost / gained by scene, all 11 pairs (sign-test p) | | | 183 / 71 (1e-12) | 239 / 20 (8e-49) | | |

The per-pair lost/gained counts and p-values are in the campaign A and C
sections below.

## Key findings

1. **On new (v2) layouts, steering lowers success, and stronger steering lowers it more.**
   For the 9 original pairs: 61% (base) → 34% (γ = 0.4) → 12% (γ = 0.8).
   Paired by scene, γ = 0.4 lost 182 scenes and gained 63 (sign test p ≈ 1e-14).
2. **On the pairs' own training layouts, γ = 0.4 is neutral.** 60/92 for both
   arms, with 21 scenes lost and 21 gained. So the harm on v2 comes mainly
   from the layouts being new, not from a broken serving pipeline.
3. **The harm grows with distance from the training layouts** *(session analysis)*:
   steered success falls 53% → 31% → 19% from near to far v2 scenes. Base
   success stays flat (66% / 55% / 61%).
4. **Steering helps only where the base policy is weak:** cheez_it (11 → 30 on v2,
   1 → 9 on v1), fork_big (0 → 8 on v2) and, earlier, FoodPacking (1% → 86%).
   Where the base policy already does well, it hurts.
5. **On v2, γ = 0.8 is never better than γ = 0.4.** It is worse for 9 of the
   11 pairs and ties for the other two (utility jug a01, 3 and 3; pitcher, 0 and 0).
6. **Steering turns failures into grasp failures.** On v2, the share of failures
   that happen at the grasp rises from 69% (base) to 87% (γ = 0.4) and 98% (γ = 0.8).
   The videos show the arm going into an elbow-high posture and never closing
   on the object.
7. **Milk jug a01's pair is broken in its own right.** It collapses even on its
   own training layouts (10/10 → 2/10, 8 lost and 0 gained, p = 0.008).
8. **A likely contributor is the demo style.** In every inspected training set,
   Franka joint 3 is locked at 0 and joint 5 barely moves: the 6-DOF OMY leader
   style. The steering probe confirms that the proxies pull actions toward that
   style.
9. **Open confound in the steered results: the gripper registration.** The demos
   were recorded under the OMY action registration, but all rollouts ran under
   standard DROID. γ = 0 runs are unaffected.

## Protocol (all campaigns)

- **Base model:** `pi05_droid_jointpos` (PyTorch checkpoint
  `/tmp2/leocheng/pps/runs/ckpt/pi05_droid_jointpos_pt`), served by `pps-serve`
  from `/tmp2/leocheng/pps`. At γ = 0, the base model is served through the same
  PPS server with `--mode none`.
- **Robot:** DROID Franka + Robotiq 2F-85 (`franka_robotiq_2f_85_flattened.usd`),
  standard DROID registration.
- **Rollout:** `policies/pps/run.py`, one env, one episode per scene, 8-action
  open-loop horizon at 15 Hz, 50 s timeout (750 steps), sensor video on.
- **Pairing:** each scene runs as episode 0 with the pair's training-task label
  (`--pps-task SimplePnp<pair>`), so all three γ arms share noise keys and can
  be compared scene by scene. A side effect: all 50 scenes of one object share
  one noise sequence.
- **Hardware:** two RTX 4090s, each with one PPS server (~10 GB) and **one**
  Isaac client. Two clients per GPU ran out of memory. CPUs pinned to
  `0-13,20-33`.
- **Statistics:** "lost" means base succeeded and steered failed; "gained" is
  the reverse. p-values are exact two-sided sign tests on lost vs gained.
  95% intervals are Wilson intervals.
- **Failure stage:** taken from the task's subtask reason. `object_grabbed` not
  satisfied counts as a **grasp** failure; `object_resting_in_container` not
  satisfied counts as a **place** failure.

### Preconditions verified before campaign A

- **v2 scene pool:** all 6,550 eligible slots pass geometry, physics (5/5 runs),
  Franka reach and, for handle objects, visibility. Getting there took 153 slot
  replacements over 20 top-up rounds, 38 of them cream cheese. Committed as
  `e1ef9000` (layouts) and `18dea617` (qualified tasks in
  `robolab/tasks/single_pnp_v2/`).
- **Success-condition check** (`tools/simple_pnp_v2/success_check.py`): one
  random layout per eligible object, 131 tasks, **131/131 passed**. Success never
  fired on an untouched scene. It always fired after the object was dropped in
  the bin, 9–28 steps later (median 11).
- **Smoke test:** 2 cheez_it scenes at γ = 0.8 confirmed that the server accepted
  the pair label and prompt, and that results, videos, HDF5 and event logs
  were written.

### Reference point: FoodPacking1BoxesTask (2026-10-18)

An earlier pair, evaluated on the same single benchmark scene its demos came
from, with 100 episodes per γ
(`/tmp2/leocheng/pps/runs/droid/FoodPacking1BoxesTask/fast_eager_gpu0_gamma_sweep_20261018/`):

| γ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Success | 1% | 1% | 71% | 83% | 86% | 77% |

## Campaign A — 9 pairs on 50 v2 layouts each

Output: [`output/pps_v2_eval_20261104/gamma_<g>/<pair>/`](../output/pps_v2_eval_20261104/).
Each pair was trained on 50 demos (OMY leader + MimicGen) from about 10 v1
layouts of its object. v2 layouts were generated to be at least 3 cm from every
v1 layout.

### Success per object (out of 50)

| Object | v2, γ = 0 | v2, γ = 0.4 | v2, γ = 0.8 | v2, lost / gained (γ = 0.4, p) | v2, lost / gained (γ = 0.8, p) |
| --- | ---: | ---: | ---: | --- | --- |
| Milk jug a01 | 45 | 1 | 0 | 44 / 0 (1e-13) | 45 / 0 (6e-14) |
| Granola bars | 40 | 30 | 7 | 13 / 3 (0.021) | 34 / 1 (2e-9) |
| Plastic pail a01 | 38 | 28 | 13 | 16 / 6 (0.053) | 28 / 3 (5e-6) |
| Chocolate pudding mix | 30 | 20 | 6 | 18 / 8 (0.076) | 26 / 2 (3e-6) |
| Milk carton | 30 | 28 | 13 | 13 / 11 (0.84) | 19 / 2 (2e-4) |
| Utility jug a01 | 30 | 3 | 3 | 28 / 1 (1e-7) | 28 / 1 (1e-7) |
| Utility jug a02 | 26 | 3 | 1 | 25 / 2 (6e-6) | 25 / 0 (6e-8) |
| Milk jug a02 | 23 | 11 | 1 | 20 / 8 (0.036) | 23 / 1 (3e-6) |
| **Cheez_it** | 11 | **30** | 8 | **5 / 24 (5e-4)** | 10 / 7 (0.63) |
| **Total (of 450)** | **273 (61%)** | **154 (34%)** | **52 (12%)** | **182 / 63 (1e-14)** | **238 / 17 (5e-51)** |
| 95% CI | 56–65% | 30–39% | 9–15% | | |

### Failure stage (all 9 objects)

| Arm (v2) | Failures | Grasp | Place | Grasp share |
| --- | ---: | ---: | ---: | ---: |
| γ = 0 | 177 | 122 | 55 | 69% |
| γ = 0.4 | 296 | 257 | 39 | 87% |
| γ = 0.8 | 398 | 390 | 8 | 98% |

Steering does not mainly hurt placement; it stops the arm from grasping at all.
Cheez_it is the clearest example of a gain: its base failures were mostly at the
place step (20 of 39), and at γ = 0.4 those fall to 6.

### Failure videos (γ = 0.4)

248 side-by-side videos, base on the left and γ = 0.4 on the right, frame-synced,
in H.264:
[`failure_videos_gamma_0.4/<pair>/<grasp|place>/`](../output/pps_v2_eval_20261104/failure_videos_gamma_0.4/),
with an `index.json`. They were rendered by `render_failures.py`, which also
accepts `--gamma 0.8`.

**Stacked videos (γ = 0 / 0.4 / 0.8, top to bottom):** all 296 γ = 0.4 failures
in
[`failure_videos_gamma_stack/<pair>/<stage at γ = 0.4>/<Task>__g0_<S|F>_g0.4_<S|F>_g0.8_<S|F>.mp4`](../output/pps_v2_eval_20261104/failure_videos_gamma_stack/),
1280 × 1092, H.264, with an `index.json`. Rendered by
`output/pps_v2_eval_20261104/render_gamma_stack.py`.

The typical pattern, from milk jug a01 layout 00: both arms start the same way.
By about 6 s the steered arm lifts its elbow high and twists the wrist, hovers
beside the jug with the gripper open, then drifts away until the timeout. The
base arm reaches straight down, grasps, and places the jug.

## Campaign B — the same 9 pairs on their own v1 training layouts

Output: [`output/pps_v2_eval_20261104/v1train/`](../output/pps_v2_eval_20261104/v1train/).
92 layouts (10–11 per object), the same ones each pair's demos were recorded on.
The v1 scenes never went through the v2 physics, reach and visibility gates,
but both arms see the same scenes.

Without the flag, γ = 0 and γ = 0.4 ran on these scenes. With the render rate
limiter disabled (`--kit_args "--/app/runLoops/main/rateLimitEnabled=false"`),
γ = 0 ran again (campaign B′) along with γ = 0.8. **Compare each γ only with the
γ = 0 run made under the same flag setting.**

| Object | v1, γ = 0 | v1, γ = 0.4 | Lost / gained (γ = 0.4) | v1, γ = 0, flag | v1, γ = 0.8, flag | Lost / gained (γ = 0.8, vs flagged γ = 0) | v2, γ = 0 | v2, γ = 0.4 |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Chocolate pudding mix | 9/11 | 8/11 | 3 / 2 | 5/11 | 10/11 | 0 / 5 | 30 | 20 |
| Granola bars | 11/11 | 10/11 | 1 / 0 | 11/11 | 11/11 | 0 / 0 | 40 | 30 |
| Milk carton | 5/10 | 7/10 | 2 / 4 | 6/10 | 9/10 | 0 / 3 | 30 | 28 |
| Milk jug a01 | 10/10 | **2/10** | **8 / 0** (p = 0.008) | 10/10 | 5/10 | 5 / 0 (p = 0.06) | 45 | 1 |
| Milk jug a02 | 4/10 | 4/10 | 2 / 2 | 7/10 | 9/10 | 1 / 3 | 23 | 11 |
| Plastic pail a01 | 10/10 | 10/10 | 0 / 0 | 10/10 | 9/10 | 1 / 0 | 38 | 28 |
| Utility jug a01 | 6/10 | 4/10 | 4 / 2 | 9/10 | 8/10 | 2 / 1 | 30 | 3 |
| Utility jug a02 | 4/10 | 6/10 | 1 / 3 | 6/10 | 6/10 | 3 / 3 | 26 | 3 |
| **Cheez_it** | 1/10 | **9/10** | **0 / 8** (p = 0.008) | 2/10 | 9/10 | **0 / 7** (p = 0.016) | 11 | 30 |
| **Total** | **60/92 (65%)** | **60/92 (65%)** | **21 / 21 (p = 1)** | **66/92 (72%)** | **76/92 (83%)** | **12 / 22 (p = 0.12)** | 273/450 | 154/450 |

### Flag check (campaign B′): repeated runs are not identical

Running γ = 0 with the flag on the same 92 scenes and the same noise keys gave
66/92, against 60/92 without it. Scene by scene, 8 were lost and 14 gained
(p = 0.29). So the totals don't differ significantly, but the episodes
themselves differ:

- **End steps:** of the 52 scenes that succeeded in both runs, only 5 ended on
  the same step.
- **Outcomes:** 22 of 92 scenes changed outcome. Chocolate pudding mix alone
  went from 9/11 to 5/11.

The earlier evidence that the flag doesn't change outcomes came from a 128-step
**open-loop** action replay, which gave bit-identical values in all 42 numeric
HDF5 datasets
([pps_evaluation_performance_20261106.md](pps_evaluation_performance_20261106.md)).
In closed loop, the policy sees rendered images, and small differences there
compound. **It is not yet known whether the flag causes the divergence, or
whether closed-loop runs never repeat exactly, flag or not.** Unflagged runs
have never been repeated. Either way, a single episode per scene carries
run-to-run noise of several scenes per object, beyond the paired-noise design.

**What it means for γ = 0.8 on v1:** compared with the flagged γ = 0 run, the
apparent gain shrinks from 10 lost / 26 gained (p = 0.011, against the
unflagged base) to 12 lost / 22 gained (p = 0.12), which is not significant.
Per object, the strong effects persist: cheez_it is still helped (2 → 9, 0 / 7)
and milk jug a01 still hurt (10 → 5, 5 / 0). Two other observations: γ = 0.8 had
no place failures on v1 (16 grasp, 0 place), and its median successful
episode was short (118–157 steps, against 124–189 at γ = 0.4).

**What campaign B shows:**
- **New layouts are the main source of the γ = 0.4 harm.** Training layouts are
  neutral (21/21), while v2 is strongly negative (182/63).
- **The utility jugs show this most clearly:** roughly unchanged on v1
  (6 → 4, 4 → 6), but collapsing on v2 (30 → 3, 26 → 3).
- **Two pairs behave the same everywhere:** milk jug a01 hurts and cheez_it
  helps, on both v1 and v2.
- **γ = 0.8 on training layouts is not clearly better than no steering**
  (76 vs 66 of 92 under the same flag setting, p = 0.12), while on v2 it is the
  worst arm (52/450). The gap between training and new layouts is at least as
  large as at γ = 0.4.
- **Sample size:** about 10 scenes per object with one episode each, so
  per-object differences of 2–3 scenes are noise.

## Distance from the training layouts *(session analysis)*

Distance is the larger of the target's and the bin's offset to the nearest
training layout, using positions only (no yaw). This was computed interactively
on 2026-11-06 and no script was saved.

**How far v2 is from the training layouts:**

| Per pair | Training layouts | Unseen v1 layouts | v2 layouts |
| --- | --- | --- | --- |
| Median distance to the nearest training layout | 0.09–0.15 m (to the nearest *other* one) | 0.10–0.16 m | 0.15–0.19 m |
| Target outside the area spanned by the training targets | n/a | 38–70% | 76–94% |

v2 positions spread about 10–45% wider than v1 in each direction, so most v2
scenes need the proxies to extrapolate.

**Success by distance (all 450 v2 scenes, in thirds):**

| Distance to nearest training layout | γ = 0 | γ = 0.4 | Lost / gained |
| --- | ---: | ---: | --- |
| Near (≤ 0.14 m) | 66% | 53% | 47 / 27 |
| Middle (0.14–0.20 m) | 55% | 31% | 62 / 25 |
| Far (0.20–0.43 m) | 61% | 19% | 73 / 11 |

Leaving out milk jug a01 and cheez_it gives the same pattern: steered success
54% → 28% → 23%, against a base of 71% → 57% → 58%. Even the nearest v2 scenes
lose more than they gain (47/27), while the training layouts break even. So
distance does not explain everything. Not yet measured: target yaw, and the v2
scene-construction changes (contact starts, kept root rotations, ground plane
at the table bottom). Any of these may shift the camera views relative to the
v1 demos.

## Campaign C — fork_big and pitcher on 50 v2 layouts each

Output: [`output/pps_v2_eval_20261106_fork_pitcher/`](../output/pps_v2_eval_20261106_fork_pitcher/).
Both pairs were trained on data collected on this server, 50 demos each from
8–10 v1 layouts, exported with `export_session` (`43e5b2a7`):
`output/omy_pps/20261105_142025_fork_big/pps.hdf5` (8,927 rows) and
`.../20261105_145739_pitcher/pps.hdf5` (7,960 rows). The pairs' training
manifests match these files' hashes.

| Object | v2, γ = 0 | v2, γ = 0.4 | v2, γ = 0.8 | v2, lost / gained (γ = 0.4, p) | v2, lost / gained (γ = 0.8, p) |
| --- | ---: | ---: | ---: | --- | --- |
| Fork_big | 0 | **8** | 3 | 0 / 8 (0.008) | 0 / 3 (0.25) |
| Pitcher | 1 | 0 | 0 | 1 / 0 | 1 / 0 |

| Object, arm | Grasp failures | Place failures |
| --- | ---: | ---: |
| Fork_big γ = 0 / 0.4 / 0.8 | 50 / 41 / 45 | 0 / 1 / 2 |
| Pitcher γ = 0 / 0.4 / 0.8 | 20 / 30 / 50 | 29 / 20 / 0 |

- **Fork_big:** the base never grasps the fork. γ = 0.4 succeeds on layouts 04,
  17, 20, 21, 23, 44, 48 and 49. γ = 0.8 succeeds on 08, 14 and 35, with no
  overlap. With one episode per layout, the difference between 0.4 and 0.8 is
  not established.
- **Pitcher:** the base often grasps but fails to place it. Steering moves
  failures back to the grasp stage, and at γ = 0.8 every episode fails at the
  grasp. The single base success is layout 30, at step 717. Videos of that
  scene for base and γ = 0.4 are in
  [`videos/`](../output/pps_v2_eval_20261106_fork_pitcher/videos/)
  (`pitcher30_base_vs_gamma0.4.mp4` and stills), plus all three arms stacked
  in `SimplePnpV2303Pitcher30Task__gamma_stack.mp4`.

## Mechanism analyses

### Training data vs Franka rollouts

Checked on the five datasets available locally (chocolate pudding mix, granola
bars, milk carton, cheez_it, plastic pail) plus fork_big and pitcher. The four
jug datasets are not on this host.

- **Joint 3 is locked.** The commanded `panda_joint3` is exactly 0 in every frame
  of every demo, and the measured value stays within ±0.06 rad. The base model
  uses −0.49 to +0.58 rad in its v2 runs. The PPS exporter reports joint 3 as
  the only constant column.
- **Joint 5 barely moves.** Its spread is 0.06–0.12 rad, against 0.25–0.68 rad
  for the other joints.
- **Start pose and gripper:** the start pose matches the evaluation reset. The
  gripper encoding is OMY ±1 in the raw demos and 0/1 in the PPS exports.

### Steering probe (`steer_probe.py`, `steer_probe.json`)

30 demo frames per object were sent to the server twice with identical noise,
once unsteered and once at γ = 0.4 with the object's pair. The returned 15-step
action chunks were compared against the demo actions.

| Pair | Mean \|joint 3\| target, base → γ = 0.4 (rad) | Reduction in error to demo actions, per joint 1–7 |
| --- | --- | --- |
| Chocolate pudding mix | 0.092 → 0.035 | 51, 55, 62, 27, 37, 58, 27% |
| Granola bars | 0.073 → 0.030 | 31, 41, 60, 28, 44, 54, 17% |
| Milk carton | 0.105 → 0.034 | 46, 46, 67, 48, 54, 50, 45% |
| Cheez_it | 0.103 → 0.039 | 47, 10, 62, 8, 30, 29, 24% |
| Plastic pail a01 | 0.039 → 0.016 | 21, 41, 57, 22, 47, 49, 21% |

On training frames, steering moves the action chunk toward the demos on every
joint, by 8–67%, most strongly on joint 3. **The pairs do what they were trained
to do**, and the server, prompts, arm selection and noise pairing behave
correctly. This makes a serving bug unlikely. It also means the steering pulls
toward the constrained 6-DOF OMY motion style.

### Current interpretation

The proxies were trained on about 10 layouts each, with joint 3 locked and joint
5 nearly fixed. Near those layouts they are roughly neutral. Farther away they
pull a base policy that would otherwise work into postures that never grasp.
Where the base policy has no working strategy (fork_big, cheez_it,
FoodPacking), that prior is useful; where it already has a good 7-DOF solution,
it gets in the way.

This is not proven. A corrected claim from the sessions: the OMY-vs-Franka
mismatch was first put forward as the main cause. That was withdrawn because the
FoodPacking pair had the same OMY-sourced data and still helped, and because
campaign B was neutral on training layouts.

### Gripper registration (open confound)

The PPS export spec (§1.1) requires rollouts of OMY-recorded data to use the
OMY action registration (`OmyDroidActionsCfg`). It also requires the client to
convert the server's 0/1 gripper output to the OMY command: −1 (close) if
g ≥ 0.5, otherwise +1 (open). All campaigns here ran under the standard DROID
registration, whose gripper speed limit differs, and
`policies/pi0_family/client.py` sends the rounded 0/1 value unchanged. The γ = 0
arms are unaffected; every steered result carries this mismatch. Adding the OMY
rollout option to the client was proposed but not decided.

## Throughput

Medians of per-episode `it_per_sec` (control steps per second) and means of
per-step timings, from `timing` in the results:

| Run | Env step (ms) | Policy inference (ms) | Steps/s |
| --- | ---: | ---: | ---: |
| v2, γ = 0 (base) | 162–175 | 22.2–22.9 | 5.0–5.3 |
| v2, γ = 0.4 / 0.8 (steered) | 157–175 | 31.6–35.4 | 4.7–5.1 |
| v1, γ = 0.4, no flag | 164–168 | 35.4–37.1 | 4.8–5.0 |
| v1, γ = 0.8, **rate limiter disabled** | 108–119 | 35.0–35.7 | 6.2–6.8 |

- **Steering cost:** about 12 ms per request (+50% inference), small next to
  the environment step.
- **Rate limiter:** disabling it cut environment time by about 30%. Failed
  episodes take about 107 s instead of 152 s.
- **Episode length:** failed episodes run the full 750 steps, so campaign
  wall time scales with the failure rate. It was about 2.5 min per failed
  episode without the flag.
- **Profiling:** see
  [pps_evaluation_performance_20261106.md](pps_evaluation_performance_20261106.md).

## Pending work and suggested next steps

1. **Measure run-to-run repeatability.** Repeat the unflagged γ = 0 run on the
   92 v1 scenes. If it diverges as much as B′ did, closed-loop runs simply don't
   repeat and the flag is safe to use; if it repeats exactly, the flag changes
   behaviour. Either way, run every arm under the same flag setting from now
   on, and consider several episodes per scene.
2. **Unseen-v1 control:** about 40 v1 layouts per object that the pairs didn't
   train on, base and γ = 0.4, roughly 720 episodes (about 8–10 h with the flag).
   This separates "new layout" from "v2-specific construction".
3. **Offline proxy diagnostics (test 2):** compare the reference proxy against
   the base model's velocity on demo frames and on Franka rollout frames,
   binned by layout distance. This can run now for fork_big and pitcher, whose
   training data is local; the original nine pairs' corpora are only on the HPC.
4. **Resolve the gripper registration:** add the OMY rollout option, then
   re-run at least fork_big and cheez_it.
5. **Better pairs:** train on more diverse layouts, for example demos on part
   of the 50 v2 layouts per object, evaluated on the held-out rest. Optionally
   unlock joint 3, or filter joints 3 and 5 out of the steering residual.
6. **Finer γ:** γ = 0.1 and 0.2, plus a naive (task-proxy-only) ablation, on
   cheez_it and fork_big where steering helps.

## Artifacts and reproduction

| Item | Path |
| --- | --- |
| Campaign A launchers, logs, results | `output/pps_v2_eval_20261104/` (`server.sh`, `worker.sh`, `jobs_*`, `gamma_<g>/<pair>/episode_results.jsonl`) |
| Campaign B / B′ | `output/pps_v2_eval_20261104/v1train/` (`worker.sh`, `worker_fast.sh`, `worker_flagcheck.sh`, `queue_flagcheck.sh`) |
| Campaign C | `output/pps_v2_eval_20261106_fork_pitcher/` (`start_after_test1.sh`, `server.sh`, `worker.sh`, `tasks/`) |
| Result summarizer (all tables in this report) | `output/pps_v2_eval_20261104/summarize_results.py OUT.json` |
| Job completion check | `output/pps_v2_eval_20261104/check_done.py` |
| Failure-video renderer | `output/pps_v2_eval_20261104/render_failures.py` |
| Steering probe | `output/pps_v2_eval_20261104/steer_probe.py`, `steer_probe.json` |
| Proxy pairs | `/tmp2/leocheng/pps/runs/droid/proxy_pairs_20261104/`, `/tmp2/leocheng/pps/runs/droid/proxy_pairs_20261106_fork_big_pitcher/` |
| v2 qualification record | `output/simple_pnp_v2_validation/production/` (`final_report.json`, `gate_status.json`, `passed_tasks.txt`) |
| Success-check results | `output/simple_pnp_v2_validation/success_check/sample/` |

Per-episode rollout command (campaigns A and C; B uses `--task-dirs single_pnp`,
and the flagged runs add the `--kit_args` value):

```bash
OMNI_KIT_ACCEPT_EULA=YES taskset -c 0-13,20-33 timeout 21600 \
  .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:$G \
  policies/pps/run.py --task $(cat tasks/$pair.txt) --task-dirs single_pnp_v2 \
  --num-envs 1 --num-runs 1 --headless --video-mode sensor --renderer realtime \
  --mode pps --gamma $gamma --arm $pair \
  --pps-task SimplePnp$pair --open-loop-horizon 8 \
  --remote-host 127.0.0.1 --remote-port 81${G}0 --output-folder-name $out
# γ = 0: --mode none instead of the three steering flags
# fast runs: --kit_args "--/app/runLoops/main/rateLimitEnabled=false"
```

`policies/pps/run.py` and `policies/pps/client.py` are **untracked in git**.
Commit them to make these runs reproducible. PPS server: `/tmp2/leocheng/pps`
at `b9356ad`.
