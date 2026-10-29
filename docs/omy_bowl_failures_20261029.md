# Bowl grasp failures — 2026-10-29

The dominant observed failure is **an unstable rim grasp that tips or briefly lifts the bowl, then loses it while the close command remains active**. Preventing visible penetration addresses collision exclusion; it does not ensure that the contacts can support the bowl's weight and resist rotation. These recordings do not support blaming an absent close command, low material friction, or the earlier 16-iteration configuration.

All original recordings and production assets/configuration were left unchanged. Analysis, rendered saved states, and 12 separate diagnostic rollouts are in [`output/omy_diagnostics/20261029_bowl_failures`](../output/omy_diagnostics/20261029_bowl_failures).

## What is actually in the folder

| Recording | Attempts | Recorded successes | Other outcomes |
| --- | ---: | ---: | --- |
| `20261028_164247_bowl` | 0 | 0 | Preparation only |
| `20261028_164415_bowl` | 8 | 0 | Seven attempts containing a close command; one reset without closing |
| `20261028_164902_wooden_bowl` | 10 | 3 | Seven failed attempts |

There are 2,230 original action/state rows. All recorded leader samples are valid, and every recorded position-iteration cap is 32. The wooden-bowl successes are layouts 00, 01 and 03. Failures generally ended through manual reset/layout skip, not automatic task termination. This is a selected troubleshooting set, not a collection success-rate estimate.

The resolved configuration uses CUDA, TGS, 120 Hz physics, 15 Hz commands, two GPU partitions, original finger colliders and a 5 rad/s gripper velocity limit. The robot's 64-iteration request is capped at 32 by the scene. The close action is binary and targets approximately 0.7854 rad.

## Many apparent lifts are just tipping

I transformed every source bowl mesh vertex with each recorded object pose and measured its lowest world height relative to the initial lowest vertex. This is a visual-mesh clearance estimate, not a measurement of cooked-collider separation. Table motion is not used to infer contact forces.

**Ten of the fourteen failed attempts containing a close command never raised the lowest vertex more than 5 mm during closure.** Four did achieve more clearance before losing the bowl. Root height alone substantially overstates successful lifting because it increases as a bowl tips onto an edge.

| Original attempt | Maximum root rise | Maximum lowest-vertex rise | Result |
| --- | ---: | ---: | --- |
| Red bowl, layout 00, demo 0 | 42.2 mm | 0.5 mm | Tips against table, slips out |
| Red bowl, layout 00, demo 2 | 37.6 mm | 0.2 mm | Essentially no table clearance |
| Red bowl, layout 02, demo 1 | 171.2 mm | 123.9 mm | Actual temporary lift, then loss |
| Wooden bowl, layout 02, demo 1 | 24.1 mm | 0.2 mm | Tips, slips out |
| Wooden bowl, layout 00, demo 0 | 174.4 mm | 119.9 mm | Successful placement |
| Wooden bowl, layout 03, demo 0 | 331.0 mm | 256.3 mm | Successful placement |

![Root motion versus actual visual-mesh clearance](../output/omy_diagnostics/20261029_bowl_failures/tipping_vs_lifting.png)

For red-bowl layout 00/demo 0, the close command begins at action row 40 (2.667 s) and remains active through row 75; opening begins at row 76 (5.067 s). The bowl tilts about 59° by post-action row 64 and 73° by row 68. The master joint reaches near-full closure at row 66, even though the bowl has not cleared the tabletop. It then falls back upright before opening is requested. This is loss of useful contact during closure, not an early release command.

Rendered images restore original states without integrating the action trajectory. Images are **pre-action**; original `states` rows are **post-action**. Thus image row `i+1` corresponds to state row `i`.

- [Red-bowl failure, wrist sequence](../output/omy_diagnostics/20261029_bowl_failures/bowl00_demo0_wrist_cam.jpg)
- [Same failure, external camera](../output/omy_diagnostics/20261029_bowl_failures/bowl00_demo0_over_shoulder_left_camera.jpg)
- [Wooden-bowl failure, wrist sequence](../output/omy_diagnostics/20261029_bowl_failures/wood02_demo1_wrist_cam.jpg)
- [Wooden-bowl success, wrist sequence](../output/omy_diagnostics/20261029_bowl_failures/wood00_demo0_wrist_cam.jpg)

The successful wooden grasp keeps the wall captured between the fingers as the bowl rotates toward vertical and is transported. In the failed wooden example, the wall moves out of the finger region and the jaws finish closing. This supports a contact-placement/retention explanation, though images alone cannot certify an unpenetrated grasp.

## Contact measurements explain how it slips

The original recordings do not contain contact points, normals or per-finger forces. I made a diagnostic copy of the existing contact benchmark and added per-physics-step logs for both inner fingers. Saved states omit solver/contact caches; these are **new rollouts**, not retroactive force measurements of the originals.

Two baseline replays of red-bowl layout 00/demo 0 reproduced early loss. Maximum reported finger–bowl overlap was 0.426 and 0.402 mm. Thus a similar failure occurs without large penetration. These measurements cover the two inner-finger bodies against the bowl; they do not bound every robot/object collision.

The first baseline replay shows:

- During action rows 42–48, the left inner finger contacts the bowl, while the right has no measured bowl contact. Closure began at row 40.
- At the end of row 52, force-weighted contact heights are about **22.6 mm and 58.8 mm** above the world origin: approximately **36 mm apart vertically**, rather than a matched pinch across the thin wall at one height.
- Mean normal-force sums during row 52 are approximately **1.9 N and 1.3 N**. Active-substep medians over the rollout are 1.67 and 1.98 N; peaks are 9.91 and 9.13 N. These are sums of reported normal-force magnitudes, excluding friction forces, not total grasp wrenches.
- Contact locations move as the bowl rolls. Both fingers eventually lose useful contact; the last contact above 0.1 N occurs during row 70 in repeat 0 and row 68 in repeat 1, before opening at row 76.

![Both finger forces during a separate baseline replay](../output/omy_diagnostics/20261029_bowl_failures/baseline_contact_forces.png)

The source assets explain why this is demanding. Both bowls have mass **0.5 kg**, corresponding to about **4.9 N** weight. The red bowl is approximately 161 mm in diameter and 55 mm high, with roughly **1 mm side walls** and a thicker lip. The wooden bowl is approximately 280 mm in diameter and 130 mm high; sampled wall sections are about **11–13 mm** thick. Both diameters exceed the nominal 85 mm gripper opening, so the observed strategy grasps a wall/rim rather than enclosing the whole diameter.

A rim grasp carries the center of mass away from the contacts. In an approximately horizontal pose, the gravitational moment can be on the order of **0.4 N m** for the red bowl and **0.7 N m** for the wooden bowl, using the radius as an illustrative lever arm. Actual moments depend on orientation and contact location. The fingers need a contact arrangement that resists rotation as well as downward motion. Contacts on sloped surfaces at different heights can tip the bowl, move toward the lip, and disappear as the jaws close. Nonpenetration alone does not supply the missing resisting moment.

Both bowl and finger materials already author static/dynamic friction **2.0**; the finger material uses `max` combining. There is no evidence of accidentally frictionless materials. More friction is not a demonstrated remedy for an unfavorable or disappearing contact arrangement.

## The gripper constraints contribute, but simple parameter changes are not qualified fixes

The runtime drive still resolves to stiffness **5729.58 N m/rad**, damping **0.01146 N m s/rad**, and effort cap **16.5 N m**. The inner follower joints use compliant mimic constraints with natural frequency 1000 rad/s and damping ratio 0.05. A strong master motor does not guarantee equally strong, well-positioned opposing fingertip contacts. Small angular deflections can matter for a 1 mm shell, although mimic residuals alone do not measure penetration or prove the cause.

The original failed grasps have peak mimic residuals approximately 0.93–3.84°; successful wooden grasps reach 1.57–3.88°. Therefore a residual threshold alone does not separate success from failure.

I ran two repeats per comparison below on the **same red-bowl layout 00/demo 0 commands**. All keep 120 Hz, two GPU partitions, and the source initial state. The 64-iteration comparison ran on the other GPU. These are mechanism probes, not throughput measurements or statistically established improvements. Task success is not the right metric here: the source trajectory abandons the grasp instead of completing a placement.

| Diagnostic change | Contact/retention result before opening at row 76 | Maximum overlap, repeats 0 / 1 |
| --- | --- | ---: |
| Baseline, 32 iterations | Loses contact during rows 70 / 68 | 0.43 / 0.40 mm |
| 64 iterations | Still loses contact during rows 70 / 69 | 0.51 / 0.56 mm |
| Existing mass-preserving finger-hull profile | Still loses contact during row 67 in both | 0.80 / 1.10 mm |
| Rigid mimic constraints, natural frequency 0 | Contact persists until commanded opening; lower mimic error, but master reaches full closure, so retained contact is not proof of a normal opposed-pad pinch | 1.04 / 1.54 mm |
| Drive stiffness 275, damping 0.06 | Carries bowl until opening, with severe overlap and larger linkage errors | 8.78 / 54.53 mm |

The rigid-mimic result shows that constraint compliance affects retention in this setup; the increased overlap prevents calling it a validated solution. The lower-gain result is especially important: apparently better lifting can result from invalid contact. It must not be accepted on task/height metrics alone. NVIDIA's [Robotiq tuning guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html) similarly treats drive and mimic tuning as coupled and requires inspecting collision geometry, penetration and grasp behavior.

Two additional baseline replays cover wooden-bowl layout 02/demo 1. The first reproduced the slip with 0.60 mm maximum overlap. The second diverged, reporting a transient 39.82 mm overlap during action row 49 and a different trajectory. This is evidence that the replay/contact pipeline is sensitive and must not be substituted for the original wooden-bowl recording. The original recorded poses/images, not that divergent rollout, establish the reported failure.

## What I would address next

1. **Establish a stable rim grasp in a minimal scene.** Place the two pad contact regions opposite one another across the wall, with enough insertion to retain it during rotation. Test slow closure, a stationary hold and a vertical lift. Capture both fingers' forces, contact locations, bowl clearance and rotation. The successful wooden attempt is a useful reference, not a universal grasp template.
2. **Tune follower compliance together with the drive and pad geometry.** The current compliant linkage can change retention; simply increasing the scene cap or switching all four finger meshes to hulls did not resolve this example. Any candidate must retain and release without penetration or geometric hooking. The 275/0.06 trial fails that requirement.
3. **Verify the actual cooked rim/pad shapes and mass model.** The thin red shell and the uniformly assigned 0.5 kg mass make this a demanding rim-grasp test. Validate intended mass/COM rather than reducing mass merely to obtain success.
4. **Use a two-finger grasp criterion.** [`contact_gripper`](../robolab/robots/droid.py#L184) currently observes only `left_inner_finger`. One-sided contact can be recognized as grabbed even while the bowl is table-supported. Separate sensors plus a hold/lift criterion would distinguish contact from a secure grasp; this improves detection, not physical grip strength.

## Reproducibility and limits

- [Original episode measurements and source hashes](../output/omy_diagnostics/20261029_bowl_failures/recording_summary.json)
- [Mesh dimensions and sampled thickness](../output/omy_diagnostics/20261029_bowl_failures/asset_geometry.json)
- [Probe comparison](../output/omy_diagnostics/20261029_bowl_failures/probe_comparison.json)
- [Read-only recording analysis](../output/omy_diagnostics/20261029_bowl_failures/analyze_recordings.py), [mesh transformation analysis](../output/omy_diagnostics/20261029_bowl_failures/analyze_geometry.py), [probe summarization](../output/omy_diagnostics/20261029_bowl_failures/summarize_probes.py), and [diagnostic benchmark copy](../output/omy_diagnostics/20261029_bowl_failures/contact_probe.py).

Run recording analysis before mesh analysis, then probe summarization to regenerate the numerical reports/plots. Mesh analysis requires bundled `omni.usd.libs-*` on `PYTHONPATH`, its `bin` and the matching Python `lib` on `LD_LIBRARY_PATH`. Each probe's `summary.json` records the full arguments and resolved physics/drive/mass properties; `detailed*.npz` stores contact rows as `[substep, side, normal_force, point_xyz, normal_xyz, separation]` and states as `[13 joint angles, object xyz, object xyzw]`. Sides follow the contact view's requested order, left then right. Contact timestamps are `(substep+1)/120` s.

No production fix was applied. The evidence establishes the observed tipping/slipping mechanism and demonstrates sensitivity to constraint settings. It does not isolate one universal root cause or qualify a robust replacement configuration.
