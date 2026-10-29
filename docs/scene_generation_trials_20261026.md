# Scene qualification trials — 2026-10-26

The repaired pail layout passed physical starts, saved-scene reloads, a rim
grasp, transport, and supported release into the bin. The block stack passed
after enabling PhysX stabilization in its diagnostic profile. The original pail
layout reproduced the reported collision. These results cover these two
candidates and the original failure control, not the full scene library.

[Consolidated evidence](../output/scene_trials_20261026/summary.json) records the
qualified scene paths and hashes, reports, settings, and limitations. Each run
also retains its environment configuration, execution log, and rendered views.

| Trial | Result |
| --- | --- |
| Original `pnp_242_plasticpail_a01_15.usda` | Failed 3/3 starts; approximately 99.41 N unintended pail/bin contact; pail did not achieve tabletop support. |
| Repaired pail, authored poses | Passed 3/3 five-second starts; maximum object translation below 1.4 mm; no observed unintended contacts. |
| Repaired pail, exported poses reopened | Passed 3/3 five-second starts; the reloaded scene bytes match the final grasp/transfer scene. |
| Pail task predicates | Initial table and airborne controls stayed false. Released drops succeeded after 0.8 s, with matching subtask credit. Reset required a fresh settling interval. |
| Pail robot trial | Final rim grasp held the pail at least 99.4 mm above its start for one second with both fingers engaged. Transport retained contact; success remained false while held. Seating before opening produced released, stable success with neither finger touching. |
| Stack, default OMY physics | Failed all three original and all three reload trials on the 10 mm/s final-window velocity limit, despite very small pose drift. |
| Stack, PhysX stabilization enabled | Passed 3/3 starts and 3/3 saved-scene reloads; top-block final-window maximum speed below 3.5 mm/s. |
| Stack grasp and predicates under stabilization | Passed a 119 mm top-block lift/hold; both fingers maintained contact. Released stack and airborne controls behaved as expected. |
| Rendering | Both candidates and the original collision control rendered and were visually reviewed. Object colors, geometry, table texture, and robot were visible. MDL metallic/roughness warnings remain in logs; this is not photometric calibration. |

The physics protocol used Isaac Sim 5.0.0.0, Isaac Lab 2.2.0, DROID/OMY actors
and joint actions, 120 Hz physics, 15 Hz control, position-iteration cap 16, and
reset seeds 0–2. Each start ran for five simulated seconds with a final continuous
0.5-second stability window. Limits were 20 mm total translation, 15° tilt,
10 mm/s linear speed, 0.5 rad/s angular speed, and window drift within 1 mm/0.5°.
Diagnostics stepped physics directly to continue observing after task success;
they are not collection demonstrations or a policy success-rate evaluation.

The stack **requires** `cfg.sim.physx.enable_stabilization = True` for the passing
profile. Saving its USD does not encode that environment setting. The global
production default remains unchanged, and its default-profile failure remains
in the reports. Geometry checks on the exported stack found a 0.209 mm bounding
envelope overlap at the intended contact, a 22.5 mm center support margin, and
325.7 mm table-edge clearance; measured physics confirmed support.

The grasp trials exposed a misleading frame assumption: the flattened DROID
asset's fingers extend along `base_link` +X and close along Y. The old optional
+Z fingertip-offset comments were removed. The final pail candidate used yaw
90°, a rim offset of 53 mm, and 25 mm fingertip depth below the object's top.
The runner waits for bounded pose convergence before closing, samples unintended
robot contact at every physics step, and records both inner fingers separately.

Earlier attempts are retained: a shallow grasp slipped during transfer, and
opening a loaded rim grasp in midair left one finger hooked under the lip.
Converging at the approach pose, smoothing transport, and seating the pail on the
bin floor before opening resolved the tested transfer. A lift-only pass would
not have revealed the release problem.

The scene skill's validation reference now calls for inspecting actual tool
frames and finger coverage, distinguishes sandbox GPU visibility from host
availability, and bounds simulator execution and teardown. The trial runner
uses the repository's GPU-isolation lock, a 600-second execution limit, and
bounded teardown. Initial harness/export errors are preserved but excluded from
qualification evidence.

Reproduce the successful pail protocol from the repository root:

```bash
.venv/bin/python -B scripts/qualify_scene_trial.py \
  --host-gpu 1 --kind pail \
  --scene output/scene_trials_20261026/pail_reload/pail_pickup.settled.usda \
  --output-dir output/pail_requalification \
  --grasps 1 --candidate 1 --grasp-depth 0.025 --transfer
```

For the stack, use the exact scene path in `summary.json`, `--kind stack`,
`--stabilization`, and a new output directory. Read gate statuses in `report.json`:
a completed diagnostic process can intentionally contain failed trials, such as
the original pail control. Existing library scenes and tasks were not replaced.

Three reset seeds establish repeatability without randomizing initial poses.
These bounded scripted trials do not establish robustness across all poses,
robots, policies, or the other task-skill example scenes. The deliberately
pre-stacked fixture is suitable for initial-state qualification; its
`released_stack` predicate does not itself enforce sustained stability.
