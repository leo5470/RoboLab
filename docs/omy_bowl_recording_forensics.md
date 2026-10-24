# Bowl recording forensics — 2026-10-24

The reported bowl recording used a 16-position-iteration scene cap. It also combines a roughly 1 mm bowl wall, high contact friction, and an inherited gripper drive whose effective stiffness is **5,729.58 N m/rad**. Recorded states show the driven finger opening fully while the bowl remains attached to one finger. These observations identify several concrete contributors; the recording alone cannot establish which change prevents penetration.

This report describes the original recording and read-only asset analysis. Subsequent simulator measurements are referenced explicitly where used to resolve drive units. The accompanying diagnostic replays are separate rollouts and must not replace the original recorded states or images.

## Evidence and provenance

- Original recording: [`omy_demos.hdf5`](../output/omy_teleop/20261024_141641_bowl/SimplePnp291Bowl00Task/part_000/omy_demos.hdf5), group `data/demo_1`.
- Original resolved configuration: [`env_cfg.json`](../output/omy_teleop/20261024_141641_bowl/SimplePnp291Bowl00Task/part_000/env_cfg.json).
- Original timings: [`attempt_0001_performance.json`](../output/omy_teleop/20261024_141641_bowl/SimplePnp291Bowl00Task/part_000/attempt_0001_performance.json) and the adjacent `attempt_0001_timing.npz`.
- Original predicate/event evidence: [`collection_events.jsonl`](../output/omy_teleop/20261024_141641_bowl/SimplePnp291Bowl00Task/part_000/collection_events.jsonl).
- Camera materialization: `output/omy_teleop/20261024_141641_bowl/postprocess/20261024T064002_3ae34b2d16a5/rendered/SimplePnp291Bowl00Task/part_000/SimplePnp291Bowl00Task_20261024T061714097282Z_a0002/images.hdf5`.
- Bowl source: [`assets/objects/ycb/bowl.usd`](../assets/objects/ycb/bowl.usd).
- Robot source: [`assets/robots/franka_robotiq_2f_85_flattened.usd`](../assets/robots/franka_robotiq_2f_85_flattened.usd).

SHA-256 values at analysis time:

| File | SHA-256 |
| --- | --- |
| Original `omy_demos.hdf5` | `e6f2fe6e0bf8403ec0f4616bf36690e7f22f08d398e9b037db2fe553ccdcbc19` |
| `bowl.usd` | `4674da9935073ec0355cf3e27ead3b87ab98588e1d9bbb116eedaa6fad3ce98e` |
| `franka_robotiq_2f_85_flattened.usd` | `f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41` |

`demo_0` contains four samples and ended for `leader_stale`. The relevant `demo_1` contains 1,800 samples, has no invalid recorded leader samples, and ended with `terminal_failure`. The postprocessing quality report marks its timing and structural alignment valid and retains its recorded failure outcome. Structural validity does not certify contact fidelity.

## Physics settings actually used

The HDF5 `data` attributes contain `robolab_omy_teleop` JSON and simulator version strings. They agree with the resolved configuration:

| Setting | Original collection |
| --- | --- |
| Isaac Sim / Isaac Lab | 5.0.0.0 / 2.2.0 |
| Physics device | `cuda:0` |
| Profile label | `cpu`; explicit device selection nevertheless retained CUDA physics |
| Solver | TGS (`solver_type=1`) |
| Physics timestep | 1/120 s |
| Command rate / decimation | 15 simulated Hz / eight physics steps per action |
| Scene position iteration range | 1–16 |
| Robot articulation position request | 64, limited by the scene cap |
| Scene velocity iteration range | 0–1 |
| Robot articulation velocity request | 0 |
| Scene CCD | Enabled |
| Stabilization | Disabled |
| Stepping | `held-target`, no fallback reasons |
| Render interval | Four physics steps; nominally two renders per command |
| Viewing | SteamVR; resolution multiplier 0.8 |
| Recording | Compact, state only, deferred images |
| Contact sensors | Required profile, seven of seven selected, history length zero |
| Kit/TBB workers / physics workers | Eight / two |
| Gripper velocity limit | `velocity_limit_sim=5.0` rad/s |

The gripper velocity-limit correction was already present in this recording. The legacy ignored `velocity_limit` field does not explain this example. The 16-iteration scene cap is a real accuracy reduction even though the USD articulation still requests 64.

## Recorded failure sequence

Rows below are zero-based action indices. Action row `i` begins at episode time `i/15`; `states/...` row `i` is measured after that action, at `(i+1)/15`. Camera images in the materialized dataset restore the pre-action state. Thus a sheet cell intentionally shows a pre-action image together with a clearly labeled post-action state; they are separated by one 1/15 s action interval.

| Event | Row | Episode time | Recorded evidence |
| --- | ---: | ---: | --- |
| Close command begins | 53 | 3.533 s | Action changes from `+1` to `-1` |
| Closing temporarily stalls | 54–58 | Actions begin 3.600–3.867 s | Driven joint remains approximately 0.376–0.382 rad |
| Almost full closure reached | 65 | State at 4.400 s | Driven joint 0.7847 rad; bowl root height 41.29 mm |
| Open command begins | 100 | 6.667 s | Action changes from `-1` to `+1` |
| Driven joint reaches approximately zero | 103 | State at 6.933 s | Bowl root remains at 248.86 mm |
| Open gripper continues carrying bowl | 104–1799 | Through 120 s | Images show a bowl hanging from one finger |

After row 103 the bowl's minimum root height is 119.23 mm. Its final position is approximately `(0.35642, -0.02622, 0.27849)` m. After row 240 the arm varies by less than 0.00134 rad per joint, while the bowl position varies by less than 0.00067 m per axis for the remaining episode. This is persistent attachment after the open command, rather than a brief release delay.

The driven joint's sampled maximum closing velocity is 4.99885 rad/s, consistent with the enforced 5 rad/s limit. Maximum sampled opening speed is about 2.97042 rad/s. These are action-boundary samples and can miss peaks within the eight physics steps.

### Visual evidence

The sheets were extracted from the existing materialized images with Pillow. No physics was rerun to create them. Images are resized for review; original camera frames remain in the materialized HDF5.

![Grasp, release and persistent attachment from the wrist camera](../output/omy_diagnostics/20261024_bowl_contacts/evidence/bowl_wrist_sheet.png)

![Closure transition around rows 58–73](../output/omy_diagnostics/20261024_bowl_contacts/evidence/bowl_transition_sheet.png)

![Full episode sampled every 90 actions](../output/omy_diagnostics/20261024_bowl_contacts/evidence/bowl_contact_sheet.png)

The images establish visible attachment and support the reported symptom. They do not measure the surfaces of the cooked collision hulls. A rim hanging on a geometric recess, frictional locking, and actual collider penetration require further contact measurements to separate.

## Gripper constraint errors

The recorded joint order is seven arm joints followed by:

```text
finger_joint
right_outer_knuckle_joint
left_inner_finger_joint
right_inner_finger_joint
left_inner_finger_knuckle_joint
right_inner_finger_knuckle_joint
```

Using the asset's mimic gearing, the expected follower signs relative to `finger_joint` are `[+1, -1, +1, -1, -1]`. The residual is the measured follower angle minus its expected signed reference angle:

| Follower | Maximum absolute residual | Row | Median residual after row 103 |
| --- | ---: | ---: | ---: |
| Right outer knuckle | 0.03084 rad / 1.77° | 65 | +0.000034 rad |
| Left inner finger | 0.08653 rad / 4.96° | 65 | −0.000365 rad |
| Right inner finger | 0.05022 rad / 2.88° | 101 | +0.02027 rad / 1.16° |
| Left inner finger knuckle | 0.01517 rad / 0.87° | 64 | −0.000213 rad |
| Right inner finger knuckle | 0.00995 rad / 0.57° | 72 | −0.002346 rad |

There is clear constraint error during grasp and release, with a persistent right inner finger error after the driven joint has opened. The source records joint coordinates, not all individual link poses or joint-anchor separation, so this table does not bound total gripper deformation.

## Bowl and finger collision geometry

The bowl default prim is `/bowl`, with collision source `/bowl/obj_000013_Mesh`. The stage uses metres. The mesh contains 8,323 source vertices and 15,728 triangular faces; bounds are approximately `±80.72 × ±80.56 × ±27.485` mm. After merging duplicate seam vertices it has 7,866 vertices, consistent winding, a single watertight component, and Euler characteristic two. Unmerged topology has duplicate seam vertices, so an unprocessed boundary-edge count would incorrectly suggest a hole.

The authored bowl collision approximation is `convexDecomposition`, with `maxConvexHulls=256`, `errorPercentage=1.0`, and `shrinkWrap=true`. **256 is a maximum cooking request, not an observed runtime hull count.** Both inner fingers also use convex decomposition for their finger and fingertip geometry. Collision complexity therefore deserves direct cooked-shape inspection before selecting a simpler representation.

Radial ray intersections through the source bowl mesh show a shell approximately 1 mm thick. For each listed local height, 36 rays originate on the bowl axis and point horizontally at equally spaced azimuths. All 36 rays at each height have two forward intersections. Their radial separation gives:

| Local height | Median wall thickness | Minimum | Maximum |
| ---: | ---: | ---: | ---: |
| −20 mm | 1.223 mm | 1.001 mm | 1.457 mm |
| −10 mm | 1.130 mm | 0.975 mm | 1.347 mm |
| 0 mm | 1.014 mm | 0.775 mm | 1.298 mm |
| +10 mm | 1.020 mm | 0.844 mm | 1.390 mm |
| +20 mm | 0.992 mm | 0.716 mm | 1.385 mm |

These are sampled radial thicknesses, not minimum normal thickness over the whole surface, and do not describe the cooked convex hulls. At 5 rad/s and 120 Hz the driven angle can advance 0.04167 rad per physics step. Depending on its lever arm, a fingertip can sweep several millimetres while meeting this approximately 1 mm source shell. This makes collision detection, shape approximation, and gripper speed meaningful variables even if solver iterations are increased.

The scene and both inner finger bodies already have CCD enabled; the bowl also authors `physxRigidBody:enableCCD=true`. Adding the same flag again is not a demonstrated fix.

## Contact materials and drive units

The bowl authors mass 0.5 kg, static and dynamic friction 2.0, and restitution 0.1. Both inner fingers bind the Robotiq `PhysicsMaterial`, which also has static and dynamic friction 2.0 and `frictionCombineMode="max"`. A friction trial must modify the finger material as well as the object, or the pair can retain friction 2.0 through the maximum combine rule.

The gripper config leaves stiffness and damping unspecified, so the values come from the robot USD. Angular USD drive gains must be distinguished from Isaac Lab's runtime gains:

| Gain | Authored angular USD value | Effective runtime value |
| --- | ---: | ---: |
| Stiffness | 100 N m/degree | **5,729.578125 N m/rad** |
| Damping | 0.0002 N m s/degree | **0.01145915594 N m s/rad** |
| Maximum force/torque | 16.5 N m | 16.5 N m |

The conversion is multiplication by `180/pi`. The effective values were read from `robot.data` in the later, unmodified-gain diagnostics; see the `gripper` object in [`cpu32/summary.json`](../output/omy_diagnostics/20261024_bowl_contacts/cpu32/summary.json) and [`cpu64/summary.json`](../output/omy_diagnostics/20261024_bowl_contacts/cpu64/summary.json), with corresponding `collider_drive_audit.json` files for the authored values. This is a subsequent runtime audit of the same asset/configuration, rather than a new field retroactively present in the original recording.

Reading the authored number as 100 N m/rad understates the effective stiffness by about 57.3 times. Gain experiments must use the intended API's units and log the resolved runtime values. The combination of high stiffness, little damping, force saturation and mimic constraints deserves testing alongside solver count.

The effective stiffness closely matches the approximately 5,700 N m/rad Robotiq case in NVIDIA's [gripper tuning guide, drive stiffness and damping](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html#drive-stiffness-damping). That guide documents the USD-to-PhysX `180/pi` conversion and relates excessive stiffness to the gripper's small inertia and simulation rate. Its worked example derives stiffness around 275 and damping around 0.06; these are candidates to evaluate with this asset and timestep, not calibrated production values. See also the broader [contact physics source review](omy_contact_physics_sources.md).

## Release detection covers one finger

[`contact_gripper` in `droid.py`](../robolab/robots/droid.py) maps the logical `gripper` to only `left_inner_finger`. Its adjacent comment explains that Isaac Lab's filtered contact sensor requires one body per environment; matching both fingers in one sensor breaks the filtered force matrix.

In this episode, `placement_check` events report `gripper_detached=true` after row 101 even while visual evidence shows continued attachment. The persistent recorded mimic residual is on the right inner finger, consistent with attachment on a side the selected sensor does not cover. This is a detection gap independent of whether the physical attachment originates in solver error, friction, or geometry.

The eventual episode outcome is still correctly failure: no supported, settled placement in the bin occurs. A future detection change should use separate per-finger sensors or a supported aggregation path, and require both sides to detach; changing the regex to match both bodies in the existing sensor is insufficient. Other gripper links may also matter for geometric hooking.

## Collection performance

The original full attempt's `active_control` telemetry contains 1,800 steps:

| Metric | Value |
| --- | ---: |
| Wall control rate | 10.087 Hz |
| Real-time factor | 0.6724 |
| Work mean / p50 / p95 / p99 | 98.189 / 100.907 / 112.716 / 116.032 ms |
| Physics mean | 56.734 ms |
| Rendering mean | 22.196 ms |
| Recording mean | 8.006 ms |
| Task checks mean | 6.998 ms |
| Sensors mean | 1.664 ms |
| Scene writes mean | 1.092 ms |
| Observations mean | 0.233 ms |
| Mean render interval | 49.561 ms |
| Render rate | 20.173 frames/s |
| Longest render gap | 1,038.433 ms |

All 1,800 commands exceed the 66.667 ms nominal command budget. Physics accounts for approximately 58% of mean work; the other regions total approximately 41.5 ms. Increasing iterations globally without recovering time elsewhere has little existing margin in this recording. These regions are host elapsed timings, including synchronization, and are not isolated GPU kernel durations.

The first four-row attempt is unsuitable for steady-state speed comparisons. Its task-check/recording startup costs substantially inflate its average.

## Reproducible analysis methods

Read HDF5 with the repository's `.venv/bin/python`, `h5py`, and NumPy; no simulator startup is required. This minimal snippet verifies the action transitions, opening state and mimic residuals:

```python
import json
import h5py
import numpy as np

path = "output/omy_teleop/20261024_141641_bowl/SimplePnp291Bowl00Task/part_000/omy_demos.hdf5"
with h5py.File(path) as f:
    metadata = json.loads(f["data"].attrs["robolab_omy_teleop"])
    episode = f["data/demo_1"]
    actions = episode["actions"][:]
    joints = episode["states/articulation/robot/joint_position"][:]
    bowl = episode["states/rigid_object/bowl/root_pose"][:]
    print(metadata["performance"]["solver"])
    changes = np.flatnonzero(np.diff(actions[:, -1])) + 1
    print([(int(i), float(actions[i, -1])) for i in changes])
    print("open state", joints[103, 7], "minimum later bowl z", bowl[103:, 2].min())
    signs = np.array([1, -1, 1, -1, -1])
    residuals = joints[:, 8:13] - joints[:, 7:8] * signs
    print("maximum residual degrees", np.rad2deg(np.abs(residuals).max(axis=0)))
```

The USD analysis uses the bundled `pxr` libraries without launching Kit. For this environment, add `isaacsim/extscache/omni.usd.libs-*` to Python's import path, and its `bin` directory plus the matching Python distribution's `lib` to `LD_LIBRARY_PATH` before starting Python. Read-only `Usd.Stage.Open` then exposes schema attributes and mesh points. Use `trimesh.Trimesh(vertices, faces, process=True)` to merge duplicate seam vertices before checking topology. Radial thickness uses `mesh.ray.intersects_location(origins, directions, multiple_hits=True)` and the difference between the two sorted intersection radii. These computations are in metres.

The figures sample the existing `obs/image_obs/wrist_cam` and `obs/image_obs/over_shoulder_left_camera` datasets. Full-episode rows are `range(0, 1800, 90)`, transition rows are `range(58, 74)`, and the wrist overview uses rows `[40, 50, 53, 54, 55, 56, 57, 60, 65, 75, 95, 100, 101, 102, 103, 105, 110, 120, 150, 240]`.

## Interpretation and remaining measurements

The source establishes that the lower solver cap was active and that meaningful gripper constraint errors occurred. It also establishes that reopening the driven joint did not release the bowl. It does **not** isolate the scene iteration cap as the sole cause.

Useful controlled comparisons preserve the source initial state and action timing while varying scene cap, gripper speed, resolved stiffness/damping, both contacting materials and collision representation separately. The first 240–300 actions cover grasp, commanded release and a short hold; the remaining original actions are largely stationary. A cold state restoration omits solver/contact caches, and open-loop replay can diverge even at the source cap, so repeat trials and inspect the actual failure mechanism rather than relying only on replay success.

To distinguish contact convergence from a missed collision or geometric hook, record both fingertips' contact points, normals, forces and separation at physics substeps, inspect the cooked collision shapes against the visual mesh, and monitor individual link poses and mimic/anchor error. The original recording lacks these measurements. CPU/GPU or timestep changes also alter dynamics and need separate validation before collection settings are promoted.

## Appendix: mimic compliance, passive speed and inertia audit

All five authored mimic constraints reference `finger_joint`. Their source properties are:

| Follower joint | Property instance | Gearing | Natural frequency, rad/s | Damping ratio |
| --- | --- | ---: | ---: | ---: |
| `right_outer_knuckle_joint` | `rotZ` | −1 | 1,000,000 | 0 |
| `right_inner_finger_joint` | `rotX` | −1 | 1,000 | 0.05 |
| `left_inner_finger_joint` | `rotX` | +1 | 1,000 | 0.05 |
| `right_inner_finger_knuckle_joint` | `rotX` | +1 | 1,000 | 0.05 |
| `left_inner_finger_knuckle_joint` | `rotX` | +1 | 1,000 | 0.05 |

The installed `omni.physxtests/tests/PhysxMimicJointAPI.py`, function `predictMimicJointComplianceOutcome`, enables compliance only when natural frequency and damping ratio are both positive. Its model uses `kp = omega**2 / r` and `kd = 2*omega*dampingRatio / r`, where `r` is the effective inverse response. Consequently the right outer knuckle uses a hard constraint despite its large frequency value; the other four have finite compliance and 5% of critical damping.

At 120 Hz, the compliant constraints' `omega*dt` is 8.33. NVIDIA's [compliant mimic tuning guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#tuning-a-compliant-mimic-joint) recommends starting around `omega*dt=1`, avoiding damping ratios below one because of oscillation, then increasing frequency as needed to reduce lag. This supports controlled frequency/damping sweeps; it does not establish that the authored values cause the observed failure. Turning compliance off is a useful diagnostic comparison, with the separate risk that hard mimic constraints compete with contact constraints. CPU/GPU replay differences alone do not establish a known backend bug.

Direct attribute authoring was verified offline against this source asset using the bundled USD libraries and a session edit layer. The exact existing attributes are `physxMimicJoint:<instance>:naturalFrequency` and `physxMimicJoint:<instance>:dampingRatio`; both are non-custom `float` attributes. For each of the five constraints, these calls succeeded and read back the requested values:

```python
stage.SetEditTarget(stage.GetSessionLayer())
prefix = "physxMimicJoint:" + instance + ":"
frequency = joint_prim.GetAttribute(prefix + "naturalFrequency")
damping = joint_prim.GetAttribute(prefix + "dampingRatio")
assert frequency and damping
assert frequency.Set(120.0)
assert damping.Set(1.0)
assert frequency.Get() == 120.0 and damping.Get() == 1.0
```

The test did not save the source layer or launch simulation. Direct `GetAttribute(...).Set(...)` is sufficient here because the source already authors these attributes; a generated `PhysxMimicJointAPI.CreateNaturalFrequencyAttr` method is not required. Other assets without these attributes would need explicit creation with the appropriate type.

The 5 rad/s driven-joint cap also does not bound every passive joint: original action-boundary samples contain **9.598 rad/s** on `left_inner_finger_knuckle_joint` at row 65 and −5.705 rad/s on `right_outer_knuckle_joint` at row 102. Passive joint velocity limits are authored as 10,000 degrees/s, approximately 174.53 rad/s. Surface-speed and collision tests should therefore monitor all gripper links, rather than interpreting the driven-joint cap as a global speed bound.

Traversing the robot with `Usd.TraverseInstanceProxies()` finds **11 Robotiq collision source meshes**: seven convex hull sources and four convex decomposition sources. The sources contain 85,986 triangles in total; the base contributes 62,230. Excluding the base leaves ten moving finger/knuckle sources. Per side, the inner finger body/pad contribute 2,590 and 742 triangles. These counts include instance proxies omitted by an ordinary stage traversal; they still do not reveal how many hulls the runtime decompositions produce. Convex source triangle count must not be interpreted as the number of runtime collision triangles.

Authored gripper component mass/inertia data are symmetric between sides:

| Component | Mass, kg | Principal inertias, kg m² |
| --- | ---: | --- |
| Base | 0.609891 | `(3.629e-4, 4.747e-4, 5.407e-4)` |
| Outer knuckle | 0.012735 | `(3.195e-7, 2.058e-6, 2.135e-6)` |
| Outer finger | 0.039179 | `(4.293e-6, 1.088e-5, 1.322e-5)` |
| Inner finger body | 0.019493 | `(9.274e-7, 1.767e-6, 1.960e-6)` |
| Inner fingertip pad | 0.019762 | `(9.985e-7, 2.839e-6, 3.210e-6)` |
| Inner knuckle | 0.027307 | `(4.488e-6, 8.005e-6, 1.184e-5)` |

Inner finger body plus pad mass is about 0.039255 kg per side. Their inertias require transforms and the parallel-axis theorem to aggregate; adding the listed principal components would be incorrect. No joint armature is authored in the source asset, and the original actuator configuration leaves armature unspecified.

The Franka source additionally warrants an inertia sanity check: several links author masses of 2–3 kg with principal inertias around `0.9e-6` to `3.3e-6 kg m²` and centres of mass at zero. `panda_link7` authors mass 0.405291 kg and principal inertias approximately `(6.317e-8, 6.320e-8, 1.061e-7) kg m²`. Such combinations imply submillimetre radii of gyration for sizeable robot links. Subsequent `body_properties` reads in [`cuda16_tuned_d06_contacts/summary.json`](../output/omy_diagnostics/20261024_bowl_contacts/cuda16_tuned_d06_contacts/summary.json) **confirm these unusually small values in the runtime tensors**, including full body-frame tensors. Comparison with coherent physical reference data remains necessary; this is an asset-validation concern, not a demonstrated cause of the bowl failure or CPU replay instability. Collider-derived inertia experiments did not, by themselves, fix the recorded grasp.
