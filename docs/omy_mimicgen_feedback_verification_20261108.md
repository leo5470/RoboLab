**MimicGen feedback verification — 2026-11-08**

The feedback is substantially correct. Enforcing tier quotas repairs the scheduler, but explicit workspace targets account for the larger estimated spatial gain. The generator's phase transition and available source behaviors are credible obstacles to realizing that gain. The implementation plan has been revised to establish motion feasibility before building the full production profile and export machinery.

All verification here used CPU calculations and existing files. No Isaac Sim process, GPU computation, policy rollout or training was started. Collision, reachability under the actual controller, and policy improvement remain unvalidated.

Evidence: [independent audit script](../output/omy_diversity_verification_20261108/audit.py), [audit JSON](../output/omy_diversity_verification_20261108/audit.json), [archived feedback scripts](../output/omy_diversity_verification_20261108/feedback_scripts/), and [revised plan](omy_mimicgen_diversity_plan_20261108.md).

**Scheduler and actual releases**

Running the supplied `sim_plan.py` against the real `Pipeline.plan`, with fake successful job results and no simulator, reproduces:

| Ten humans, 40 generated slots | Accepted near | Moderate | Wide |
|---|---:|---:|---:|
| Candidate factor 1.5 | 60 | 0 | 0 |
| Candidate factor 3 | 80 | 40 | 0 |

The parent total can be filled before all tiers run; `pipeline.py:593`, `:605`, `:615` explain why. A higher candidate factor alone is insufficient. The script's additional partial-acceptance case is a synthetic scheduling exercise, not an estimate of real acceptance rates.

Both fork/pitcher published pools have 50 candidates and 50 selected episodes. Their generating script was `output/omy_pipeline/20261105_pitcher_fork_gpu0/run.py`, whose `make_job` uses only near augmentation and whose completion loop stops once the total can fill 50 slots. It did not use `Pipeline`. Qualification uses `--no-randomize` despite the range fields present in those job JSON files; `scripts/run_omy_mimic_collection.py:79` confirms this distinction.

The reported displacement maxima reproduce **per coordinate**: about 2.47 cm for object XY and 0.99 cm for bin XY, with object yaw at most 4.8°. Their Euclidean XY maxima are about 3.04 cm and 1.25 cm respectively. This unit distinction does not alter the near-only diagnosis.

**Spatial estimates and their limitations**

The distance is `min_demo max(norm(object_xy_delta), norm(bin_xy_delta))`. Both objects must match the same demonstration. Yaw, grasp orientation, trajectory and robot posture are excluded.

| Recorded data, scored on all 50 authored v2 layouts | Fork median (m) | Pitcher median (m) |
|---|---:|---:|
| Human demonstrations only | 0.2150 | 0.2014 |
| Published 50-demo pool | 0.2079 | 0.1999 |

These reproduce independently from saved HDF5 initial states and v2 generation records. The feedback's rounded table combines different computational protocols; its “current pool” row is resampled near-only data, not the measured pool above.

| Sampling estimate | Fork median (m) | Pitcher median (m) | Protocol |
|---|---:|---:|---|
| Near only | 0.2058 | 0.1973 | Median across 300 random 25/25 v2 splits |
| Local 50/25/25 mixture | 0.1878 | 0.1870 | Same split protocol |
| Ten explicit target layouts | 0.1149 | 0.1259 | Targets sampled from one half; score on the other |
| Ten farthest-first target layouts | 0.1097 | 0.1196 | Same split protocol, different target selector |
| All generated candidates wide | 0.1759 | 0.1649 | 400 local sampling trials, score all 50 v2 layouts |

These scripts assume that sampled poses yield successful accepted demonstrations. They do not simulate the generator, preparation rejection, IK, contacts, source-dependent success or the actual surplus-candidate selector. Fork has 40 generated slots; pitcher has 42, so “all 40 wide” is shorthand. The hard-coded source-task lists match the saved qualified-source manifests.

The unlimited-box calculation needs an additional qualification:

| Optimistic continuous wide-box bound | Fork median (m) | Pitcher median (m) |
|---|---:|---:|
| Perturb around every human, including disqualified sources | 0.1277 | 0.1047 |
| Perturb only qualified sources; retain all original human demos | 0.1374 | 0.1508 |

The first line reproduces the feedback; the second uses the current generation eligibility. Neither is achievable coverage promised by any finite dataset, and neither accounts for geometry or successful execution. The difference reinforces the importance of recovering missing source behaviors.

The extra `dose.py` reproduces the reported tail improvement. At 10→20 targets, median-over-splits p90 drops from 0.189→0.163 m for fork and 0.190→0.157 m for pitcher. Corresponding median distances are 0.115→0.101 m and 0.127→0.113 m. With all 25 available training targets they reach about 0.099/0.105 m. This is saturation of that finite split experiment, not a fundamental 0.10 m floor. Object-only distance falls from roughly 0.134/0.148 m to 0.062/0.060 m at ten targets in this dose study, whose zero-target local mixture is approximately one-third per tier.

The 25/25 split avoids exact target overlap within each trial, but reuses the known v2 collection for design. It is exploratory evidence, not a fresh held-out policy evaluation. The revised plan preserves v2 as a regression set and requires fresh final-test layouts.

**Phase transition, approach and reachability**

`grasp_index` in `mimicgen.py:107` triggers when a closed-gripper object first rises more than 2 cm. `mimic_cfg` then switches from object-relative to bin-relative motion. Installed Isaac Lab `data_generator.py:505` onward starts the bridge at the previous commanded target and merges the next segment using linear interpolation; the local default is one second. This does create a long, low transfer when object-to-bin arrangements change.

The feedback's relative-vector mismatch reproduces: median best-source change is 0.178 m for fork and 0.595 m for pitcher. Independently reading the USD joint chain and applying NumPy FK to source joint commands gives planned end-effector bridge XY lengths of **0.185 m and 0.598 m** at zero boundary offset. These include full source/target pose transforms, rather than just root XY offsets. FK at the DROID home agrees with the authored robot position to about 0.011 mm. This verifies commanded geometry, not controller tracking or success.

The lift numbers need precision. Qualified-source median lift at the trigger is 2.55 cm for fork and 2.20 cm for pitcher. The boundary also includes a random 0–3-frame delay; at the latest included source frame the corresponding medians are 5.70/4.20 cm. Commanded end-effector poses, measured source object heights and achieved generated clearances are different quantities. The source peak-lift medians are 28.1/22.5 cm. A later clearance-aware phase boundary is justified, but simply choosing the global peak can include substantial source lateral transport and is not a validated repair.

All five qualified pitcher source layouts require +y carry, and 29 of 50 v2 targets require −y carry. An independent object-root line proxy from the target object toward the source's bin-relative starting position intersects the bin's XY bounds on all 29. That supports the collision concern. It does **not** prove 29 actual collisions: the real path includes grasp offsets, rotations, object extent, tracking errors and boundary randomization. A literal table drag is also not established from XY geometry alone; the object has some clearance, while the bin rim is about 10.5 cm high.

The moved initial waypoint is a real code behavior. `data_generator.py:453` uses `is_first_subtask or generation_transform_first_robot_pose`, so setting the second operand false does not prevent transforming the first source robot pose. The physical reset remains fixed, and the robot interpolates from it to the transformed initial waypoint. The supplied planar approximation gives median displacement 0.133/0.282 m and counts 7/9 targets within 0.25 m of the base axis. Full source-pose/FK reconstruction gives 0.132/0.282 m and counts **7/8**; the threshold count depends on the approximation and chosen source. These are warning diagnostics, not infeasibility certificates. The planar points beyond the table's robot-side edge do not establish collision or reachability. “Within reach” remains unverified for the six-joint generator.

Each “best source” metric minimizes independently: the source minimizing bridge length need not minimize the initial-waypoint shift or satisfy grasp/IK constraints. Simultaneously satisfying those requirements can be harder than either optimistic metric suggests.

The reach mismatch is explicit in the code. `tools/simple_pnp_v2/reach_check.py` states that OMY limits are ignored and yaw is free; `reach_map.py` solves over seven joints. `mimicgen.py:162` removes joint 3 from IK and targets the full source-derived orientation. Passing the v2 screen is insufficient evidence for executing the transformed trajectory.

**Qualification and selection**

| Source loss | Saved evidence |
|---|---|
| Pitcher 02 and 06 | Zero generated motion; preparation rejects 9.1 mm and 10.8 mm footprint gaps against the 20 mm rule. |
| Pitcher 37 | Both generated reproductions fail; no closed-gripper lift. |
| Fork 04, 24 and 31 | All six reproduction attempts achieve task success but fail `GRIPPER_HIT_TABLE`. Seven of ten humans remain qualified; six supply the final generated pool. |
| Snickers | Only one of ten humans qualifies. Seven excluded sources have at least one table-contact rejection; the other two have unsuccessful reproductions. Six of those seven contact-rejected sources have task-successful attempts. Some also have velocity or preparation failures. |

The missing pitcher directions are therefore real. The records distinguish a successful human demonstration from the generator's ability to reproduce it. They do not prove all rejected contact events were harmless or that the generator reproduced the human motion faithfully: it uses Cartesian targets, interpolation and IK, and human pool rows explicitly lack generated-contact/IK checks. Recover sources by diagnosing preparation and reproduction, not by automatically disabling gates.

The selector correction is valid with one nuance: `final_dataset.py:109` prioritizes task representation and then farthest feature distance; parent constraints can apply first. It favors distant eligible candidates and may over-select wide examples, but does not guarantee selecting the widest spatial cases. The plan now motivates exact tier quotas as preserving the experimental mixture.

The other corrections check out: legacy wide bin yaw is ±5° versus zero at deployment; 25% overlap is for object XY boxes only; v1→v2 scene construction confounds a coverage comparison; and the report's nearby-v2 count is 47 losses versus 27 gains. Pitcher base success is 1/50 and steered success 0/50. It is useful for transport diagnostics but weak as the sole PPS success benchmark. Joint 3 remains constrained, and the report's limited joint-5 variation is an unresolved policy-transfer hypothesis, not a proven cause cured by these changes.

**Plan changes**

The revised order is: prepare a small feasibility pilot and behavior audit; when GPU work is allowed, measure current-generator failures on fixed cases; repair and validate lift/transfer and approach construction; diagnose missing sources; add generator-specific reach/path screening; then build production target generation and provenance. The CPU scheduler repair remains independently useful. Test 25% and 50% explicit-target shares before considering larger datasets, and control scene construction and generator changes separately in policy experiments.

Reproduction of the independent audit uses only USD reading, HDF5 and NumPy:

```bash
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  tools/simple_pnp_v2/usd_python.sh output/omy_diversity_verification_20261108/audit.py
```

The archived feedback scripts are unchanged copies; some contain absolute imports of their original scratchpad directory. They were inspected and executed there for verification. `audit.json` records input JSON hashes and source-HDF5 hashes as declared in the saved manifests. The audit does not recertify every recording's full-file hash or validate a newly generated dataset.
