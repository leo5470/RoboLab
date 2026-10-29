# Scene validation and evidence

Keep three independent results: **geometry**, **physics**, and **grasp**. Each is
passed, failed, or pending, with its scope and evidence. Generating a file, getting
a solver success, running for a fixed time, or producing a screenshot does not
establish the later results. A physics failure keeps the candidate unqualified
even when it eventually comes to rest.

## 1. Static geometry

For separate objects directly on one axis-aligned rectangular tabletop, use the
shared repository checker with every tabletop object/container:

```bash
python tools/audit_simple_pnp_scenes.py \
  --scene assets/scenes/generated/pail_pickup.usda \
  --objects plasticpail_a01 grey_bin \
  --report output/pail_pickup.geometry.json
```

For an existing Simple-PnP library:

```bash
python tools/audit_simple_pnp_scenes.py \
  --scene-dir assets/scenes/simple_pnp \
  --pattern '*plasticpail_a01*' \
  --report output/plasticpail_a01.geometry.json
```

Run from the repository root with numpy and `pxr` available. Exit 0 means the
listed geometry checks passed; exit 1 means issues were found. Read the report,
including audit errors; a missing dependency is not an inspected geometry failure
and should leave the qualification stage pending. A library report applies only
to the scenes its rows cover. Single-scene reports include the scene and composed
USD layer hashes; verify hashes before reusing evidence.

The checker uses transformed bounding envelopes, includes root scales/pivots,
checks pair separation and complete table footprint, and checks authored geometry
bottoms against the support top. Defaults are 20 mm pair clearance, 10 mm edge
clearance, and bottom gaps from −1 to +5 mm. Start new placements at +2 mm. These
are screening thresholds, not exact mesh penetration or robot grasp checks.
The support's bounding rectangle must actually describe a usable tabletop;
a shelf assembly, hollow bin, or arbitrary mesh is not automatically a support.

For stacks, filled containers, or shelves, do **not** run every object through the
flat-table gate or waive its failures to claim a pass. Instead check each intended
support/contact relationship, support footprints, cavity floors/walls/openings,
and all unintended intersections. Inspect collision geometry as needed. If no
implemented checker covers a relation, record the manual geometry evidence and
its limitations, or leave that part pending. Keep general checks (resolved assets,
units, complete poses, and unintended collisions) for every arrangement.

Material/texture dependency checks are separate from geometric composition. A
missing shader or image can leave appearance/render validation pending while
geometry passes, if it cannot affect geometry or physics. Identify that limitation;
do not generalize it into permission to ignore missing meshes, collision payloads,
or USD composition errors. Some material search paths resolve only inside Kit.

## 2. Physics qualification

Use the intended simulator, robot/fixture, and physics settings, with the arm
stationary and the gripper open at its intended starting pose. Record those
settings and reset seeds. Check robot/start collisions before advancing physics.
Where possible, observe contacts and poses per physics step so transient launches
or tipping cannot disappear behind a later stationary pose.

For ordinary tabletop pickup scenes, a useful initial protocol is three reset
seeds, at least five simulated seconds of observation per reset, and a final
continuous 0.5 s stable window. These are qualification defaults, not a universal
profile for intentional dynamic/unstable tasks. Declare a task-specific profile
when the requested initial state legitimately differs; do not weaken it after
seeing failures just to qualify a layout.

Require actual support on the intended surface, no unintended object/robot
contacts, upright destination where required, and a retained graspable target pose.
Track displacement from the intended start as well as recent drift: an object
launched to another resting position must not pass. Example flat-start limits
are 20 mm total translation, 15° change in local-up direction, 0.01 m/s linear
speed, 0.5 rad/s angular speed, and stable-window drift within 1 mm / 0.5°.
Report chosen limits and measured maxima; do not label these values as results.

The existing OMY `ScenePreparation` gate has shorter minimum/stable windows and
supports the collection profile described in `docs/omy_xr_collection.md`. A
`ready` report proves only its implemented checks, not the longer protocol above
or grasp feasibility. Read the current code/profile before reusing its evidence.

`assets/scenes/_utils/settle_scenes.py` can produce a settled candidate:

```bash
python assets/scenes/_utils/settle_scenes.py \
  --scene assets/scenes/generated/pail_pickup.usda --replace --screenshot \
  --screenshot-dir output/pail_pickup_images
```

Use `--replace` on a **candidate copy in the same directory** to retain correct
relative paths. Keep an original backup during repairs. Without `--replace`, the
utility writes into the central scene directory, which can break nested paths.
It advances 300 updates and exports; it does not implement the qualification
protocol or reject failed scenes. Do not treat completion as a pass. Check output
paths and every object's measured outcome. Re-audit exported geometry, reload the
saved poses in the intended environment, and repeat the physics checks. The
utility strips velocities/PhysicsScene artifacts; verify the resulting scene in
the consumer configuration rather than assuming identical simulation behavior.

Distinguish unavailable hardware from sandbox device visibility: a failed GPU
query inside a restricted sandbox is not evidence that the host has no GPU. Use
the permitted host/GPU launch mechanism when the simulation is authorized. Run
simulator jobs with a wall-clock timeout, persist partial results, and bound
teardown too; an initialization error must not leave a GPU job running indefinitely.

If Isaac/GPU access is unavailable, save the candidate and static report with
physics pending. Do not fabricate a settled pose or copy a pass from another
layout, robot profile, or older asset version.

## 3. Grasp qualification

For the requested robot/gripper, check reachable pregrasp/grasp/lift poses, opening
width and finger thickness, table/bin clearance along approach and closing, and
an actual short lift/hold trial. A stable object may lie flat with no finger access;
large objects may require a handle or rim grasp. Inspect those affordances rather
than comparing object width alone to the gripper opening.

Inspect the selected robot asset's actual tool axes and finger geometry before
building a grasp pose. Generic gripper conventions and comments can be wrong:
the current flattened DROID/Robotiq asset extends along `base_link` +X and closes
along Y. Measure the pad/tip offset at the tested opening. Record contact sampling
cadence and which finger bodies were observed; a one-finger task contact mapping
does not establish two-finger grasp retention or release.

Record the tested grasp, robot profile, and outcome. If no robot was selected or
no grasp trial ran, mark grasp pending/robot unspecified. Robot-agnostic scene
authoring can finish at a lower stated validation level; do not claim collection
readiness or automatically run a policy rollout.

## Evidence handoff and changes

Keep report files outside task discovery and associate them with the exact scene
bytes and composed USD layer hashes. A compact handoff records scene path/hash,
asset hashes, intended relationships, generation seed, check commands, limits,
status/reasons for each gate, and links to measured outputs. Include robot,
simulator version, physics configuration, and reset seeds for dynamic results.

After any pose, scale, asset, or layer edit, invalidate prior geometry/physics/grasp
results for that scene and rerun affected gates. Changed robot/physics settings
invalidate the corresponding dynamic evidence. Screenshots and previous success
rates cannot replace matching validation reports. Metadata is a record of executed
checks; do not manufacture a passing status to make a task generator proceed.
