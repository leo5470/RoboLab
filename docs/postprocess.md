# OMY demonstration postprocessing for PPS

This document describes a proposed offline pipeline for turning Quest + OMY-L100 teleoperation recordings into training data compatible with TritiumR's [Proxy Policy Steering (PPS)](https://github.com/TritiumR/pps) DROID joint-position training path. It is a design note; it does not add or change processing code.

## Recommendation

Keep state-only HDF5 recordings as the permanent source of truth. After collection, validate and select attempts, render policy cameras from recorded states, then export a PPS-compatible LeRobot dataset. This keeps rendering and encoding out of the live teleoperation loop and allows the training export to be rebuilt later.

## PPS training representation

PPS's Isaac Lab demonstration converter writes one row per sample with these features:

| Feature | Shape and meaning |
| --- | --- |
| `exterior_image_1_left` | RGB image, 180 × 320 × 3; external camera |
| `wrist_image_left` | RGB image, 180 × 320 × 3; wrist camera |
| `joint_position` | Seven measured Franka joint angles (`float32`) |
| `gripper_position` | One measured gripper value (`float32`) |
| `actions` | Seven arm values and one gripper command (`float32[8]`) |
| `task` | Language instruction attached to each episode |

The converter declares 15 FPS. It also defines microphone features, but the ordinary RGB DROID policy does not require them. The selected PPS policy transforms resize images to 224 × 224. Its joint-position training configuration treats arm actions as absolute joint targets: it subtracts the measured current arm position to form deltas, while leaving the gripper action absolute. Training examples contain a future action sequence; the inspected π0.5 proxy uses a horizon of 15 actions.

Sources: [PPS Isaac Lab converter](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/examples/Isaaclab/convert_isaaclab_data_to_lerobot.py), [DROID policy transforms](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/policies/droid_policy.py), [training configuration](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/training/config.py), and [data loader](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/training/data_loader.py). These code links refer to revision `a22f9ad` inspected on 2026-10-01.

## Alignment and units to settle

Our recorder stores an action at step `t` and the resulting scene state after that action. Observations are pre-action: observation 0 belongs to the initial scene state, and observation `t > 0` belongs to post-step state `t − 1`. Therefore, the PPS row at time `t` must pair the pre-action image and measured joint state with the action sent at `t`.

The PPS example converter shifts its `joint_actions` observation forward one sample. That is specific to its recorder's convention; applying the shift to our direct action recording would misalign the data. Export the mapped and limited Franka command actually sent to the environment, not raw L100 readings and not the robot's achieved joint positions as a substitute for the command.

Our OMY gripper command uses `+1 = open` and `−1 = closed`. The inspected PPS joint-position environment accepts `0 = open` and `1 = closed`, so its command conversion is:

```text
pps_gripper_command = (1 - omy_gripper_command) / 2
```

Measured gripper state needs a distinct mapping. The PPS pot converter reads a finger-joint angle, while PPS evaluation can take the eighth raw articulation position. Those paths do not establish a single normalized convention. Keep the raw finger position and normalized closure available internally, and explicitly choose the field and units to match the selected PPS training/evaluation configuration. Do not assume the command encoding is also the correct measured-state encoding.

Sources: [PPS binary gripper action](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/binary_zero_one_joint_actions.py), [PPS pot observations](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pot/mdp/observations.py), and [PPS evaluation input handling](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/eval_steering.py).

## Sampling rate

OMY collection defaults to 60 control steps per simulated second and also supports `--control-hz 30` and `--control-hz 15`, with physics fixed at 120 Hz. The inspected PPS pot environment runs at 15 control steps per simulated second. Setting export metadata to 15 FPS for a 60 Hz recording without resampling would make that trajectory appear four times longer. A native 15 Hz collection already has one command and one pre-action observation for each 1/15 simulated second; its FPS should still be checked against recorded simulation timestamps.

Retain every raw sample in HDF5 and make export frequency configurable. For a PPS profile, select samples using simulation timestamps and declare the resulting rate accurately. Keep wall-clock timestamps for latency analysis; they should not redefine simulated motion. Downsampling a 60 Hz recording discards intermediate commands and may affect contact-rich grasps. Native 15 Hz collection avoids that export loss but gives the operator fewer new commands per second; compare grasp and placement quality before adopting it as the collection-wide rate.

## Proposed offline pipeline

1. **Index and validate.** Read the HDF5 recording, `env_cfg.json`, collection events, and per-attempt timing files. Check sample-count alignment, finite values, joint order, episode boundaries, and missing or interrupted data. Write a manifest that records each attempt's outcome and any rejection reason.

2. **Select attempts and samples.** Initially include valid successful attempts for imitation learning. Preserve failed and interrupted attempts in the source archive for debugging and possible future uses. Exclude READY and reset handling; keep useful pauses during manipulation. Record each exported sample's source run, episode, and original step.

3. **Render policy cameras offline.** Restore recorded scene states and render the fixed external and wrist cameras from those states. Do not regenerate images by replaying actions through physics, because contact-rich trajectories can diverge. The Quest viewpoint is for the operator; the dataset should use stable, explicitly configured policy cameras. Review frames around grasps and object transfers before processing the whole collection.

4. **Export a PPS-specific LeRobot dataset.** Map camera, joint-state, action, gripper, task, timestamp, and episode fields explicitly. Preserve absolute Franka joint-position commands so PPS can apply its configured delta transform. Use a LeRobot version verified against the chosen PPS training setup; legacy import paths alone do not establish that our existing LeRobot v3 exporter is compatible.

5. **Report quality and split by episode.** Produce preview videos, timing summaries, rejection reasons, and train/validation episode lists. Keep whole episodes in one split, so frames from a trajectory cannot leak across training and validation. Make rendering and export resumable per episode and cache rendered frames.

## Existing RoboLab pieces

The offline camera materializer in [`scripts/materialize_demo_images.py`](../scripts/materialize_demo_images.py) and [`robolab/core/replay/materialize.py`](../robolab/core/replay/materialize.py) already rebuilds an environment and renders observations from recorded states with pre-action alignment.

RoboLab also has a generic LeRobot v3 exporter in [`robolab/core/export/lerobot_exporter.py`](../robolab/core/export/lerobot_exporter.py), invoked by [`scripts/convert_to_lerobot.py`](../scripts/convert_to_lerobot.py). It expects `data.hdf5` or `run_*.hdf5`, uses different feature names, and pairs actions with post-step joint positions. It also generates timestamps from the configured FPS without resampling. These differences mean it needs an OMY/PPS export profile and alignment handling before it can serve this collection.

## First validation milestone

Process three successful BananaInBowl attempts end to end. Inspect the rendered frames around grasps, compare every exported action and observation against the source HDF5, check the gripper's state and command mappings, and load a batch through the intended PPS training data loader. Once those checks pass, scale the pipeline to the rest of the collection.

## Training target still to choose

The raw archive can support both general policy training and PPS's full base/reference/task-proxy workflow. For the full PPS workflow, normalization must also match the frozen base policy and the proxies: PPS supports reusing teacher normalization during distillation, so independently normalizing their action spaces can make their comparison inconsistent. The intended model/checkpoint determines which normalization and compatibility checks the exporter should enforce.

Source: [PPS distillation script](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/scripts/distill_pytorch.py).
