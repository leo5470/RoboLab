"""Reconstruct a recorded DROID task for the Mimic source-review environment."""

import json
from pathlib import Path

import gymnasium as gym

from robolab.core.environments.config import parse_env_cfg
from robolab.core.replay.env_config import apply_recorded_env_cfg
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs


def source_review_config(source: str | Path, metadata: dict, device="cpu"):
    task = metadata.get("task")
    if not task:
        raise ValueError("Source metadata must identify its task")
    postfix = "MimicSourceReview"
    auto_register_droid_rel_ik_envs(task=task, env_postfix=postfix,
                                  include_viewport_camera=False, include_policy_cameras=False)
    cfg = parse_env_cfg(task + postfix, device=device, seed=metadata.get("base_seed", 0), num_envs=1, use_fabric=True)
    recorded = json.loads(Path(source).with_name("env_cfg.json").read_text())
    # create_env adds this display-only metadata after registration. Preserve it
    # explicitly; the resolved /instruction is restored by the existing overlay.
    if "_instruction_variants" in recorded:
        cfg._instruction_variants = recorded["_instruction_variants"]
    skipped = apply_recorded_env_cfg(cfg, recorded)
    # Scoring structures serialize functools.partial as strings. They are not
    # used by this replay. All other unresolved fields need investigation.
    critical = [field for field in skipped if not field.startswith("/subtasks")]
    if critical:
        raise ValueError("Recorded config has unresolved fields: " + "; ".join(critical))
    cfg.recorders = None
    cfg.viewer.resolution = (960, 540)
    name = f"RoboLab-{task}-DroidMimicReview-v0"
    if name not in gym.registry:
        gym.register(id=name, entry_point="robolab.datagen.envs.droid_mimic_env:DroidMimicEnv",
                     disable_env_checker=True)
    return name, cfg, skipped
