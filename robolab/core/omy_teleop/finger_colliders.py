"""OMY-only finger hulls with an explicit reference for cooked mass properties."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


REFERENCE_PATH = Path(__file__).with_name("robotiq_finger_mass.json")


def finger_collider_identity():
    """Portable provenance for recording/resume, without importing the simulator."""
    content = REFERENCE_PATH.read_bytes()
    reference = json.loads(content)
    return {"reference_sha256": hashlib.sha256(content).hexdigest(),
            "asset_sha256": reference["asset_sha256"], "stack": reference["stack"]}


def configure_finger_colliders(cfg, profile):
    """Modify only this copied OMY spawn config; the source USD stays intact."""
    if profile not in ("original", "hulls"):
        raise ValueError(f"Unknown OMY finger collider profile: {profile}")
    cfg.omy_finger_colliders = profile
    cfg.omy_finger_collider_metadata = {"profile": profile}
    if profile == "original":
        return
    if not str(cfg.sim.device).startswith("cuda") or cfg.scene.num_envs != 1:
        raise ValueError("OMY finger hulls require one CUDA environment")
    spawn = cfg.scene.robot.spawn
    if spawn.scale is not None or spawn.mass_props is not None:
        raise ValueError("OMY finger hulls do not support robot scale or mass overrides")
    reference = json.loads(REFERENCE_PATH.read_text())
    source = Path(spawn.usd_path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference["asset_sha256"]:
        raise ValueError("Robot USD differs from the finger mass reference; use --finger-colliders original")
    if any(importlib.metadata.version(name) != version for name, version in reference["stack"].items()):
        raise ValueError("Finger mass reference requires its recorded Isaac Sim/Isaac Lab versions")
    cfg.omy_finger_collider_metadata.update(finger_collider_identity(), mass_properties_applied=False)
    original_spawn = spawn.func

    def spawn_finger_hulls(*args, **kwargs):
        from pxr import Usd, UsdPhysics

        prim = original_spawn(*args, **kwargs)
        changed = []
        for child in Usd.PrimRange(prim):
            if "_inner_finger/" in str(child.GetPath()) and child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set("convexHull")
                changed.append(str(child.GetPath()))
        if len(changed) != 4:
            raise ValueError(f"Expected four finger/pad collider sources, found {len(changed)}")
        return prim

    spawn.func = spawn_finger_hulls


def restore_finger_mass_properties(env):
    """Restore reference masses, COM frames and full tensors after cooking hulls."""
    if getattr(env.cfg, "omy_finger_colliders", "original") != "hulls":
        return
    import torch

    identity = finger_collider_identity()
    metadata = env.cfg.omy_finger_collider_metadata
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("Finger mass reference changed while creating the environment")
    reference = json.loads(REFERENCE_PATH.read_text())
    robot = env.scene["robot"]
    indices = {name: robot.body_names.index(name) for name in reference["bodies"]}
    view = robot.root_physx_view
    for key in ("masses", "coms", "inertias"):
        values = getattr(view, f"get_{key}")().clone()
        for name, body in reference["bodies"].items():
            values[0, indices[name]] = torch.as_tensor(body[key], dtype=values.dtype, device=values.device)
        getattr(view, f"set_{key}")(values, torch.tensor([0], dtype=torch.int32))
        actual = getattr(view, f"get_{key}")()
        if not torch.allclose(actual, values, rtol=1e-5, atol=1e-9):
            raise ValueError(f"Could not preserve finger {key} after cooking hulls")
    # Default-based future resets/randomizers must see the preserved model too.
    for attr, getter in (("default_mass", "get_masses"), ("default_inertia", "get_inertias")):
        if hasattr(robot.data, attr):
            target = getattr(robot.data, attr)
            target.copy_(getattr(view, getter)().to(target.device))
    metadata["mass_properties_applied"] = True
    metadata["bodies"] = list(indices)
