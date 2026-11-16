#!/usr/bin/env python3
"""Normalize generated-task assets whose rigid body lives below the root prim.

The original source assets are preserved.  Flattened repaired copies are written
under ``assets/objects/repaired`` and generated simple-PnP scenes/catalog entries
are redirected to those copies.
"""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
REPAIRS = {
    "apple": Path("assets/objects/objaverse/apple_02.usd"),
    "gregorys_coffee_cup": Path("assets/objects/objaverse/gregorys_coffee_cup.usd"),
    "lunchbag": Path("assets/objects/objaverse/lunchbag.usd"),
    "red_bell_pepper": Path("assets/objects/objaverse/red_bell_pepper.usd"),
    "snickers_bar": Path("assets/objects/objaverse/snickers_bar.usd"),
}


def _copy_attr(source, destination) -> None:
    value = source.Get()
    if value is not None:
        destination.Set(value)


def main() -> None:
    launcher = AppLauncher(headless=True)
    from pxr import Usd, UsdPhysics

    repaired_dir = ROOT / "assets/objects/repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}

    for name, relative_source in REPAIRS.items():
        source = ROOT / relative_source
        destination = repaired_dir / relative_source.name

        source_stage = Usd.Stage.Open(str(source))
        if source_stage is None:
            raise RuntimeError(f"Could not open {source}")
        source_stage.Flatten().Export(str(destination))

        stage = Usd.Stage.Open(str(destination))
        root = stage.GetDefaultPrim()
        if not root:
            raise RuntimeError(f"No default prim in {destination}")

        nested_bodies = [
            prim
            for prim in Usd.PrimRange(root)
            if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not nested_bodies:
            raise RuntimeError(f"Expected at least one nested rigid body in {source}")

        source_rb = UsdPhysics.RigidBodyAPI(nested_bodies[0])
        root_rb = UsdPhysics.RigidBodyAPI.Apply(root)
        for source_attr, destination_attr in (
            (source_rb.GetRigidBodyEnabledAttr(), root_rb.CreateRigidBodyEnabledAttr()),
            (source_rb.GetKinematicEnabledAttr(), root_rb.CreateKinematicEnabledAttr()),
            (source_rb.GetStartsAsleepAttr(), root_rb.CreateStartsAsleepAttr()),
            (source_rb.GetVelocityAttr(), root_rb.CreateVelocityAttr()),
            (source_rb.GetAngularVelocityAttr(), root_rb.CreateAngularVelocityAttr()),
        ):
            _copy_attr(source_attr, destination_attr)
        mass_bodies = [prim for prim in nested_bodies if prim.HasAPI(UsdPhysics.MassAPI)]
        if mass_bodies:
            source_mass = UsdPhysics.MassAPI(mass_bodies[0])
            root_mass = UsdPhysics.MassAPI.Apply(root)
            for source_attr, destination_attr in (
                (source_mass.GetMassAttr(), root_mass.CreateMassAttr()),
                (source_mass.GetDensityAttr(), root_mass.CreateDensityAttr()),
                (source_mass.GetCenterOfMassAttr(), root_mass.CreateCenterOfMassAttr()),
                (source_mass.GetDiagonalInertiaAttr(), root_mass.CreateDiagonalInertiaAttr()),
                (source_mass.GetPrincipalAxesAttr(), root_mass.CreatePrincipalAxesAttr()),
            ):
                _copy_attr(source_attr, destination_attr)

        for nested in nested_bodies:
            nested.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if nested.HasAPI(UsdPhysics.MassAPI):
                nested.RemoveAPI(UsdPhysics.MassAPI)

        stage.GetRootLayer().Save()

        verify_stage = Usd.Stage.Open(str(destination))
        verify_root = verify_stage.GetDefaultPrim()
        remaining_nested = [
            prim
            for prim in Usd.PrimRange(verify_root)
            if prim != verify_root and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not verify_root.HasAPI(UsdPhysics.RigidBodyAPI) or remaining_nested:
            raise RuntimeError(f"Rigid-body normalization failed for {destination}")

        old_catalog_path = relative_source.as_posix()
        new_catalog_path = destination.relative_to(ROOT).as_posix()
        replacements[old_catalog_path] = new_catalog_path
        print(f"Repaired {name}: {old_catalog_path} -> {new_catalog_path}")

    catalog_path = ROOT / "assets/objects/object_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for entry in catalog:
        if entry.get("name") in REPAIRS:
            entry["usd_path"] = replacements[entry["usd_path"]]
            entry["rigid_body"] = True
            entry["static_body"] = False
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    scene_dir = ROOT / "assets/scenes/simple_pnp"
    changed = 0
    for scene_path in scene_dir.glob("*.usda"):
        text = scene_path.read_text()
        updated = text
        for old_path, new_path in replacements.items():
            old_payload = "../../" + old_path.removeprefix("assets/")
            new_payload = "../../" + new_path.removeprefix("assets/")
            updated = updated.replace(old_payload, new_payload)
        if updated != text:
            scene_path.write_text(updated)
            changed += 1

    print(f"Updated {changed} generated scenes")
    if changed != 250:
        raise RuntimeError(f"Expected to update 250 scenes, updated {changed}")

    launcher.app.close()


if __name__ == "__main__":
    main()
