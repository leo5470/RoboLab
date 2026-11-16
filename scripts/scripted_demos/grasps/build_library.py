"""Phase 2 step 2: build the grasp library from GraspDataGen's validated grasps (D5).

For each object: the grasps that held under the mass-scaled test (``gdg/sim_mass``), mapped to DROID's base_link
with the measured frame map, plus their 180° twins; ``passed_1g`` marks those that also held under the default 1 g
test (``gdg/sim_1g``); ``pad_penetration`` is how deep the mesh reaches into DROID's open fingertip pads (DROID's open
gap is 4 mm narrower than GraspDataGen's, so grasps placed with GraspDataGen's 1 mm standoff can graze).

    scripts/scripted_demos/run.sh --cpu scripts/scripted_demos/grasps/build_library.py [--objects 303 301 ...]

Writes ``<grasp_data>/library/<idx>_<name>.npz`` and ``<grasp_data>/library/manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

from robolab.core.scripted_demos.grasps import frames, library, objects
from robolab.core.scripted_demos import paths, select

GRIPPER_DIR = "robotiq_2f_85"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=[str(o.index) for o in objects.BATCH])
    args = parser.parse_args()

    root = objects.grasp_data_root()
    frame_map_path = root / "gripper_frame_map.json"
    T_gdg_droid = frames.load_frame_map(frame_map_path)
    masses = json.loads((root / "masses.json").read_text())
    commit = subprocess.run(["git", "-C", str(paths.graspdatagen_dir()), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    mesh_manifest = json.loads((root / "meshes" / "manifest.json").read_text())
    out_dir = root / "library"
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in args.objects:
        obj = objects.get(key)
        paths = {run: root / "gdg" / run / f"grasp_{'guess' if run == 'guess' else 'sim'}_data" / GRIPPER_DIR / f"{obj.key}.yaml"
                 for run in ("guess", "sim_mass", "sim_1g")}
        guess = library.parse_isaac_grasp_yaml(paths["guess"])
        mass = library.parse_isaac_grasp_yaml(paths["sim_mass"])
        one_g = library.parse_isaac_grasp_yaml(paths["sim_1g"])
        passed_1g_names = set(one_g["names"][one_g["confidence"] > 0].tolist())
        passed_1g = np.array([n in passed_1g_names for n in mass["names"]])
        lib = library.build(mass, T_gdg_droid, also_passed={"1g": passed_1g})
        vertices = objects.mesh_vertices(obj)
        lib["pad_penetration"] = select.pad_penetration(lib["T_obj_grasp"], vertices[:: max(1, len(vertices) // 8000)])
        digest = library.save(out_dir / f"{obj.key}.npz", lib)
        n_valid = int((mass["confidence"] > 0).sum())
        approach_z = lib["T_obj_grasp"][:, 2, frames.APPROACH_AXIS]
        manifest[obj.key] = {
            "library": f"{obj.key}.npz", "library_sha256": digest,
            "graspdatagen_commit": commit, "gripper_config": "robotiq_2f_85",
            "gripper_usd_sha256": library.file_sha256(root / "bots" / "robotiq_2f_85.usd"),
            "obj": mesh_manifest[obj.key]["obj"], "obj_sha256": mesh_manifest[obj.key]["obj_sha256"],
            "frame_map": json.loads(frame_map_path.read_text())["translation_gdg_from_droid_m"],
            "cli": {"grasp_guess": {"num_grasps": int(len(guess["names"])) if len(guess["names"]) else None,
                                    "seed": guess["header"].get("seed")},
                    "grasp_sim_mass": {"force_magnitude_g": masses[obj.key]},
                    "grasp_sim_1g": {"force_magnitude_g": 1.0}},
            "counts": {"guesses": int(len(guess["names"])), "held_mass_scaled": n_valid,
                       "held_1g": int(len(passed_1g_names)), "held_both": int(passed_1g.sum()),
                       "library_with_twins": int(len(lib["T_obj_grasp"]))},
            "pad_penetration_mm_percentiles": {str(q): float(np.percentile(lib["pad_penetration"], q) * 1000)
                                               for q in (50, 90, 99, 100)},
            "pad_penetration_over_2mm": int((lib["pad_penetration"] > 0.002).sum()),
            "approach_z_hist": np.histogram(approach_z, bins=[-1.01, -0.7, -0.3, 0.3, 0.7, 1.01])[0].tolist(),
        }
        print(f"{obj.key}: {len(guess['names'])} guesses, {n_valid} held (mass-scaled), "
              f"{len(passed_1g_names)} held (1 g), library {len(lib['T_obj_grasp'])} with twins")
    library.write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
