"""Phase 1 step 1: export each batch object's mesh as an OBJ in the RoboLab object frame (D4).

The object frame is the frame whose pose ``WorldState.get_pose`` reports: the rigid-body prim that the scene
references (the asset's default prim). Scenes override that prim's own xformOps with the layout pose, so mesh
points are taken relative to it (all child transforms, including any scale, are applied and baked in).

    scripts/scripted_demos/run.sh --pxr scripts/scripted_demos/grasps/export_meshes.py [--objects 303 301 ...]

Writes ``<grasp_data>/meshes/<idx>_<name>.obj``, ``object_scales.json`` (all 1.0) and ``meshes/manifest.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos.records import file_sha256


def collect_mesh(usd_path):
    """Triangulated (vertices, faces) of every visible mesh under the default prim, in the default prim's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError(f"{usd_path}: expected Z-up and metersPerUnit 1")
    root = stage.GetDefaultPrim()
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"{usd_path}: the default prim {root.GetPath()} is not the rigid body")
    # Row-vector convention (pxr): p_world = p_local @ M.  p_root = p_world @ inv(M_root).
    root_inv = np.linalg.inv(np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0)))
    vertices, faces, meshes, offset = [], [], [], 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() not in ("default", "render"):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(points) == 0:
            continue
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)) @ root_inv
        P = (np.c_[points, np.ones(len(points))] @ M)[:, :3]
        tris, start = [], 0
        for n in counts:  # fan triangulation
            poly = indices[start : start + n]
            tris.extend((poly[0], poly[k], poly[k + 1]) for k in range(1, n - 1))
            start += n
        T = np.asarray(tris, dtype=np.int64)
        if np.linalg.det(M[:3, :3]) < 0:  # mirrored transform: keep outward winding
            T = T[:, ::-1]
        vertices.append(P)
        faces.append(T + offset)
        offset += len(P)
        meshes.append({"prim": str(prim.GetPath()), "points": int(len(P)), "triangles": int(len(T))})
    if not vertices:
        raise ValueError(f"{usd_path}: no visible meshes")
    return np.vstack(vertices), np.vstack(faces), meshes, str(root.GetPath())


def write_obj(path, vertices, faces) -> None:
    with open(path, "w") as f:
        f.write("# exported by scripts/scripted_demos/grasps/export_meshes.py (RoboLab object frame, meters)\n")
        np.savetxt(f, vertices, fmt="v %.7f %.7f %.7f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=[str(o.index) for o in objects.BATCH])
    args = parser.parse_args()

    out = objects.grasp_data_root() / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    scales = {}
    for key in args.objects:
        obj = objects.get(key)
        usd = objects.ROBOLAB_DIR / obj.usd_path
        vertices, faces, meshes, root = collect_mesh(usd)
        obj_path = out / f"{obj.key}.obj"
        write_obj(obj_path, vertices, faces)
        scales[obj_path.name] = 1.0
        manifest[obj.key] = {
            "usd_path": obj.usd_path, "usd_sha256": file_sha256(usd), "root_prim": root, "meshes": meshes,
            "obj": obj_path.name, "obj_sha256": file_sha256(obj_path),
            "bounds_min": vertices.min(0).round(5).tolist(), "bounds_max": vertices.max(0).round(5).tolist(),
            "extent_mm": ((vertices.max(0) - vertices.min(0)) * 1000).round(1).tolist(),
        }
        print(f"{obj.key}: {len(vertices)} vertices, {len(faces)} triangles, extent {manifest[obj.key]['extent_mm']} mm")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    all_scales = {f"{k}.obj": 1.0 for k in manifest}
    (out / "object_scales.json").write_text(json.dumps(all_scales, indent=1))


if __name__ == "__main__":
    main()
