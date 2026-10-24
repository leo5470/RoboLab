"""Pivot-aware support placement shared by collection and scene generation."""

from itertools import product
import math

import numpy as np


def supported_root_height(root_z, bottom_z, support_z, clearance=0.002):
    if not all(math.isfinite(float(v)) for v in (root_z, bottom_z, support_z, clearance)) or clearance < 0:
        raise ValueError("Invalid support placement bounds")
    return float(root_z + support_z + clearance - bottom_z)


def bounds_corners(lower, upper, pose=(0, 0, 0, 1, 0, 0, 0), scale=(1, 1, 1)):
    lower, upper, pose, scale = (np.asarray(v, dtype=float) for v in (lower, upper, pose, scale))
    if (lower.shape != (3,) or upper.shape != (3,) or pose.shape != (7,) or scale.shape != (3,)
            or not all(np.isfinite(v).all() for v in (lower, upper, pose, scale))
            or np.any(upper < lower) or np.any(scale == 0) or np.linalg.norm(pose[3:]) < 1e-8):
        raise ValueError("Invalid transformed asset bounds")
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    corners = np.array(list(product(*zip(lower, upper)))) * scale
    return corners @ rotation.T + pose[:3]


def asset_local_bounds(path):
    """Read child transforms and geometry after USD is available; no simulation."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError(f"Asset has no valid default prim: {path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeUntransformedBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError(f"Asset has no finite geometry bounds: {path}")
    scale = Gf.Transform(UsdGeom.Xformable(stage.GetDefaultPrim()).GetLocalTransformation()).GetScale()
    return {"lower": list(bounds.GetMin()), "upper": list(bounds.GetMax()), "scale": list(scale)}
