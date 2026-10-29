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


def _footprint(corners):
    """Convex XY envelope of transformed bounds, including tilted boxes."""
    points = sorted(set(map(tuple, np.asarray(corners, dtype=float)[:, :2])))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    chains = []
    for sequence in (points, points[::-1]):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.extend(chain[:-1])
    if len(chains) < 3:
        raise ValueError("Degenerate tabletop footprint")
    return np.asarray(chains)


def footprint_separation(first, second):
    """Largest separating-axis gap; negative means bounding envelopes overlap.

    This conservative broad-phase check does not prove mesh penetration or
    graspability. A positive gap is a lower bound on geometric clearance.
    """
    polygons = [_footprint(points) for points in (first, second)]
    gaps = []
    for polygon in polygons:
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            a, b = (points @ axis for points in polygons)
            gaps.append(max(b.min() - a.max(), a.min() - b.max()))
    return float(max(gaps))


def flat_layout_issues(objects, table, *, pair_clearance=0.02, edge_clearance=0.01):
    """Validate separate tabletop objects before settling or accepting a start.

    Uses complete transformed bounds, not root positions or catalog dimensions.
    The support is an axis-aligned rectangular tabletop (the Simple-PnP table).
    """
    if any(not math.isfinite(v) or v < 0 for v in (pair_clearance, edge_clearance)):
        raise ValueError("Invalid layout clearance")
    for points in [table, *objects.values()]:
        if np.asarray(points).shape != (8, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid layout bounds")
    lo, hi = np.min(table, axis=0), np.max(table, axis=0)
    issues = []
    for name, points in objects.items():
        if (np.any(np.min(points, axis=0)[:2] < lo[:2] + edge_clearance)
                or np.any(np.max(points, axis=0)[:2] > hi[:2] - edge_clearance)):
            issues.append(f"{name}: footprint outside table edge clearance")
    names = list(objects)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            gap = footprint_separation(objects[first], objects[second])
            if gap < pair_clearance:
                issues.append(f"{first}/{second}: footprint clearance {gap:.4f} m < {pair_clearance:.4f} m")
    return issues


def tilt_change_deg(initial, current):
    """Change in local-up direction; yaw and quaternion sign do not matter."""
    def up(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError("Invalid orientation")
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)])
    return math.degrees(math.acos(float(np.clip(up(initial) @ up(current), -1, 1))))
