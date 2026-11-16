"""Phase 2 step 3: visual check of the library. Overlays 20 grasps per object (DROID palm, pad and finger slice
boxes; red handle, blue rim, green body grasps) on the object mesh, from three views, and counts library grasps whose
open fingertip pads the mesh enters by more than 3 mm (gate: none).

    scripts/scripted_demos/run.sh --pxr scripts/scripted_demos/grasps/render_grasps.py [--objects ...] [--count 20]   (needs matplotlib)

Writes ``<grasp_data>/library/renders/<idx>_<name>.png`` and ``renders/pad_intersections.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robolab.core.scripted_demos.grasps import frames, library, objects  # noqa: E402
from robolab.core.scripted_demos import select  # noqa: E402

EDGES = [(a, b) for a, b in itertools.combinations(range(8), 2) if bin(a ^ b).count("1") == 1]


GATE_DEPTH = 0.003  # 1.75 mm open-gap difference + GraspDataGen's 1 mm standoff, rounded up
COLORS = {"handle": "tab:red", "rim": "tab:blue", "body": "tab:green"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="*", default=[str(o.index) for o in objects.BATCH])
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    root = objects.grasp_data_root()
    out = root / "library" / "renders"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "pad_intersections.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}  # other objects' rows are kept
    rng = np.random.default_rng(0)
    for key in args.objects:
        obj = objects.get(key)
        lib = library.load(root / "library" / f"{obj.key}.npz")
        v = objects.mesh_vertices(obj)
        pts = v[:: max(1, len(v) // 6000)]
        depth = lib["pad_penetration"] if "pad_penetration" in lib else select.pad_penetration(lib["T_obj_grasp"], pts)
        hits = depth > GATE_DEPTH
        n = len(lib["T_obj_grasp"])
        approach_ok = lib["T_obj_grasp"][:, 2, 0] <= select.Params().max_approach_up  # not from below (upright object)
        kind = library.classify(lib["T_obj_grasp"], v, obj.index)
        pick = []
        for k, share in (("handle", 0.4), ("rim", 0.3), ("body", 0.3)):
            pool = np.nonzero((lib["twin"] == 0) & approach_ok & (kind == k))[0]
            pick += list(rng.choice(pool, size=min(int(round(args.count * share)), len(pool)), replace=False))
        approach = lib["T_obj_grasp"][:, :3, frames.APPROACH_AXIS]
        report[obj.key] = {"library": int(n), "pad_intersections": int(hits.sum()),
                           "pad_penetration_mm_percentiles": {str(q): float(np.percentile(depth, q) * 1000)
                                                              for q in (50, 90, 99, 100)},
                           "pad_intersection_fraction": float(hits.mean()) if n else None,
                           "approach_down_fraction": float((approach[:, 2] < -0.7).mean()) if n else None,
                           "approach_side_fraction": float((np.abs(approach[:, 2]) <= 0.3).mean()) if n else None,
                           "class_counts": {k: int((kind == k).sum()) for k in COLORS},
                           "class_counts_not_from_below": {k: int(((kind == k) & approach_ok).sum()) for k in COLORS},
                           "rendered": [int(i) for i in pick]}
        fig = plt.figure(figsize=(18, 6))
        views = [("top (x right, y up)", 90, -90), ("front (x right, z up)", 0, -90), ("side (y right, z up)", 0, 0)]
        for k, (title, elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(*pts[::3].T, s=0.3, c="0.6", alpha=0.5)
            for i in pick:
                T = lib["T_obj_grasp"][i]
                color = COLORS[kind[i]]
                for lo, hi in [select.PALM, *select.PADS, *select.FINGERS]:
                    c = frames.transform_points(T, select.box_points((lo, hi)))
                    for a, b in EDGES:
                        ax.plot(*np.stack([c[a], c[b]]).T, color=color, lw=0.8)
            span = (v.max(0) - v.min(0)).max() / 2 + 0.06
            mid = (v.max(0) + v.min(0)) / 2
            ax.set_xlim(mid[0] - span, mid[0] + span)
            ax.set_ylim(mid[1] - span, mid[1] + span)
            ax.set_zlim(mid[2] - span, mid[2] + span)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{obj.key}: {title}")
        counts = report[obj.key]["class_counts_not_from_below"]
        fig.suptitle(f"{obj.key}: {len(pick)} of {n} library grasps (red handle {counts['handle']}, blue rim "
                     f"{counts['rim']}, green body {counts['body']}; not from below); pads entered > 3 mm: {hits.sum()}")
        fig.tight_layout()
        fig.savefig(out / f"{obj.key}.png", dpi=110)
        plt.close(fig)
        print(obj.key, json.dumps({k: v for k, v in report[obj.key].items() if k != "rendered"}))
    report_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
