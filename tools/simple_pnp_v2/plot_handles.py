#!/usr/bin/env python3
"""Plot each handle record over its asset's rest-frame points (top and side views).

    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/plot_handles.py \
        --inputs output/simple_pnp_v2_validation/inputs/inputs.json \
        --handles output/simple_pnp_v2_validation/inputs/handle_records.json \
        --out output/simple_pnp_v2_validation/inputs/handle_plots

Grey: asset points. Red: the record's handle points. Arrow: the recorded
direction. Use these to confirm or correct each record before freezing.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from handle_records import rest_points

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--inputs", type=Path, required=True); parser.add_argument("--handles", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True); parser.add_argument("--classes", default="ABC")
args = parser.parse_args()
inputs = json.loads(args.inputs.read_text()); E = {e["catalog_index"]: e for e in inputs["entries"]}
recs = json.loads(args.handles.read_text())["records"]; args.out.mkdir(parents=True, exist_ok=True)
for k, r in recs.items():
    if r["class"] not in args.classes:
        continue
    e = E[int(k)]; X = rest_points(ROOT / e["payload"], e["geometry"]["root_rotation_wxyz"], e["scale"])
    X = X[np.random.default_rng(0).choice(len(X), min(len(X), 6000), replace=False)]; H = np.array(r["points"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, (i, j, t) in zip(ax, ((0, 1, "top (X right, Y up)"), (0, 2, "front (X right, Z up)"), (1, 2, "side (Y right, Z up)"))):
        a.scatter(X[:, i], X[:, j], s=1, c="0.7"); a.scatter(H[:, i], H[:, j], s=4, c="r"); a.set_aspect("equal"); a.set_title(t)
    d = r.get("handle_direction_local_deg")
    if d is not None:
        c = np.array(r["body_center_xy"]); L = np.ptp(X[:, :2], 0).max() / 2
        ax[0].annotate("", c + L * np.array([np.cos(np.radians(d)), np.sin(np.radians(d))]), c, arrowprops=dict(color="b"))
    fig.suptitle(f"{k} {r['prim_name']} class {r['class']} dir={d}"); fig.tight_layout()
    fig.savefig(args.out / f"{int(k):03d}_{r['prim_name']}.png", dpi=70); plt.close(fig)
print("wrote", args.out)
