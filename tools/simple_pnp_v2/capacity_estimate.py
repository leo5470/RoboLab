#!/usr/bin/env python3
"""Rough Monte Carlo estimate of how many v2 layouts each retained entry can admit.

    .venv/bin/python tools/simple_pnp_v2/capacity_estimate.py \
        --reach-map output/simple_pnp_v2_validation/reach_map.json

Samples bin centres (yaw 0, 42 x 28 cm footprint) and target positions/yaws
uniformly over the measured table. It applies the 10 mm edge clearance, 20 mm
footprint clearance (axis-aligned boxes from catalog dims), and the strict v1
novelty rule: target, bin, and offset each >= 3 cm from every v1 layout of that
entry (v1 root positions from the manifest). With --reach-map, the target must be
reachable at 0.03 and 0.20 m and the bin centre at release height (0.20 m, or
0.35 m for targets taller than 18 cm). Rough model: no handle, grasp, or
within-v2 diversity constraints, and catalog boxes rather than measured
footprints. Use it to rank risk, not as a capacity result.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLE = (0.1971, 0.8971, -0.5, 0.5)
BIN_HALF = (0.21, 0.14)


def reach_lookup(path):
    data = json.loads(Path(path).read_text())
    xs, ys = np.array(data["xs"]), np.array(data["ys"])
    grids = {float(k): np.array(v, bool) for k, v in data["reachable"].items()}
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]

    def ok(z, x, y):  # conservative: all four surrounding grid cells reachable
        G = grids[z]
        i = np.clip(np.floor((x - xs[0]) / step_x).astype(int), 0, len(xs) - 2)
        j = np.clip(np.floor((y - ys[0]) / step_y).astype(int), 0, len(ys) - 2)
        inside = (x >= xs[0]) & (x <= xs[-1]) & (y >= ys[0]) & (y <= ys[-1])
        return inside & G[i, j] & G[i + 1, j] & G[i, j + 1] & G[i + 1, j + 1]
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reach-map", type=Path, help="JSON from reach_map.py with heights 0.03, 0.20, 0.35")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--tries", type=int, default=20)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    ok = reach_lookup(args.reach_map) if args.reach_map else None
    manifest = json.loads((ROOT / "assets/scenes/simple_pnp/manifest.json").read_text())
    refs = collections.defaultdict(list); dims = {}; names = {}
    for s in manifest["scenes"]:
        k = s["catalog_index"]; dims[k] = s["target"]["dims"]; names[k] = s["target"]["name"]
        refs[k].append((s["target"]["x"], s["target"]["y"], s["destination"]["x"], s["destination"]["y"]))
    X0, X1, Y0, Y1 = TABLE; N = args.samples; rng = np.random.default_rng(0); rows = []
    for k, ref in sorted(refs.items()):
        R = np.array(ref); w, d, h = dims[k]
        yaw = rng.uniform(0, np.pi, N)
        hx = (abs(np.cos(yaw)) * w + abs(np.sin(yaw)) * d) / 2; hy = (abs(np.sin(yaw)) * w + abs(np.cos(yaw)) * d) / 2
        bx = rng.uniform(X0 + BIN_HALF[0] + .01, X1 - BIN_HALF[0] - .01, N); by = rng.uniform(Y0 + BIN_HALF[1] + .01, Y1 - BIN_HALF[1] - .01, N)
        tx = rng.uniform(X0, X1, N); ty = rng.uniform(Y0, Y1, N)
        g = (tx - hx >= X0 + .01) & (tx + hx <= X1 - .01) & (ty - hy >= Y0 + .01) & (ty + hy <= Y1 - .01)
        g &= np.maximum(np.abs(tx - bx) - hx - BIN_HALF[0], np.abs(ty - by) - hy - BIN_HALF[1]) >= 0.02
        if ok:
            g &= ok(0.03, tx, ty) & ok(0.20, tx, ty) & ok(0.35 if h > 0.18 else 0.20, bx, by)
        dT = np.min(np.hypot(tx[:, None] - R[:, 0], ty[:, None] - R[:, 1]), 1)
        dB = np.min(np.hypot(bx[:, None] - R[:, 2], by[:, None] - R[:, 3]), 1)
        dO = np.min(np.hypot((tx - bx)[:, None] - (R[:, 0] - R[:, 2]), (ty - by)[:, None] - (R[:, 1] - R[:, 3])), 1)
        p = float((g & (dT >= .03) & (dB >= .03) & (dO >= .03)).mean())
        rows.append({"catalog_index": k, "name": names[k], "accept_rate": p,
                     "resolve_within_tries": 1 - (1 - p) ** args.tries})
    p = np.array([r["accept_rate"] for r in rows]); res = np.array([r["resolve_within_tries"] for r in rows])
    print(f"{len(rows)} entries ({'with' if ok else 'without'} reach): per-try acceptance median {np.median(p):.3f}; "
          f"P(resolve in {args.tries} uniform tries) median {np.median(res):.2f}, <0.5: {(res < .5).sum()}, <0.1: {(res < .1).sum()}, "
          f"no accepted sample: {(p == 0).sum()}")
    print("lowest:", [(r["catalog_index"], r["name"], round(r["accept_rate"], 4)) for r in sorted(rows, key=lambda r: r["accept_rate"])[:8]])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"model": "rough Monte Carlo; see docstring", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
