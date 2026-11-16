#!/usr/bin/env python3
"""Summarise start_probe.py output; optionally compare two probe runs.

    .venv/bin/python tools/simple_pnp_v2/analyze_start_probe.py PROBE_DIR [OTHER_PROBE_DIR]

Per case: preparation status, table rise, and per object the bottom gap measured
against the table's *current* top (so a table that moves is not mistaken for
object motion), total motion, tilt, and peak speeds. Within a run, every case is
compared with the first case of the same task (max position/angle difference
step by step). With a second directory, matching cases are compared across
runs. Quaternion angle differences below ~0.1° are float32 rounding, not motion.
"""
import json
from pathlib import Path
import sys

import numpy as np


def angle(a, b):
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def trace_diff(t1, t2, names):
    k = min(len(t1), len(t2)); pos = ang = 0.0
    for a, b in zip(t1[:k], t2[:k]):
        for n in names:
            pos = max(pos, float(np.linalg.norm(np.array(a[n]["pose"][:3]) - np.array(b[n]["pose"][:3]))) * 1000)
            ang = max(ang, angle(a[n]["pose"][3:], b[n]["pose"][3:]))
    return pos, ang, len(t1), len(t2)


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    first = Path(sys.argv[1]); second = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    for f in sorted(first.glob("*.json")):
        if f.name == "meta.json":
            continue
        row = json.loads(f.read_text())
        print(f"\n== {row['task']}" + ("  ERROR" if row.get("error") else "") + (f"  candidate={row['candidate_scene']}" if row.get("candidate_scene") else ""))
        if row.get("error"):
            print(row["error"][-600:])
        cases = row["cases"]
        if not cases:
            continue
        names = [n for n in cases[0]["trace"][0] if n != "phase"]
        objects = [n for n in names if n != "table"]
        for c in cases:
            tr = c["trace"]
            rise = [s["table"]["bottom_gap_mm"] - tr[0]["table"]["bottom_gap_mm"] if "table" in s else 0.0 for s in tr]
            reasons = "" if c["status"] == "ready" else " | " + "; ".join(c["reasons"] or [])[:160]
            print(f"  clearance {c['clearance_m'] * 1000:.2f} mm seed {c['seed']}: prep {c['status']} {c['elapsed_s']:.3f}s,"
                  f" table rise {max(rise, key=abs):+.2f} mm, wall {c['wall_s']:.1f}s{reasons}")
            for n in objects:
                rel = [s[n]["bottom_gap_mm"] - r for s, r in zip(tr, rise)]
                last = tr[-1][n]
                print(f"    {n:28s} gap to table: start {rel[0]:+.2f} min {min(rel):+.2f} final {rel[-1]:+.2f} mm |"
                      f" motion {last.get('disp_mm', 0):.2f} mm tilt {last.get('tilt_deg', 0):.2f} deg |"
                      f" peak {max(s[n]['lin'] for s in tr[1:]):.4f} m/s {max(s[n]['ang'] for s in tr[1:]):.3f} rad/s")
        for i, c in enumerate(cases[1:], 1):
            p, a, l1, l2 = trace_diff(cases[0]["trace"], c["trace"], names)
            print(f"  case 0 vs case {i} (clearance {c['clearance_m']}, seed {c['seed']}): max {p:.4f} mm / {a:.4f} deg (steps {l1} vs {l2})")
        if second and (second / f.name).exists():
            other = json.loads((second / f.name).read_text())["cases"]
            for i, (a_case, b_case) in enumerate(zip(cases, other)):
                p, a, _, _ = trace_diff(a_case["trace"], b_case["trace"], names)
                print(f"  across runs, case {i}: max {p:.4f} mm / {a:.4f} deg")


if __name__ == "__main__":
    main()
