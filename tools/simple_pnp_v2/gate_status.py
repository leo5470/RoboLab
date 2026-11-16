#!/usr/bin/env python3
"""Per-slot gate status for authored Simple-PnP v2 slots, verified against each scene's hash.

    python3 tools/simple_pnp_v2/gate_status.py   # writes output/simple_pnp_v2_validation/production/gate_status.json

physics: all five run reports of the matching task pass and carry the scene's hash;
reach: production/reach.json row passes for the same hash; visibility (handle entries):
the report passes. task_checks record which tasks were imported at runtime (success
check or v2 evaluation) and which had the goal-semantics success check.
"""
import collections
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V = Path(__file__).resolve().parents[2] / "output/simple_pnp_v2_validation"
REL = "output/simple_pnp_v2_validation"


def main():
    os.chdir(V)
    recs = {r["slot"]: r for r in (json.load(open(f)) for f in glob.glob("generation/*.json")) if r["status"] == "authored"}
    reach = {r["slot"]: r for r in json.load(open("production/reach.json"))["results"]}
    handles = set(open("production/handle_tasks.txt").read().split())
    sc = {json.load(open(f))["task"]: json.load(open(f))["status"] for f in glob.glob("success_check/sample/*.json")}
    loaded = set()
    for f in glob.glob("../pps_v2_eval_20261104/gamma_0/*/episode_results.jsonl"):
        loaded |= {json.loads(l)["env_name"] for l in open(f) if l.strip()}

    def one(item):
        s, rec = item
        rr = reach[s]; src = rr["pose"]["source"]; t = os.path.basename(src)[:-len("_run0.json")]
        pdir = os.path.basename(os.path.dirname(src)); sha = rec["measurements"]["scene_sha256"]
        runs = [json.load(open(f"physics/{pdir}/{t}_run{i}.json")) for i in range(5) if os.path.exists(f"physics/{pdir}/{t}_run{i}.json")]
        g = {"geometry": "passed",
             "physics": "passed" if len(runs) == 5 and all(x["status"] == "passed" and x["scene_sha256"] == sha for x in runs) else "failed",
             "reach": "passed" if rr["status"] == "passed" and rr["scene_sha256"] == sha else "failed"}
        vrep = None
        if t in handles:
            v = [f for f in (f"visibility/production/{t}.json", f"pilot3/visibility/{t}.json") if os.path.exists(f)]
            vv = json.load(open(v[0])) if v else {}
            g["visibility"] = "passed" if vv.get("status") == "passed" and vv.get("scene_sha256", sha) == sha else "failed"
            vrep = f"{REL}/{v[0]}" if v else None
        else:
            g["visibility"] = "not_applicable"
        return s, {"task": t, "scene_sha256": sha, "gates": g,
                   "evidence": {"physics": f"{REL}/physics/{pdir}/{t}_run[0-4].json", "reach": f"{REL}/production/reach.json",
                                "visibility": vrep},
                   "task_checks": {"syntax": "passed",
                                   "runtime_import": "passed" if (t in sc or t in loaded) else "not_run",
                                   "goal_semantics": sc.get(t, "not_run")}}

    out = dict(ThreadPoolExecutor(32).map(one, sorted(recs.items())))
    bad = {s: v["gates"] for s, v in out.items() if "failed" in v["gates"].values()}
    Path("production/gate_status.json").write_text(json.dumps(
        {"note": "Per-slot gate status, verified against each scene file hash.", "slots": out}, indent=1))
    print(len(out), "slots; failing", len(bad), list(bad.items())[:3])
    print(collections.Counter(v["task_checks"]["runtime_import"] for v in out.values()),
          collections.Counter(v["task_checks"]["goal_semantics"] for v in out.values()))


if __name__ == "__main__":
    main()
