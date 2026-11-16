"""L/T arm selection, evaluation-layout coincidence and explicit-target scheduling (no Isaac Sim)."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robolab.core.omy_teleop import layout_arms
from robolab.core.omy_teleop.final_dataset import parent_key
from robolab.core.omy_teleop.layout_arms import ARMS, ExplicitTargets, choose_strata, coincident, stratum_of
from robolab.core.omy_teleop.pipeline import source_key

NAMES = ("obj", "bin")


def _row(kind, i, parent=None, target=None, tier=None, start=(0.5, 0.2, 0.5, -0.2)):
    rng = np.random.default_rng(i)
    row = {"kind": kind, "task": f"T{i % 7}", "episode": f"demo_{i}", "file": f"/runs/{kind}/{i}/omy_demos.hdf5",
           "fingerprint": f"f{kind}{i}", "blocks": [rng.normal(size=4), rng.normal(size=3), rng.normal(size=2), rng.normal(size=2)],
           "generation_target": target, "generation_ranges": None,
           "start_poses": {"obj": [start[0], start[1], 0, 1, 0, 0, 0], "bin": [start[2], start[3], 0, 1, 0, 0, 0]}}
    if kind == "generated":
        row["generation_source"] = {"file": f"/collected/P{parent}/part_000/omy_demos.hdf5", "episode": "demo_0"}
    return row


def _pool(per_tier=60, explicit=60, parents=10):
    rows, tiers = [], {}
    rows += [_row("collected", i) for i in range(10)]
    i = 100
    for tier, n in (("near", per_tier), ("moderate", per_tier // 2), ("wide", per_tier // 2)):
        for k in range(n):
            row = _row("generated", i, parent=k % parents)
            tiers[str(Path(row["file"]).resolve())] = tier
            rows.append(row)
            i += 1
    for k in range(explicit):
        rows.append(_row("generated", i, parent=k % parents, target=f"t{k // 2:03d}"))  # two demos per target
        i += 1
    return rows, tiers


def test_each_arm_gets_its_exact_stratum_counts_balanced_over_parents():
    rows, tiers = _pool()
    strata = [stratum_of(r, tiers) for r in rows]
    for arm, quotas in ARMS.items():
        chosen, _ = choose_strata(rows, quotas, strata, distinct={"explicit": lambda r: r["generation_target"]})
        assert Counter(stratum_of(r, tiers) for r in chosen) == Counter(quotas)
        assert len({r["fingerprint"] for r in chosen}) == 50
        if "explicit" in quotas:
            targets = [r["generation_target"] for r in chosen if r["generation_target"]]
            assert len(targets) == len(set(targets)) == 20  # one demo per layout
        for stratum in quotas:
            if stratum == "collected":
                continue
            parents = Counter(parent_key(r) for r in chosen if stratum_of(r, tiers) == stratum)
            assert max(parents.values()) - min(parents.values()) <= 1 and len(parents) == min(10, quotas[stratum])


def test_a_short_stratum_is_refused_not_backfilled():
    rows, tiers = _pool(explicit=30)  # 15 distinct targets < 20
    strata = [stratum_of(r, tiers) for r in rows]
    with pytest.raises(ValueError, match="'explicit' needs 20; 15 available"):
        choose_strata(rows, ARMS["T"], strata, distinct={"explicit": lambda r: r["generation_target"]})
    choose_strata(rows, ARMS["L"], strata)  # L needs no explicit targets


def test_starts_at_evaluation_layouts_are_found():
    layouts = [{"id": "v2_00", "poses": {"obj": [0.3, 0.1, 0], "bin": [0.6, -0.2, 0]}}]
    rows = [_row("generated", 1, parent=0, start=(0.31, 0.1, 0.6, -0.2)),   # 10 mm away
            _row("generated", 2, parent=0, start=(0.3, 0.1, 0.63, -0.2)),   # bin 30 mm away
            _row("generated", 3, parent=0, start=(0.5, 0.3, 0.6, -0.2))]
    assert coincident(rows, layouts, NAMES) == {0: "v2_00"}


def _work(tmp_path, keys=("A", "B", "C", "D")):
    sources = [{"task": f"Task{k}", "input": f"/collected/Task{k}/part_000", "episode": "demo_0"} for k in keys]
    work = SimpleNamespace(sources=sources, qualified={source_key(s) for s in sources[:3]}, family="obj",
                           folder=tmp_path / "generated")
    (work.folder / "jobs").mkdir(parents=True)
    return work


def test_explicit_jobs_use_ranked_qualified_sources_with_fallback(tmp_path):
    work = _work(tmp_path)
    manifest = tmp_path / "targets.json"
    manifest.write_text('{"targets": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]}')
    ranking = {"targets": [{"id": "t1", "ranked_sources": ["TaskD:demo_0", "TaskB:demo_0", "TaskA:demo_0", "TaskC:demo_0"]},
                           {"id": "t2", "ranked_sources": []},
                           {"id": "t3", "ranked_sources": ["TaskC:demo_0"]}]}
    runner = ExplicitTargets(work, manifest, ranking, [0], "p4", need=2, max_sources=2)
    # D is not qualified; at most two sources per target.
    assert [s["task"] for s in runner.ranked["t1"]] == ["TaskB", "TaskA"]
    job = runner.next_job()
    assert job["target_id"] == "t1" and job["task"] == "TaskB" and job["source_rank"] == 0
    assert job["stage"] == "generate" and job["tier"] == "explicit" and job["generator_profile"] == "p4"
    assert job["id"].startswith(layout_arms.EXPLICIT_PREFIX) and job["target_manifest"] == str(manifest.resolve())
    runner.results["t1"] = [{**job, "accepted": [], "errors": []}]
    job = runner.next_job()
    assert (job["target_id"], job["task"], job["source_rank"]) == ("t1", "TaskA", 1)  # fallback to the next source
    runner.results["t1"].append({**job, "accepted": ["demo_0"], "errors": []})
    job = runner.next_job()
    assert job["target_id"] == "t3"  # t1 done; t2 has no feasible source
    runner.results["t3"] = [{**job, "accepted": ["demo_0"], "errors": []}]
    assert runner.next_job() is None  # need reached
    assert runner.report()["accepted_targets"] == 2 and runner.report()["accepted_by_source_rank"] == {1: 1, 0: 1}


def test_generator_profile_identity():
    from robolab.core.omy_teleop.pipeline import Pipeline, implementation, implementation_tag
    from robolab.core.omy_teleop.repairs import GENERATOR_PROFILES, profile_settings
    plain, p4 = implementation(), implementation("p4")
    assert "generator_profile" not in plain and p4["generator_profile"]["args"] == list(GENERATOR_PROFILES["p4"])
    assert implementation_tag(plain) != implementation_tag(p4)
    settings = profile_settings("p4")
    assert settings.over_bin and settings.lift_clearance_m == 0.03 and not settings.symmetric_yaw
    nowhere = Path("/nonexistent")
    with pytest.raises(ValueError, match="Unknown generator profile"):
        Pipeline({"summary": nowhere / "s", "pools": nowhere, "state": nowhere}, [0], dry_run=True,
                 generator_profile="p9")
    p = Pipeline({"summary": nowhere / "s", "pools": nowhere, "state": nowhere}, [0], dry_run=True,
                 generator_profile="p4")
    assert p.profile_field() == {"generator_profile": "p4"} and p.impl == p4
