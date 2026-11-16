"""Tier quotas of the single-source pipeline scheduler, without generation."""

from collections import Counter
from pathlib import Path

import pytest

from robolab.core.omy_teleop import pipeline
from robolab.core.omy_teleop.pipeline import TIERS, Pipeline, SessionWork, allocate, source_key, tier_of


def _ranges(tier):
    spec = TIERS[tier]
    return {"pose_range": {"x": [-spec["pose_range"], spec["pose_range"]], "y": [0, 0],
                           "yaw": [-spec["yaw_range"], spec["yaw_range"]]},
            "object_pose_range": {"x": [-spec["object_pose_range"], spec["object_pose_range"]], "y": [0, 0],
                                  "yaw": [-spec["object_yaw_range"], spec["object_yaw_range"]]}}


def _setup(parents, factor, humans=10, count=50):
    nowhere = Path("/nonexistent")
    p = Pipeline({"summary": nowhere / "s.json", "pools": nowhere, "state": nowhere}, [0], count=count,
                 candidate_factor=factor, dry_run=True)
    work = SessionWork("s")
    work.family = "f"
    work.sources = [{"task": f"T{i:02d}", "input": f"/x/T{i:02d}/part_000", "episode": "demo_0"} for i in range(parents)]
    work.qualified = {source_key(s) for s in work.sources}
    work.rows = [{"kind": "collected"}] * humans
    return p, work


def _row(job, tier=None):
    return {"kind": "generated", "generation_ranges": _ranges(tier or job["tier"]),
            "generation_source": {"file": job["input"] + "/omy_demos.hdf5", "episode": job["episode"]}}


def _finish(work, job, accepted, attempts=None):
    key = source_key(job)
    work.results[key].append({"tier": job["tier"], "attempts": job["num_demos"] if attempts is None else attempts,
                              "accepted": [f"demo_{i}" for i in range(accepted)], "errors": []})
    work.rows += [_row(job)] * accepted


def _run(p, work, accept=lambda job: job["num_demos"], limit=2000):
    jobs = []
    for _ in range(limit):
        step = p.plan(work)
        if not isinstance(step, dict):
            return step, jobs
        jobs.append(step)
        _finish(work, step, accept(step))
    raise AssertionError("scheduler did not finish")


def _by(work, field):
    counts = Counter()
    for row in work.rows:
        if row["kind"] == "generated":
            key = (str(Path(row["generation_source"]["file"]).parent), row["generation_source"]["episode"])
            counts[{"tier": tier_of(row["generation_ranges"]), "parent": key}[field]] += 1
    return counts


@pytest.mark.parametrize("parents,factor,expected", [
    (10, 1.5, {"near": 30, "moderate": 15, "wide": 15}),
    (10, 3, {"near": 60, "moderate": 30, "wide": 30}),
    (9, 1.5, {"near": 30, "moderate": 15, "wide": 15}),
    (9, 3, {"near": 60, "moderate": 30, "wide": 30}),
])
def test_every_tier_reaches_its_global_total(parents, factor, expected):
    # The previous scheduler stopped at 60 near / 0 / 0 (factor 1.5) and 80 / 40 / 0 (factor 3) for ten parents.
    p, work = _setup(parents, factor)
    step, _ = _run(p, work)
    assert step == "select" and work.tier_shortfall == {}
    assert dict(_by(work, "tier")) == expected
    totals, targets = p.tier_targets(work)
    have = Counter()
    for row in work.rows[10:]:
        have[(str(Path(row["generation_source"]["file"]).parent), "demo_0"), tier_of(row["generation_ranges"])] += 1
    assert have == Counter({k: v for k, v in targets.items() if v})


def test_integer_allocation_is_exact():
    assert allocate(60, [.5, .25, .25]) == [30, 15, 15]
    assert allocate(7, [.5, .25, .25]) == [3, 2, 2]
    assert allocate(5, [1, 1, 1]) == [2, 2, 1]
    assert allocate(0, [1, 1]) == [0, 0]
    p, work = _setup(7, 1.5, humans=7, count=40)  # 33 generated -> 50 candidates
    totals, targets = p.tier_targets(work)
    assert sum(totals.values()) == 50 and totals == {"near": 25, "moderate": 13, "wide": 12}
    for tier, total in totals.items():
        shares = [targets[k, tier] for k in sorted(work.qualified)]
        assert sum(shares) == total and max(shares) - min(shares) <= 1


def test_rejected_candidates_are_replaced():
    p, work = _setup(10, 1.5)
    produced = [0]

    def every_other(job):
        """Quality rejects every second candidate of the whole run."""
        accepted = sum((produced[0] + i) % 2 for i in range(job["num_demos"]))
        produced[0] += job["num_demos"]
        return accepted

    step, jobs = _run(p, work, accept=every_other)
    assert step == "select" and dict(_by(work, "tier")) == {"near": 30, "moderate": 15, "wide": 15}
    assert len(jobs) > 30  # more jobs than with every candidate accepted


def test_an_exhausted_parent_share_moves_to_other_parents_not_other_tiers():
    p, work = _setup(10, 1.5)
    exhausted = source_key(work.sources[0])
    work.results[exhausted].append({"tier": "near", "attempts": p.max_attempts_per_source, "accepted": []})
    step, jobs = _run(p, work)
    assert step == "select" and dict(_by(work, "tier")) == {"near": 30, "moderate": 15, "wide": 15}
    assert all(source_key(j) != exhausted for j in jobs)


def test_a_dead_tier_is_reported_when_no_parent_can_fill_it():
    p, work = _setup(3, 1.5, humans=3, count=13)  # 10 generated -> 15 candidates: 7 / 4 / 4
    for source in work.sources:
        work.results[source_key(source)].append({"tier": "wide", "attempts": 20, "accepted": []})
    step, jobs = _run(p, work)
    assert {j["tier"] for j in jobs} == {"near", "moderate"}
    assert step == "select" and work.tier_shortfall == {"wide": 4}  # 11 candidates still cover 10 slots
    p, work = _setup(3, 1.5, humans=3, count=13)
    for source in work.sources:
        for tier in ("moderate", "wide"):
            work.results[source_key(source)].append({"tier": tier, "attempts": 20, "accepted": []})
    step, _ = _run(p, work)
    assert step == "shortfall" and work.tier_shortfall == {"moderate": 4, "wide": 4}


def test_concurrent_jobs_never_reserve_more_than_a_tier_needs():
    p, work = _setup(10, 1.5)
    reserved = Counter()
    while isinstance(step := p.plan(work), dict):
        work.active[step["id"]] = step
        reserved[step["tier"]] += step["num_demos"]
    totals, targets = p.tier_targets(work)
    assert step is None and len(work.active) == 10  # one job per parent at a time, then wait
    assert all(reserved[t] <= totals[t] for t in totals)
    assert all(j["num_demos"] <= targets[source_key(j), j["tier"]] for j in work.active.values())
    # Finishing the jobs releases their reservations to the accepted counts.
    for job in list(work.active.values()):
        work.active.pop(job["id"])
        _finish(work, job, job["num_demos"])
    step, _ = _run(p, work)
    assert step == "select" and dict(_by(work, "tier")) == {"near": 30, "moderate": 15, "wide": 15}


def test_resume_reads_tiers_from_recorded_ranges():
    p, work = _setup(10, 1.5)
    first = {"input": "/x/T00/part_000", "episode": "demo_0", "tier": "wide"}
    work.rows += [_row(first)] * 15 + [{**_row(first), "generation_ranges": None}] * 5
    step, jobs = _run(p, work)
    assert step == "select" and not [j for j in jobs if j["tier"] == "wide"]
    assert _by(work, "tier")[None] == 5  # untiered candidates (e.g. explicit targets) count toward no tier


def test_tier_is_none_for_unknown_or_unrandomized_ranges():
    assert tier_of(None) is None and tier_of({"pose_range": None, "object_pose_range": None}) is None
    assert {tier_of(_ranges(t)) for t in TIERS} == set(TIERS)
    odd = _ranges("near")
    odd["pose_range"]["x"][1] = .015
    assert tier_of(odd) is None

