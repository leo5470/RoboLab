"""Exercise the collector's finite benchmark without launching Isaac Sim."""

import ast
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from robolab.core.teleop.performance import LoopTiming, RealtimePacer
from robolab.core.teleop.views import inset_resolution


@pytest.fixture
def benchmark(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])

    def advance(seconds):
        now[0] += seconds

    sim = SimpleNamespace(step=lambda: advance(.01), render=lambda: advance(.02))
    exported = []
    recorder = SimpleNamespace(
        export_episodes=lambda env_ids=None: exported.append(env_ids),
        set_success_to_episodes=lambda ids, values: None,
    )
    env = SimpleNamespace(
        sim=sim, device="cpu", step_dt=.0667, all_terminated=False,
        recorder_manager=recorder, reset=lambda seed: None,
        cfg=SimpleNamespace(sim=SimpleNamespace(render_interval=8),
                            viewer=SimpleNamespace(resolution=(960, 540))),
    )

    def step(action):
        sim.step()
        sim.render()
        advance(.005)

    env.step = step
    args = SimpleNamespace(profile=False, benchmark_steps=3, benchmark_warmup=2,
                           benchmark_realtime=True, seed=0, defer_images=True,
                           teleop_profile="wifi", view_layout="single")
    torch = SimpleNamespace(zeros=lambda *a, **kw: 0, bool=bool, no_grad=nullcontext)
    scope = dict(os=os, time=time, json=json, torch=torch, args_cli=args, LoopTiming=LoopTiming,
                 RealtimePacer=lambda period: RealtimePacer(period, clock=lambda: now[0], sleep=advance),
                 simulation_app=SimpleNamespace(is_running=lambda: True), inset_resolution=inset_resolution)
    source = Path(__file__).resolve().parents[2] / "robolab/core/teleop/collector.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_benchmark", "_mark_attempt_failed")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return lambda env, output: scope["_benchmark"](env, output, args, scope["simulation_app"]), env, args, exported


@pytest.mark.parametrize("paced", [True, False])
def test_benchmark_excludes_warmup_and_restores_wrappers(benchmark, tmp_path, paced):
    run, env, args, exported = benchmark
    args.benchmark_realtime = paced
    original_step, original_render = env.sim.step, env.sim.render
    original_export = env.recorder_manager.export_episodes
    assert run(env, str(tmp_path)) == 0
    result = json.loads((tmp_path / "benchmark.json").read_text())
    assert result["steps"] == 3
    assert result["steps_per_second"] == pytest.approx(1 / .035)
    assert result["wall_steps_per_second"] == pytest.approx(1 / (.0667 if paced else .035))
    assert result["physics_ms_mean"] == pytest.approx(10)
    assert result["render_ms_mean"] == pytest.approx(20)
    assert result["other_ms_mean"] == pytest.approx(5)
    assert result["over_budget_steps"] == 0
    assert env.sim.step is original_step
    assert env.sim.render is original_render
    assert env.recorder_manager.export_episodes is original_export
    assert exported == [[0]]


def test_benchmark_discards_and_restores_after_error(benchmark, tmp_path):
    run, env, args, exported = benchmark
    original_step = env.sim.step
    original_export = env.recorder_manager.export_episodes
    def fail(action):
        raise RuntimeError("step failed")
    env.step = fail
    with pytest.raises(RuntimeError, match="step failed"):
        run(env, str(tmp_path))
    assert exported == [[0]]
    assert env.sim.step is original_step
    assert env.recorder_manager.export_episodes is original_export
    assert not (tmp_path / "benchmark.json").exists()


def test_benchmark_inset_metadata_survives_shared_collector_extraction(benchmark, tmp_path):
    run, env, args, _ = benchmark
    args.view_layout = "inset"
    args.inset_camera = "wrist"
    assert run(env, str(tmp_path)) == 0
    result = json.loads((tmp_path / "benchmark.json").read_text())
    assert result["inset_resolution"] == [320, 180]
    assert result["inset_camera"] == "wrist"
