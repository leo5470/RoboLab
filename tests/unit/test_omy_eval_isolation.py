"""The OMY scheduler remains instance-local and outside standard policy evaluation."""

import ast
from pathlib import Path

import pytest

from robolab.core.omy_teleop.performance import RenderPacer


@pytest.mark.parametrize("relative_path", [
    "policies/pi0_family/run.py",
    "robolab/eval/runner.py",
    "robolab/eval/episode.py",
    "robolab/registrations/droid/auto_env_registrations_jointpos.py",
    "robolab/core/environments/runtime.py",
    "robolab/core/environments/env.py",
])
def test_standard_evaluation_does_not_install_omy_scheduler(relative_path):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / relative_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        assert not any("omy_teleop" in module for module in modules), relative_path


def test_omy_pacer_does_not_patch_another_simulator_or_its_class():
    class Sim:
        def render(self):
            return "rendered"

    teleop_sim, eval_sim = Sim(), Sim()
    original = Sim.render
    waits = []
    pacer = RenderPacer(teleop_sim, 1 / 30, clock=lambda: 0.0, sleep=waits.append)
    try:
        with pacer.step():
            assert eval_sim.render() == "rendered"
        assert waits == []
        assert Sim.render is original
        assert "render" not in vars(eval_sim)
    finally:
        pacer.close()
    assert "render" not in vars(teleop_sim)
