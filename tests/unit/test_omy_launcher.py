"""CLI shutdown/error reporting without loading Isaac Sim or opening the leader."""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("outcome, expected", [(0, 0), (130, 130), (RuntimeError("test failure"), 1)])
def test_status_is_set_before_kit_shutdown(monkeypatch, capsys, outcome, expected):
    calls = []
    app = SimpleNamespace(
        app=SimpleNamespace(post_quit=lambda status: calls.append(("quit", status))),
        close=lambda: calls.append(("close", None)),
    )

    class Launcher:
        def __init__(self, args):
            self.app = app

        @staticmethod
        def add_app_launcher_args(parser):
            pass

    def run(args, simulation_app):
        assert simulation_app is app
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake_isaac = ModuleType("isaaclab")
    fake_isaac_app = ModuleType("isaaclab.app")
    fake_isaac_app.AppLauncher = Launcher
    fake_runner = ModuleType("robolab.core.omy_teleop.runner")
    fake_runner.run = run
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaac)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_isaac_app)
    monkeypatch.setitem(sys.modules, "robolab.core.omy_teleop.runner", fake_runner)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.0")
    monkeypatch.setattr(sys, "argv", ["teleop_omy.py"])
    path = Path(__file__).resolve().parents[2] / "examples" / "teleop_omy.py"
    spec = importlib.util.spec_from_file_location("omy_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == expected
    assert calls == [("quit", expected), ("close", None)]
    if isinstance(outcome, Exception):
        assert "test failure" in capsys.readouterr().err
