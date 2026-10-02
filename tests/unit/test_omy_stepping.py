"""Physics cadence, renders, and fallback preserve every substep."""

from types import SimpleNamespace

import pytest

from robolab.core.omy_teleop.stepping import advance_physics, runtime_reasons


@pytest.mark.parametrize("held,interval", [(True, 2), (True, 4), (False, 2), (False, 4)])
def test_physics_loop_contract(held, interval):
    calls = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(decimation=8, sim=SimpleNamespace(render_interval=interval)),
        _sim_step_counter=0, physics_dt=1 / 120, step_dt=1 / 15,
        action_manager=SimpleNamespace(apply_action=lambda: calls.append("target")),
        scene=SimpleNamespace(write_data_to_sim=lambda: calls.append("write"),
                              update=lambda dt: calls.append(("update", dt))),
        sim=SimpleNamespace(has_gui=lambda: True, has_rtx_sensors=lambda: False,
                            step=lambda render: calls.append(("physics", render)),
                            render=lambda: calls.append(("render", env._sim_step_counter))),
    )
    advance_physics(env, held_target=held)
    assert env._sim_step_counter == 8
    assert calls.count(("physics", False)) == 8
    assert calls.count("write") == calls.count("target") == (1 if held else 8)
    assert [v[1] for v in calls if isinstance(v, tuple) and v[0] == "render"] == list(range(interval, 9, interval))
    updates = [v[1] for v in calls if isinstance(v, tuple) and v[0] == "update"]
    assert updates == pytest.approx([1 / 15] if held else [1 / 120] * 8)
    assert sum(updates) == pytest.approx(1 / 15)
    if held:
        assert calls[-1] == ("update", 1 / 15)


def test_external_wrench_rejects_runtime_optimization():
    robot = SimpleNamespace(has_external_wrench=False)
    object = SimpleNamespace(has_external_wrench=False)
    env = SimpleNamespace(scene=SimpleNamespace(articulations={"robot": robot}, rigid_objects={"banana": object}))
    assert runtime_reasons(env) == []
    object.has_external_wrench = True
    assert runtime_reasons(env) == ["external wrench requires submission on every physics substep"]
