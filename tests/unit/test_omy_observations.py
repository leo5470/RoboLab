"""Observation caching lifecycle without starting Kit or accessing hardware."""

from types import SimpleNamespace

import pytest

from robolab.core.omy_teleop import observations as obs


def test_one_pose_calculation_per_pass_and_no_stale_reset(monkeypatch):
    calls = []
    env = SimpleNamespace(pose=1)

    def poses(env):
        calls.append(env.pose)
        return tuple(env.pose + i for i in range(4))

    class Manager:
        def compute(self, *, update_history=False):
            return (obs.ee_pos(env), obs.ee_quat(env), obs.eef_pos(env), obs.eef_quat(env))

    monkeypatch.setattr(obs, "_compute_poses", poses)
    env.observation_manager = Manager()
    cache = obs.ObservationPassCache(env)
    try:
        assert env.observation_manager.compute(update_history=True) == (1, 2, 3, 4)
        assert calls == [1]
        env.pose = 10  # reset/teleport at the same simulation timestamp
        assert env.observation_manager.compute() == (10, 11, 12, 13)
        assert calls == [1, 10]
        env.pose = 20
        assert obs.ee_pos(env) == 20  # calls outside compute must not reuse poses
        assert cache.values is None
    finally:
        cache.close()
    cache.close()
    assert not hasattr(env, "_omy_observation_pass")
    assert "compute" not in vars(env.observation_manager)


def test_exception_and_existing_instance_method_are_restored(monkeypatch):
    env = SimpleNamespace()
    monkeypatch.setattr(obs, "_compute_poses", lambda env: (1, 2, 3, 4))

    def compute():
        obs.ee_pos(env)
        raise ValueError("observation failed")

    env.observation_manager = SimpleNamespace(compute=compute)
    cache = obs.ObservationPassCache(env)
    with pytest.raises(ValueError, match="observation failed"):
        env.observation_manager.compute()
    assert cache.values is None
    cache.close()
    assert env.observation_manager.compute is compute


def test_duplicate_install_rejected():
    env = SimpleNamespace(observation_manager=SimpleNamespace(compute=lambda: {}))
    cache = obs.ObservationPassCache(env)
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            obs.ObservationPassCache(env)
    finally:
        cache.close()
