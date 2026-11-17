"""Locked joints in the scripted collector's arm chain: planning IK, the command emulation's DLS step and (through the
same Jacobian) the simulator-side controller leave them where they start."""

import numpy as np

from robolab.core.scripted_demos.kinematics import Chain


def _frame(xyz, axis=None, angle=0.0):
    out = np.eye(4)
    if axis is not None:
        c, s = np.cos(angle), np.sin(angle)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        out[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c] if axis != 1 else [c, s, -s, c]
    out[:3, 3] = xyz
    return out


def _chain():
    """Seven revolute joints (each about its local x) with the joint axes turned so the arm spans 3D, plus a tool."""
    half = np.pi / 2
    T0 = [_frame([0, 0, 0.33], 1, -half), _frame([0, 0, 0], 2, half), _frame([0.32, 0, 0], 2, -half),
          _frame([0, 0.08, 0], 2, half), _frame([0.38, -0.08, 0], 2, -half), _frame([0, 0, 0], 2, half),
          _frame([0, 0.09, 0], 2, half), _frame([0.2, 0, 0])]
    K = len(T0)
    return Chain(T0=np.stack(T0), T1inv=np.stack([np.eye(4)] * K), revolute=np.array([True] * 7 + [False]),
                 limits=np.array([[-2.9, 2.9]] * 7 + [[0.0, 0.0]]), T_root=np.eye(4), T_end_base=np.eye(4))


HOME = np.array([0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0])


def test_locked_jacobian_columns_are_zero_and_the_rest_unchanged():
    chain = _chain()
    locked = chain.with_locked([2])
    J, T = chain.jacobian(HOME + 0.1)
    Jl, Tl = locked.jacobian(HOME + 0.1)
    assert np.array_equal(T, Tl)
    assert np.all(Jl[:, 2] == 0) and np.abs(J[:, 2]).max() > 0.1
    assert np.array_equal(np.delete(J, 2, 1), np.delete(Jl, 2, 1))


def test_locked_ik_reaches_six_joint_targets_without_moving_the_locked_joint():
    chain, locked = _chain(), _chain().with_locked([2])
    rng = np.random.default_rng(0)
    q_targets = HOME + rng.uniform(-0.3, 0.3, (50, 7)) * np.array([1, 1, 0, 1, 1, 1, 1])
    T = chain.fk(q_targets)
    q, converged, _, _ = locked.ik(T, HOME)
    assert converged.mean() > 0.9
    assert np.all(q[:, 2] == HOME[2])
    q_free, *_ = chain.ik(T, HOME)
    assert np.abs(q_free[:, 2] - HOME[2]).max() > 1e-3  # unlocked, redundancy moves joint 3 too


def test_locked_dls_step_leaves_the_joint():
    locked = _chain().with_locked([2])
    T = _chain().fk(HOME + 0.2)
    dq, _ = locked.dls_delta(np.repeat(HOME[None], 3, 0), np.repeat(T[None], 3, 0))
    assert np.all(dq[:, 2] == 0) and np.abs(dq).max() > 0
