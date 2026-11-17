"""Numpy forward/inverse kinematics of the DROID arm (Franka + Robotiq 2F-85) up to the gripper ``base_link``.

The chain (joint local frames, axes, limits) comes from RoboLab's ``tools/simple_pnp_v2/reach_map.py:load_chain``,
exported once by ``scripts/scripted_demos/grasps/export_robot_geometry.py``. All functions are batched over a leading dimension so
the grasp selector can check many candidates at once. Poses are in the robot root frame (= the env-local frame,
the robot spawns at the env origin with identity rotation; checked in Phase 3 against the simulator).
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

HOME = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])  # DroidCfg init_state


def _rotx(q: np.ndarray) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    T = np.zeros(q.shape + (4, 4))
    T[..., 0, 0] = 1.0
    T[..., 1, 1], T[..., 1, 2], T[..., 2, 1], T[..., 2, 2] = c, -s, s, c
    T[..., 3, 3] = 1.0
    return T


def orientation_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) taking ``R`` to ``R_target``, batched (..., 3, 3) → (..., 3)."""
    R_err = R_target @ np.swapaxes(R, -1, -2)
    c = np.clip((np.trace(R_err, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(c)
    v = np.stack([R_err[..., 2, 1] - R_err[..., 1, 2], R_err[..., 0, 2] - R_err[..., 2, 0],
                  R_err[..., 1, 0] - R_err[..., 0, 1]], -1)
    s = np.sin(angle)
    small = s < 1e-6
    scale = np.where(small, 0.5, angle / (2 * np.where(small, 1.0, s)))
    out = v * scale[..., None]
    # Near 180° the antisymmetric part vanishes; use the symmetric part's dominant axis.
    flip = (angle > np.pi - 1e-3)
    if np.any(flip):
        B = (R_err[flip] + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diagonal(B, axis1=-2, axis2=-1), 0, None))
        out[flip] = axis * np.pi
    return out


@dataclasses.dataclass
class Chain:
    T0: np.ndarray  # (K, 4, 4) parent → joint frame
    T1inv: np.ndarray  # (K, 4, 4) joint frame → child
    revolute: np.ndarray  # (K,) bool; revolute joints rotate about their local X
    limits: np.ndarray  # (K, 2)
    T_root: np.ndarray  # (4, 4) panda_link0 in the robot frame
    T_end_base: np.ndarray  # (4, 4) chain end → gripper base_link
    # Arm joints (0-based) every IK step leaves where they are: their Jacobian columns are zero, so ``ik``,
    # ``dls_delta`` and the simulator-side ``sim.ArmIK`` solve over the other joints and hold these at the value they
    # start from (home after a reset). ``(2,)`` holds panda_joint3 as the OMY MimicGen generator does.
    locked: tuple[int, ...] = ()

    @classmethod
    def load(cls, path: pathlib.Path) -> "Chain":
        d = np.load(path)
        return cls(d["T0"], d["T1inv"], d["revolute"].astype(bool), d["limits"], d["T_root"], d["T_end_base"])

    def with_locked(self, joints) -> "Chain":
        return dataclasses.replace(self, locked=tuple(int(j) for j in joints))

    @property
    def joint_limits(self) -> np.ndarray:
        return self.limits[self.revolute]

    def _forward(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float64)
        batch = q.shape[:-1]
        T = np.broadcast_to(self.T_root, batch + (4, 4)).copy()
        axes, origins, k = [], [], 0
        for T0, T1inv, rev in zip(self.T0, self.T1inv, self.revolute):
            T = T @ T0
            if rev:
                axes.append(T[..., :3, 0].copy())
                origins.append(T[..., :3, 3].copy())
                T = T @ _rotx(q[..., k])
                k += 1
            T = T @ T1inv
        return T @ self.T_end_base, np.stack(axes, -2), np.stack(origins, -2)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """``base_link`` pose(s) for joint vector(s) ``q`` (..., 7) → (..., 4, 4)."""
        return self._forward(q)[0]

    def jacobian(self, q: np.ndarray):
        """Geometric Jacobian at ``base_link`` (..., 6, 7) and the pose; the ``locked`` joints' columns are zero."""
        T, axes, origins = self._forward(q)
        p = T[..., None, :3, 3]
        Jv = np.cross(axes, p - origins)
        J = np.concatenate([np.swapaxes(Jv, -1, -2), np.swapaxes(axes, -1, -2)], -2)
        if self.locked:
            J[..., list(self.locked)] = 0.0
        return J, T

    def ik(self, T_target: np.ndarray, q0: np.ndarray, *, iterations: int = 200, damping: float = 0.05,
           max_step: float = 0.2, pos_tol: float = 0.002, rot_tol: float = np.radians(2.0),
           rot_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Damped-least-squares IK from ``q0``, batched.

        Returns ``(q, converged, pos_err_m, rot_err_rad)``. Starting from the current configuration mirrors what the
        simulator-side controller does, so a solution found here is one the executor can track to.
        """
        T_target = np.asarray(T_target, dtype=np.float64)
        q = np.broadcast_to(np.asarray(q0, dtype=np.float64), T_target.shape[:-2] + (7,)).copy()
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
        done = np.zeros(q.shape[:-1], dtype=bool)
        for _ in range(iterations):
            J, T = self.jacobian(q)
            e = np.concatenate([T_target[..., :3, 3] - T[..., :3, 3],
                                orientation_error(T[..., :3, :3], T_target[..., :3, :3])], -1)
            pe, re = np.linalg.norm(e[..., :3], axis=-1), np.linalg.norm(e[..., 3:], axis=-1)
            done = (pe < pos_tol) & (re < rot_tol)
            if done.all():
                break
            Jw, ew = W @ J, e @ W
            A = Jw @ np.swapaxes(Jw, -1, -2) + damping**2 * np.eye(6)
            dq = np.einsum("...ji,...j->...i", Jw, np.linalg.solve(A, ew[..., None])[..., 0])
            dq = np.clip(dq, -max_step, max_step)
            dq[done] = 0.0
            q = np.clip(q + dq, lo, hi)
        T = self.fk(q)
        pe = np.linalg.norm(T_target[..., :3, 3] - T[..., :3, 3], axis=-1)
        re = np.linalg.norm(orientation_error(T[..., :3, :3], T_target[..., :3, :3]), axis=-1)
        return q, (pe < pos_tol) & (re < rot_tol), pe, re

    def dls_delta(self, q: np.ndarray, T_target: np.ndarray, lam: float = 0.01):
        """IsaacLab's DLS step (``DifferentialIKController``, ``ik_method="dls"``, default lambda 0.01), batched:
        ``J^T (J J^T + lam^2 I)^-1 e`` with ``e`` = position error and the axis-angle of ``R_target R^T``.
        Returns ``(dq, T)`` where ``T`` is the pose at ``q``."""
        J, T = self.jacobian(q)
        e = np.concatenate([T_target[..., :3, 3] - T[..., :3, 3],
                            orientation_error(T[..., :3, :3], T_target[..., :3, :3])], -1)
        A = J @ np.swapaxes(J, -1, -2) + lam**2 * np.eye(6)
        return np.einsum("...ji,...j->...i", J, np.linalg.solve(A, e[..., None])[..., 0]), T
