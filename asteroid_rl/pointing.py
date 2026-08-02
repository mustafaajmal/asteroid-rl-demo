"""Attitude helpers to point a body axis at a world target.

Used as a scripted outer loop (not part of the PPO action space) so the
instrument camera can face the asteroid before / during descent.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from Basilisk.utilities import RigidBodyKinematics as rbk

# Instrument / thrust geometry: camera and landing approach use -body z.
DEFAULT_BORESIGHT_B = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, or a default -z axis if ``vector`` is near zero.

    Args:
        vector: Arbitrary 3-vector.

    Returns:
        Shape ``(3,)`` unit vector.
    """
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return DEFAULT_BORESIGHT_B.copy()
    return v / n


def line_of_sight_N(position_N: Sequence[float], target_N: Sequence[float]) -> np.ndarray:
    """Inertial unit vector from spacecraft position toward a target.

    Args:
        position_N: Hub position in the inertial frame, meters.
        target_N: Target position in the inertial frame, meters.

    Returns:
        Shape ``(3,)`` unit line-of-sight vector.
    """
    return unit(
        np.asarray(target_N, dtype=np.float64).reshape(3)
        - np.asarray(position_N, dtype=np.float64).reshape(3)
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of ``v``.

    Args:
        v: Length-3 vector.

    Returns:
        Shape ``(3, 3)`` skew matrix.
    """
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def dcm_align_a_to_b(a_B: np.ndarray, b_N: np.ndarray) -> np.ndarray:
    """Build ``C_BN`` such that ``C_BN @ a_B`` aligns with ``b_N``.

    Args:
        a_B: Body-frame axis to align (e.g. camera boresight).
        b_N: Desired inertial direction for that axis.

    Returns:
        Shape ``(3, 3)`` DCM ``C_BN`` (inertial vector → body: ``v_B = C_BN v_N``
        is NOT this mapping; Basilisk ``MRP2C(sigma_BN)`` gives ``v_B = C_BN v_N``.
        Here we return the DCM ``R`` satisfying ``R @ a_B = b_N``, then convert
        with ``sigma = C2MRP(R.T)`` so ``MRP2C(sigma) @ b_N ≈ a_B``).
    """
    a = unit(a_B)
    b = unit(b_N)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        # 180 deg: pick any orthogonal axis.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, a))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = unit(np.cross(a, helper))
        # Rodrigues for 180 about axis: R = 2*axis*axis^T - I
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    k = _skew(v)
    return np.eye(3, dtype=np.float64) + k + k @ k * ((1.0 - c) / (s * s))


def mrp_point_boresight_at(
    position_N: Sequence[float],
    target_N: Sequence[float],
    boresight_B: Sequence[float] = DEFAULT_BORESIGHT_B,
) -> Tuple[float, float, float]:
    """Compute body MRP ``sigma_BN`` that aims ``boresight_B`` at ``target_N``.

    Args:
        position_N: Hub inertial position, meters.
        target_N: Target inertial position, meters.
        boresight_B: Body-frame camera / approach axis to align with LOS.

    Returns:
        Length-3 MRP tuple ``sigma_BN`` suitable for ``hub.setAttitude``.
    """
    los_N = line_of_sight_N(position_N, target_N)
    # R maps body axis → inertial LOS: R @ boresight_B = los_N
    r_body_to_inertial = dcm_align_a_to_b(
        np.asarray(boresight_B, dtype=np.float64), los_N
    )
    # Basilisk: v_B = C_BN v_N  ⇒  C_BN = R.T
    c_bn = r_body_to_inertial.T
    sigma = np.asarray(rbk.C2MRP(c_bn), dtype=np.float64).reshape(3)
    return float(sigma[0]), float(sigma[1]), float(sigma[2])


def apply_pointing(hub, position_N: Sequence[float], target_N: Sequence[float]) -> None:
    """Set hub attitude to point the default boresight at ``target_N``.

    Args:
        hub: MuJoCo body handle exposing ``setAttitude``.
        position_N: Current hub inertial position, meters.
        target_N: Target inertial position, meters.
    """
    if not hasattr(hub, "setAttitude"):
        return
    sigma = mrp_point_boresight_at(position_N, target_N)
    hub.setAttitude(list(sigma))
    if hasattr(hub, "setAttitudeRate"):
        hub.setAttitudeRate([0.0, 0.0, 0.0])
