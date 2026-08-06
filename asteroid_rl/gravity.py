"""Gravity SysModels for the asteroid landing environment.

``ConstantGravity`` matches Phase-1 (fixed inertial force). ``CentralGravity``
applies asteroid-centered point-mass attraction so elliptical orbits are valid.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import RigidBodyKinematics as rbk

# MuJoCo asteroid body pose in sat_ast_landing.xml.
DEFAULT_ASTEROID_COM_N = (0.0, 0.0, -150.0)
# Demo µ: circular orbit near ~250 m has period of a few minutes (not real Itokawa).
DEFAULT_MU = 15000.0
# Approx hub mass (density 200 * 2x2x2 m box); used to convert accel -> force.
DEFAULT_SPACECRAFT_MASS_REF = 1600.0


def hover_throttle_central(
    position_N: Sequence[float],
    *,
    mu: float = DEFAULT_MU,
    mass: float = DEFAULT_SPACECRAFT_MASS_REF,
    max_thrust: float = 2500.0,
    com_N: Sequence[float] = DEFAULT_ASTEROID_COM_N,
) -> float:
    """Throttle fraction that cancels central-gravity weight at ``position_N``.

    Under point-mass gravity, ``g = µ / r^2`` varies strongly with altitude, so a
    fixed hover fraction (e.g. pad-level ~0.66) will *climb forever* higher up
    where true hover is ~0.2–0.4. Scripted settle must use this estimate.

    Args:
        position_N: Hub inertial position, meters.
        mu: Gravitational parameter, m^3/s^2.
        mass: Spacecraft mass, kg.
        max_thrust: Thruster force at throttle 1.0, Newtons.
        com_N: Asteroid COM inertial position, meters.

    Returns:
        Hover throttle in ``[0, 1]``.
    """
    pos = np.asarray(position_N, dtype=np.float64).reshape(3)
    com = np.asarray(com_N, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(pos - com))
    if r < 1.0:
        r = 1.0
    weight = float(mass) * float(mu) / (r * r)
    thr = float(max_thrust)
    if thr <= 1e-9:
        return 1.0
    return float(np.clip(weight / thr, 0.0, 1.0))


class ConstantGravity(sysModel.SysModel):
    """Basilisk SysModel that applies a constant inertial force as "gravity".

    Reads the application-site attitude, rotates ``force_N`` into the site
    frame, and publishes a ``ForceAtSiteMsg`` for an ``MJForceActuator``.

    Attributes:
        force_N: Constant force vector in the inertial frame, Newtons.
        frameInMsg: Reader for the site/spacecraft state message.
        forceOutMsg: Output force message consumed by the force actuator.
    """

    def __init__(self, force_N: Sequence[float], *args: Any):
        """Create the constant-gravity model.

        Args:
            force_N: Length-3 inertial force vector in Newtons.
            *args: Forwarded to ``sysModel.SysModel.__init__``.
        """
        super().__init__(*args)
        self.force_N = force_N
        self.frameInMsg = messaging.SCStatesMsgReader()
        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Recompute and publish the site-frame force for this integrator step.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        frame: messaging.SCStatesMsgPayload = self.frameInMsg()
        dcm_BN = rbk.MRP2C(frame.sigma_BN)
        force_B = np.dot(dcm_BN, self.force_N)
        payload = messaging.ForceAtSiteMsgPayload(force_S=force_B)
        self.forceOutMsg.write(payload, CurrentSimNanos, self.moduleID)


class CentralGravity(sysModel.SysModel):
    """Point-mass gravity toward an asteroid center of mass.

    Publishes ``F_B = C_BN @ (-m * µ * r_hat / r^2)`` with ``r`` from the
    asteroid COM to the spacecraft in the inertial frame.

    Attributes:
        mu: Gravitational parameter, m^3/s^2.
        mass: Spacecraft mass used for force scaling, kg.
        com_N: Asteroid COM inertial position, meters.
        frameInMsg: Reader for spacecraft states.
        forceOutMsg: Force message for the MuJoCo force actuator.
    """

    def __init__(
        self,
        mu: float = DEFAULT_MU,
        mass: float = DEFAULT_SPACECRAFT_MASS_REF,
        com_N: Sequence[float] = DEFAULT_ASTEROID_COM_N,
        *args: Any,
    ):
        """Create central-body gravity.

        Args:
            mu: Gravitational parameter µ, m^3/s^2.
            mass: Mass used to convert acceleration to Newtons.
            com_N: Asteroid center of mass in the inertial frame, meters.
            *args: Forwarded to ``sysModel.SysModel.__init__``.
        """
        super().__init__(*args)
        self.mu = float(mu)
        self.mass = float(mass)
        self.com_N = np.asarray(com_N, dtype=np.float64).reshape(3)
        self.frameInMsg = messaging.SCStatesMsgReader()
        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Publish central gravity force for this integrator step.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        frame: messaging.SCStatesMsgPayload = self.frameInMsg()
        r_sc = np.asarray(frame.r_BN_N, dtype=np.float64).reshape(3)
        r_vec = r_sc - self.com_N
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1.0:
            r_norm = 1.0
            r_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Inertial acceleration a = -µ r / r^3 ; force = m a.
        force_N = -self.mass * self.mu * r_vec / (r_norm**3)
        dcm_BN = rbk.MRP2C(frame.sigma_BN)
        force_B = np.dot(dcm_BN, force_N)
        payload = messaging.ForceAtSiteMsgPayload(force_S=force_B.tolist())
        self.forceOutMsg.write(payload, CurrentSimNanos, self.moduleID)
