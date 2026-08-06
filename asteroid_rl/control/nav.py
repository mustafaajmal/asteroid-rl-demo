"""Navigation / sensor interface notes for upright landing (Basilisk).

This demo already has **privileged** hub state from the MuJoCo/Basilisk
recorder (``r_BN_N``, ``v_BN_N``, ``sigma_BN``). That is enough to compute
altitude, speed, and thruster-vs-local-up tilt. Adding Basilisk sensor
*modules* does not unlock upright landing by itself — it adds noise models
for realism. The limiting factor is **actuation + GNC**:

- One body-+z thruster: divert and vertical brake share one force direction.
- Instant MRP pointing (demo) ≈ perfect reaction-wheel slew: attitude can
  snap, but thrust while pointed wrong still accelerates the wrong way.
- Central gravity: hover throttle **falls with altitude** (``g = µ/r²``). A
  fixed pad-level hover (~0.66 at 2.5 kN) will hang or climb at 50–100 m
  where true hover is ~0.2–0.4 — use ``hover_throttle_central`` / env
  ``info["hover_throttle"]``.

Basilisk pieces that map to a flight-like stack (for later wiring):

Sensors (https://avslab.github.io/basilisk/Documentation/simulation/sensors/):

- ``imuSensor`` — body rates + non-grav accel (gyro/accel).
- ``starTracker`` — attitude quaternion / MRP with noise.
- ``camera`` — already used for Vizard / VLM path.
- Optional: lidar / altimeter style range (or heightmap truth as today).

Actuators (dynamics):

- ``reactionWheels`` — attitude control without thruster (true upright hold).
- ``Thrusters`` / force actuator — translational Δv (already present).

``estimate_nav_from_truth`` below is the privileged stand-in for an IMU +
star-tracker fusion filter until those modules are attached to ``build_sim``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from asteroid_rl.dynamics.pointing import local_up_N, thruster_up_tilt_deg, unit


def estimate_nav_from_truth(
    *,
    position_N: Sequence[float],
    velocity_N: Sequence[float],
    sigma_BN: Sequence[float],
    com_N: Sequence[float],
    altitude_m: float,
) -> Dict[str, Any]:
    """Build a nav dict from privileged truth (IMU/ST stand-in).

    Args:
        position_N: Hub inertial position, m.
        velocity_N: Hub inertial velocity, m/s.
        sigma_BN: Attitude MRP.
        com_N: Asteroid COM, m.
        altitude_m: Surface / pad altitude, m.

    Returns:
        Dict with ``local_up``, ``tilt_deg``, ``v_horiz``, ``speed``, etc.
    """
    pos = np.asarray(position_N, dtype=np.float64).reshape(3)
    vel = np.asarray(velocity_N, dtype=np.float64).reshape(3)
    up = local_up_N(pos, com_N)
    v_horiz = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
    return {
        "local_up_N": up.astype(np.float32),
        "tilt_deg": float(thruster_up_tilt_deg(sigma_BN, up)),
        "v_horiz": float(np.linalg.norm(v_horiz)),
        "v_up": float(np.dot(vel, up)),
        "speed": float(np.linalg.norm(vel)),
        "altitude_m": float(altitude_m),
        "boresight_ground_N": unit(-up).astype(np.float32),
    }
