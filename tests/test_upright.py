"""Upright / tilt helpers and Phase-1 gate default."""

from __future__ import annotations

import numpy as np

from asteroid_rl.environment.gym_env import LandingEnvConfig
from asteroid_rl.dynamics.gravity import hover_throttle_central
from asteroid_rl.dynamics.pointing import (
    boresight_tilt_deg,
    local_up_N,
    mrp_point_boresight_along,
)


def test_phase1_require_upright_default_off():
    cfg = LandingEnvConfig()
    assert cfg.require_upright is False


def test_autonomous_defaults_enable_upright_and_mission():
    cfg = LandingEnvConfig().apply_autonomous_defaults()
    assert cfg.require_upright is True
    assert cfg.enable_mission_search is True
    assert cfg.orbit_start_mode == "approach"
    assert cfg.action_mode == "point_throttle"
    # Fresh Phase-1 untouched.
    p1 = LandingEnvConfig()
    assert p1.require_upright is False
    assert p1.gravity_mode == "constant"


def test_local_up_points_away_from_com():
    up = local_up_N([0.0, 0.0, 0.0], [0.0, 0.0, -150.0])
    assert up[2] > 0.9


def test_thruster_tilt_zero_when_firing_along_up():
    """Upright brake: boresight looks at ground (−up) ⇒ thrust along +up."""
    from asteroid_rl.dynamics.pointing import thruster_up_tilt_deg

    up = np.array([0.0, 0.0, 1.0])
    # Point boresight along −up (look down).
    sigma = mrp_point_boresight_along(-up)
    tilt = thruster_up_tilt_deg(sigma, up)
    assert tilt < 2.0


def test_tilt_zero_when_boresight_aligned_with_up():
    up = np.array([0.0, 0.0, 1.0])
    sigma = mrp_point_boresight_along(up)
    tilt = boresight_tilt_deg(sigma, up)
    assert tilt < 1.0


def test_hover_throttle_drops_with_altitude():
    """Central-g hover at high altitude must be well below pad-level hover."""
    com = (0.0, 0.0, -150.0)
    pad = hover_throttle_central(
        (0.0, 0.0, -25.0), max_thrust=2500.0, com_N=com
    )
    high = hover_throttle_central(
        (0.0, 0.0, 55.0), max_thrust=2500.0, com_N=com
    )
    assert pad > 0.55
    assert high < 0.30
    assert high < pad - 0.25
