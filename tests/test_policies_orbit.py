"""Unit tests for scripted orbital GNC (no Basilisk)."""

from __future__ import annotations

import numpy as np

from asteroid_rl.policies import scripted_action, scripted_orbit_action


def test_scripted_action_shape_and_bounds():
    obs = np.array([20.0, -1.0, 25.0, 1.5, 0.0], dtype=np.float32)
    info = {
        "altitude": 20.0,
        "vertical_velocity": -1.0,
        "speed": 1.5,
        "perception": {
            "target_visible": True,
            "landing_site_box": [0.4, 0.4, 0.6, 0.6],
            "hazard_score": 0.05,
            "progress_assessment": "ok",
        },
    }
    act = scripted_action(obs, info)
    assert act.shape == (1,)
    assert 0.0 <= float(act[0]) <= 1.0


def test_scripted_orbit_deorbit_opposes_velocity():
    # Fast +x motion → energy blend still opposes velocity → point +x.
    obs = np.array(
        [
            100.0,
            0.0,
            50.0,  # rel
            5.0,
            0.0,
            0.0,  # vel
            80.0,  # alt
            5.0,  # speed
            0.0,
        ],
        dtype=np.float32,
    )
    act = scripted_orbit_action(obs, {})
    assert act.shape == (4,)
    assert 0.0 <= float(act[0]) <= 1.0
    point = act[1:4]
    assert float(point[0]) > 0.3
    assert float(act[0]) >= 0.5


def test_scripted_orbit_divert_toward_site_when_slow():
    # Nearly still far +x → velocity target toward site → accel −x → point +x.
    obs = np.array(
        [100.0, 0.0, 50.0, 0.2, 0.0, 0.0, 80.0, 0.2, 0.0],
        dtype=np.float32,
    )
    act = scripted_orbit_action(obs, {})
    point = act[1:4]
    assert float(point[0]) > 0.4


def test_scripted_orbit_terminal_corridor_uses_landing_throttle():
    # Nearly above pad, descending moderately.
    obs = np.array(
        [2.0, 1.0, 15.0, 0.0, 0.0, -1.0, 15.0, 1.0, 0.5],
        dtype=np.float32,
    )
    act = scripted_orbit_action(obs, {})
    assert act.shape == (4,)
    assert 0.0 <= float(act[0]) <= 1.0


def test_scripted_autonomous_acquire_low_throttle():
    from asteroid_rl.policies import scripted_autonomous_action

    obs = np.array(
        [100.0, 0.0, 50.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0], dtype=np.float32
    )
    act = scripted_autonomous_action(
        obs,
        {
            "mission_mode": "acquire",
            "pointing_command": np.array([0.0, 0.0, -1.0]),
        },
    )
    assert act.shape == (4,)
    assert float(act[0]) <= 0.25
