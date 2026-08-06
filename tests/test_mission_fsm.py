"""Unit tests for multi-phase mission FSM (no Basilisk)."""

from __future__ import annotations

import numpy as np

from asteroid_rl.mission import (
    MissionConfig,
    MissionState,
    mission_pointing_command,
    mission_throttle_gate,
    update_mission,
)


def test_disabled_mission_forces_land():
    state = MissionState(mode="search")
    cfg = MissionConfig(enabled=False)
    out = update_mission(
        state,
        perception={"target_visible": False, "hazard_score": 1.0},
        position_N=np.array([0.0, 0.0, 100.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        altitude_m=100.0,
        config=cfg,
    )
    assert out.mode == "land"


def test_not_visible_enters_acquire():
    state = MissionState(mode="search")
    cfg = MissionConfig(
        enabled=True,
        max_search_steps=999,
        divert_force_altitude_m=50.0,
        divert_force_lateral_m=20.0,
    )
    out = update_mission(
        state,
        perception={"target_visible": False, "hazard_score": 0.5},
        position_N=np.array([0.0, 0.0, 200.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        altitude_m=200.0,
        config=cfg,
        lateral_m=150.0,
    )
    assert out.mode == "acquire"


def test_low_hazard_commits_divert():
    state = MissionState(mode="search")
    cfg = MissionConfig(enabled=True, hazard_commit_threshold=0.10)
    out = update_mission(
        state,
        perception={
            "target_visible": True,
            "hazard_score": 0.05,
            "landing_site_box": [0.4, 0.4, 0.6, 0.6],
        },
        position_N=np.array([0.0, 0.0, 100.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        altitude_m=100.0,
        config=cfg,
        lateral_m=40.0,
    )
    assert out.mode == "divert"
    assert out.committed is True


def test_near_pad_enters_upright_after_commit():
    state = MissionState(mode="divert", committed=True)
    cfg = MissionConfig(enabled=True, upright_altitude_m=70.0, upright_lateral_m=35.0)
    out = update_mission(
        state,
        perception={"target_visible": True, "hazard_score": 0.05},
        position_N=np.array([5.0, 5.0, 20.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        altitude_m=50.0,
        config=cfg,
        lateral_m=10.0,
    )
    assert out.mode == "upright"


def test_throttle_gate_caps_search():
    state = MissionState(mode="search")
    cfg = MissionConfig(enabled=True)
    thr, reason = mission_throttle_gate(0.9, state=state, altitude_m=80.0, config=cfg)
    assert thr <= 0.25
    assert reason == "search"


def test_near_approach_auto_commits_divert():
    state = MissionState(mode="search")
    cfg = MissionConfig(
        enabled=True,
        divert_force_altitude_m=120.0,
        divert_force_lateral_m=90.0,
        hazard_commit_threshold=0.01,
    )
    out = update_mission(
        state,
        perception={"target_visible": True, "hazard_score": 0.9},
        position_N=np.array([10.0, 0.0, 50.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        altitude_m=80.0,
        config=cfg,
        lateral_m=20.0,
    )
    assert out.mode in ("divert", "upright")
    assert out.committed is True
    state = MissionState(mode="acquire")
    cfg = MissionConfig(asteroid_com_N=(0.0, 0.0, -150.0))
    cmd = mission_pointing_command(
        state,
        position_N=np.array([0.0, 0.0, 0.0]),
        target_N=np.array([0.0, 0.0, -30.0]),
        config=cfg,
    )
    assert cmd[2] < 0  # toward COM below
