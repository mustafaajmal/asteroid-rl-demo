"""Regression: Phase-1 defaults must stay constant-gravity + throttle."""

from __future__ import annotations

from asteroid_rl.env import LandingEnvConfig


def test_phase1_defaults_unchanged():
    cfg = LandingEnvConfig()
    assert cfg.gravity_mode == "constant"
    assert cfg.action_mode == "throttle"
    assert cfg.orbital_reset is False
    assert cfg.obs_mode == "truth"
    assert cfg.use_flat_surface is False


def test_orbital_defaults_do_not_mutate_fresh_phase1():
    orbital = LandingEnvConfig().apply_orbital_defaults()
    assert orbital.gravity_mode == "central"
    assert orbital.action_mode == "point_throttle"
    assert orbital.orbital_reset is True
    assert orbital.use_flat_surface is True

    phase1 = LandingEnvConfig()
    assert phase1.gravity_mode == "constant"
    assert phase1.action_mode == "throttle"
    assert phase1.use_flat_surface is False
