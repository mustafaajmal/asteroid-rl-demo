"""Integration smokes that need Basilisk (may be slow)."""

from __future__ import annotations

import numpy as np
import pytest

basilisk = pytest.importorskip("Basilisk")


def test_phase1_scripted_can_safe_land():
    from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
    from asteroid_rl.episode import run_episode
    from asteroid_rl.policies import scripted_action

    env = AsteroidLandingEnv(LandingEnvConfig(seed=0, reuse_sim=False))
    summary = run_episode(
        "scripted",
        scripted_action,
        env,
        "logs/test_phase1_scripted.csv",
        print_every=10**9,
        max_steps=800,
    )
    env.close()
    assert summary.get("termination_reason") == "safe_landing"


def test_orbital_reset_and_step_shapes():
    from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
    from asteroid_rl.policies import scripted_orbit_action

    cfg = LandingEnvConfig(seed=1).apply_orbital_defaults()
    env = AsteroidLandingEnv(cfg)
    obs, info = env.reset()
    assert obs.shape == (9,)
    assert env.action_space.shape == (4,)
    assert info.get("altitude", 1e9) < 1e5  # not heightmap sentinel
    act = scripted_orbit_action(obs, info)
    obs2, reward, term, trunc, info2 = env.step(act)
    assert obs2.shape == (9,)
    assert np.isfinite(reward)
    env.close()


def test_autonomous_env_step_exposes_mission_and_tilt():
    from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
    from asteroid_rl.policies import scripted_autonomous_action

    cfg = LandingEnvConfig(seed=3).apply_autonomous_defaults()
    cfg.orbit_start_mode = "approach"
    env = AsteroidLandingEnv(cfg)
    obs, info = env.reset()
    assert obs.shape == (9,)
    assert "mission_mode" in info
    act = scripted_autonomous_action(obs, info)
    obs2, reward, term, trunc, info2 = env.step(act)
    assert "tilt_deg" in info2
    assert "pointing_command" in info2
    assert np.isfinite(reward)
    env.close()


def test_orbital_scripted_reduces_distance_over_window():
    """Velocity-target divert should generally shrink range over many steps."""
    from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
    from asteroid_rl.policies import scripted_orbit_action

    cfg = LandingEnvConfig(seed=2).apply_orbital_defaults()
    cfg.orbit_start_mode = "approach"
    env = AsteroidLandingEnv(cfg)
    obs, info = env.reset()
    d0 = float(info["distance_to_target"])
    distances = [d0]
    for _ in range(120):
        act = scripted_orbit_action(obs, info)
        obs, _r, term, trunc, info = env.step(act)
        distances.append(float(info["distance_to_target"]))
        if term or trunc:
            break
    env.close()
    early = float(np.mean(distances[:20]))
    late = float(np.mean(distances[-20:]))
    assert late < early * 1.15 or min(distances) < d0 * 0.85
