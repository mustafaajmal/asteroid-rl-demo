"""Quick environment sanity check: reset plus a few fixed-throttle steps.

Verifies that ``AsteroidLandingEnv`` can reset, step, and return observations
(including surface altitude), rewards, and info flags without training or Vizard.

Also supports ``--orbital`` for central-gravity + point/throttle smoke.
"""

from __future__ import annotations

import argparse

import numpy as np

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.policies import scripted_orbit_action


def _run_mode(mode: str, steps: int = 3) -> None:
    """Smoke a few steps under one observation mode.

    Args:
        mode: ``truth``, ``sensors``, or ``perception``.
        steps: Number of fixed-throttle steps to run.
    """
    env = AsteroidLandingEnv(config=LandingEnvConfig(obs_mode=mode, reuse_sim=True))
    obs, info = env.reset()
    print(f"=== obs_mode={mode} dim={obs.shape} ===")
    print("reset obs:", obs)
    assert info.get("truth_state") is not None
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.5], dtype=np.float32)
        )
        print(
            f"step {i}: obs={obs} reward={reward:.3f} "
            f"reason={info.get('termination_reason')}"
        )
        if terminated or truncated:
            break
    env.close()


def _run_orbital(steps: int = 8) -> None:
    """Smoke elliptical reset + scripted_orbit actions."""
    cfg = LandingEnvConfig(seed=0).apply_orbital_defaults()
    env = AsteroidLandingEnv(config=cfg)
    obs, info = env.reset()
    print(f"=== orbital dim={obs.shape} action={env.action_space} ===")
    assert obs.shape == (9,)
    assert float(info["altitude"]) < 1.0e5
    d0 = float(info["distance_to_target"])
    for i in range(steps):
        action = scripted_orbit_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"orb {i}: alt={info['altitude']:.1f} dist={info['distance_to_target']:.1f} "
            f"thr={info['throttle']:.2f} reward={reward:.2f}"
        )
        if terminated or truncated:
            break
    print(f"distance start={d0:.1f} end={float(info['distance_to_target']):.1f}")
    env.close()


def main() -> None:
    """Reset the env and run short smokes for Phase-1 and optional orbital."""
    parser = argparse.ArgumentParser(description="Env smoke tests")
    parser.add_argument(
        "--orbital",
        action="store_true",
        help="Also run elliptical / point-throttle smoke",
    )
    parser.add_argument(
        "--orbital-only",
        action="store_true",
        help="Skip Phase-1 obs-mode smokes; only orbital",
    )
    args = parser.parse_args()

    if not args.orbital_only:
        for mode in ("truth", "sensors", "perception"):
            _run_mode(mode)
    if args.orbital or args.orbital_only:
        _run_orbital()


if __name__ == "__main__":
    main()
