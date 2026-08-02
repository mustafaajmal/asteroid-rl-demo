"""Quick environment sanity check: reset plus a few fixed-throttle steps.

Verifies that ``AsteroidLandingEnv`` can reset, step, and return observations
(including surface altitude), rewards, and info flags without training or Vizard.
"""

from __future__ import annotations

import numpy as np

from asteroid_rl.env import AsteroidLandingEnv


def _run_mode(mode: str, steps: int = 3) -> None:
    """Smoke a few steps under one observation mode.

    Args:
        mode: ``truth``, ``sensors``, or ``perception``.
        steps: Number of fixed-throttle steps to run.
    """
    from asteroid_rl.env import LandingEnvConfig

    env = AsteroidLandingEnv(config=LandingEnvConfig(obs_mode=mode, reuse_sim=True))
    obs, info = env.reset()
    print(f"=== obs_mode={mode} dim={obs.shape} ===")
    print("reset obs:", obs)
    assert info.get("truth_state") is not None
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.5], dtype=np.float32)
        )
        print(f"step {i}: obs={obs} reward={reward:.3f} reason={info.get('termination_reason')}")
        if terminated or truncated:
            break
    env.close()


def main() -> None:
    """Reset the env and run a few fixed-throttle steps for each obs mode.

    Prints observation, reward, and termination reason. Also asserts that
    privileged ``truth_state`` remains available in ``info`` for reward logging.
    """
    for mode in ("truth", "sensors", "perception"):
        _run_mode(mode)


if __name__ == "__main__":
    main()
