"""Quick environment sanity check: reset plus a few fixed-throttle steps.

Verifies that ``AsteroidLandingEnv`` can reset, step, and return observations
(including surface altitude), rewards, and info flags without training or Vizard.
"""

from __future__ import annotations

import numpy as np

from asteroid_rl.env import AsteroidLandingEnv


def main() -> None:
    """Reset the env and run up to 10 steps at throttle ``0.5``.

    Prints observation, reward, termination flags, reward terms, and info
    flags each step. Stops early if the episode terminates or truncates.
    """
    env = AsteroidLandingEnv()
    obs, info = env.reset()
    print("reset obs:", obs)
    print("reset info keys:", sorted(info.keys()))

    for i in range(10):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print("--- step", i, "---")
        print("obs:", obs)
        print("reward:", reward)
        print("terminated:", terminated, "truncated:", truncated)
        print("termination_reason:", info.get("termination_reason"))
        print(
            "reward terms:",
            {
                "reward_total": info.get("reward_total"),
                "reward_progress": info.get("reward_progress"),
                "reward_speed_penalty": info.get("reward_speed_penalty"),
                "reward_fuel_penalty": info.get("reward_fuel_penalty"),
                "reward_terminal": info.get("reward_terminal"),
            },
        )
        print(
            "flags:",
            {
                "success": info.get("success"),
                "crash": info.get("crash"),
                "escape": info.get("escape"),
                "timeout": info.get("timeout"),
                "thrust_N": info.get("thrust_N"),
            },
        )
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
