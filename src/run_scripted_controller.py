import csv
import os

import numpy as np
from asteroid_landing_env import AsteroidLandingEnv


def choose_throttle(obs):
    altitude, vertical_velocity, distance, speed, prev_throttle = obs

    # Simple braking logic.
    # vertical_velocity is radial relative to target proxy.
    if vertical_velocity < -1.0:
        return 1.0
    elif vertical_velocity < -0.5:
        return 0.75
    elif distance < 10.0 and speed > 0.5:
        return 0.65
    else:
        return 0.25


def main():
    os.makedirs("logs", exist_ok=True)
    env = AsteroidLandingEnv()
    obs, info = env.reset()

    rows = []

    for _ in range(400):
        throttle = choose_throttle(obs)
        obs, reward, terminated, truncated, info = env.step(
            np.array([throttle], dtype=np.float32)
        )

        row = {
            "time": info["sim_time_sec"],
            "altitude": float(obs[0]),
            "vertical_velocity": float(obs[1]),
            "distance": float(obs[2]),
            "speed": float(obs[3]),
            "throttle": float(throttle),
            "reward": float(reward),
            "termination_reason": info["termination_reason"],
        }
        rows.append(row)
        print(row)

        if terminated or truncated:
            break

    if rows:
        with open("logs/scripted_controller_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print("Saved logs/scripted_controller_log.csv")


if __name__ == "__main__":
    main()
