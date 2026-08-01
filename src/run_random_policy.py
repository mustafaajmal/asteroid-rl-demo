import csv
import os

from asteroid_landing_env import AsteroidLandingEnv


def main():
    os.makedirs("logs", exist_ok=True)
    env = AsteroidLandingEnv()
    obs, info = env.reset()
    rows = []

    for _ in range(400):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        row = {
            "time": info["sim_time_sec"],
            "altitude": float(obs[0]),
            "vertical_velocity": float(obs[1]),
            "distance": float(obs[2]),
            "speed": float(obs[3]),
            "throttle": float(action[0]),
            "reward": float(reward),
            "termination_reason": info["termination_reason"],
        }
        rows.append(row)
        print(row)
        if terminated or truncated:
            break

    if rows:
        with open("logs/random_policy_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Saved logs/random_policy_log.csv")


if __name__ == "__main__":
    main()
