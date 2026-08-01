import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from asteroid_landing_env import AsteroidLandingEnv


def evaluate(model, env, csv_path="logs/ppo_eval_log.csv"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    obs, info = env.reset()
    rows = []

    for _ in range(400):
        action, _ = model.predict(obs, deterministic=True)
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
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved {csv_path}")
    if rows:
        print("Final row:", rows[-1])


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = AsteroidLandingEnv()

    try:
        check_env(env, warn=True)
    except Exception as exc:
        print("check_env failed; continuing with training anyway.")
        print(exc)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
    )

    # Intentionally small proof-of-life training budget.
    model.learn(total_timesteps=1_000)

    model.save("outputs/ppo_asteroid_fixed_site.zip")
    print("Saved outputs/ppo_asteroid_fixed_site.zip")

    eval_env = AsteroidLandingEnv()
    evaluate(model, eval_env)


if __name__ == "__main__":
    main()
