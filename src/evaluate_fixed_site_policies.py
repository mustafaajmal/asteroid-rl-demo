import csv
import os
from typing import List

import numpy as np
from stable_baselines3 import PPO

from asteroid_landing_env import AsteroidLandingEnv, LandingEnvConfig
from policy_utils import (
    ensure_dirs,
    random_action,
    run_episode,
    scripted_action,
    write_summary_markdown,
)


PPO_V2_PATH = "outputs/ppo_asteroid_fixed_site_v2.zip"
PPO_V1_PATH = "outputs/ppo_asteroid_fixed_site.zip"


def main():
    ensure_dirs()
    config = LandingEnvConfig(seed=0, randomize_reset=False)
    summaries: List[dict] = []

    # Random
    env = AsteroidLandingEnv(config=config)
    rng = np.random.default_rng(0)
    summaries.append(
        run_episode(
            "random",
            lambda obs: random_action(obs, rng=rng),
            env,
            "logs/eval_random_episode_0.csv",
        )
    )

    # Scripted
    env = AsteroidLandingEnv(config=config)
    summaries.append(
        run_episode(
            "scripted",
            scripted_action,
            env,
            "logs/eval_scripted_episode_0.csv",
        )
    )

    # PPO if available
    ppo_path = None
    if os.path.isfile(PPO_V2_PATH):
        ppo_path = PPO_V2_PATH
    elif os.path.isfile(PPO_V1_PATH):
        ppo_path = PPO_V1_PATH

    if ppo_path is None:
        summaries.append(
            {
                "policy": "ppo",
                "steps": None,
                "final_time": None,
                "final_distance": None,
                "final_speed": None,
                "termination_reason": "skipped_no_checkpoint",
                "total_reward": None,
                "csv_path": None,
                "min_distance": None,
                "max_speed": None,
                "avg_throttle": None,
                "max_throttle": None,
                "initial_distance": None,
            }
        )
        print("No PPO checkpoint found; skipping PPO evaluation.")
    else:
        print(f"Evaluating PPO checkpoint: {ppo_path}")
        model = PPO.load(ppo_path, device="cpu")
        env = AsteroidLandingEnv(config=config)

        def ppo_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        summaries.append(
            run_episode(
                "ppo",
                ppo_action,
                env,
                "logs/eval_ppo_episode_0.csv",
            )
        )

    summary_csv = "outputs/fixed_site_eval_summary.csv"
    fieldnames = list(summaries[0].keys())
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {summary_csv}")

    summary_md = "outputs/fixed_site_eval_summary.md"
    write_summary_markdown(summaries, summary_md)
    print(f"Saved {summary_md}")

    for s in summaries:
        print(
            s.get("policy"),
            "distance=",
            s.get("final_distance"),
            "speed=",
            s.get("final_speed"),
            "reason=",
            s.get("termination_reason"),
            "reward=",
            s.get("total_reward"),
        )


if __name__ == "__main__":
    main()
