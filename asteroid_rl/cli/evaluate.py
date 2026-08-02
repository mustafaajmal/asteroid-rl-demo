"""Compare random / scripted / PPO on the fixed-site landing environment.

Runs one episode per available policy, writes per-policy CSVs under ``logs/``,
and writes tabular summaries under ``outputs/``.
"""

from __future__ import annotations

import csv
import os
from typing import List

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode, write_summary_markdown
from asteroid_rl.policies import make_action_fn


PPO_V2_PATH = "outputs/ppo_asteroid_fixed_site_v2.zip"
PPO_V1_PATH = "outputs/ppo_asteroid_fixed_site.zip"


def main() -> None:
    """Evaluate random, scripted, and PPO (if a checkpoint exists).

    Preference order for PPO checkpoints is ``PPO_V2_PATH`` then
    ``PPO_V1_PATH``. If neither exists, PPO is recorded as skipped.
    """
    ensure_dirs()
    config = LandingEnvConfig(seed=0, randomize_reset=False)
    summaries: List[dict] = []

    for policy in ("random", "scripted"):
        env = AsteroidLandingEnv(config=config)
        summaries.append(
            run_episode(
                policy,
                make_action_fn(policy, seed=0),
                env,
                f"logs/eval_{policy}_episode_0.csv",
            )
        )

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
        summaries.append(
            run_episode(
                "ppo",
                make_action_fn("ppo", model=model),
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
