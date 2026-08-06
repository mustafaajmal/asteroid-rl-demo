"""Compare random / scripted / PPO on the fixed-site landing environment.

Supports ``--flat-surface`` and ``--obs-mode`` so evaluation matches training.
"""

from __future__ import annotations

import argparse
import csv
import os
from argparse import Namespace
from typing import List

from stable_baselines3 import PPO

from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.environment.episode import ensure_dirs, run_episode, write_summary_markdown
from asteroid_rl.environment.observations import observation_dim
from asteroid_rl.control.policies import make_action_fn


PPO_V2_PATH = "outputs/ppo_asteroid_fixed_site_v2.zip"
PPO_V1_PATH = "outputs/ppo_asteroid_fixed_site.zip"


def parse_args() -> Namespace:
    """Parse evaluation CLI flags.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate landing policies")
    parser.add_argument("--flat-surface", action="store_true")
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=("truth", "sensors", "perception"),
        default="truth",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Evaluate random, scripted, and PPO (if a checkpoint exists)."""
    args = parse_args()
    ensure_dirs()
    config = LandingEnvConfig(
        seed=args.seed,
        randomize_reset=False,
        use_flat_surface=bool(args.flat_surface),
        obs_mode=str(args.obs_mode),
        auto_point=True,
    )
    summaries: List[dict] = []

    for policy in ("random", "scripted"):
        env = AsteroidLandingEnv(config=config)
        summaries.append(
            run_episode(
                policy,
                make_action_fn(policy, seed=args.seed),
                env,
                f"logs/eval_{policy}_episode_0.csv",
            )
        )
        env.close()

    ppo_path = args.model or None
    if not ppo_path:
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
            }
        )
        print("No PPO checkpoint found; skipping PPO evaluation.")
    else:
        print(f"Evaluating PPO checkpoint: {ppo_path}")
        model = PPO.load(ppo_path, device="cpu")
        if int(model.observation_space.shape[0]) != observation_dim(args.obs_mode):
            print(
                f"PPO obs dim {model.observation_space.shape[0]} != "
                f"obs_mode={args.obs_mode} ({observation_dim(args.obs_mode)}); skipping"
            )
            summaries.append(
                {
                    "policy": "ppo",
                    "termination_reason": "skipped_obs_mismatch",
                    "csv_path": None,
                }
            )
        else:
            env = AsteroidLandingEnv(config=config)
            summaries.append(
                run_episode(
                    "ppo",
                    make_action_fn("ppo", model=model),
                    env,
                    "logs/eval_ppo_episode_0.csv",
                )
            )
            env.close()

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
