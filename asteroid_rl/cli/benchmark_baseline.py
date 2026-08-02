"""Fixed-site baseline benchmark: scripted (+ optional light randomization).

Runs one or more episodes, prints landing stats and perception-stub summaries.
This is the planning-document “RL / controller alone with a known site” check
before Scenic or VLM.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode
from asteroid_rl.policies import make_action_fn


def parse_args() -> Namespace:
    """Parse benchmark CLI arguments.

    Returns:
        Namespace with ``episodes``, ``seed``, and ``randomize`` fields.
    """
    parser = argparse.ArgumentParser(description="Fixed-site baseline benchmark")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable light start-state randomization (not Scenic)",
    )
    parser.add_argument(
        "--flat-surface",
        action="store_true",
        help="Benchmark on a flat plane (PDF first-benchmark style)",
    )
    return parser.parse_args()


def main() -> None:
    """Run scripted baseline episodes and print compact results."""
    args = parse_args()
    ensure_dirs()

    print(
        f"Baseline benchmark: episodes={args.episodes} "
        f"light_randomize={args.randomize} flat_surface={args.flat_surface}"
    )
    for ep in range(args.episodes):
        config = LandingEnvConfig(
            seed=args.seed + ep,
            light_randomize=args.randomize,
            auto_point=True,
            reuse_sim=True,
            use_flat_surface=bool(args.flat_surface),
        )
        env = AsteroidLandingEnv(config=config)
        summary = run_episode(
            "scripted",
            make_action_fn("scripted", seed=args.seed + ep),
            env,
            f"logs/benchmark_scripted_ep{ep}.csv",
            print_every=None,
        )
        print(
            f"ep{ep}: reason={summary.get('termination_reason')} "
            f"alt={summary.get('final_altitude')} "
            f"spd={summary.get('final_speed')} "
            f"reward={summary.get('total_reward')}"
        )
        env.close()


if __name__ == "__main__":
    main()
