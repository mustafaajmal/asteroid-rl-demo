"""Run one landing episode under a chosen policy.

Entrypoint for headless or Vizard playback of scripted, random, or PPO control.
Does not train a model; for PPO it only loads an existing checkpoint.

Examples:
    python -m asteroid_rl.cli.play --policy scripted
    python -m asteroid_rl.cli.play --policy random
    python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip
    python -m asteroid_rl.cli.play --policy scripted --viz
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode
from asteroid_rl.policies import make_action_fn


def parse_args() -> Namespace:
    """Parse command-line arguments for episode playback.

    Returns:
        Parsed argparse namespace with ``policy``, ``model``, ``seed``, ``viz``,
        ``realtime_scale``, and ``csv`` fields.
    """
    parser = argparse.ArgumentParser(
        description="Play one asteroid-landing episode (optionally in Vizard)"
    )
    parser.add_argument(
        "--policy",
        choices=("scripted", "random", "ppo"),
        default="scripted",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
        help="PPO checkpoint (for --policy ppo)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Live-stream the episode to Vizard",
    )
    parser.add_argument(
        "--realtime-scale",
        type=float,
        default=0.0,
        help="Extra sleep vs control_dt (0 = no extra sleep / Vizard paces liveStream)",
    )
    parser.add_argument("--csv", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Load a policy, run one episode, and write a CSV log.

    When ``--viz`` is set, enables Basilisk liveStream and launches Vizard.
    When ``--policy ppo`` is set, loads ``--model`` from disk first.
    """
    args = parse_args()
    ensure_dirs()

    model = None
    if args.policy == "ppo":
        if not os.path.isfile(args.model):
            raise FileNotFoundError(
                f"PPO checkpoint not found: {args.model}\n"
                "Train first, or use --policy scripted."
            )
        model = PPO.load(args.model, device="cpu")

    action_fn = make_action_fn(args.policy, model=model, seed=args.seed)
    config = LandingEnvConfig(
        seed=args.seed,
        randomize_reset=False,
        reuse_sim=not args.viz,
        enable_viz=args.viz,
    )
    env = AsteroidLandingEnv(config=config)

    prefix = "viz" if args.viz else "play"
    csv_path = args.csv or f"logs/{prefix}_{args.policy}_episode.csv"

    if args.viz:
        print(f"Starting '{args.policy}' episode with Vizard liveStream...")
        print("Vizard should open and connect to tcp://localhost:5556")
        print("If it waits forever: Direct Communication + Live Streaming in Vizard.")

    summary = run_episode(
        args.policy,
        action_fn,
        env,
        csv_path,
        print_every=10 if args.viz else 1,
        step_sleep_sec=config.control_dt * args.realtime_scale,
    )
    print(
        f"Episode ended: {summary.get('termination_reason')} "
        f"(dist={summary.get('final_distance')}, speed={summary.get('final_speed')})"
    )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
