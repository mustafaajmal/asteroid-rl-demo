"""Train PPO for elliptical-orbit point+throttle GNC landing.

Uses central gravity, elliptical resets, BC warm-start from ``scripted_orbit``,
then PPO fine-tuning. Prefer the EvalCallback best zip over the final zip.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs
from asteroid_rl.imitate import warmstart_from_scripted


def parse_args() -> Namespace:
    """Parse orbital PPO training arguments."""
    parser = argparse.ArgumentParser(description="Train orbital point+throttle PPO")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--bc-episodes", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/ppo_orbital_final.zip",
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default="outputs/best_model_orbital",
    )
    return parser.parse_args()


def make_orbital_config(seed: int) -> LandingEnvConfig:
    """Build the standard orbital training env config."""
    cfg = LandingEnvConfig(seed=seed)
    cfg.apply_orbital_defaults()
    return cfg


def main() -> None:
    """BC warm-start then PPO train on elliptical-orbit landings."""
    args = parse_args()
    ensure_dirs()
    os.makedirs(args.best_dir, exist_ok=True)
    os.makedirs("outputs/checkpoints_orbital", exist_ok=True)
    os.makedirs("outputs/eval_logs_orbital", exist_ok=True)

    config = make_orbital_config(args.seed)
    env = AsteroidLandingEnv(config=config)
    eval_env = AsteroidLandingEnv(config=make_orbital_config(args.seed + 1))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
    )

    if args.bc_episodes > 0:
        print(f"BC warm-start from scripted_orbit ({args.bc_episodes} episodes)...")
        warmstart_from_scripted(
            model, config, episodes=args.bc_episodes, epochs=25, orbit=True
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.best_dir,
        log_path="outputs/eval_logs_orbital",
        eval_freq=2048,
        n_eval_episodes=2,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=4096,
        save_path="outputs/checkpoints_orbital",
        name_prefix="ppo_orbital",
    )

    print(f"Training orbital PPO for {args.timesteps} timesteps on {args.device}...")
    model.learn(total_timesteps=int(args.timesteps), callback=[eval_cb, ckpt_cb])
    model.save(args.out)
    print(f"Saved final model to {args.out}")
    print(f"Best eval model (if any) under {args.best_dir}/best_model.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
