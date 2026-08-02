"""Train fixed-site PPO on the asteroid landing environment.

Runs Stable-Baselines3 PPO against ``AsteroidLandingEnv`` with a fixed target
and truth-state observations. Saves periodic checkpoints and a final zip under
``outputs/``.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs


def parse_args() -> Namespace:
    """Parse command-line arguments for PPO training.

    Returns:
        Parsed argparse namespace with ``timesteps``, ``device``, ``seed``,
        ``out``, and ``resume`` fields.
    """
    parser = argparse.ArgumentParser(description="Train fixed-site PPO")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Construct the env, train or resume PPO, and save the final checkpoint.

    Creates ``outputs/checkpoints`` for intermediate saves. If ``--resume``
    points to an existing zip, training continues from that model.
    """
    args = parse_args()
    ensure_dirs()
    os.makedirs("outputs/checkpoints", exist_ok=True)

    config = LandingEnvConfig(seed=args.seed, randomize_reset=False, reuse_sim=True)
    env = AsteroidLandingEnv(config=config)

    tb_log = None
    try:
        import tensorboard  # noqa: F401

        tb_log = "outputs/tensorboard"
    except ImportError:
        print("tensorboard not installed; continuing without TB logging")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            seed=args.seed,
            device=args.device,
            tensorboard_log=tb_log,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=2048,
        save_path="outputs/checkpoints",
        name_prefix="ppo_fixed_site_v2",
    )

    print(f"Training PPO for {args.timesteps} timesteps on device={args.device}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=args.resume is None,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
