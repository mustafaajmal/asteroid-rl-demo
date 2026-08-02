"""Phase 1C controlled reset-randomization curriculum (no Scenic).

Trains PPO through staged amounts of start-state noise: fixed start, then small
distance, velocity, and lateral randomization. Each stage saves its own
checkpoint under ``outputs/``.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from typing import Any, Dict

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs


STAGES = [
    {
        "name": "fixed",
        "timesteps": 10000,
        "overrides": {"randomize_reset": False},
    },
    {
        "name": "distance_small",
        "timesteps": 10000,
        "overrides": {
            "randomize_reset": True,
            "randomize_initial_distance": True,
            "initial_distance_delta": 5.0,
        },
    },
    {
        "name": "velocity_small",
        "timesteps": 10000,
        "overrides": {
            "randomize_reset": True,
            "randomize_initial_distance": True,
            "initial_distance_delta": 5.0,
            "randomize_initial_vertical_velocity": True,
            "initial_vertical_velocity_delta": 0.1,
        },
    },
    {
        "name": "lateral_small",
        "timesteps": 10000,
        "overrides": {
            "randomize_reset": True,
            "randomize_initial_distance": True,
            "initial_distance_delta": 5.0,
            "randomize_initial_vertical_velocity": True,
            "initial_vertical_velocity_delta": 0.1,
            "randomize_lateral_offset": True,
            "lateral_offset_delta": 2.0,
        },
    },
]


def parse_args() -> Namespace:
    """Parse command-line arguments for curriculum training.

    Returns:
        Parsed argparse namespace with ``device``, ``seed``,
        ``timesteps_per_stage``, and ``start_from`` fields.
    """
    parser = argparse.ArgumentParser(description="Curriculum PPO training")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timesteps-per-stage",
        type=int,
        default=None,
        help="Override timesteps for every stage (useful for quick smoke runs)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
        help="Optional pretrained checkpoint to continue from",
    )
    return parser.parse_args()


def make_config(seed: int, overrides: Dict[str, Any]) -> LandingEnvConfig:
    """Build a ``LandingEnvConfig`` with curriculum stage overrides applied.

    Args:
        seed: Random seed stored on the config.
        overrides: Mapping of ``LandingEnvConfig`` field names to values to set
            after constructing the default config (for example randomization
            flags and deltas).

    Returns:
        Configured ``LandingEnvConfig`` with ``reuse_sim=True``.
    """
    cfg = LandingEnvConfig(seed=seed, reuse_sim=True)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    """Train PPO through each curriculum stage and save checkpoints.

    Optionally loads ``--start-from`` before stage 0. Saves
    ``outputs/ppo_curriculum_<stage>.zip`` after each stage and
    ``outputs/ppo_asteroid_curriculum_final.zip`` at the end.
    """
    args = parse_args()
    ensure_dirs()

    model = None
    if args.start_from and os.path.isfile(args.start_from):
        print(f"Loading pretrained checkpoint: {args.start_from}")
        boot_env = AsteroidLandingEnv(
            config=make_config(args.seed, STAGES[0]["overrides"])
        )
        model = PPO.load(args.start_from, env=boot_env, device=args.device)

    for stage in STAGES:
        name = stage["name"]
        timesteps = (
            args.timesteps_per_stage
            if args.timesteps_per_stage is not None
            else int(stage["timesteps"])
        )
        cfg = make_config(args.seed, stage["overrides"])
        env = AsteroidLandingEnv(config=cfg)
        print(
            f"=== Stage {name}: timesteps={timesteps} "
            f"overrides={stage['overrides']} ==="
        )

        if model is None:
            tb_log = "outputs/tensorboard"
            try:
                import tensorboard  # noqa: F401
            except ImportError:
                tb_log = None
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
        else:
            model.set_env(env)

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        out = f"outputs/ppo_curriculum_{name}.zip"
        model.save(out)
        print(f"Saved {out}")

    final_out = "outputs/ppo_asteroid_curriculum_final.zip"
    model.save(final_out)
    print(f"Saved {final_out}")


if __name__ == "__main__":
    main()
