"""Flat-then-mesh PPO curriculum with optional behavior-cloning warm-start.

Planning-document order: prove landing on a flat plane, then transfer to the
Itokawa heightmap. Scripted demos warm-start the policy so short runs can reach
``safe_landing`` before long home-PC fine-tuning.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from typing import Any, Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs
from asteroid_rl.imitate import warmstart_from_scripted


def default_stages(timesteps_flat: int, timesteps_mesh: int) -> List[Dict[str, Any]]:
    """Build flat → mesh curriculum stage descriptors.

    Args:
        timesteps_flat: PPO steps on the flat plane.
        timesteps_mesh: PPO steps on the Itokawa mesh.

    Returns:
        List of stage dicts with ``name``, ``timesteps``, ``overrides``.
    """
    return [
        {
            "name": "flat_fixed",
            "timesteps": int(timesteps_flat),
            "overrides": {
                "use_flat_surface": True,
                "randomize_reset": False,
            },
        },
        {
            "name": "mesh_fixed",
            "timesteps": int(timesteps_mesh),
            "overrides": {
                "use_flat_surface": False,
                "randomize_reset": False,
            },
        },
    ]


def parse_args() -> Namespace:
    """Parse curriculum training CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Flat→mesh PPO curriculum with BC warm-start"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps-flat", type=int, default=50000)
    parser.add_argument("--timesteps-mesh", type=int, default=50000)
    parser.add_argument(
        "--timesteps-per-stage",
        type=int,
        default=None,
        help="Override both stage lengths (smoke / M2)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=("truth", "sensors", "perception"),
        default="truth",
    )
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=6,
        help="Scripted demo episodes for behavior-cloning warm-start (0=skip)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="",
        help="Optional checkpoint to continue from instead of BC",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/ppo_asteroid_curriculum_final.zip",
    )
    return parser.parse_args()


def make_config(args: Namespace, overrides: Dict[str, Any]) -> LandingEnvConfig:
    """Build env config for one curriculum stage.

    Args:
        args: CLI namespace.
        overrides: Stage-specific field overrides.

    Returns:
        ``LandingEnvConfig``.
    """
    cfg = LandingEnvConfig(
        seed=args.seed,
        reuse_sim=True,
        auto_point=True,
        obs_mode=str(args.obs_mode),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    """Run BC warm-start (optional) then flat→mesh PPO stages."""
    args = parse_args()
    ensure_dirs()
    os.makedirs("outputs/best_model", exist_ok=True)

    flat_steps = (
        args.timesteps_per_stage
        if args.timesteps_per_stage is not None
        else args.timesteps_flat
    )
    mesh_steps = (
        args.timesteps_per_stage
        if args.timesteps_per_stage is not None
        else args.timesteps_mesh
    )
    stages = default_stages(flat_steps, mesh_steps)

    model = None
    first_cfg = make_config(args, stages[0]["overrides"])
    first_env = AsteroidLandingEnv(config=first_cfg)

    if args.start_from and os.path.isfile(args.start_from):
        print(f"Loading checkpoint: {args.start_from}")
        model = PPO.load(args.start_from, env=first_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            first_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            seed=args.seed,
            device=args.device,
        )
        if args.bc_episodes > 0:
            print(f"BC warm-start from scripted ({args.bc_episodes} episodes)...")
            warmstart_from_scripted(
                model, first_cfg, episodes=int(args.bc_episodes), epochs=40
            )

    for stage in stages:
        name = stage["name"]
        timesteps = int(stage["timesteps"])
        cfg = make_config(args, stage["overrides"])
        env = AsteroidLandingEnv(config=cfg)
        eval_env = AsteroidLandingEnv(
            config=make_config(args, {**stage["overrides"], "obs_noise_std": 0.0})
        )
        print(
            f"=== Stage {name}: timesteps={timesteps} "
            f"overrides={stage['overrides']} obs_mode={args.obs_mode} ==="
        )
        model.set_env(env)
        tag = f"{args.obs_mode}_{name}"
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"outputs/best_model_{tag}",
            log_path=f"outputs/eval_logs_{tag}",
            eval_freq=max(2048, timesteps // 4),
            n_eval_episodes=3,
            deterministic=True,
            render=False,
        )
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_cb],
            reset_num_timesteps=False,
        )
        out = f"outputs/ppo_curriculum_{args.obs_mode}_{name}.zip"
        model.save(out)
        print(f"Saved {out}")

    model.save(args.out)
    print(f"Saved {args.out}")
    # Only refresh the default play/evaluate path for truth-mode curricula.
    if str(args.obs_mode) == "truth":
        model.save("outputs/ppo_asteroid_fixed_site_v2.zip")
        print("Saved outputs/ppo_asteroid_fixed_site_v2.zip")
    else:
        print(
            f"Skipped overwriting ppo_asteroid_fixed_site_v2.zip "
            f"(obs_mode={args.obs_mode})"
        )


if __name__ == "__main__":
    main()
