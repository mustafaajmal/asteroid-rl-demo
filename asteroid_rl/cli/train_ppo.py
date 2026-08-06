"""Train fixed-site PPO on the asteroid landing environment.

Runs Stable-Baselines3 PPO against ``AsteroidLandingEnv`` with a fixed target.
Supports ``--obs-mode truth|sensors|perception`` so training can avoid
privileged site-distance state. Reward still uses simulator truth.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.environment.episode import ensure_dirs


class ThrottleStatCallback(BaseCallback):
    """Log mean / max throttle from completed rollout episodes.

    Attributes:
        verbose: SB3 verbosity level.
    """

    def __init__(self, verbose: int = 0):
        """Create the callback.

        Args:
            verbose: Forwarded to ``BaseCallback``.
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called each env step; always continue training.

        Returns:
            Always ``True``.
        """
        return True

    def _on_rollout_end(self) -> None:
        """Print throttle stats from the last rollout buffer when available."""
        try:
            actions = np.asarray(self.model.rollout_buffer.actions)
            if actions.size == 0:
                return
            flat = actions.reshape(-1)
            print(
                f"rollout throttle: mean={float(flat.mean()):.3f} "
                f"max={float(flat.max()):.3f} min={float(flat.min()):.3f}"
            )
        except Exception:
            pass


def parse_args() -> Namespace:
    """Parse command-line arguments for PPO training.

    Returns:
        Parsed argparse namespace with training and env option fields.
    """
    parser = argparse.ArgumentParser(description="Train fixed-site PPO")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total PPO timesteps (use 1e5+ on a desktop CPU)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--flat-surface",
        action="store_true",
        help="Train against a flat plane instead of the Itokawa heightmap",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.0,
        help="Gaussian noise std on agent observation channels (0 = off)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=("truth", "sensors", "perception"),
        default="truth",
        help="Policy observation: privileged truth, onboard-like sensors, or camera stub",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=4096,
        help="Evaluate deterministic policy every N timesteps",
    )
    return parser.parse_args()


def make_env(args: Namespace, *, for_eval: bool = False) -> AsteroidLandingEnv:
    """Build a training or evaluation environment from CLI args.

    Args:
        args: Parsed CLI namespace.
        for_eval: If True, disable obs noise for cleaner metrics.

    Returns:
        Configured ``AsteroidLandingEnv``.
    """
    config = LandingEnvConfig(
        seed=args.seed if not for_eval else args.seed + 10_000,
        randomize_reset=False,
        reuse_sim=True,
        use_flat_surface=bool(args.flat_surface),
        obs_noise_std=0.0 if for_eval else float(args.obs_noise),
        obs_mode=str(args.obs_mode),
        auto_point=True,
    )
    return AsteroidLandingEnv(config=config)


def main() -> None:
    """Construct the env, train or resume PPO, and save the final checkpoint.

    Creates ``outputs/checkpoints`` for intermediate saves and
    ``outputs/best_model`` via ``EvalCallback``. If ``--resume`` points to an
    existing zip, training continues from that model.
    """
    args = parse_args()
    ensure_dirs()
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/best_model", exist_ok=True)

    env = make_env(args, for_eval=False)
    eval_env = make_env(args, for_eval=True)

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
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            # Small entropy bonus so the policy does not collapse to always-zero throttle.
            ent_coef=0.01,
            vf_coef=0.5,
            seed=args.seed,
            device=args.device,
            tensorboard_log=tb_log,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=4096,
        save_path="outputs/checkpoints",
        name_prefix="ppo_fixed_site_v2",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="outputs/best_model",
        log_path="outputs/eval_logs",
        eval_freq=max(int(args.eval_freq), 1000),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    throttle_cb = ThrottleStatCallback()

    print(
        f"Training PPO for {args.timesteps} timesteps on device={args.device} "
        f"(obs_mode={args.obs_mode}, flat_surface={args.flat_surface}, "
        f"obs_noise={args.obs_noise})"
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, throttle_cb],
        reset_num_timesteps=args.resume is None,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}")
    print("Best eval checkpoint (if improved): outputs/best_model/best_model.zip")


if __name__ == "__main__":
    main()
