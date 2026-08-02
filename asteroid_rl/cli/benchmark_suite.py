"""Planning-doc evaluation suite: flat vs mesh, scripted vs PPO, obs modes.

Writes a compact CSV summary under ``outputs/benchmark_suite_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode
from asteroid_rl.policies import make_action_fn


def parse_args() -> Namespace:
    """Parse benchmark suite arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Flat/mesh policy benchmark suite")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
    )
    parser.add_argument(
        "--obs-modes",
        type=str,
        default="truth,sensors,perception",
        help="Comma-separated obs modes to evaluate for PPO/scripted",
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        default="flat,mesh",
        help="Comma-separated: flat and/or mesh",
    )
    return parser.parse_args()


def _run_case(
    *,
    label: str,
    policy: str,
    model: Optional[Any],
    flat: bool,
    obs_mode: str,
    seed: int,
    episodes: int,
) -> List[Dict[str, Any]]:
    """Run one policy/surface/obs_mode case for several episodes.

    Args:
        label: Row label prefix.
        policy: ``scripted`` or ``ppo``.
        model: Loaded PPO or ``None``.
        flat: Flat-surface flag.
        obs_mode: Observation mode.
        seed: Base seed.
        episodes: Episode count.

    Returns:
        List of summary row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        config = LandingEnvConfig(
            seed=seed + ep,
            reuse_sim=True,
            auto_point=True,
            use_flat_surface=flat,
            obs_mode=obs_mode,
        )
        # PPO observation dim must match training; skip mismatched cases at call site.
        env = AsteroidLandingEnv(config=config)
        action_fn = make_action_fn(policy, model=model, seed=seed + ep)
        surface = "flat" if flat else "mesh"
        csv_path = f"logs/bench_{label}_{surface}_{obs_mode}_ep{ep}.csv"
        summary = run_episode(policy, action_fn, env, csv_path)
        env.close()
        rows.append(
            {
                "case": label,
                "policy": policy,
                "surface": surface,
                "obs_mode": obs_mode,
                "episode": ep,
                "termination_reason": summary.get("termination_reason"),
                "final_altitude": summary.get("final_altitude"),
                "final_distance": summary.get("final_distance"),
                "final_speed": summary.get("final_speed"),
                "total_reward": summary.get("total_reward"),
                "success": summary.get("termination_reason") == "safe_landing",
                "csv_path": csv_path,
            }
        )
        print(
            f"{label}/{surface}/{obs_mode}/ep{ep}: "
            f"{summary.get('termination_reason')} "
            f"reward={summary.get('total_reward')}"
        )
    return rows


def main() -> None:
    """Run the flat/mesh × obs-mode benchmark matrix."""
    args = parse_args()
    ensure_dirs()
    obs_modes = [m.strip() for m in args.obs_modes.split(",") if m.strip()]
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    flat_flags = []
    if "flat" in surfaces:
        flat_flags.append(True)
    if "mesh" in surfaces:
        flat_flags.append(False)

    model = None
    if os.path.isfile(args.model):
        model = PPO.load(args.model, device="cpu")
        print(f"Loaded PPO model: {args.model}")
    else:
        print(f"No PPO checkpoint at {args.model}; scripted-only suite")

    rows: List[Dict[str, Any]] = []
    for flat in flat_flags:
        for obs_mode in obs_modes:
            rows.extend(
                _run_case(
                    label="scripted",
                    policy="scripted",
                    model=None,
                    flat=flat,
                    obs_mode=obs_mode,
                    seed=args.seed,
                    episodes=args.episodes,
                )
            )
            if model is not None:
                # Only evaluate PPO when obs_mode matches the trained space.
                trained_dim = int(model.observation_space.shape[0])
                from asteroid_rl.observations import observation_dim

                if observation_dim(obs_mode) != trained_dim:
                    print(
                        f"Skipping PPO obs_mode={obs_mode}: "
                        f"model dim={trained_dim} != {observation_dim(obs_mode)}"
                    )
                    continue
                rows.extend(
                    _run_case(
                        label="ppo",
                        policy="ppo",
                        model=model,
                        flat=flat,
                        obs_mode=obs_mode,
                        seed=args.seed,
                        episodes=args.episodes,
                    )
                )

    out_csv = "outputs/benchmark_suite_summary.csv"
    os.makedirs("outputs", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(1 for r in rows if r.get("success"))
    print(f"Saved {out_csv} ({n_ok}/{len(rows)} safe_landing)")


if __name__ == "__main__":
    main()
