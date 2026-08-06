"""Evaluate autonomous / orbital policies and report safe_landing + upright rates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode
from asteroid_rl.policies import make_action_fn


def parse_args() -> argparse.Namespace:
    """Parse CLI args for autonomous evaluation."""
    p = argparse.ArgumentParser(description="Eval autonomous landing policies")
    p.add_argument(
        "--model",
        type=str,
        default="outputs/best_model_autonomous/best_model.zip",
        help="PPO zip (prefer EvalCallback best over final).",
    )
    p.add_argument(
        "--policy",
        type=str,
        default="scripted_autonomous",
        choices=["ppo", "scripted_autonomous", "scripted_orbit"],
    )
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument(
        "--start-mode",
        type=str,
        default="",
        help="Override orbit_start_mode. Empty=autonomous default.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="outputs/eval_autonomous_summary.json",
    )
    p.add_argument(
        "--no-upright-gate",
        action="store_true",
        help="Disable require_upright for ablation.",
    )
    return p.parse_args()


def main() -> None:
    """Run N autonomous episodes and print / save landing-rate summary."""
    args = parse_args()
    ensure_dirs()
    cfg = LandingEnvConfig(seed=args.seed).apply_autonomous_defaults()
    if args.start_mode:
        cfg.orbit_start_mode = args.start_mode
    if args.no_upright_gate:
        cfg.require_upright = False

    model = None
    if args.policy == "ppo":
        path = Path(args.model)
        if not path.exists():
            raise SystemExit(f"Model not found: {path}")
        model = PPO.load(str(path))
    action_fn = make_action_fn(args.policy, model=model, seed=args.seed)

    reasons: Counter = Counter()
    modes_final: Counter = Counter()
    min_dists = []
    tilts = []
    rows = []
    for i in range(int(args.episodes)):
        cfg.seed = int(args.seed) + i
        env = AsteroidLandingEnv(cfg)
        summary = run_episode(
            f"{args.policy}_{i}",
            action_fn,
            env,
            None,
            print_every=10**9,
            max_steps=int(args.max_steps),
        )
        # Last-step extras from a quick reset/step are unavailable; use summary.
        env.close()
        reason = str(summary.get("termination_reason", "unknown"))
        reasons[reason] += 1
        min_dists.append(float(summary.get("min_distance", 1e9)))
        rows.append(
            {
                "episode": i,
                "reason": reason,
                "min_distance": float(summary.get("min_distance", 1e9)),
                "min_altitude": float(summary.get("min_altitude", 1e9)),
                "final_speed": float(summary.get("final_speed", 1e9)),
                "steps": int(summary.get("steps", 0)),
            }
        )

    n = max(1, int(args.episodes))
    land_rate = float(reasons.get("safe_landing", 0)) / n
    out = {
        "policy": args.policy,
        "model": args.model if args.policy == "ppo" else None,
        "episodes": n,
        "safe_landing_rate": land_rate,
        "require_upright": not bool(args.no_upright_gate),
        "reasons": dict(reasons),
        "mean_min_distance": float(sum(min_dists) / len(min_dists)),
        "episodes_detail": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"safe_landing_rate={land_rate:.3f}  reasons={dict(reasons)}")


if __name__ == "__main__":
    main()
