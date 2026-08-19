"""Evaluate a fixed policy against Scenic-generated approach scenarios.

MINIMUM experiment (PI notes): loop Scenic ``generate()`` for initial
conditions, run scripted (or PPO) in the Gym env, report safe-landing rates.

Example::

    python -m asteroid_rl.cli.evaluate_scenic \\
        --scenario ../Scenic/examples/basilisk/curriculum/sphere.scenic \\
        --episodes 5 --policy scripted
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from asteroid_rl.control.policies import make_action_fn
from asteroid_rl.environment.episode import ensure_dirs, run_episode
from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig


def parse_args():
    p = argparse.ArgumentParser(description="Eval policy on Scenic scenario starts")
    p.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path to a Basilisk-world .scenic file",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--policy", choices=("scripted", "random", "ppo"), default="scripted")
    p.add_argument("--model", type=str, default="", help="PPO zip when --policy ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flat-surface", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--obs-mode", default="truth", choices=("truth", "sensors"))
    p.add_argument(
        "--out",
        type=str,
        default="outputs/scenic_eval/summary.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    scenario = Path(args.scenario)
    if not scenario.is_file():
        raise SystemExit(f"Scenario not found: {scenario}")

    config = LandingEnvConfig(
        seed=args.seed,
        scenic_scenario_path=str(scenario.resolve()),
        use_flat_surface=bool(args.flat_surface),
        flat_surface_z=-30.0,
        obs_mode=str(args.obs_mode),
        auto_point=True,
        randomize_reset=False,
        enable_viz=False,
    )

    model = None
    if args.policy == "ppo":
        from stable_baselines3 import PPO

        if not args.model or not os.path.isfile(args.model):
            raise SystemExit("--policy ppo requires an existing --model zip")
        model = PPO.load(args.model, device="cpu")

    action_fn = make_action_fn(args.policy, model=model, seed=args.seed)
    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scenic scenario: {scenario}")
    print(f"Policy: {args.policy}  episodes={args.episodes}\n")

    for i in range(int(args.episodes)):
        env = AsteroidLandingEnv(config=config)
        summary = run_episode(
            f"{args.policy}_scenic",
            action_fn,
            env,
            str(out.parent / f"ep_{i:03d}.csv"),
            max_steps=400,
        )
        meta = getattr(env, "_last_scenic_meta", {}) or {}
        summary["episode"] = i
        summary["scenario"] = str(scenario)
        summary["scenic_meta"] = json.dumps(meta.get("asteroid", {}))
        summary["safe_landing"] = summary.get("termination_reason") == "safe_landing"
        rows.append(summary)
        env.close()
        print(
            f"  ep{i}: term={summary.get('termination_reason')} "
            f"alt={summary.get('final_altitude')} "
            f"speed={summary.get('final_speed')} "
            f"R={summary.get('total_reward')}"
        )

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_safe = sum(1 for r in rows if r.get("safe_landing"))
    rate = n_safe / max(len(rows), 1)
    print(f"\nSafe landings: {n_safe}/{len(rows)} ({100*rate:.0f}%)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
