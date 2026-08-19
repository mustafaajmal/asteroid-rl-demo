"""Short PPO train on Scenic sphere starts, then eval on curriculum stages.

Brings the project closer to the PI end-state: train on Scenic scenarios, then
measure transfer / degradation on harder terrain stages.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO

from asteroid_rl.control.policies import make_action_fn
from asteroid_rl.environment.episode import ensure_dirs, run_episode
from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig


def _scenic_root() -> Path:
    here = Path(__file__).resolve()
    demo = here.parents[2]
    return demo.parent / "Scenic"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=8000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train-scenario",
        type=str,
        default="",
        help="Defaults to Scenic curriculum/sphere.scenic",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/scenic_curriculum",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenic_root = _scenic_root()
    train_scen = (
        Path(args.train_scenario)
        if args.train_scenario
        else scenic_root / "examples/basilisk/curriculum/sphere.scenic"
    )
    if not train_scen.is_file():
        raise SystemExit(f"Missing train scenario: {train_scen}")

    stages = {
        "sphere": scenic_root / "examples/basilisk/curriculum/sphere.scenic",
        "ellipsoid": scenic_root / "examples/basilisk/curriculum/ellipsoid.scenic",
        "bumpy": scenic_root / "examples/basilisk/curriculum/bumpy.scenic",
    }

    print(f"Train scenario: {train_scen}")
    print(f"PPO timesteps: {args.timesteps}\n")

    train_cfg = LandingEnvConfig(
        seed=args.seed,
        scenic_scenario_path=str(train_scen.resolve()),
        use_flat_surface=False,
        obs_mode="truth",
        auto_point=True,
        enable_viz=False,
        reuse_sim=False,  # procedural mesh rebuilds every episode
        success_speed=3.5,
        success_altitude=8.0,
        min_success_altitude=0.3,
        success_lateral=40.0,
        time_limit=120.0,
    )
    env = AsteroidLandingEnv(config=train_cfg)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=args.seed,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=int(args.timesteps))
    model_path = out / "ppo_scenic_sphere.zip"
    model.save(str(model_path))
    env.close()
    print(f"Saved {model_path}\n")

    # Baseline scripted + trained PPO on each curriculum stage.
    action_fns = {
        "scripted": make_action_fn("scripted", seed=args.seed),
        "ppo": make_action_fn("ppo", model=model, seed=args.seed),
    }
    rows = []
    summary = {}
    for stage, scen in stages.items():
        summary[stage] = {}
        for policy_name, fn in action_fns.items():
            ok = 0
            for i in range(int(args.eval_episodes)):
                cfg = LandingEnvConfig(
                    seed=args.seed + i,
                    scenic_scenario_path=str(scen.resolve()),
                    use_flat_surface=False,
                    obs_mode="truth",
                    auto_point=True,
                    enable_viz=False,
                    reuse_sim=False,
                    success_speed=3.5,
                    success_altitude=8.0,
                    min_success_altitude=0.3,
                    success_lateral=40.0,
                    time_limit=120.0,
                )
                ev = AsteroidLandingEnv(config=cfg)
                ep = run_episode(
                    f"{policy_name}_{stage}",
                    fn,
                    ev,
                    str(out / f"{stage}_{policy_name}_ep{i:02d}.csv"),
                    max_steps=500,
                )
                safe = ep.get("termination_reason") == "safe_landing"
                ok += int(safe)
                rows.append(
                    {
                        "stage": stage,
                        "policy": policy_name,
                        "episode": i,
                        "safe_landing": safe,
                        "final_altitude": ep.get("final_altitude"),
                        "final_speed": ep.get("final_speed"),
                        "total_reward": ep.get("total_reward"),
                        "termination_reason": ep.get("termination_reason"),
                    }
                )
                ev.close()
            rate = ok / max(int(args.eval_episodes), 1)
            summary[stage][policy_name] = {
                "safe": ok,
                "episodes": int(args.eval_episodes),
                "safe_rate": rate,
            }
            print(f"{stage:10s} {policy_name:8s}  {ok}/{args.eval_episodes} ({100*rate:.0f}%)")

    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
