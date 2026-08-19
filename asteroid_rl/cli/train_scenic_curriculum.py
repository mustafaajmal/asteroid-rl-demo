"""MINIMUM path B: train PPO on Scenic scenarios, then eval on the curriculum.

Default: progressive curriculum — train sphere → ellipsoid → bumpy (each stage
gets ``--timesteps-per-stage``), then evaluate scripted + PPO on all stages.

Scenic provides the scenario distribution (craft ICs + procedural asteroid).
Basilisk/MuJoCo runs dynamics with the Scenic-sampled mesh rebuilt each episode.

Example::

    export SCENIC_ROOT=../Scenic PYTHONPATH=../Scenic/src:.
    python -m asteroid_rl.cli.train_scenic_curriculum \\
      --timesteps-per-stage 4000 --eval-episodes 4 --seed 2
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
    return Path(__file__).resolve().parents[2].parent / "Scenic"


def _env_cfg(scenario: Path, seed: int) -> LandingEnvConfig:
    return LandingEnvConfig(
        seed=seed,
        scenic_scenario_path=str(scenario.resolve()),
        use_flat_surface=False,
        obs_mode="truth",
        auto_point=True,
        enable_viz=False,
        reuse_sim=False,  # rebuild procedural mesh every episode
        success_speed=3.5,  # low / writeup-friendly gate
        success_altitude=8.0,
        min_success_altitude=0.3,
        success_lateral=40.0,
        time_limit=120.0,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO on Scenic curriculum, then eval transfer"
    )
    p.add_argument(
        "--timesteps-per-stage",
        type=int,
        default=4000,
        help="PPO steps per curriculum stage (progressive train)",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="If >0, overrides progressive train: train only on --train-scenario "
        "for this many steps (legacy sphere-only mode)",
    )
    p.add_argument("--eval-episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--train-scenario",
        type=str,
        default="",
        help="Only used with --timesteps >0 (single-scenario train)",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["sphere", "ellipsoid", "bumpy"],
        help="Curriculum order for progressive train + final eval",
    )
    p.add_argument("--out-dir", type=str, default="outputs/scenic_curriculum")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenic_root = _scenic_root()
    stages = {
        "sphere": scenic_root / "examples/basilisk/curriculum/sphere.scenic",
        "ellipsoid": scenic_root / "examples/basilisk/curriculum/ellipsoid.scenic",
        "bumpy": scenic_root / "examples/basilisk/curriculum/bumpy.scenic",
    }
    for name in args.stages:
        if name not in stages or not stages[name].is_file():
            raise SystemExit(f"Missing scenario for stage '{name}': {stages.get(name)}")

    progressive = int(args.timesteps) <= 0
    model = None
    model_path = out / (
        "ppo_scenic_curriculum.zip" if progressive else "ppo_scenic_sphere.zip"
    )

    if progressive:
        print(
            f"Progressive Scenic curriculum train: {args.stages} "
            f"({args.timesteps_per_stage} steps/stage)\n"
        )
        for si, stage in enumerate(args.stages):
            scen = stages[stage]
            print(f"=== train stage {si}: {stage} ({scen.name}) ===")
            env = AsteroidLandingEnv(config=_env_cfg(scen, args.seed + si * 1000))
            if model is None:
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    seed=args.seed,
                    n_steps=256,
                    batch_size=64,
                    learning_rate=3e-4,
                )
            else:
                model.set_env(env)
            model.learn(total_timesteps=int(args.timesteps_per_stage), reset_num_timesteps=False)
            stage_zip = out / f"ppo_after_{stage}.zip"
            model.save(str(stage_zip))
            print(f"  saved {stage_zip}")
            env.close()
    else:
        train_scen = (
            Path(args.train_scenario)
            if args.train_scenario
            else stages["sphere"]
        )
        if not train_scen.is_file():
            raise SystemExit(f"Missing train scenario: {train_scen}")
        print(f"Single-scenario train: {train_scen} ({args.timesteps} steps)\n")
        env = AsteroidLandingEnv(config=_env_cfg(train_scen, args.seed))
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
        env.close()

    assert model is not None
    model.save(str(model_path))
    print(f"\nSaved final model: {model_path}\n")

    action_fns = {
        "scripted": make_action_fn("scripted", seed=args.seed),
        "ppo": make_action_fn("ppo", model=model, seed=args.seed),
    }
    rows = []
    summary = {
        "train_mode": "progressive" if progressive else "single",
        "stages": list(args.stages),
        "timesteps_per_stage": int(args.timesteps_per_stage) if progressive else None,
        "timesteps": int(args.timesteps) if not progressive else None,
        "eval": {},
    }
    print("=== eval (scripted + PPO) on each stage ===")
    for stage in args.stages:
        scen = stages[stage]
        summary["eval"][stage] = {}
        for policy_name, fn in action_fns.items():
            ok = 0
            for i in range(int(args.eval_episodes)):
                cfg = _env_cfg(scen, args.seed + i)
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
            summary["eval"][stage][policy_name] = {
                "safe": ok,
                "episodes": int(args.eval_episodes),
                "safe_rate": rate,
            }
            print(f"{stage:10s} {policy_name:8s}  {ok}/{args.eval_episodes} ({100 * rate:.0f}%)")

    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(json.dumps(summary["eval"], indent=2))


if __name__ == "__main__":
    main()
