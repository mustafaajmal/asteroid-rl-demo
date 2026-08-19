"""Train a stronger Scenic PPO (optional) and record Vizard .bin demos.

Example:
  python -m asteroid_rl.cli.record_scenic_viz --timesteps 50000 --seed 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from stable_baselines3 import PPO

from asteroid_rl.control.policies import make_action_fn
from asteroid_rl.environment.episode import ensure_dirs, run_episode
from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.environment.gym_env import _find_vizard_app
from asteroid_rl.sensing.camera import launch_vizard_load_file


def _scenic_root() -> Path:
    return Path(__file__).resolve().parents[2].parent / "Scenic"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--model", type=str, default="", help="Reuse an existing PPO zip; skip train")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--out-dir", type=str, default="outputs/scenic_strong")
    p.add_argument("--open-last", action="store_true", help="Open the last recorded .bin in Vizard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    viz_dir = out / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    scenic_root = _scenic_root()
    stages = {
        "sphere": scenic_root / "examples/basilisk/curriculum/sphere.scenic",
        "ellipsoid": scenic_root / "examples/basilisk/curriculum/ellipsoid.scenic",
        "bumpy": scenic_root / "examples/basilisk/curriculum/bumpy.scenic",
    }
    for name, path in stages.items():
        if not path.is_file():
            raise SystemExit(f"Missing scenario: {path}")

    model_path = Path(args.model) if args.model else out / "ppo_scenic_strong.zip"
    model = None

    if args.skip_train or (args.model and model_path.is_file()):
        if not model_path.is_file():
            raise SystemExit(f"Missing model: {model_path}")
        print(f"Loading existing model: {model_path}")
        model = PPO.load(str(model_path), device="cpu")
    else:
        print(f"Training PPO for {args.timesteps} steps on sphere...")
        train_cfg = LandingEnvConfig(
            seed=args.seed,
            scenic_scenario_path=str(stages["sphere"].resolve()),
            use_flat_surface=False,
            obs_mode="truth",
            auto_point=True,
            enable_viz=False,
            reuse_sim=False,
            success_speed=3.5,
            success_altitude=8.0,
            min_success_altitude=0.3,
        )
        train_env = AsteroidLandingEnv(config=train_cfg)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            seed=args.seed,
        )
        model.learn(total_timesteps=int(args.timesteps))
        model.save(str(model_path))
        train_env.close()
        print(f"Saved model: {model_path.resolve()}")

    rows = []
    last_bin = None
    # Record scripted + PPO on each stage (one episode each).
    for stage, scen in stages.items():
        for policy_name in ("scripted", "ppo"):
            bin_path = viz_dir / f"{stage}_{policy_name}_UnityViz.bin"
            cfg = LandingEnvConfig(
                seed=args.seed + (0 if stage == "sphere" else 1 if stage == "ellipsoid" else 2),
                scenic_scenario_path=str(scen.resolve()),
                use_flat_surface=False,
                obs_mode="truth",
                auto_point=True,
                enable_viz=True,
                viz_mode="file",
                viz_save_file=str(bin_path),
                reuse_sim=False,
                success_speed=3.5,
                success_altitude=8.0,
                min_success_altitude=0.3,
            )
            env = AsteroidLandingEnv(config=cfg)
            action_fn = make_action_fn(
                policy_name,
                model=model if policy_name == "ppo" else None,
                seed=cfg.seed,
            )
            print(f"\nRecording {stage}/{policy_name} -> {bin_path.name}")
            result = run_episode(
                f"{policy_name}_{stage}",
                action_fn,
                env,
                str(out / f"ep_{stage}_{policy_name}.csv"),
                max_steps=900,
                print_every=50,
            )
            recorded = getattr(env.handles, "viz_bin_path", None) or str(bin_path)
            last_bin = recorded
            safe = result.get("termination_reason") == "safe_landing"
            row = {
                "stage": stage,
                "policy": policy_name,
                "success": bool(safe),
                "termination_reason": result.get("termination_reason"),
                "steps": int(result.get("steps", 0) or 0),
                "final_alt_m": float(result.get("final_altitude", float("nan"))),
                "final_speed_mps": float(result.get("final_speed", float("nan"))),
                "viz_bin": recorded,
            }
            rows.append(row)
            print(
                f"  reason={row['termination_reason']} steps={row['steps']} "
                f"alt={row['final_alt_m']:.2f} speed={row['final_speed_mps']:.2f}"
            )
            env.close()

    summary_path = out / "viz_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "viz_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Strongest Scenic model + Vizard demos ===")
    print(f"Model: {model_path.resolve()}")
    print(f"Viz folder: {viz_dir.resolve()}")
    for r in rows:
        print(
            f"  {r['stage']:10s} {r['policy']:8s} success={r['success']}  "
            f"{Path(r['viz_bin']).name}"
        )
    print(f"\nSummary: {summary_path.resolve()}")
    print(
        "Replay any demo:\n"
        f'  & "$env:BASILISK_ROOT\\..\\Utilities\\Vizard\\Vizard.exe" '
        f'-loadFile "<path-to-.bin>"'
    )

    if args.open_last and last_bin and os.path.isfile(last_bin):
        launch_vizard_load_file(last_bin, find_app_fn=_find_vizard_app)


if __name__ == "__main__":
    main()
