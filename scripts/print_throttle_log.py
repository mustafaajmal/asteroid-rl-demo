"""Print a sampled throttle-vs-altitude log for the best mesh PPO."""

from __future__ import annotations

import pandas as pd
from stable_baselines3 import PPO

from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.environment.episode import run_episode
from asteroid_rl.control.policies import make_action_fn


def main() -> None:
    model = PPO.load("outputs/best_model_truth_mesh_fixed/best_model.zip", device="cpu")
    env = AsteroidLandingEnv(LandingEnvConfig(seed=0))
    csv_path = "logs/throttle_descent_ppo_best.csv"
    summary = run_episode(
        "ppo",
        make_action_fn("ppo", model=model),
        env,
        csv_path,
        print_every=10**9,
    )
    env.close()

    df = pd.read_csv(csv_path)
    print(
        "END:",
        summary.get("termination_reason"),
        "alt=",
        round(summary["final_altitude"], 3),
        "spd=",
        round(summary["final_speed"], 3),
    )
    print()

    rows = []
    last_t = -999.0
    for _, row in df.iterrows():
        t = float(row.time)
        alt = float(row.altitude)
        take = (
            (t - last_t >= 5.0)
            or (alt < 25.0 and t - last_t >= 1.0)
            or (alt < 10.0 and t - last_t >= 0.5)
        )
        if take:
            rows.append(row)
            last_t = t
    if not rows or float(rows[-1].time) != float(df.iloc[-1].time):
        rows.append(df.iloc[-1])

    header = f"{'t_sec':>7} {'alt_m':>8} {'spd':>6} {'throttle':>8} {'thrust_N':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        thr = float(row.throttle)
        print(
            f"{float(row.time):7.2f} "
            f"{float(row.altitude):8.2f} "
            f"{float(row.speed):6.2f} "
            f"{thr:8.3f} "
            f"{thr * 275.0:8.1f}"
        )

    n_unique = int(df.throttle.round(2).nunique())
    print()
    print(f"Distinct throttle levels (0.01): {n_unique}")
    print(f"Full CSV: {csv_path}")


if __name__ == "__main__":
    main()
