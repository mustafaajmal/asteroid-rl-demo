"""Plot per-policy and overlaid comparison charts from evaluate CSVs.

Expects ``logs/eval_<policy>_episode_0.csv`` files produced by
``asteroid_rl.cli.evaluate``. Writes PNGs under ``outputs/plots`` by default.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd

from asteroid_rl.environment.episode import ensure_dirs
from asteroid_rl.analysis.plotting import plot_metric_vs_time


METRICS = ["distance", "speed", "throttle", "reward"]
POLICY_FILES = {
    "random": "logs/eval_random_episode_0.csv",
    "scripted": "logs/eval_scripted_episode_0.csv",
    "ppo": "logs/eval_ppo_episode_0.csv",
}


def parse_args() -> Namespace:
    """Parse command-line arguments for policy comparison plots.

    Returns:
        Parsed argparse namespace with an ``outdir`` field.
    """
    parser = argparse.ArgumentParser(
        description="Plot policy comparison charts from evaluate CSVs"
    )
    parser.add_argument("--outdir", type=str, default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    """Load available eval CSVs and write per-policy plus comparison plots.

    Missing policy CSV files are skipped with a printed warning.
    """
    args = parse_args()

    ensure_dirs()
    os.makedirs(args.outdir, exist_ok=True)

    loaded = {}
    for policy, path in POLICY_FILES.items():
        if not os.path.isfile(path):
            print(f"Missing {path}; skipping {policy}")
            continue
        loaded[policy] = pd.read_csv(path)
        for metric in METRICS:
            if metric in loaded[policy].columns:
                plot_metric_vs_time(
                    loaded[policy],
                    metric,
                    f"{policy} {metric} vs time",
                    os.path.join(args.outdir, f"{policy}_{metric}.png"),
                )

    for metric in METRICS:
        plt.figure()
        plotted = False
        for policy, df in loaded.items():
            if metric in df.columns:
                plt.plot(df["time"], df[metric], label=policy)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("time [s]")
        plt.ylabel(metric)
        plt.title(f"comparison_{metric}")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.outdir, f"comparison_{metric}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
