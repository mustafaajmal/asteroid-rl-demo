import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from policy_utils import ensure_dirs


METRICS = ["distance", "speed", "throttle", "reward"]
POLICY_FILES = {
    "random": "logs/eval_random_episode_0.csv",
    "scripted": "logs/eval_scripted_episode_0.csv",
    "ppo": "logs/eval_ppo_episode_0.csv",
}


def plot_series(df: pd.DataFrame, metric: str, title: str, out_path: str):
    plt.figure()
    plt.plot(df["time"], df[metric])
    plt.xlabel("time [s]")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/plots")
    args = parser.parse_args()

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
                plot_series(
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
