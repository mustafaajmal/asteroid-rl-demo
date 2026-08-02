"""Plot distance/speed/throttle/reward time series from one episode CSV.

Writes one PNG per available metric into ``--outdir`` (default ``outputs/``).
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

import pandas as pd

from asteroid_rl.plotting import plot_metric_vs_time


def parse_args() -> Namespace:
    """Parse command-line arguments for single-episode plotting.

    Returns:
        Parsed argparse namespace with ``csv_path`` and ``outdir`` fields.
    """
    parser = argparse.ArgumentParser(
        description="Plot metrics from one episode CSV"
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--outdir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    """Read the CSV and save metric-vs-time plots for standard columns.

    Plots ``distance``, ``speed``, ``throttle``, and ``reward`` when present.
    """
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    os.makedirs(args.outdir, exist_ok=True)

    for col in ["distance", "speed", "throttle", "reward"]:
        if col not in df.columns:
            continue
        plot_metric_vs_time(
            df,
            col,
            f"{col} vs time",
            os.path.join(args.outdir, f"{col}_vs_time.png"),
        )


if __name__ == "__main__":
    main()
