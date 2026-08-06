"""Shared Matplotlib helpers for episode CSV plots."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_vs_time(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: str,
    *,
    dpi: int = 150,
) -> None:
    """Save a single metric-versus-time line plot.

    Args:
        df: Episode dataframe containing ``time`` and ``metric`` columns.
        metric: Column name to plot on the y-axis.
        title: Matplotlib plot title.
        out_path: Destination PNG filesystem path.
        dpi: Output figure DPI.
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.figure()
    plt.plot(df["time"], df[metric])
    plt.xlabel("time [s]")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
