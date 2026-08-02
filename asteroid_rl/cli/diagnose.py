"""Diagnose one episode CSV for a likely failure mode.

Reads a per-step episode log (from play or evaluate) and prints aggregate
telemetry plus a coarse textual failure-mode guess.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

import pandas as pd


def diagnose(df: pd.DataFrame) -> str:
    """Infer a coarse failure mode string from an episode dataframe.

    Args:
        df: Episode log with columns including ``termination_reason``,
            ``distance``, and ``speed``. May be empty.

    Returns:
        Short human-readable diagnosis string such as ``"success"``,
        ``"high-speed impact near target"``, or
        ``"policy did not make meaningful progress toward target"``.
    """
    if df.empty:
        return "empty episode log"

    termination_reason = df["termination_reason"].iloc[-1]
    final_distance = float(df["distance"].iloc[-1])
    initial_distance = float(df["distance"].iloc[0])
    final_speed = float(df["speed"].iloc[-1])

    if termination_reason == "safe_landing":
        return "success"
    if termination_reason == "crash":
        return "high-speed impact near target"
    if termination_reason == "escaped":
        return "moved too far from target"
    if final_distance > initial_distance * 0.9:
        return "policy did not make meaningful progress toward target"
    if final_speed > 2.0:
        return "approached but remained too fast"
    if termination_reason == "timeout":
        return "timeout while hovering/stuck"
    return "timeout/unclear; inspect plots"


def parse_args() -> Namespace:
    """Parse command-line arguments for episode diagnosis.

    Returns:
        Parsed argparse namespace with a ``csv_path`` field.
    """
    parser = argparse.ArgumentParser(description="Diagnose one episode CSV")
    parser.add_argument("csv_path", type=str)
    return parser.parse_args()


def main() -> None:
    """Load ``csv_path``, print episode stats, and print the diagnosis.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    args = parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(args.csv_path)

    df = pd.read_csv(args.csv_path)
    policy = df["policy"].iloc[0] if "policy" in df.columns else "unknown"
    diagnosis = diagnose(df)

    print(f"Policy: {policy}")
    print(f"Episode length: {len(df)}")
    print(f"Final distance: {float(df['distance'].iloc[-1]):.4f}")
    print(f"Final speed: {float(df['speed'].iloc[-1]):.4f}")
    print(f"Total reward: {float(df['reward'].sum()):.4f}")
    print(f"Termination reason: {df['termination_reason'].iloc[-1]}")
    print(f"Max speed: {float(df['speed'].max()):.4f}")
    print(f"Min distance: {float(df['distance'].min()):.4f}")
    print(f"Average throttle: {float(df['throttle'].mean()):.4f}")
    print(f"Max throttle: {float(df['throttle'].max()):.4f}")
    print(f"Likely failure mode: {diagnosis}")


if __name__ == "__main__":
    main()
