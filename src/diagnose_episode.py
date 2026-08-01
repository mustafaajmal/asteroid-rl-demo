import argparse
import os

import pandas as pd


def diagnose(df: pd.DataFrame) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Diagnose one episode CSV")
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

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
