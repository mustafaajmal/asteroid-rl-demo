import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/plot_logs.py logs/scripted_controller_log.csv")
        return

    path = sys.argv[1]
    df = pd.read_csv(path)
    os.makedirs("outputs", exist_ok=True)

    for col in ["distance", "speed", "throttle", "reward"]:
        if col in df.columns:
            plt.figure()
            plt.plot(df["time"], df[col])
            plt.xlabel("time [s]")
            plt.ylabel(col)
            plt.title(f"{col} vs time")
            out = f"outputs/{col}_plot.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")
            plt.close()


if __name__ == "__main__":
    main()
