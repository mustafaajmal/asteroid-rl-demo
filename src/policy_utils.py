import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def ensure_dirs():
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/tensorboard").mkdir(parents=True, exist_ok=True)


def write_episode_csv(rows: List[dict], path: str):
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_episode(rows: List[dict]) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    distances = [float(r.get("distance", 0.0)) for r in rows]
    speeds = [float(r.get("speed", 0.0)) for r in rows]
    throttles = [float(r.get("throttle", 0.0)) for r in rows]
    return {
        "final_time": last.get("time"),
        "final_distance": last.get("distance"),
        "final_speed": last.get("speed"),
        "termination_reason": last.get("termination_reason"),
        "total_reward": sum(float(r.get("reward", 0.0)) for r in rows),
        "steps": len(rows),
        "min_distance": min(distances) if distances else None,
        "max_speed": max(speeds) if speeds else None,
        "avg_throttle": float(np.mean(throttles)) if throttles else None,
        "max_throttle": max(throttles) if throttles else None,
        "initial_distance": distances[0] if distances else None,
    }


def scripted_action(obs) -> np.ndarray:
    _altitude, vertical_velocity, distance, speed, _previous_throttle = obs
    if vertical_velocity < -1.0:
        throttle = 1.0
    elif vertical_velocity < -0.5:
        throttle = 0.75
    elif distance < 20.0 and speed > 0.75:
        throttle = 0.65
    else:
        throttle = 0.25
    return np.array([throttle], dtype=np.float32)


def random_action(_obs, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    generator = rng or np.random.default_rng()
    return np.array([float(generator.uniform(0.0, 1.0))], dtype=np.float32)


def run_episode(
    policy_name: str,
    action_fn: Callable[[Any], np.ndarray],
    env,
    csv_path: str,
    max_steps: int = 1000,
) -> dict:
    obs, info = env.reset()
    rows: List[dict] = []

    for step_idx in range(max_steps):
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "policy": policy_name,
                "step": step_idx,
                "time": info["sim_time_sec"],
                "altitude": float(obs[0]),
                "vertical_velocity": float(obs[1]),
                "distance": float(obs[2]),
                "speed": float(obs[3]),
                "previous_throttle": float(obs[4]),
                "throttle": float(info["throttle"]),
                "thrust_N": float(info.get("thrust_N", 0.0)),
                "reward": float(reward),
                "reward_progress": float(info.get("reward_progress", 0.0)),
                "reward_speed_penalty": float(info.get("reward_speed_penalty", 0.0)),
                "reward_fuel_penalty": float(info.get("reward_fuel_penalty", 0.0)),
                "reward_terminal": float(info.get("reward_terminal", 0.0)),
                "termination_reason": info.get("termination_reason"),
                "success": bool(info.get("success", False)),
                "crash": bool(info.get("crash", False)),
                "escape": bool(info.get("escape", False)),
                "timeout": bool(info.get("timeout", False)),
            }
        )
        if terminated or truncated:
            break

    write_episode_csv(rows, csv_path)
    summary = summarize_episode(rows)
    summary["policy"] = policy_name
    summary["csv_path"] = csv_path
    return summary


def write_summary_markdown(summaries: List[Dict[str, Any]], path: str):
    lines = [
        "# Fixed-Site Policy Evaluation Summary",
        "",
        "This comparison isolates the fixed-site RL control loop. It does not evaluate Scenic, VLM, or perception.",
        "",
    ]
    for s in summaries:
        lines.extend(
            [
                f"## {s.get('policy', 'unknown')}",
                "",
                f"- Steps: `{s.get('steps')}`",
                f"- Final time [s]: `{s.get('final_time')}`",
                f"- Final distance [m]: `{s.get('final_distance')}`",
                f"- Final speed [m/s]: `{s.get('final_speed')}`",
                f"- Min distance [m]: `{s.get('min_distance')}`",
                f"- Max speed [m/s]: `{s.get('max_speed')}`",
                f"- Total reward: `{s.get('total_reward')}`",
                f"- Avg throttle: `{s.get('avg_throttle')}`",
                f"- Termination reason: `{s.get('termination_reason')}`",
                f"- Episode CSV: `{s.get('csv_path')}`",
                "",
            ]
        )

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
