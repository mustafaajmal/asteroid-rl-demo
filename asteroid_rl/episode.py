"""Episode execution, CSV logging, and summary helpers.

Centralizes the reset/step loop used by play and evaluate so every runner
writes the same CSV schema and summary fields. Also provides filesystem
helpers for ``logs/`` and ``outputs/``.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def ensure_dirs() -> None:
    """Create standard ``logs/`` and ``outputs/`` directories if missing.

    Creates ``logs``, ``outputs``, ``outputs/plots``, and
    ``outputs/tensorboard`` relative to the current working directory.
    """
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/tensorboard").mkdir(parents=True, exist_ok=True)


def write_episode_csv(rows: List[dict], path: str) -> None:
    """Write per-step episode rows to a CSV file.

    Args:
        rows: List of row dictionaries. All rows should share the same keys;
            the first row defines the CSV header. If empty, the function
            returns without writing.
        path: Destination CSV filesystem path. Parent directories are created
            when needed.
    """
    if not rows:
        return
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_episode(rows: List[dict]) -> dict:
    """Compute aggregate metrics for one episode from its CSV-style rows.

    Args:
        rows: Per-step episode dictionaries containing at least ``time``,
            ``distance``, ``speed``, ``throttle``, ``reward``, and
            ``termination_reason`` keys (as produced by ``run_episode``).

    Returns:
        A summary dictionary with final state, totals, min/max distance and
        speed, and throttle statistics. Returns an empty dict if ``rows`` is
        empty.
    """
    if not rows:
        return {}
    last = rows[-1]
    distances = [float(r.get("distance", 0.0)) for r in rows]
    speeds = [float(r.get("speed", 0.0)) for r in rows]
    throttles = [float(r.get("throttle", 0.0)) for r in rows]
    altitudes = [float(r.get("altitude", 0.0)) for r in rows]
    return {
        "final_time": last.get("time"),
        "final_altitude": last.get("altitude"),
        "final_distance": last.get("distance"),
        "final_speed": last.get("speed"),
        "termination_reason": last.get("termination_reason"),
        "total_reward": sum(float(r.get("reward", 0.0)) for r in rows),
        "steps": len(rows),
        "min_altitude": min(altitudes) if altitudes else None,
        "min_distance": min(distances) if distances else None,
        "max_speed": max(speeds) if speeds else None,
        "avg_throttle": float(np.mean(throttles)) if throttles else None,
        "max_throttle": max(throttles) if throttles else None,
        "initial_distance": distances[0] if distances else None,
        "initial_altitude": altitudes[0] if altitudes else None,
    }


def run_episode(
    policy_name: str,
    action_fn: Callable[[Any], np.ndarray],
    env,
    csv_path: str,
    max_steps: int = 2000,
    *,
    print_every: Optional[int] = None,
    step_sleep_sec: float = 0.0,
) -> dict:
    """Roll out one episode, log steps to CSV, and return a summary.

    Args:
        policy_name: Label stored in each CSV row and in the returned summary
            (for example ``"scripted"`` or ``"ppo"``).
        action_fn: Callable mapping the current observation to a throttle
            action array compatible with ``env.action_space``.
        env: Environment instance exposing Gymnasium-style ``reset`` and
            ``step`` (typically ``AsteroidLandingEnv``).
        csv_path: Filesystem path where the per-step CSV will be written.
        max_steps: Maximum number of control steps before forcing a stop if
            the environment never terminates or truncates.
        print_every: If not ``None``, print a short telemetry line every this
            many steps (and on step 0). If ``None``, no per-step printing.
        step_sleep_sec: Optional wall-clock sleep after each step, useful to
            slow playback when Vizard is not pacing the sim.

    Returns:
        Episode summary dict from ``summarize_episode``, plus ``policy`` and
        ``csv_path`` keys.
    """
    obs, info = env.reset()
    rows: List[dict] = []

    for step_idx in range(max_steps):
        try:
            action = action_fn(obs, info)
        except TypeError:
            action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        perception = info.get("perception") or {}
        box = perception.get("landing_site_box") or [None, None, None, None]
        # Prefer privileged info telemetry — agent ``obs`` may be sensors/perception.
        truth = info.get("truth_state")
        if truth is not None and len(truth) >= 5:
            altitude = float(truth[0])
            vertical_velocity = float(truth[1])
            distance = float(truth[2])
            speed = float(truth[3])
            previous_throttle = float(truth[4])
        else:
            altitude = float(info.get("altitude", 0.0))
            vertical_velocity = float(info.get("vertical_velocity", 0.0))
            distance = float(info.get("distance_to_target", 0.0))
            speed = float(info.get("speed", 0.0))
            previous_throttle = float(info.get("throttle", 0.0))
        row = {
            "policy": policy_name,
            "step": step_idx,
            "time": info["sim_time_sec"],
            "obs_mode": info.get("obs_mode", ""),
            "altitude": altitude,
            "vertical_velocity": vertical_velocity,
            "distance": distance,
            "speed": speed,
            "previous_throttle": previous_throttle,
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
            "target_visible": bool(info.get("target_visible", False)),
            "hazard_score": float(info.get("hazard_score", 1.0)),
            "site_box_xmin": box[0],
            "site_box_ymin": box[1],
            "site_box_xmax": box[2],
            "site_box_ymax": box[3],
            "progress_assessment": perception.get("progress_assessment"),
        }
        rows.append(row)

        if print_every is not None and step_idx % print_every == 0:
            print(
                f"t={row['time']:.2f}s  alt={row['altitude']:.1f}m  "
                f"dist={row['distance']:.1f}m  spd={row['speed']:.2f}  "
                f"thr={row['throttle']:.2f}"
            )

        if step_sleep_sec > 0:
            time.sleep(step_sleep_sec)

        if terminated or truncated:
            break

    write_episode_csv(rows, csv_path)
    summary = summarize_episode(rows)
    summary["policy"] = policy_name
    summary["csv_path"] = csv_path
    return summary


def write_summary_markdown(summaries: List[Dict[str, Any]], path: str) -> None:
    """Write a human-readable Markdown comparison of episode summaries.

    Args:
        summaries: List of summary dictionaries as returned by
            ``run_episode`` / ``summarize_episode``.
        path: Destination Markdown filesystem path. Parent directories are
            created when needed.
    """
    lines = [
        "# Fixed-Site Policy Evaluation Summary",
        "",
        "This comparison isolates the fixed-site RL control loop. "
        "It does not evaluate Scenic, VLM, or perception.",
        "",
    ]
    for s in summaries:
        lines.extend(
            [
                f"## {s.get('policy', 'unknown')}",
                "",
                f"- Steps: `{s.get('steps')}`",
                f"- Final time [s]: `{s.get('final_time')}`",
                f"- Final altitude [m]: `{s.get('final_altitude')}`",
                f"- Final distance [m]: `{s.get('final_distance')}`",
                f"- Final speed [m/s]: `{s.get('final_speed')}`",
                f"- Min altitude [m]: `{s.get('min_altitude')}`",
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
