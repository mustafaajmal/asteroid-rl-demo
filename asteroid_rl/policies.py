"""Action helpers used by play / evaluate tooling.

Provides scripted / random / PPO mappings from observation (+ optional info)
to throttle ``[0, 1]`` or point+throttle ``[throttle, dx, dy, dz]``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from asteroid_rl.perception import perception_feature_vector
from asteroid_rl.pointing import unit


def scripted_action(obs, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Choose throttle with altitude + perception-aware braking.

    Reads privileged altitude/speed from ``info`` when available so the
    scripted baseline still works under ``sensors`` / ``perception`` obs modes.
    Uses the geometry perception stub for visibility / hazard gating.

    Args:
        obs: Environment observation (layout depends on ``obs_mode``; unused
            when ``info`` carries telemetry).
        info: Optional env info dict containing ``altitude`` / ``speed`` and a
            ``perception`` stub.

    Returns:
        A length-1 ``float32`` array containing throttle in ``[0, 1]``.
    """
    info = info or {}
    truth = info.get("truth_state")
    if truth is not None and len(truth) >= 4:
        altitude = float(truth[0])
        vertical_velocity = float(truth[1])
        speed = float(truth[3])
    elif "altitude" in info:
        altitude = float(info["altitude"])
        vertical_velocity = float(info.get("vertical_velocity", 0.0))
        speed = float(info.get("speed", 0.0))
    else:
        altitude = float(obs[0])
        vertical_velocity = float(obs[1])
        speed = float(obs[3])
    descent_rate = -float(vertical_velocity)
    perception = info.get("perception")
    feats = perception_feature_vector(perception)
    visible, _cu, _cv, hazard = [float(x) for x in feats]

    if altitude < 6.0 and speed <= 0.75:
        throttle = 0.15
    elif altitude < 10.0:
        throttle = 1.0 if descent_rate > 0.5 or speed > 0.9 else 0.55
    elif altitude < 25.0:
        throttle = 1.0 if descent_rate > 1.0 or speed > 1.5 else 0.45
    elif altitude < 60.0:
        throttle = 0.85 if descent_rate > 1.5 or speed > 2.0 else 0.25
    elif descent_rate > 2.0 or speed > 2.5:
        throttle = 0.7
    elif visible < 0.5 and altitude > 40.0:
        throttle = 0.0
    else:
        throttle = 0.0

    if hazard > 0.45 and altitude > 15.0:
        throttle = min(1.0, throttle + 0.15)
    if hazard < 0.10 and visible >= 0.5 and altitude < 20.0:
        throttle = max(throttle, 0.35 if descent_rate > 0.4 else throttle)

    if info.get("mission_mode") == "search" and altitude > 25.0:
        throttle = min(throttle, 0.25)

    return np.array([float(np.clip(throttle, 0.0, 1.0))], dtype=np.float32)


def scripted_orbit_action(obs, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Point+throttle GNC: anti-velocity far out, then site-point landing.

    Args:
        obs: Orbital observation
            ``[rel_xyz(3), vel_xyz(3), altitude, speed, prev_throttle]`` when
            available; otherwise falls back to ``info``.
        info: Env info with ``altitude``, ``speed``, positions when present.

    Returns:
        Shape ``(4,)`` ``float32``: ``[throttle, dx, dy, dz]`` (point body -z).
    """
    info = info or {}
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    if o.size >= 9:
        rel = o[0:3]
        vel = o[3:6]
        altitude = float(o[6])
        speed = float(o[7])
    else:
        truth = info.get("truth_state")
        altitude = float(
            truth[0] if truth is not None else info.get("altitude", 100.0)
        )
        speed = float(truth[3] if truth is not None and len(truth) > 3 else info.get("speed", 1.0))
        init_p = np.asarray(info.get("initial_position_N", [0.0, 0.0, 120.0]), dtype=np.float64)
        # Without orbital obs, aim toward -z / brake opposite default descent.
        rel = np.array([0.0, 0.0, altitude], dtype=np.float64)
        vel = np.array([0.0, 0.0, -speed], dtype=np.float64)
        del init_p

    approach_range = float(info.get("orbit_approach_range_m", 80.0))
    range_to_site = float(np.linalg.norm(rel))

    # Near the surface: always aim at the pad and use landing throttle bands.
    if altitude < 60.0:
        to_site = -rel
        point = (
            unit(to_site)
            if float(np.linalg.norm(to_site)) > 1e-6
            else np.array([0.0, 0.0, -1.0])
        )
        fake_info = {
            "altitude": altitude,
            "vertical_velocity": float(vel[2]),
            "speed": speed,
            "perception": info.get("perception"),
            "mission_mode": info.get("mission_mode"),
        }
        throttle = float(scripted_action(np.zeros(5), fake_info)[0])
        # If still far laterally, keep braking along LOS a bit harder.
        if range_to_site > approach_range:
            throttle = max(throttle, 0.45)
    elif range_to_site > approach_range:
        # Far: point camera (-z) along +velocity so thrust (+z) is anti-velocity.
        point = (
            unit(vel)
            if float(np.linalg.norm(vel)) > 1e-6
            else np.array([0.0, 0.0, 1.0])
        )
        throttle = 0.65 if speed > 1.5 else 0.45
    else:
        to_site = -rel
        point = (
            unit(to_site)
            if float(np.linalg.norm(to_site)) > 1e-6
            else np.array([0.0, 0.0, -1.0])
        )
        fake_info = {
            "altitude": altitude,
            "vertical_velocity": float(vel[2]),
            "speed": speed,
            "perception": info.get("perception"),
            "mission_mode": info.get("mission_mode"),
        }
        throttle = float(scripted_action(np.zeros(5), fake_info)[0])

    return np.array(
        [float(np.clip(throttle, 0.0, 1.0)), float(point[0]), float(point[1]), float(point[2])],
        dtype=np.float32,
    )


def random_action(
    _obs,
    rng: Optional[np.random.Generator] = None,
    info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sample a uniform random throttle in ``[0, 1]``."""
    del info
    generator = rng or np.random.default_rng()
    return np.array([float(generator.uniform(0.0, 1.0))], dtype=np.float32)


def random_orbit_action(
    _obs,
    rng: Optional[np.random.Generator] = None,
    info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sample random throttle and pointing direction."""
    del info
    generator = rng or np.random.default_rng()
    direction = generator.normal(0.0, 1.0, size=3)
    direction = unit(direction)
    return np.array(
        [
            float(generator.uniform(0.0, 1.0)),
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        ],
        dtype=np.float32,
    )


def make_action_fn(
    policy: str,
    *,
    model=None,
    seed: int = 0,
) -> Callable:
    """Build an ``(obs, info=None) -> action`` callable for the named policy.

    Args:
        policy: ``scripted``, ``scripted_orbit``, ``random``, ``random_orbit``,
            or ``ppo``.
        model: Loaded SB3 PPO model when ``policy == "ppo"``.
        seed: RNG seed for random policies.

    Returns:
        Callable mapping ``(obs, info=None)`` to an action array.
    """
    if policy == "scripted":
        return scripted_action
    if policy == "scripted_orbit":
        return scripted_orbit_action
    if policy == "random":
        rng = np.random.default_rng(seed)

        def _random(obs, info=None):
            return random_action(obs, rng=rng, info=info)

        return _random
    if policy == "random_orbit":
        rng = np.random.default_rng(seed)

        def _random_orbit(obs, info=None):
            return random_orbit_action(obs, rng=rng, info=info)

        return _random_orbit
    if policy == "ppo":
        if model is None:
            raise ValueError("PPO policy requires a loaded model")

        def ppo_action(obs, info=None):
            del info
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return ppo_action
    raise ValueError(f"Unknown policy: {policy}")
