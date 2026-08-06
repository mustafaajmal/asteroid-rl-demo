"""Action helpers used by play / evaluate tooling.

Provides scripted / random / PPO mappings from observation (+ optional info)
to throttle ``[0, 1]`` or point+throttle ``[throttle, dx, dy, dz]``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from asteroid_rl.gravity import (
    DEFAULT_ASTEROID_COM_N,
    DEFAULT_MU,
    DEFAULT_SPACECRAFT_MASS_REF,
    hover_throttle_central,
)
from asteroid_rl.perception import perception_feature_vector
from asteroid_rl.pointing import unit


def _estimate_hover(info: Dict[str, Any], rel: np.ndarray, altitude: float) -> float:
    """Altitude-aware hover throttle (central gravity varies with r)."""
    if "hover_throttle" in info:
        return float(np.clip(float(info["hover_throttle"]), 0.05, 1.0))
    pos = info.get("position_N")
    if pos is None:
        # Reconstruct approx position above flat pad at z=-30.
        pos = np.array(
            [float(rel[0]), float(rel[1]), -30.0 + float(altitude)],
            dtype=np.float64,
        )
    return float(
        hover_throttle_central(
            pos,
            mu=float(info.get("gravity_mu", DEFAULT_MU)),
            mass=float(info.get("gravity_mass_ref", DEFAULT_SPACECRAFT_MASS_REF)),
            max_thrust=float(info.get("max_thrust", 2500.0)),
            com_N=info.get("asteroid_com_N", DEFAULT_ASTEROID_COM_N),
        )
    )


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
    """Point+throttle GNC: velocity-target divert toward the fixed pad.

    One thruster along body **+z** (boresight **−z**). Desired inertial accel
    ``a_cmd`` is realized by pointing ``-z`` along ``-a_cmd``.

    Phases:
    1. **Terminal corridor** — low alt, small lateral, moderate speed: point at
       site and reuse Phase-1 altitude-band braking.
    2. **Velocity targeting** — command a closing velocity toward the site
       (``v_des = -v_close * unit(rel)``) and thrust along ``v_des - vel``.
       This *diverts* cross-track error, not just anti-velocity braking.
    3. **Energy cap** — if very fast or climbing hard, blend more anti-velocity
       and raise throttle so we do not escape the demo sphere.

    Throttle is scheduled from ``|v_err|`` (never ``m*|a|/F_max`` saturation).

    Args:
        obs: Orbital observation
            ``[rel_xyz(3), vel_xyz(3), altitude, speed, prev_throttle]``.
        info: Env info (optional perception / mission fields).

    Returns:
        Shape ``(4,)`` ``float32``: ``[throttle, dx, dy, dz]``.
    """
    info = info or {}
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    if o.size >= 9:
        rel = o[0:3].copy()
        vel = o[3:6].copy()
        altitude = float(o[6])
        speed = float(o[7])
    else:
        truth = info.get("truth_state")
        altitude = float(
            truth[0] if truth is not None else info.get("altitude", 100.0)
        )
        speed = float(
            truth[3]
            if truth is not None and len(truth) > 3
            else info.get("speed", 1.0)
        )
        rel = np.array([0.0, 0.0, altitude], dtype=np.float64)
        vel = np.array([0.0, 0.0, -speed], dtype=np.float64)

    range_to_site = float(np.linalg.norm(rel))
    lateral = float(np.linalg.norm(rel[:2]))
    r_hat = unit(rel) if range_to_site > 1e-9 else np.array([0.0, 0.0, 1.0])
    v_hat = unit(vel) if speed > 1e-9 else np.array([0.0, 0.0, -1.0])
    climbing = float(np.dot(r_hat, vel)) > 0.35

    # --- Phase 1: near-pad ---
    # Outside success cone: divert (thrust toward pad / cancel lateral).
    # Inside cone: point at site and brake (thrust away along LOS).
    if altitude < 80.0:
        if lateral > 15.0:
            lat_vec = np.array([rel[0], rel[1], 0.0], dtype=np.float64)
            if float(np.linalg.norm(lat_vec)) < 1e-9:
                lat_vec = rel.copy()
            lat_hat = unit(lat_vec)
            v_lat = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
            # Accel toward pad in XY + damp lateral rate; soft vertical brake.
            a_cmd = -0.6 * lat_hat - 0.8 * v_lat
            if float(vel[2]) < -1.0:
                a_cmd = a_cmd + np.array([0.0, 0.0, 0.8])
            a_hat = unit(a_cmd)
            point = -a_hat
            throttle = 0.55 if speed < 1.5 else 0.75
            if speed > 3.0:
                throttle = 0.90
            return np.array(
                [
                    float(np.clip(throttle, 0.0, 1.0)),
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                ],
                dtype=np.float32,
            )

        # Inside lateral cone: LOS brake toward soft-land gates.
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
        if altitude > 12.0 and speed > 0.9:
            throttle = max(throttle, 0.80)
        if altitude > 6.0 and speed > 1.5:
            throttle = 1.0
        if altitude <= 8.0 and speed > 0.75:
            throttle = 1.0
        if 0.5 <= altitude <= 5.0 and speed <= 0.85 and lateral <= 18.0:
            throttle = min(throttle, 0.35)  # settle into success band
        return np.array(
            [
                float(np.clip(throttle, 0.0, 1.0)),
                float(point[0]),
                float(point[1]),
                float(point[2]),
            ],
            dtype=np.float32,
        )

    # --- Phase 2: velocity targeting toward the site (divert) ---
    # Closing speed grows with range but stays modest so we do not overshoot.
    approach_range = float(info.get("orbit_approach_range_m", 80.0))
    if range_to_site > 200.0:
        v_close = 1.6
    elif range_to_site > approach_range:
        v_close = 1.1
    elif range_to_site > 40.0:
        v_close = 0.7
    else:
        v_close = 0.35
    # Near the surface, prefer mostly vertical closing (kill lateral first).
    if altitude < 90.0 and lateral > 15.0:
        lateral_hat = unit(np.array([rel[0], rel[1], 0.0]))
        v_des = -1.4 * lateral_hat - 0.4 * np.array([0.0, 0.0, 1.0 if rel[2] > 0 else -1.0])
        v_des = unit(v_des) * min(v_close, 1.5)
    else:
        v_des = -v_close * r_hat

    v_err = v_des - vel
    # Blend anti-velocity when too energetic so divert does not add escape Δv.
    if speed > 5.5 or climbing:
        brake_w = 0.70 if speed > 7.0 else 0.45
        v_err = (1.0 - brake_w) * v_err + brake_w * (-vel)
    elif speed > 4.0:
        v_err = 0.75 * v_err + 0.25 * (-vel)

    err_norm = float(np.linalg.norm(v_err))
    if err_norm < 1e-9:
        v_err = -v_hat
        err_norm = 1.0
    a_hat = v_err / err_norm
    # Point boresight (-z) opposite desired accel so +z thrust = a_hat.
    point = -a_hat

    # --- Throttle schedule from tracking error (bounded) ---
    if speed > 6.0 or climbing:
        throttle = 0.88
    elif err_norm > 3.0:
        throttle = 0.75
    elif err_norm > 1.5:
        throttle = 0.58
    elif altitude < 100.0 and lateral > 20.0:
        throttle = 0.55
    else:
        throttle = 0.40
    if range_to_site > 700.0:
        throttle = min(throttle, 0.32)
    if range_to_site > 1200.0:
        throttle = min(throttle, 0.22)
    # Soft cap when climbing away from pad to limit escapes.
    if climbing and altitude > 60.0 and range_to_site > 80.0:
        throttle = min(throttle, 0.65)

    return np.array(
        [
            float(np.clip(throttle, 0.0, 1.0)),
            float(point[0]),
            float(point[1]),
            float(point[2]),
        ],
        dtype=np.float32,
    )


def scripted_autonomous_action(obs, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Multi-phase planning-doc policy: acquire → divert → upright soft-land.

    Uses ``info["mission_mode"]`` when present (env FSM). Falls back to
    kinematics heuristics if mission is disabled.

    Returns 4-D ``[throttle, dx, dy, dz]`` for ``point_throttle`` envs.
    """
    info = info or {}
    mode = str(info.get("mission_mode", "divert")).lower()
    pointing_cmd = info.get("pointing_command")

    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    if o.size >= 9:
        rel = o[0:3].copy()
        vel = o[3:6].copy()
        altitude = float(o[6])
        speed = float(o[7])
    else:
        truth = info.get("truth_state")
        altitude = float(
            truth[0] if truth is not None else info.get("altitude", 100.0)
        )
        speed = float(
            truth[3]
            if truth is not None and len(truth) > 3
            else info.get("speed", 1.0)
        )
        rel = np.array([0.0, 0.0, altitude], dtype=np.float64)
        vel = np.array([0.0, 0.0, -speed], dtype=np.float64)

    lateral = float(np.linalg.norm(rel[:2]))
    # Prefer env-provided pointing command (mission FSM).
    if pointing_cmd is not None:
        cmd = unit(np.asarray(pointing_cmd, dtype=np.float64).reshape(3))
    else:
        cmd = unit(-rel) if float(np.linalg.norm(rel)) > 1e-9 else np.array([0.0, 0.0, -1.0])

    # Heuristic mode if mission disabled / unknown.
    if mode not in ("search", "acquire", "divert", "upright", "land"):
        if altitude < 70.0 and lateral <= 35.0:
            mode = "upright"
        else:
            mode = "divert"

    if mode in ("search", "acquire"):
        # Slew toward body/pad; low thrust (env also gates).
        point = cmd
        throttle = 0.15 if mode == "acquire" else 0.10
        return np.array(
            [throttle, float(point[0]), float(point[1]), float(point[2])],
            dtype=np.float32,
        )

    if mode in ("upright", "land") or (altitude < 100.0 and lateral <= 45.0):
        # One-thruster soft-land:
        # A) Close lateral / kill horiz speed first (do NOT LOS-brake yet —
        #    LOS brake pushes *away* from the pad and freezes lateral miss).
        # B) When nearly overhead, look at pad and track a descent profile
        #    relative to *altitude-dependent* hover (central g ≠ constant).
        v_horiz = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
        v_h = float(np.linalg.norm(v_horiz))
        lat_vec = np.array([rel[0], rel[1], 0.0], dtype=np.float64)
        lat_n = float(np.linalg.norm(lat_vec))
        climbing = float(vel[2]) > 0.20
        hover = _estimate_hover(info, rel, altitude)

        # Near the pad, prefer vertical settle over chasing tiny lateral miss,
        # but still kill leftover horizontal speed (one thruster must tilt briefly).
        need_divert = lat_n > 16.0 or v_h > 0.40
        if altitude < 8.0 and lat_n <= 24.0 and v_h < 0.25:
            need_divert = False
        if need_divert:
            lat_hat = lat_vec / max(lat_n, 1e-9)
            # Accel toward pad in XY + damp horiz rate; soft upward if diving.
            a_cmd = -0.85 * lat_hat - 1.1 * v_horiz
            if float(vel[2]) < -0.8:
                a_cmd = a_cmd + np.array([0.0, 0.0, 0.9])
            if climbing:
                a_cmd = -vel  # anti-velocity to stop escape
            a_hat = unit(a_cmd)
            point = -a_hat
            throttle = max(hover + 0.12, 0.35)
            if lat_n > 20.0 or v_h > 0.8:
                throttle = max(throttle, hover + 0.25)
            if speed > 2.5 or climbing:
                throttle = max(throttle, min(hover + 0.35, 0.95))
            if climbing and altitude > 40.0:
                # At high alt, true hover is low — do not fire pad-level hover.
                throttle = min(throttle, max(hover + 0.15, 0.40))
            return np.array(
                [
                    float(np.clip(throttle, 0.0, 1.0)),
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                ],
                dtype=np.float32,
            )

        # Overhead corridor: point thruster along local-up (look at ground).
        # Do NOT look-at-pad here — with lateral miss L at altitude h the
        # look-at-pad tilt is atan(L/h) (e.g. 8 m / 6 m ≈ 53°), which fails
        # the upright gate and couples huge lateral accel into the settle.
        # Mission divert pointing is also look-at-pad — ignore it while settling.
        if mode in ("upright", "land") and pointing_cmd is not None:
            point = cmd
        else:
            point = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        v_up = float(vel[2])
        # Desired vertical rate (negative = descend). Taper into the success band.
        if altitude > 50.0:
            v_des = -1.8
        elif altitude > 25.0:
            v_des = -1.1
        elif altitude > 12.0:
            v_des = -0.70
        elif altitude > 5.05:
            # Many timeouts hung at 5.2–6.5 m; push through the 5 m ceiling.
            v_des = -0.55
        elif altitude > 2.5:
            v_des = -0.15
        else:
            v_des = -0.05

        # Hold nearly still once inside the success gates (alt/speed/lateral).
        in_band = 0.5 <= altitude <= 5.0 and lat_n <= 24.0
        if in_band and speed <= 0.80:
            v_des = 0.0

        # throttle ≈ hover + K*(v_des - v_up): too-fast descent → more thrust.
        throttle = hover + 0.70 * (v_des - v_up)
        # Kill horizontal residual with a bit of extra thrust while upright
        # (cannot cancel horiz with pointing without tilting — accept and
        # wait for damping / prior divert). Prefer vertical speed kill.
        if in_band and speed > 0.80:
            throttle = hover + 0.90 * max(-v_up, 0.0) + 0.20 * speed
            throttle = min(throttle, hover + 0.40)
        if (not in_band) and altitude <= 8.0 and lat_n <= 24.0:
            # Push into the ≤5 m band; residual horiz speed is OK if ≤ success.
            if speed > 1.2:
                throttle = max(throttle, hover + 0.20)
            else:
                throttle = min(throttle, hover - 0.05)
        if climbing and altitude > 5.0:
            throttle = max(throttle, hover + 0.10)
        if altitude < 1.2 and v_up < -0.35:
            throttle = max(throttle, hover + 0.30)

        return np.array(
            [
                float(np.clip(throttle, 0.0, 1.0)),
                float(point[0]),
                float(point[1]),
                float(point[2]),
            ],
            dtype=np.float32,
        )

    # Divert: reuse orbital velocity-target GNC.
    return scripted_orbit_action(obs, info)


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
    direction = generator.normal(size=3)
    direction = unit(direction)
    throttle = float(generator.uniform(0.0, 1.0))
    return np.array(
        [throttle, float(direction[0]), float(direction[1]), float(direction[2])],
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
        policy: ``scripted``, ``scripted_orbit``, ``scripted_autonomous``,
            ``random``, ``random_orbit``, or ``ppo``.
        model: Loaded SB3 PPO model when ``policy == "ppo"``.
        seed: RNG seed for random policies.

    Returns:
        Callable mapping ``(obs, info=None)`` to an action array.
    """
    if policy == "scripted":
        return scripted_action
    if policy == "scripted_orbit":
        return scripted_orbit_action
    if policy == "scripted_autonomous":
        return scripted_autonomous_action
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
