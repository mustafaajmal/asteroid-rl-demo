"""Agent observation packing: truth vs sensors vs perception vs orbital.

Reward and termination always use simulator truth. These helpers build the
vector the *policy* is allowed to see under each ``obs_mode`` / action mode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from asteroid_rl.perception import perception_policy_features

OBS_MODES = ("truth", "sensors", "perception", "orbital")


def validate_obs_mode(mode: str) -> str:
    """Normalize and validate an observation mode name.

    Args:
        mode: Requested mode string.

    Returns:
        Lowercased mode in ``OBS_MODES``.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    key = str(mode).strip().lower()
    if key not in OBS_MODES:
        raise ValueError(f"Unknown obs_mode={mode!r}; expected one of {OBS_MODES}")
    return key


def observation_dim(mode: str) -> int:
    """Return the policy observation vector length for ``mode``.

    Args:
        mode: One of ``truth``, ``sensors``, ``perception``, ``orbital``.

    Returns:
        Observation dimensionality.
    """
    mode = validate_obs_mode(mode)
    if mode == "orbital":
        return 9
    if mode == "truth":
        return 5
    if mode == "sensors":
        return 5
    return 6  # perception


def pack_truth_vector(
    *,
    altitude: float,
    vertical_velocity: float,
    distance: float,
    speed: float,
    previous_throttle: float,
) -> np.ndarray:
    """Pack privileged simulator state used for reward / termination.

    Args:
        altitude: Hub altitude above terrain, meters.
        vertical_velocity: Inertial ``v_z``, m/s.
        distance: Distance to landing site, meters.
        speed: Speed magnitude, m/s.
        previous_throttle: Last applied throttle in ``[0, 1]``.

    Returns:
        Shape ``(5,)`` ``float32`` truth vector.
    """
    return np.array(
        [altitude, vertical_velocity, distance, speed, previous_throttle],
        dtype=np.float32,
    )


def pack_orbital_vector(
    *,
    position_N: Sequence[float],
    velocity_N: Sequence[float],
    target_N: Sequence[float],
    altitude: float,
    previous_throttle: float,
) -> np.ndarray:
    """Pack privileged relative state for orbital / point-throttle policies.

    Args:
        position_N: Hub inertial position, meters.
        velocity_N: Hub inertial velocity, m/s.
        target_N: Landing-site inertial position, meters.
        altitude: Altitude above terrain, meters.
        previous_throttle: Last throttle in ``[0, 1]``.

    Returns:
        Shape ``(9,)`` ``float32``:
        ``[rel_xyz(3), vel_xyz(3), altitude, speed, previous_throttle]``.
    """
    r = np.asarray(position_N, dtype=np.float64).reshape(3)
    v = np.asarray(velocity_N, dtype=np.float64).reshape(3)
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    rel = r - target
    speed = float(np.linalg.norm(v))
    return np.array(
        [
            float(rel[0]),
            float(rel[1]),
            float(rel[2]),
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(altitude),
            speed,
            float(previous_throttle),
        ],
        dtype=np.float32,
    )


def encode_agent_observation(
    mode: str,
    truth: np.ndarray,
    *,
    perception: Optional[Dict[str, Any]] = None,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    orbital: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the policy observation from truth (+ optional perception / orbital).

    Modes:
        ``truth``: Privileged 5-D state (optional isotropic Gaussian noise).
        ``sensors``: Onboard-like scalars — altimeter, vertical rate, speed,
            closing rate proxy, previous throttle. No site distance / pose.
        ``perception``: Camera-pipeline stub features.
        ``orbital``: Relative site vector, velocity, altitude, speed, throttle.

    Args:
        mode: Observation mode name.
        truth: Clean privileged vector from ``pack_truth_vector``.
        perception: Dict from perception stub (for ``perception`` mode).
        noise_std: Gaussian noise std (throttle channel re-clipped).
        rng: RNG for noise.
        orbital: Pre-packed orbital vector (required for ``orbital`` mode).

    Returns:
        ``float32`` policy observation of length ``observation_dim(mode)``.
    """
    mode = validate_obs_mode(mode)
    t = np.asarray(truth, dtype=np.float32).reshape(-1)
    if t.shape[0] < 5:
        raise ValueError(f"truth vector must have length >= 5, got {t.shape}")

    if mode == "orbital":
        if orbital is None:
            raise ValueError("orbital obs_mode requires orbital= vector")
        obs = np.asarray(orbital, dtype=np.float32).reshape(-1)
        if obs.shape[0] != 9:
            raise ValueError(f"orbital vector must have length 9, got {obs.shape}")
        return _maybe_noise(obs, noise_std, rng, clip_throttle_idx=8)

    if mode == "truth":
        obs = t[:5].copy()
        return _maybe_noise(obs, noise_std, rng, clip_throttle_idx=4)

    if mode == "sensors":
        altitude = float(t[0])
        vertical_velocity = float(t[1])
        speed = float(t[3])
        previous_throttle = float(t[4])
        closing_rate = max(0.0, -vertical_velocity)
        obs = np.array(
            [altitude, vertical_velocity, speed, closing_rate, previous_throttle],
            dtype=np.float32,
        )
        return _maybe_noise(obs, noise_std, rng, clip_throttle_idx=4)

    feats = perception_policy_features(perception)
    previous_throttle = float(t[4])
    obs = np.concatenate(
        [feats, np.array([previous_throttle], dtype=np.float32)]
    )
    if noise_std > 0.0:
        generator = rng or np.random.default_rng()
        mask = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        obs = obs + (
            generator.normal(0.0, noise_std, size=obs.shape).astype(np.float32) * mask
        )
        obs[0] = float(np.clip(obs[0], 0.0, 1.0))
        obs[3] = float(np.clip(obs[3], 0.0, 1.0))
        obs[5] = float(np.clip(obs[5], 0.0, 1.0))
    return obs.astype(np.float32)


def _maybe_noise(
    obs: np.ndarray,
    noise_std: float,
    rng: Optional[np.random.Generator],
    *,
    clip_throttle_idx: int,
) -> np.ndarray:
    """Add isotropic Gaussian noise and clip the throttle channel."""
    out = np.asarray(obs, dtype=np.float32).copy()
    if noise_std > 0.0:
        generator = rng or np.random.default_rng()
        out = out + generator.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out[clip_throttle_idx] = float(np.clip(out[clip_throttle_idx], 0.0, 1.0))
    return out
