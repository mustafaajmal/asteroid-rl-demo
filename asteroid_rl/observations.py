"""Agent observation packing: truth vs sensors vs perception.

Reward and termination always use simulator truth. These helpers build the
vector the *policy* is allowed to see under each ``obs_mode``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from asteroid_rl.perception import perception_policy_features

OBS_MODES = ("truth", "sensors", "perception")


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
        mode: One of ``truth``, ``sensors``, ``perception``.

    Returns:
        Observation dimensionality.
    """
    mode = validate_obs_mode(mode)
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


def encode_agent_observation(
    mode: str,
    truth: np.ndarray,
    *,
    perception: Optional[Dict[str, Any]] = None,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build the policy observation from truth (+ optional perception).

    Modes:
        ``truth``: Privileged 5-D state (optional isotropic Gaussian noise).
        ``sensors``: Onboard-like scalars — altimeter, vertical rate, speed,
            closing rate proxy, previous throttle. No site distance / pose.
        ``perception``: Camera-pipeline stub features — visibility, site
            image center, hazard, normalized depth, previous throttle.

    Args:
        mode: Observation mode name.
        truth: Clean privileged vector from ``pack_truth_vector``.
        perception: Dict from ``build_perception_stub`` (required for
            ``perception`` mode; ignored otherwise).
        noise_std: Gaussian noise std for ``truth`` / ``sensors`` channels
            (throttle channel is re-clipped to ``[0, 1]``).
        rng: RNG for noise; required when ``noise_std > 0``.

    Returns:
        ``float32`` policy observation of length ``observation_dim(mode)``.
    """
    mode = validate_obs_mode(mode)
    t = np.asarray(truth, dtype=np.float32).reshape(-1)
    if t.shape[0] < 5:
        raise ValueError(f"truth vector must have length >= 5, got {t.shape}")

    if mode == "truth":
        obs = t[:5].copy()
        return _maybe_noise(obs, noise_std, rng, clip_throttle_idx=4)

    if mode == "sensors":
        # Altimeter + IMU-ish rates. Closing-rate proxy uses truth vertical
        # rate only (no lateral navigation / site range).
        altitude = float(t[0])
        vertical_velocity = float(t[1])
        speed = float(t[3])
        previous_throttle = float(t[4])
        closing_rate = max(0.0, -vertical_velocity)  # positive when descending
        obs = np.array(
            [altitude, vertical_velocity, speed, closing_rate, previous_throttle],
            dtype=np.float32,
        )
        return _maybe_noise(obs, noise_std, rng, clip_throttle_idx=4)

    # perception mode — no privileged range-to-site / altitude channel
    feats = perception_policy_features(perception)
    previous_throttle = float(t[4])
    obs = np.concatenate(
        [feats, np.array([previous_throttle], dtype=np.float32)]
    )
    # Light optional noise on continuous perception channels (not visibility bit).
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
    """Add isotropic Gaussian noise and clip the throttle channel.

    Args:
        obs: Observation vector.
        noise_std: Noise standard deviation (0 disables).
        rng: Random generator.
        clip_throttle_idx: Index of the previous-throttle channel.

    Returns:
        Noisy ``float32`` observation.
    """
    out = np.asarray(obs, dtype=np.float32).copy()
    if noise_std > 0.0:
        generator = rng or np.random.default_rng()
        out = out + generator.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out[clip_throttle_idx] = float(np.clip(out[clip_throttle_idx], 0.0, 1.0))
    return out
