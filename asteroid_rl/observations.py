"""Agent observation packing: truth vs sensors vs perception.

Reward and termination always use simulator truth. These helpers build the
vector the *policy* is allowed to see under each ``obs_mode``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

OBS_MODES = ("truth", "sensors", "perception")

# Approximate scales for normalizing perception depth into ~[0, 1].
_DEPTH_REF_M = 200.0


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


def perception_policy_features(
    perception: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Pack camera-stub fields into a fixed feature vector for RL.

    Args:
        perception: Dict from ``build_perception_stub``, or ``None``.

    Returns:
        Shape ``(5,)`` ``float32``:
        ``[visible, box_center_u, box_center_v, hazard_score, inv_depth]``.
    """
    if not perception:
        return np.zeros(5, dtype=np.float32)
    box = perception.get("landing_site_box") or [0, 0, 0, 0]
    visible = 1.0 if perception.get("target_visible") else 0.0
    cu = 0.5 * (float(box[0]) + float(box[2]))
    cv = 0.5 * (float(box[1]) + float(box[3]))
    hazard = float(perception.get("hazard_score", 1.0))
    depth = float(perception.get("site_depth_m", 0.0))
    # Inverse depth in ~[0, 1]: nearer → larger (useful landing cue).
    inv_depth = float(np.clip(_DEPTH_REF_M / max(depth, 1.0), 0.0, 1.0))
    if visible < 0.5:
        inv_depth = 0.0
        cu, cv = 0.5, 0.5
    return np.array([visible, cu, cv, hazard, inv_depth], dtype=np.float32)


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


def mode_description(mode: str) -> str:
    """One-line human description of an observation mode.

    Args:
        mode: Observation mode name.

    Returns:
        Short description string.
    """
    mode = validate_obs_mode(mode)
    return {
        "truth": "privileged simulator state (cheat / scaffolding)",
        "sensors": "altimeter + rate-like scalars (no site distance)",
        "perception": "camera-stub features only (VLM schema path)",
    }[mode]
