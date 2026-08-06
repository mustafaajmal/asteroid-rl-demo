"""Scenic-*like* randomized initial states (no Scenic dependency required).

Matches the planning-document intent: satellite and asteroid start within
mutual visibility range, but the satellite may not point at the body and
approach velocity may vary. Full Scenic scenario graphs can replace this
sampler later without changing the Gym API.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Keep independent of env.py to avoid import cycles.
_DEFAULT_POSITION = (0.0, 0.0, 120.0)
_DEFAULT_VELOCITY = (0.0, 0.0, -1.5)


def sample_scenic_like_start(
    target_N: np.ndarray,
    rng: np.random.Generator,
    *,
    min_range_m: float = 80.0,
    max_range_m: float = 160.0,
    max_lateral_frac: float = 0.35,
    speed_min: float = 0.8,
    speed_max: float = 2.2,
    miss_pointing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a randomized approach start near the landing site.

    Args:
        target_N: Landing-site inertial position, meters.
        rng: NumPy random generator.
        min_range_m: Minimum start distance from the site, meters.
        max_range_m: Maximum start distance from the site, meters.
        max_lateral_frac: Lateral offset as a fraction of range.
        speed_min: Minimum approach-speed magnitude, m/s.
        speed_max: Maximum approach-speed magnitude, m/s.
        miss_pointing: If True, return a random MRP so the camera may not
            face the site (caller applies attitude).

    Returns:
        Tuple ``(position_N, velocity_N, sigma_BN)``.
    """
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    # Nominal approach is +z above the site (matches default demo geometry).
    range_m = float(rng.uniform(min_range_m, max_range_m))
    lateral = max_lateral_frac * range_m
    offset = np.array(
        [
            float(rng.uniform(-lateral, lateral)),
            float(rng.uniform(-lateral, lateral)),
            range_m,
        ],
        dtype=np.float64,
    )
    position = target + offset

    # Velocity mostly toward the site with some dispersion.
    to_site = target - position
    to_site = to_site / max(float(np.linalg.norm(to_site)), 1e-6)
    speed = float(rng.uniform(speed_min, speed_max))
    jitter = rng.normal(0.0, 0.15, size=3)
    direction = to_site + jitter
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    velocity = direction * speed

    if miss_pointing:
        sigma = rng.uniform(-0.4, 0.4, size=3).astype(np.float64)
    else:
        sigma = np.zeros(3, dtype=np.float64)
    return position, velocity, sigma


def scenic_like_or_default(
    target_N: np.ndarray,
    rng: Optional[np.random.Generator],
    *,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return scenic-like start or the fixed default approach.

    Args:
        target_N: Landing site.
        rng: RNG (required when ``enabled``).
        enabled: If False, return defaults and ``sigma=None`` (keep auto_point).

    Returns:
        Tuple ``(position, velocity, sigma_or_none)``.
    """
    if not enabled:
        return (
            np.array(_DEFAULT_POSITION, dtype=np.float64),
            np.array(_DEFAULT_VELOCITY, dtype=np.float64),
            None,
        )
    if rng is None:
        rng = np.random.default_rng()
    pos, vel, sigma = sample_scenic_like_start(target_N, rng)
    return pos, vel, sigma
