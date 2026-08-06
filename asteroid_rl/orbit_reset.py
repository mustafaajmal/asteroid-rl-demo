"""Asteroid-centered elliptical orbit initial-state sampler.

Produces Keplerian ``r, v`` about the asteroid COM so episodes can start on a
true ellipse under ``CentralGravity`` (not the Phase-1 constant-force model).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from asteroid_rl.gravity import DEFAULT_ASTEROID_COM_N, DEFAULT_MU


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle_rad``."""
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = a
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def elements_to_rv(
    *,
    a: float,
    e: float,
    inclination_rad: float,
    raan_rad: float,
    arg_periapsis_rad: float,
    true_anomaly_rad: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert classical orbital elements to COM-relative ``r, v``.

    Args:
        a: Semi-major axis, meters.
        e: Eccentricity in ``[0, 1)``.
        inclination_rad: Inclination, radians.
        raan_rad: Right ascension of ascending node, radians.
        arg_periapsis_rad: Argument of periapsis, radians.
        true_anomaly_rad: True anomaly, radians.
        mu: Gravitational parameter, m^3/s^2.

    Returns:
        Tuple ``(r_rel, v_rel)`` in the asteroid-centered inertial frame.
    """
    e = float(np.clip(e, 0.0, 0.95))
    p = a * (1.0 - e * e)
    cnu = float(np.cos(true_anomaly_rad))
    snu = float(np.sin(true_anomaly_rad))
    r_pqw = np.array([p * cnu / (1.0 + e * cnu), p * snu / (1.0 + e * cnu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    r1 = _rotation_matrix(np.array([0.0, 0.0, 1.0]), arg_periapsis_rad)
    r2 = _rotation_matrix(np.array([1.0, 0.0, 0.0]), inclination_rad)
    r3 = _rotation_matrix(np.array([0.0, 0.0, 1.0]), raan_rad)
    rot = r3 @ r2 @ r1
    return rot @ r_pqw, rot @ v_pqw


def sample_elliptical_start(
    rng: np.random.Generator,
    *,
    mu: float = DEFAULT_MU,
    com_N: Sequence[float] = DEFAULT_ASTEROID_COM_N,
    a_min_m: float = 220.0,
    a_max_m: float = 300.0,
    e_min: float = 0.05,
    e_max: float = 0.25,
    periapsis_floor_m: float = 175.0,
    inclination_max_deg: float = 35.0,
    miss_pointing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an elliptical orbit start about the asteroid COM.

    Rejects samples whose periapsis ``a(1-e)`` is below ``periapsis_floor_m``
    so the craft does not lithobrake on the first orbit.

    Args:
        rng: NumPy random generator.
        mu: Gravitational parameter matching ``CentralGravity``.
        com_N: Asteroid COM inertial position, meters.
        a_min_m: Minimum semi-major axis, meters.
        a_max_m: Maximum semi-major axis, meters.
        e_min: Minimum eccentricity.
        e_max: Maximum eccentricity.
        periapsis_floor_m: Minimum allowed periapsis radius from COM, meters.
        inclination_max_deg: Max ``|i|`` sampled uniformly, degrees.
        miss_pointing: If True, return a random initial MRP.

    Returns:
        Tuple ``(position_N, velocity_N, sigma_BN)`` in the inertial frame.
    """
    com = np.asarray(com_N, dtype=np.float64).reshape(3)
    for _ in range(64):
        a = float(rng.uniform(a_min_m, a_max_m))
        e = float(rng.uniform(e_min, e_max))
        if a * (1.0 - e) < periapsis_floor_m:
            e = max(0.0, 1.0 - periapsis_floor_m / a)
        i = float(np.deg2rad(rng.uniform(-inclination_max_deg, inclination_max_deg)))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        argp = float(rng.uniform(0.0, 2.0 * np.pi))
        nu = float(rng.uniform(0.0, 2.0 * np.pi))
        r_rel, v_rel = elements_to_rv(
            a=a,
            e=e,
            inclination_rad=i,
            raan_rad=raan,
            arg_periapsis_rad=argp,
            true_anomaly_rad=nu,
            mu=mu,
        )
        if float(np.linalg.norm(r_rel)) >= periapsis_floor_m * 0.98:
            break
    else:
        # Fallback near-circular equatorial.
        r_rel, v_rel = elements_to_rv(
            a=0.5 * (a_min_m + a_max_m),
            e=0.05,
            inclination_rad=0.0,
            raan_rad=0.0,
            arg_periapsis_rad=0.0,
            true_anomaly_rad=0.5 * np.pi,
            mu=mu,
        )

    position = com + r_rel
    velocity = v_rel
    if miss_pointing:
        sigma = rng.uniform(-0.5, 0.5, size=3).astype(np.float64)
    else:
        sigma = np.zeros(3, dtype=np.float64)
    return position, velocity, sigma


def sample_approach_start(
    rng: np.random.Generator,
    *,
    target_N: Sequence[float],
    range_min_m: float = 45.0,
    range_max_m: float = 90.0,
    closing_speed_min: float = 0.3,
    closing_speed_max: float = 0.9,
    miss_pointing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a near-pad inbound approach (curriculum / BC-friendly).

    Places the craft on a random ray from the fixed landing site with a mostly
    inbound velocity. Harder than Phase-1 hover but much easier than a full
    ellipse — used to teach divert + soft-land before full orbital starts.

    Args:
        rng: NumPy random generator.
        target_N: Fixed landing site inertial position, meters.
        range_min_m: Minimum initial range to site, meters.
        range_max_m: Maximum initial range to site, meters.
        closing_speed_min: Min inbound speed magnitude, m/s.
        closing_speed_max: Max inbound speed magnitude, m/s.
        miss_pointing: If True, return a random initial MRP.

    Returns:
        Tuple ``(position_N, velocity_N, sigma_BN)``.
    """
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    direction = rng.normal(0.0, 1.0, size=3)
    # Bias "above" the pad (+z) so altitude stays positive on flat/mesh.
    direction[2] = abs(float(direction[2])) + 0.8
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    range_m = float(rng.uniform(range_min_m, range_max_m))
    position = target + direction * range_m
    closing = float(rng.uniform(closing_speed_min, closing_speed_max))
    # Mostly inbound, small lateral jitter so divert must correct.
    jitter = rng.normal(0.0, 0.12, size=3)
    velocity = -closing * direction + jitter
    if miss_pointing:
        sigma = rng.uniform(-0.4, 0.4, size=3).astype(np.float64)
    else:
        sigma = np.zeros(3, dtype=np.float64)
    return position, velocity, sigma


def orbital_or_default(
    rng: Optional[np.random.Generator],
    *,
    enabled: bool,
    mu: float = DEFAULT_MU,
    com_N: Sequence[float] = DEFAULT_ASTEROID_COM_N,
    start_mode: str = "ellipse",
    target_N: Optional[Sequence[float]] = None,
    approach_prob: float = 0.55,
    scenic_prob: float = 0.25,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return an orbital/approach/scenic start when enabled, else nulls.

    Args:
        rng: Random generator (created if ``None`` and enabled).
        enabled: If False, returns nulls so the caller uses Phase-1 defaults.
        mu: Gravitational parameter.
        com_N: Asteroid COM.
        start_mode: ``ellipse`` | ``approach`` | ``mixed`` | ``autonomous``.
        target_N: Landing site (required for approach / mixed / autonomous).
        approach_prob: Probability of approach starts when ``start_mode=mixed``.
        scenic_prob: Extra probability of scenic-like starts in ``autonomous``.
        **kwargs: Forwarded to ``sample_elliptical_start``.

    Returns:
        ``(position, velocity, sigma)`` or ``(None, None, None)``.
    """
    if not enabled:
        return None, None, None
    generator = rng or np.random.default_rng()
    mode = str(start_mode).lower()

    if mode == "autonomous" and target_N is not None:
        u = float(generator.random())
        # ~45% approach, ~25% scenic-like miss-pointing, rest ellipse.
        ap = float(np.clip(approach_prob, 0.0, 0.9))
        sp = float(np.clip(scenic_prob, 0.0, 1.0 - ap))
        if u < ap:
            return sample_approach_start(generator, target_N=target_N)
        if u < ap + sp:
            from asteroid_rl.scenic_reset import sample_scenic_like_start

            return sample_scenic_like_start(
                np.asarray(target_N, dtype=np.float64),
                generator,
                miss_pointing=True,
            )
        pos, vel, sigma = sample_elliptical_start(
            generator, mu=mu, com_N=com_N, **kwargs
        )
        return pos, vel, sigma

    use_approach = mode == "approach" or (
        mode == "mixed"
        and target_N is not None
        and float(generator.random()) < float(approach_prob)
    )
    if use_approach and target_N is not None:
        pos, vel, sigma = sample_approach_start(generator, target_N=target_N)
        return pos, vel, sigma
    pos, vel, sigma = sample_elliptical_start(
        generator, mu=mu, com_N=com_N, **kwargs
    )
    return pos, vel, sigma
