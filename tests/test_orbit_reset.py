"""Unit tests for Keplerian orbit sampling (no Basilisk required)."""

from __future__ import annotations

import numpy as np

from asteroid_rl.gravity import DEFAULT_ASTEROID_COM_N, DEFAULT_MU
from asteroid_rl.orbit_reset import elements_to_rv, sample_elliptical_start


def test_elements_circular_equatorial_energy():
    mu = DEFAULT_MU
    a = 250.0
    r, v = elements_to_rv(
        a=a,
        e=0.0,
        inclination_rad=0.0,
        raan_rad=0.0,
        arg_periapsis_rad=0.0,
        true_anomaly_rad=0.0,
        mu=mu,
    )
    assert abs(float(np.linalg.norm(r)) - a) < 1e-6
    # Circular: |v| = sqrt(mu/a)
    assert abs(float(np.linalg.norm(v)) - np.sqrt(mu / a)) < 1e-5
    # Specific energy ≈ -mu/(2a)
    energy = 0.5 * float(np.dot(v, v)) - mu / float(np.linalg.norm(r))
    assert abs(energy + mu / (2.0 * a)) < 1e-4


def test_sample_elliptical_respects_periapsis_floor():
    rng = np.random.default_rng(0)
    floor = 175.0
    com = np.asarray(DEFAULT_ASTEROID_COM_N, dtype=np.float64)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pos, vel, sigma = sample_elliptical_start(
            rng,
            mu=DEFAULT_MU,
            com_N=com,
            periapsis_floor_m=floor,
            miss_pointing=True,
        )
        r_rel = pos - com
        assert float(np.linalg.norm(r_rel)) >= floor * 0.95
        assert sigma.shape == (3,)
        assert vel.shape == (3,)


def test_sample_approach_is_inbound_near_site():
    from asteroid_rl.orbit_reset import sample_approach_start

    rng = np.random.default_rng(3)
    target = np.array([0.0, 0.0, -20.0])
    pos, vel, sigma = sample_approach_start(rng, target_N=target)
    rel = pos - target
    assert 45.0 <= float(np.linalg.norm(rel)) <= 90.0
    # Velocity should have a component toward the site.
    assert float(np.dot(vel, rel)) < 0.0
    assert sigma.shape == (3,)


def test_orbital_or_default_autonomous_can_emit_scenic():
    from asteroid_rl.orbit_reset import orbital_or_default

    rng = np.random.default_rng(7)
    target = np.array([0.0, 0.0, -30.0])
    saw = False
    for _ in range(40):
        pos, vel, sigma = orbital_or_default(
            rng,
            enabled=True,
            start_mode="autonomous",
            target_N=target,
            approach_prob=0.2,
            scenic_prob=0.8,
        )
        # Scenic-like starts sit ~80-160m above pad with +z bias.
        if 70.0 <= float(pos[2] - target[2]) <= 170.0:
            saw = True
            break
    assert saw
