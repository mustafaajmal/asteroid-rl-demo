"""Unit tests for observation packing."""

from __future__ import annotations

import numpy as np
import pytest

from asteroid_rl.environment.observations import (
    encode_agent_observation,
    observation_dim,
    pack_orbital_vector,
    pack_truth_vector,
    validate_obs_mode,
)


def test_observation_dims():
    assert observation_dim("truth") == 5
    assert observation_dim("sensors") == 5
    assert observation_dim("perception") == 6
    assert observation_dim("orbital") == 9


def test_validate_obs_mode_rejects_unknown():
    with pytest.raises(ValueError):
        validate_obs_mode("nope")


def test_pack_and_encode_truth_sensors():
    truth = pack_truth_vector(
        altitude=10.0,
        vertical_velocity=-1.0,
        distance=12.0,
        speed=1.2,
        previous_throttle=0.5,
    )
    assert truth.shape == (5,)
    obs_t = encode_agent_observation("truth", truth)
    assert obs_t.shape == (5,)
    obs_s = encode_agent_observation("sensors", truth)
    assert obs_s.shape == (5,)
    # sensors must not carry privileged site distance as channel 2
    assert abs(float(obs_s[2]) - 1.2) < 1e-6  # speed


def test_pack_orbital_and_encode():
    orb = pack_orbital_vector(
        position_N=[1.0, 2.0, 100.0],
        velocity_N=[0.1, -0.2, -1.0],
        target_N=[0.0, 0.0, -30.0],
        altitude=50.0,
        previous_throttle=0.25,
    )
    assert orb.shape == (9,)
    assert abs(orb[0] - 1.0) < 1e-6
    assert abs(orb[2] - 130.0) < 1e-6
    out = encode_agent_observation("orbital", pack_truth_vector(
        altitude=50.0,
        vertical_velocity=-1.0,
        distance=130.0,
        speed=1.0,
        previous_throttle=0.25,
    ), orbital=orb)
    assert out.shape == (9,)


def test_orbital_encode_requires_vector():
    truth = pack_truth_vector(
        altitude=1.0,
        vertical_velocity=0.0,
        distance=1.0,
        speed=0.0,
        previous_throttle=0.0,
    )
    with pytest.raises(ValueError):
        encode_agent_observation("orbital", truth, orbital=None)
