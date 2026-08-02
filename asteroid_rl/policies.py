"""Throttle action helpers used by play / evaluate tooling.

Provides three ways to map an environment observation to a scalar throttle in
``[0, 1]``: a hand-written braking heuristic (``scripted``), uniform random
sampling (``random``), and a loaded Stable-Baselines3 PPO model (``ppo``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def scripted_action(obs) -> np.ndarray:
    """Choose throttle with an altitude-aware braking heuristic.

    Interprets the 5-D observation as altitude above terrain, inertial
    ``v_z``, distance to the surface site, speed, and previous throttle.
    Thrust increases when descending quickly and eases near a soft touchdown.

    Args:
        obs: Environment observation array of shape ``(5,)``. Expected layout is
            ``[altitude, vertical_velocity, distance, speed, previous_throttle]``.

    Returns:
        A length-1 ``float32`` array containing throttle in ``[0, 1]``.
    """
    altitude, vertical_velocity, _distance, speed, _previous_throttle = obs
    descent_rate = -float(vertical_velocity)  # positive when falling

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
    else:
        # Coast for most of the long approach so the camera sees the asteroid.
        throttle = 0.0
    return np.array([throttle], dtype=np.float32)


def random_action(_obs, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample a uniform random throttle in ``[0, 1]``.

    Args:
        _obs: Environment observation (ignored; accepted for a uniform
            ``obs -> action`` call signature).
        rng: Optional NumPy random generator. If ``None``, a fresh default
            generator is created for this call.

    Returns:
        A length-1 ``float32`` array containing throttle in ``[0, 1]``.
    """
    generator = rng or np.random.default_rng()
    return np.array([float(generator.uniform(0.0, 1.0))], dtype=np.float32)


def make_action_fn(
    policy: str,
    *,
    model=None,
    seed: int = 0,
) -> Callable:
    """Build an ``obs -> action`` callable for the named policy.

    Args:
        policy: Policy identifier. One of ``"scripted"``, ``"random"``, or
            ``"ppo"``.
        model: Loaded Stable-Baselines3 PPO model. Required when
            ``policy == "ppo"``; ignored otherwise.
        seed: Random seed used to construct the RNG for ``"random"`` policy.
            Ignored for ``"scripted"`` and ``"ppo"``.

    Returns:
        A callable that maps an observation to a throttle action array.

    Raises:
        ValueError: If ``policy`` is unknown, or if ``policy == "ppo"`` and
            ``model`` is ``None``.
    """
    if policy == "scripted":
        return scripted_action
    if policy == "random":
        rng = np.random.default_rng(seed)
        return lambda obs: random_action(obs, rng=rng)
    if policy == "ppo":
        if model is None:
            raise ValueError("PPO policy requires a loaded model")

        def ppo_action(obs):
            """Predict a deterministic PPO action for one observation.

            Args:
                obs: Environment observation array.

            Returns:
                Throttle action as a 1-D ``float32`` array.
            """
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return ppo_action
    raise ValueError(f"Unknown policy: {policy}")
