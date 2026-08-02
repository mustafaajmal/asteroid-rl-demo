"""Throttle action helpers used by play / evaluate tooling.

Provides three ways to map an environment observation to a scalar throttle in
``[0, 1]``: a hand-written braking heuristic (``scripted``), uniform random
sampling (``random``), and a loaded Stable-Baselines3 PPO model (``ppo``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def scripted_action(obs) -> np.ndarray:
    """Choose throttle with a simple braking heuristic.

    Interprets the 5-D observation as altitude proxy, radial vertical velocity,
    distance to target, speed, and previous throttle. Thrust is increased when
    closing speed is high and eased when nearer/slower.

    Args:
        obs: Environment observation array of shape ``(5,)``. Expected layout is
            ``[altitude, vertical_velocity, distance, speed, previous_throttle]``.

    Returns:
        A length-1 ``float32`` array containing throttle in ``[0, 1]``.
    """
    _altitude, vertical_velocity, distance, speed, _previous_throttle = obs
    if vertical_velocity < -1.0:
        throttle = 1.0
    elif vertical_velocity < -0.5:
        throttle = 0.75
    elif distance < 20.0 and speed > 0.75:
        throttle = 0.65
    else:
        throttle = 0.25
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
