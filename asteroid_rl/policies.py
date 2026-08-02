"""Throttle action helpers used by play / evaluate tooling.

Provides scripted / random / PPO mappings from observation (+ optional info with
perception stub) to scalar throttle in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from asteroid_rl.perception import perception_feature_vector


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
        # Legacy fallback: assume privileged truth-shaped observation.
        altitude = float(obs[0])
        vertical_velocity = float(obs[1])
        speed = float(obs[3])
    descent_rate = -float(vertical_velocity)  # positive when falling
    perception = info.get("perception")
    feats = perception_feature_vector(perception)
    visible, _cu, _cv, hazard = [float(x) for x in feats]

    # Always brake if closing too fast — even when the site is briefly out of frame.
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
        # Site not in frame and still high: coast while pointing recovers FOV.
        throttle = 0.0
    else:
        throttle = 0.0

    if hazard > 0.45 and altitude > 15.0:
        throttle = min(1.0, throttle + 0.15)
    if hazard < 0.10 and visible >= 0.5 and altitude < 20.0:
        # Plan: commit when hazard is low and site is visible.
        throttle = max(throttle, 0.35 if descent_rate > 0.4 else throttle)

    return np.array([float(np.clip(throttle, 0.0, 1.0))], dtype=np.float32)


def random_action(
    _obs,
    rng: Optional[np.random.Generator] = None,
    info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sample a uniform random throttle in ``[0, 1]``.

    Args:
        _obs: Environment observation (ignored; accepted for a uniform
            ``obs -> action`` call signature).
        rng: Optional NumPy random generator. If ``None``, a fresh default
            generator is created for this call.
        info: Optional env info (ignored; accepted for call-signature parity).

    Returns:
        A length-1 ``float32`` array containing throttle in ``[0, 1]``.
    """
    del info
    generator = rng or np.random.default_rng()
    return np.array([float(generator.uniform(0.0, 1.0))], dtype=np.float32)


def make_action_fn(
    policy: str,
    *,
    model=None,
    seed: int = 0,
) -> Callable:
    """Build an ``(obs, info=None) -> action`` callable for the named policy.

    Args:
        policy: Policy identifier. One of ``"scripted"``, ``"random"``, or
            ``"ppo"``.
        model: Loaded Stable-Baselines3 PPO model. Required when
            ``policy == "ppo"``; ignored otherwise.
        seed: Random seed used to construct the RNG for ``"random"`` policy.
            Ignored for ``"scripted"`` and ``"ppo"``.

    Returns:
        A callable that maps ``(obs, info=None)`` to a throttle action array.

    Raises:
        ValueError: If ``policy`` is unknown, or if ``policy == "ppo"`` and
            ``model`` is ``None``.
    """
    if policy == "scripted":
        return scripted_action
    if policy == "random":
        rng = np.random.default_rng(seed)

        def _random(obs, info=None):
            """Random throttle wrapper with info-compatible signature.

            Args:
                obs: Environment observation.
                info: Optional info dict (ignored).

            Returns:
                Throttle action array.
            """
            return random_action(obs, rng=rng, info=info)

        return _random
    if policy == "ppo":
        if model is None:
            raise ValueError("PPO policy requires a loaded model")

        def ppo_action(obs, info=None):
            """Predict a deterministic PPO action for one observation.

            Args:
                obs: Environment observation array.
                info: Optional info dict (ignored by the vector policy).

            Returns:
                Throttle action as a 1-D ``float32`` array.
            """
            del info
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return ppo_action
    raise ValueError(f"Unknown policy: {policy}")
