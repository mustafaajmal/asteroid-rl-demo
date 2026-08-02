"""BSK-RL-shaped observation / env adapter (partial, no full BSK-RL dependency).

Exposes dictionary observations closer to a BSK-RL training loop while reusing
``AsteroidLandingEnv`` underneath. This is an API shape adapter — not a full
``bsk_rl`` integration.

When the underlying env uses ``obs_mode="perception"`` / ``"sensors"``, the
Dict space omits privileged site-distance channels so policies cannot cheat.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.observations import validate_obs_mode
from asteroid_rl.perception import perception_feature_vector, perception_policy_features


class BskRlDictObservationEnv(gym.Env):
    """Wrap ``AsteroidLandingEnv`` with a Dict observation space.

    Privileged keys (``altitude``, ``distance``, …) are included only when the
    base env ``obs_mode`` is ``truth``. Sensor / perception modes expose only
    channels consistent with that mode.

    Attributes:
        env: Underlying ``AsteroidLandingEnv``.
        action_space: Same scalar throttle box as the base env.
        observation_space: Gymnasium ``Dict`` space.
        obs_mode: Active observation mode mirrored from config.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[LandingEnvConfig] = None):
        """Create the adapter.

        Args:
            config: Optional config forwarded to ``AsteroidLandingEnv``.
        """
        super().__init__()
        self.env = AsteroidLandingEnv(config=config)
        self.obs_mode = validate_obs_mode(self.env.config.obs_mode)
        self.action_space = self.env.action_space
        self.observation_space = self._make_observation_space()

    def _make_observation_space(self) -> spaces.Dict:
        """Build the Dict space for the active ``obs_mode``.

        Returns:
            Gymnasium ``Dict`` space.
        """
        if self.obs_mode == "perception":
            return spaces.Dict(
                {
                    "perception": spaces.Box(
                        -np.inf, np.inf, shape=(5,), dtype=np.float32
                    ),
                    "target_visible": spaces.Discrete(2),
                    "hazard_score": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
                    "previous_throttle": spaces.Box(
                        0.0, 1.0, shape=(), dtype=np.float32
                    ),
                }
            )
        if self.obs_mode == "sensors":
            return spaces.Dict(
                {
                    "altimeter": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                    "vertical_velocity": spaces.Box(
                        -np.inf, np.inf, shape=(), dtype=np.float32
                    ),
                    "speed": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                    "closing_rate": spaces.Box(
                        -np.inf, np.inf, shape=(), dtype=np.float32
                    ),
                    "previous_throttle": spaces.Box(
                        0.0, 1.0, shape=(), dtype=np.float32
                    ),
                }
            )
        return spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "vertical_velocity": spaces.Box(
                    -np.inf, np.inf, shape=(), dtype=np.float32
                ),
                "distance": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "speed": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "previous_throttle": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
                "perception": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                "target_visible": spaces.Discrete(2),
                "hazard_score": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            }
        )

    def _to_dict(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vector obs + info into the Dict observation.

        Args:
            obs: Base env agent observation for the active mode.
            info: Base env info dict (expects optional ``perception``).

        Returns:
            Dictionary observation for this wrapper.
        """
        perception = info.get("perception")
        if self.obs_mode == "perception":
            feats = perception_policy_features(perception)
            return {
                "perception": feats,
                "target_visible": int(
                    bool(perception and perception.get("target_visible"))
                ),
                "hazard_score": np.float32(
                    float(perception.get("hazard_score", 1.0)) if perception else 1.0
                ),
                "previous_throttle": np.float32(obs[-1]),
            }
        if self.obs_mode == "sensors":
            return {
                "altimeter": np.float32(obs[0]),
                "vertical_velocity": np.float32(obs[1]),
                "speed": np.float32(obs[2]),
                "closing_rate": np.float32(obs[3]),
                "previous_throttle": np.float32(obs[4]),
            }
        feats = perception_feature_vector(perception)
        return {
            "altitude": np.float32(obs[0]),
            "vertical_velocity": np.float32(obs[1]),
            "distance": np.float32(obs[2]),
            "speed": np.float32(obs[3]),
            "previous_throttle": np.float32(obs[4]),
            "perception": feats,
            "target_visible": int(bool(perception and perception.get("target_visible"))),
            "hazard_score": np.float32(
                float(perception.get("hazard_score", 1.0)) if perception else 1.0
            ),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the base env and return a Dict observation.

        Args:
            seed: Optional episode seed.
            options: Forwarded Gymnasium options.

        Returns:
            Tuple ``(dict_obs, info)``.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_dict(obs, info), info

    def step(self, action):
        """Step the base env and return a Dict observation.

        Args:
            action: Throttle action in ``[0, 1]``.

        Returns:
            Gymnasium step tuple with Dict observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_dict(obs, info), reward, terminated, truncated, info

    def render(self):
        """Forward ``render`` to the base env.

        Returns:
            RGB frame or ``None``.
        """
        return self.env.render()

    def close(self) -> None:
        """Close the base env."""
        self.env.close()


def make_bsk_rl_env(config: Optional[LandingEnvConfig] = None) -> BskRlDictObservationEnv:
    """Factory for the BSK-RL-shaped Dict observation env.

    Args:
        config: Optional ``LandingEnvConfig``.

    Returns:
        ``BskRlDictObservationEnv`` instance.
    """
    return BskRlDictObservationEnv(config=config)
