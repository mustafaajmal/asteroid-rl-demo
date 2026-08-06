"""Fixed-site asteroid landing RL demo package.

Exposes the Gymnasium environment and its configuration for import as::

    from asteroid_rl import AsteroidLandingEnv, LandingEnvConfig

Observation modes (``truth`` / ``sensors`` / ``perception``)::

    from asteroid_rl.environment.observations import OBS_MODES

BSK-RL-shaped Dict observations::

    from asteroid_rl import make_bsk_rl_env

Command-line tools live under ``asteroid_rl.cli`` (play, train, evaluate, etc.).
"""

from asteroid_rl.adapters.bsk_rl_api import BskRlDictObservationEnv, make_bsk_rl_env
from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.environment.observations import OBS_MODES
from asteroid_rl.sensing.vlm import PerceptionBackend

__all__ = [
    "AsteroidLandingEnv",
    "LandingEnvConfig",
    "BskRlDictObservationEnv",
    "make_bsk_rl_env",
    "OBS_MODES",
    "PerceptionBackend",
]
