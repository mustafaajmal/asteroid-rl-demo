"""Fixed-site asteroid landing RL demo package.

Exposes the Gymnasium environment and its configuration for import as::

    from asteroid_rl import AsteroidLandingEnv, LandingEnvConfig

Command-line tools live under ``asteroid_rl.cli`` (play, train, evaluate, etc.).
"""

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig

__all__ = ["AsteroidLandingEnv", "LandingEnvConfig"]
