"""Compatibility shim: ``asteroid_rl.env`` → ``asteroid_rl.environment.gym_env``.

Older Scenic / notebooks imported ``asteroid_rl.env`` before the 2026-08 package
reorg. Keep that path working without duplicating code.
"""

from asteroid_rl.environment.gym_env import *  # noqa: F401,F403
from asteroid_rl.environment import gym_env as _gym_env

__all__ = [name for name in dir(_gym_env) if not name.startswith("_")]
