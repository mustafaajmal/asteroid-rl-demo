"""Simulation build helpers re-exported for a clearer package layout.

The Basilisk/MuJoCo construction currently lives in ``asteroid_rl.env`` (shared
with the Gym wrapper). Import from here when you only need the sim factory:

    from asteroid_rl.sim import build_sim, LandingEnvConfig
"""

from __future__ import annotations

from asteroid_rl.env import (
    AST_OBJ_PATH,
    DEFAULT_INITIAL_POSITION,
    DEFAULT_INITIAL_VELOCITY,
    DEFAULT_TARGET,
    SIM_DT,
    SPACECRAFT_BODY_NAME,
    XML_PATH,
    ConstantGravity,
    LandingEnvConfig,
    SimHandles,
    ThrusterVizMessageWriter,
    build_sim,
)

__all__ = [
    "AST_OBJ_PATH",
    "DEFAULT_INITIAL_POSITION",
    "DEFAULT_INITIAL_VELOCITY",
    "DEFAULT_TARGET",
    "SIM_DT",
    "SPACECRAFT_BODY_NAME",
    "XML_PATH",
    "ConstantGravity",
    "LandingEnvConfig",
    "SimHandles",
    "ThrusterVizMessageWriter",
    "build_sim",
]
