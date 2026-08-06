"""Gymnasium environment, episode runner, observations, and surface queries.

Prefer submodule imports, e.g.::

    from asteroid_rl.environment.gym_env import AsteroidLandingEnv
"""

__all__ = [
    "AsteroidLandingEnv",
    "LandingEnvConfig",
    "OBS_MODES",
    "default_landing_site",
    "ensure_dirs",
    "get_surface_map",
    "run_episode",
    "write_summary_markdown",
]


def __getattr__(name: str):
    if name in {"AsteroidLandingEnv", "LandingEnvConfig"}:
        from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig

        return {"AsteroidLandingEnv": AsteroidLandingEnv, "LandingEnvConfig": LandingEnvConfig}[
            name
        ]
    if name in {"ensure_dirs", "run_episode", "write_summary_markdown"}:
        from asteroid_rl.environment import episode as _episode

        return getattr(_episode, name)
    if name == "OBS_MODES":
        from asteroid_rl.environment.observations import OBS_MODES

        return OBS_MODES
    if name in {"default_landing_site", "get_surface_map"}:
        from asteroid_rl.environment import surface as _surface

        return getattr(_surface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
