"""External API adapters (BSK-RL-shaped Dict observations, etc.)."""

__all__ = [
    "BskRlDictObservationEnv",
    "make_bsk_rl_env",
]


def __getattr__(name: str):
    if name in {"BskRlDictObservationEnv", "make_bsk_rl_env"}:
        from asteroid_rl.adapters import bsk_rl_api as _api

        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
