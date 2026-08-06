"""Gravity models, attitude pointing, and episode start sampling."""

__all__ = [
    "CentralGravity",
    "ConstantGravity",
    "DEFAULT_ASTEROID_COM_N",
    "DEFAULT_MU",
    "orbital_or_default",
    "scenic_like_or_default",
]


def __getattr__(name: str):
    if name in {
        "CentralGravity",
        "ConstantGravity",
        "DEFAULT_ASTEROID_COM_N",
        "DEFAULT_MU",
    }:
        from asteroid_rl.dynamics import gravity as _gravity

        return getattr(_gravity, name)
    if name == "orbital_or_default":
        from asteroid_rl.dynamics.orbit_reset import orbital_or_default

        return orbital_or_default
    if name == "scenic_like_or_default":
        from asteroid_rl.dynamics.scenic_reset import scenic_like_or_default

        return scenic_like_or_default
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
