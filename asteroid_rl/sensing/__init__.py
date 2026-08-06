"""Camera helpers, geometry perception stub, and VLM backend."""

__all__ = [
    "DEFAULT_VLM_MODEL",
    "PerceptionBackend",
]


def __getattr__(name: str):
    if name in {"DEFAULT_VLM_MODEL", "PerceptionBackend"}:
        from asteroid_rl.sensing import vlm as _vlm

        return getattr(_vlm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
