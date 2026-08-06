"""Scripted / learned policies, mission FSM, and BC warm-start."""

__all__ = [
    "MissionConfig",
    "MissionState",
    "make_action_fn",
]


def __getattr__(name: str):
    if name in {"MissionConfig", "MissionState"}:
        from asteroid_rl.control import mission as _mission

        return getattr(_mission, name)
    if name == "make_action_fn":
        from asteroid_rl.control.policies import make_action_fn

        return make_action_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
