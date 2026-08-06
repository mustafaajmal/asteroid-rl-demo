"""Multi-phase mission FSM for autonomous asteroid landing (planning doc).

Modes:
  ``search``   — visible but hazard too high; coast / light thrust
  ``acquire``  — asteroid not in frame; slew to bring body into view
  ``divert``   — committed approach toward the fixed pad
  ``upright``  — near-pad cone; point local-up and soft-brake
  ``land``     — legacy alias for committed soft-land (throttle uncapped)

Isolation rule: success/reward still use the configured fixed pad. Candidate
sites from perception nudge mission memory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from asteroid_rl.pointing import local_up_N, unit

MISSION_MODES = ("search", "acquire", "divert", "upright", "land")


@dataclass
class MissionState:
    """Mutable mission memory across control steps.

    Attributes:
        mode: One of ``MISSION_MODES``.
        best_hazard: Lowest hazard observed so far.
        candidate_site_N: Best candidate inertial site, or ``None``.
        steps_in_search: Steps spent in search/acquire before divert.
        committed: True once divert/upright/land has been entered.
    """

    mode: str = "search"
    best_hazard: float = 1.0
    candidate_site_N: Optional[np.ndarray] = None
    steps_in_search: int = 0
    committed: bool = False


@dataclass
class MissionConfig:
    """Tunables for the autonomous mission FSM.

    Attributes:
        enabled: Master switch (False → force ``land``).
        hazard_commit_threshold: Commit divert when hazard ≤ this (PDF: 0.10).
        max_search_steps: Force divert after this many search/acquire steps.
        search_min_altitude_m: Force divert below this altitude.
        upright_altitude_m: Enter upright mode below this altitude.
        upright_lateral_m: Enter upright mode inside this lateral miss.
        asteroid_com_N: COM used for acquire LOS / local-up.
    """

    enabled: bool = False
    hazard_commit_threshold: float = 0.10
    max_search_steps: int = 240
    search_min_altitude_m: float = 25.0
    upright_altitude_m: float = 70.0
    upright_lateral_m: float = 35.0
    # Auto-commit divert when already on a near-pad approach.
    divert_force_altitude_m: float = 120.0
    divert_force_lateral_m: float = 90.0
    asteroid_com_N: Tuple[float, float, float] = (0.0, 0.0, -150.0)


def _nudge_candidate(
    *,
    perception: Dict[str, Any],
    position_N: np.ndarray,
    target_N: np.ndarray,
    visible: bool,
) -> np.ndarray:
    """Nudge a candidate site from perception bbox (does not retarget success)."""
    box = perception.get("landing_site_box") or [0, 0, 0, 0]
    cu = 0.5 * (float(box[0]) + float(box[2]))
    cv = 0.5 * (float(box[1]) + float(box[3]))
    site = np.asarray(target_N, dtype=np.float64).reshape(3).copy()
    if visible:
        site[0] = float(position_N[0]) + (cu - 0.5) * 40.0
        site[1] = float(position_N[1]) + (cv - 0.5) * 40.0
    return site


def update_mission(
    state: MissionState,
    *,
    perception: Dict[str, Any],
    position_N: np.ndarray,
    target_N: np.ndarray,
    altitude_m: float,
    config: MissionConfig,
    lateral_m: Optional[float] = None,
) -> MissionState:
    """Advance the mission FSM from perception + kinematics.

    Args:
        state: Previous mission state.
        perception: Geometry or VLM JSON dict.
        position_N: Hub inertial position.
        target_N: Fixed landing pad (success target).
        altitude_m: Altitude above terrain/pad.
        config: Mission tunables.
        lateral_m: Optional lateral miss to pad (computed if None).

    Returns:
        Updated ``MissionState``.
    """
    if not config.enabled:
        state.mode = "land"
        state.committed = True
        return state

    visible = bool(perception.get("target_visible", False))
    hazard = float(perception.get("hazard_score", 1.0))
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    pos = np.asarray(position_N, dtype=np.float64).reshape(3)
    if lateral_m is None:
        lateral_m = float(np.linalg.norm(pos[:2] - target[:2]))

    if hazard < state.best_hazard:
        state.best_hazard = hazard
        state.candidate_site_N = _nudge_candidate(
            perception=perception,
            position_N=pos,
            target_N=target,
            visible=visible,
        )

    # Near-pad cone → upright soft-land (once committed or forced low).
    in_upright_cone = (
        float(altitude_m) < float(config.upright_altitude_m)
        and float(lateral_m) <= float(config.upright_lateral_m)
    )

    if state.committed or state.mode in ("divert", "upright", "land"):
        state.committed = True
        if in_upright_cone:
            state.mode = "upright"
        else:
            state.mode = "divert"
        return state

    # Still searching / acquiring.
    state.steps_in_search += 1
    near_approach = (
        float(altitude_m) < float(config.divert_force_altitude_m)
        and float(lateral_m) <= float(config.divert_force_lateral_m)
    )
    force_divert = (
        near_approach
        or state.steps_in_search >= int(config.max_search_steps)
        or float(altitude_m) < float(config.search_min_altitude_m)
        or (visible and hazard <= float(config.hazard_commit_threshold))
    )
    if force_divert:
        state.committed = True
        state.mode = "upright" if in_upright_cone else "divert"
        return state

    if not visible:
        state.mode = "acquire"
    else:
        state.mode = "search"
    return state


def mission_pointing_command(
    state: MissionState,
    *,
    position_N: Sequence[float],
    target_N: Sequence[float],
    config: MissionConfig,
) -> np.ndarray:
    """Suggest inertial boresight direction for the current mission mode.

    Args:
        state: Current mission state.
        position_N: Hub position.
        target_N: Fixed pad.
        config: Mission config (COM for local-up / acquire).

    Returns:
        Shape ``(3,)`` unit inertial direction for body −z.
    """
    pos = np.asarray(position_N, dtype=np.float64).reshape(3)
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    com = np.asarray(config.asteroid_com_N, dtype=np.float64).reshape(3)
    mode = str(state.mode)

    if mode in ("upright", "land"):
        # Look at the ground (−local_up) so +z thruster fires along local_up.
        return unit(-local_up_N(pos, com))
    if mode == "acquire":
        # Point at asteroid COM to bring the body into frame.
        return unit(com - pos)
    # search / divert: look at the pad (or candidate for camera framing).
    aim = state.candidate_site_N if state.candidate_site_N is not None else target
    return unit(np.asarray(aim, dtype=np.float64).reshape(3) - pos)


def mission_throttle_gate(
    throttle: float,
    *,
    state: MissionState,
    altitude_m: float,
    config: MissionConfig,
) -> Tuple[float, str]:
    """Gate throttle by mission mode.

    Args:
        throttle: Proposed throttle in ``[0, 1]``.
        state: Current mission state.
        altitude_m: Current altitude.
        config: Mission tunables.

    Returns:
        Tuple ``(gated_throttle, reason)``.
    """
    if not config.enabled:
        return float(np.clip(throttle, 0.0, 1.0)), "land"
    mode = str(state.mode)
    if mode in ("divert", "upright", "land"):
        return float(np.clip(throttle, 0.0, 1.0)), mode
    if altitude_m < config.search_min_altitude_m:
        return float(np.clip(throttle, 0.0, 1.0)), "force_land_low"
    # search / acquire: light thrust only.
    return float(np.clip(min(throttle, 0.25), 0.0, 1.0)), mode
