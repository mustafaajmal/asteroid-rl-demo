"""Hazard-gated search-then-land mission logic (planning-document §3).

While hazard is above the commit threshold (default 0.10), the lander stays in
``search`` mode: prefer coasting / holding altitude rather than committing to
touchdown. Once a low-hazard visible site is seen, mode becomes ``land`` and
normal braking is allowed. Candidate sites can be nudged from the perception
bbox center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MissionState:
    """Mutable mission memory across control steps.

    Attributes:
        mode: ``search`` or ``land``.
        best_hazard: Lowest hazard observed so far.
        candidate_site_N: Best candidate inertial site, or ``None``.
        steps_in_search: Steps spent searching.
    """

    mode: str = "search"
    best_hazard: float = 1.0
    candidate_site_N: Optional[np.ndarray] = None
    steps_in_search: int = 0


@dataclass
class MissionConfig:
    """Tunables for search-then-land.

    Attributes:
        enabled: Master switch.
        hazard_commit_threshold: Commit to land when hazard is below this
            (planning doc example: 0.10).
        max_search_steps: After this many search steps, commit to the best
            hazard seen so far (even if above threshold).
        search_min_altitude_m: Do not force search braking below this altitude.
    """

    enabled: bool = False
    hazard_commit_threshold: float = 0.10
    max_search_steps: int = 240
    search_min_altitude_m: float = 25.0


def update_mission(
    state: MissionState,
    *,
    perception: Dict[str, Any],
    position_N: np.ndarray,
    target_N: np.ndarray,
    altitude_m: float,
    config: MissionConfig,
) -> MissionState:
    """Update search/land mode from the latest perception stub.

    Args:
        state: Previous mission state.
        perception: Perception dict (geometry or VLM).
        position_N: Current hub position.
        target_N: Nominal configured landing site.
        altitude_m: Current altitude above terrain.
        config: Mission tunables.

    Returns:
        Updated ``MissionState``.
    """
    if not config.enabled:
        state.mode = "land"
        return state

    visible = bool(perception.get("target_visible", False))
    hazard = float(perception.get("hazard_score", 1.0))
    state.steps_in_search += 1

    if hazard < state.best_hazard:
        state.best_hazard = hazard
        # Nudge candidate laterally from bbox center vs image center.
        box = perception.get("landing_site_box") or [0, 0, 0, 0]
        cu = 0.5 * (float(box[0]) + float(box[2]))
        cv = 0.5 * (float(box[1]) + float(box[3]))
        site = np.asarray(target_N, dtype=np.float64).reshape(3).copy()
        if visible:
            site[0] = float(position_N[0]) + (cu - 0.5) * 40.0
            site[1] = float(position_N[1]) + (cv - 0.5) * 40.0
        state.candidate_site_N = site

    if state.mode == "search":
        if visible and hazard <= config.hazard_commit_threshold:
            state.mode = "land"
        elif state.steps_in_search >= config.max_search_steps:
            state.mode = "land"
        elif altitude_m < config.search_min_altitude_m:
            # Too low to keep searching — commit to best site.
            state.mode = "land"
    return state


def mission_throttle_gate(
    throttle: float,
    *,
    state: MissionState,
    altitude_m: float,
    config: MissionConfig,
) -> Tuple[float, str]:
    """Optionally suppress aggressive braking while still searching.

    Args:
        throttle: Proposed throttle in ``[0, 1]``.
        state: Current mission state.
        altitude_m: Current altitude.
        config: Mission tunables.

    Returns:
        Tuple ``(gated_throttle, reason)``.
    """
    if not config.enabled or state.mode == "land":
        return float(np.clip(throttle, 0.0, 1.0)), "land"
    if altitude_m < config.search_min_altitude_m:
        return float(np.clip(throttle, 0.0, 1.0)), "force_land_low"
    # Search: allow light braking only to avoid runaway speed.
    return float(np.clip(min(throttle, 0.25), 0.0, 1.0)), "search"
