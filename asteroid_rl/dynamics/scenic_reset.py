"""Scenic-*like* and real Scenic initial-state sampling.

``sample_scenic_like_start`` needs no Scenic install (PDF-style random starts).
``sample_scenic_scenario_start`` loads a ``.scenic`` file, calls ``generate()``,
and returns craft pose/velocity for Gym resets — the MINIMUM hook for training
or evaluating policies against Scenic-generated scenarios.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Keep independent of gym_env to avoid import cycles.
_DEFAULT_POSITION = (0.0, 0.0, 120.0)
_DEFAULT_VELOCITY = (0.0, 0.0, -1.5)


def sample_scenic_like_start(
    target_N: np.ndarray,
    rng: np.random.Generator,
    *,
    min_range_m: float = 80.0,
    max_range_m: float = 160.0,
    max_lateral_frac: float = 0.35,
    speed_min: float = 0.8,
    speed_max: float = 2.2,
    miss_pointing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a randomized approach start near the landing site."""
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    range_m = float(rng.uniform(min_range_m, max_range_m))
    lateral = max_lateral_frac * range_m
    offset = np.array(
        [
            float(rng.uniform(-lateral, lateral)),
            float(rng.uniform(-lateral, lateral)),
            range_m,
        ],
        dtype=np.float64,
    )
    position = target + offset

    to_site = target - position
    to_site = to_site / max(float(np.linalg.norm(to_site)), 1e-6)
    speed = float(rng.uniform(speed_min, speed_max))
    jitter = rng.normal(0.0, 0.15, size=3)
    direction = to_site + jitter
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    velocity = direction * speed

    if miss_pointing:
        sigma = rng.uniform(-0.4, 0.4, size=3).astype(np.float64)
    else:
        sigma = np.zeros(3, dtype=np.float64)
    return position, velocity, sigma


def scenic_like_or_default(
    target_N: np.ndarray,
    rng: Optional[np.random.Generator],
    *,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return scenic-like start or the fixed default approach."""
    if not enabled:
        return (
            np.array(_DEFAULT_POSITION, dtype=np.float64),
            np.array(_DEFAULT_VELOCITY, dtype=np.float64),
            None,
        )
    if rng is None:
        rng = np.random.default_rng()
    pos, vel, sigma = sample_scenic_like_start(target_N, rng)
    return pos, vel, sigma


def _ensure_scenic_on_path() -> None:
    """Prefer a sibling Scenic checkout for the basilisk-simulator interface."""
    env = os.environ.get("SCENIC_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # asteroid_rl/dynamics/scenic_reset.py → repo root → sibling Scenic
    demo_root = here.parents[2]
    candidates.append(demo_root.parent / "Scenic")
    for cand in candidates:
        src = cand / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            return


def sample_scenic_scenario_start(
    scenario_path: str | Path,
    *,
    max_iterations: int = 80,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Generate one Scenic scene and extract craft inertial state.

    Args:
        scenario_path: Path to a ``.scenic`` file using the Basilisk world model.
        max_iterations: Rejection-sampling budget for ``generate()``.
        params: Optional Scenic global parameter overrides.

    Returns:
        ``(position_N, velocity_N, sigma_BN_or_None, meta)`` where ``meta`` holds
        asteroid pose/radii when a ``ProceduralAsteroid`` is present.
    """
    _ensure_scenic_on_path()
    import scenic
    from scenic.simulators.basilisk.utils import scenic_orientation_to_mrp

    path = Path(scenario_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenic scenario not found: {path}")

    p = dict(params or {})
    p.setdefault("enable_viz", False)
    scenario = scenic.scenarioFromFile(str(path), params=p)
    scene, _ = scenario.generate(maxIterations=int(max_iterations))
    craft = scene.egoObject
    pos = np.array(
        [float(craft.position.x), float(craft.position.y), float(craft.position.z)],
        dtype=np.float64,
    )
    if getattr(craft, "velocity", None) is not None:
        vel = np.array(
            [float(craft.velocity.x), float(craft.velocity.y), float(craft.velocity.z)],
            dtype=np.float64,
        )
    else:
        vel = np.array(_DEFAULT_VELOCITY, dtype=np.float64)
    try:
        sigma = scenic_orientation_to_mrp(craft.orientation)
    except Exception:
        sigma = None

    meta: Dict[str, Any] = {"scenario": str(path.resolve()), "seed_note": "scenic.generate"}
    for obj in scene.objects:
        kind = getattr(obj, "basiliskKind", None)
        if kind == "procedural_asteroid":
            meta["asteroid"] = {
                "position": (
                    float(obj.position.x),
                    float(obj.position.y),
                    float(obj.position.z),
                ),
                "radii": (
                    float(getattr(obj, "radiusX", 45.0)),
                    float(getattr(obj, "radiusY", 40.0)),
                    float(getattr(obj, "radiusZ", 35.0)),
                ),
                "detailSeed": float(getattr(obj, "detailSeed", 0.0)),
                "noiseAmp": float(getattr(obj, "noiseAmp", 0.0)),
            }
            break
    return pos, vel, sigma, meta
