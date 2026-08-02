"""Geometry-based perception stub matching the planning-document JSON schema.

Produces the same fields a VLM will later fill from Basilisk camera frames:
``target_visible``, ``landing_site_box``, ``hazard_score``, and
``progress_assessment``. This module uses truth-state geometry only (no VLM).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from Basilisk.utilities import RigidBodyKinematics as rbk

from asteroid_rl.camera import (
    DEFAULT_CAMERA_FOV_DEG,
    DEFAULT_CAMERA_POS_B,
    DEFAULT_CAMERA_SIGMA_CB,
)
from asteroid_rl.surface import get_surface_map


def _clamp01(value: float) -> float:
    """Clamp a scalar to ``[0, 1]``.

    Args:
        value: Arbitrary float.

    Returns:
        Value restricted to the unit interval.
    """
    return float(min(1.0, max(0.0, value)))


def project_point_to_camera(
    point_N: Sequence[float],
    position_N: Sequence[float],
    sigma_BN: Sequence[float],
    *,
    camera_pos_B: Sequence[float] = DEFAULT_CAMERA_POS_B,
    sigma_CB: Sequence[float] = DEFAULT_CAMERA_SIGMA_CB,
    field_of_view_deg: float = DEFAULT_CAMERA_FOV_DEG,
) -> Tuple[bool, float, float, float]:
    """Project an inertial point into the instrument camera normalized image.

    Args:
        point_N: World point to project, meters.
        position_N: Hub inertial position, meters.
        sigma_BN: Hub attitude MRP.
        camera_pos_B: Camera position in the body frame, meters.
        sigma_CB: Camera attitude MRP relative to the body frame.
        field_of_view_deg: Vertical edge-to-edge FOV, degrees.

    Returns:
        Tuple ``(in_front, u, v, range_c)`` where ``u,v`` are normalized image
        coordinates with ``(0,0)`` top-left / ``(1,1)`` bottom-right style
        center-referenced mapping (0.5, 0.5) = optical center, and ``range_c``
        is depth along the camera +z axis in meters. If the point is behind
        the camera, ``in_front`` is False.
    """
    r_n = np.asarray(position_N, dtype=np.float64).reshape(3)
    p_n = np.asarray(point_N, dtype=np.float64).reshape(3)
    c_bn = np.asarray(rbk.MRP2C(list(sigma_BN)), dtype=np.float64)
    c_cb = np.asarray(rbk.MRP2C(list(sigma_CB)), dtype=np.float64)
    p_b = c_bn @ (p_n - r_n) - np.asarray(camera_pos_B, dtype=np.float64).reshape(3)
    p_c = c_cb @ p_b
    depth = float(p_c[2])
    if depth <= 1e-3:
        return False, 0.5, 0.5, depth

    half_fov = np.deg2rad(float(field_of_view_deg) * 0.5)
    scale = 0.5 / np.tan(half_fov)
    u = 0.5 + scale * float(p_c[0]) / depth
    v = 0.5 + scale * float(p_c[1]) / depth
    return True, float(u), float(v), depth


def estimate_angular_bbox_half_size(
    range_m: float,
    *,
    object_radius_m: float = 120.0,
    field_of_view_deg: float = DEFAULT_CAMERA_FOV_DEG,
) -> float:
    """Estimate a normalized half-width for a bbox around a spherical object.

    Args:
        range_m: Distance from camera to object, meters.
        object_radius_m: Characteristic object radius, meters.
        field_of_view_deg: Camera vertical FOV, degrees.

    Returns:
        Half-size in normalized image coordinates, clipped to a useful range.
    """
    range_m = max(float(range_m), 1.0)
    ang = np.arctan(float(object_radius_m) / range_m)
    half = float(ang / np.deg2rad(field_of_view_deg))
    return float(np.clip(half, 0.02, 0.45))


def local_hazard_score(
    site_xy: Sequence[float],
    *,
    lateral_miss_m: float,
    map_radius_m: float = 10.0,
) -> float:
    """Heuristic hazard in ``[0, 1]`` from lateral miss and local terrain roughness.

    Args:
        site_xy: Landing-site ``(x, y)`` in the inertial frame, meters.
        lateral_miss_m: Horizontal distance from hub ground-track to site, meters.
        map_radius_m: Half-window used for heightmap roughness, meters.

    Returns:
        Hazard score where lower is safer (plan threshold example: ``< 0.10``).
    """
    surface = get_surface_map()
    x0, y0 = float(site_xy[0]), float(site_xy[1])
    samples = []
    for dx in (-map_radius_m, 0.0, map_radius_m):
        for dy in (-map_radius_m, 0.0, map_radius_m):
            samples.append(surface.surface_z(x0 + dx, y0 + dy))
    arr = np.asarray(samples, dtype=np.float64)
    roughness = float(np.std(arr)) if arr.size else 0.0
    # ~5 m std → high hazard; lateral miss 50 m → high hazard.
    score = 0.55 * _clamp01(roughness / 5.0) + 0.45 * _clamp01(lateral_miss_m / 50.0)
    return _clamp01(score)


def build_perception_stub(
    *,
    position_N: Sequence[float],
    velocity_N: Sequence[float],
    sigma_BN: Sequence[float],
    target_N: Sequence[float],
    altitude_m: float,
    camera_pos_B: Sequence[float] = DEFAULT_CAMERA_POS_B,
    sigma_CB: Sequence[float] = DEFAULT_CAMERA_SIGMA_CB,
    field_of_view_deg: float = DEFAULT_CAMERA_FOV_DEG,
) -> Dict[str, Any]:
    """Build the planning-document perception JSON from truth-state geometry.

    Args:
        position_N: Hub inertial position, meters.
        velocity_N: Hub inertial velocity, m/s (reserved for future cues).
        sigma_BN: Hub attitude MRP.
        target_N: Surface landing-site position, meters.
        altitude_m: Hub altitude above local terrain, meters.
        camera_pos_B: Camera mount position in body frame, meters.
        sigma_CB: Camera MRP relative to body.
        field_of_view_deg: Camera vertical FOV, degrees.

    Returns:
        Dict with ``target_visible``, ``landing_site_box`` ``[xmin,ymin,xmax,ymax]``,
        ``hazard_score``, ``progress_assessment``, plus debug fields
        ``site_uv`` and ``site_depth_m``.
    """
    del velocity_N  # reserved
    target = np.asarray(target_N, dtype=np.float64).reshape(3)
    position = np.asarray(position_N, dtype=np.float64).reshape(3)
    lateral = float(np.linalg.norm(position[:2] - target[:2]))

    in_front, u, v, depth = project_point_to_camera(
        target,
        position,
        sigma_BN,
        camera_pos_B=camera_pos_B,
        sigma_CB=sigma_CB,
        field_of_view_deg=field_of_view_deg,
    )
    margin = 0.02
    target_visible = bool(in_front and (-margin <= u <= 1.0 + margin) and (-margin <= v <= 1.0 + margin))

    half = estimate_angular_bbox_half_size(depth if in_front else altitude_m)
    if target_visible:
        box = [
            _clamp01(u - half),
            _clamp01(v - half),
            _clamp01(u + half),
            _clamp01(v + half),
        ]
    else:
        box = [0.0, 0.0, 0.0, 0.0]

    hazard = local_hazard_score(target[:2], lateral_miss_m=lateral)

    if not target_visible:
        assessment = "target not visible in camera frame; slew / search required"
    elif abs(u - 0.5) < 0.08 and abs(v - 0.5) < 0.08:
        assessment = "site is visible and near center of frame"
    elif u < 0.5:
        assessment = "site is visible and slightly left of center"
    else:
        assessment = "site is visible and slightly right of center"
    if hazard < 0.10 and target_visible:
        assessment += "; hazard low enough to commit to landing"
    elif hazard > 0.45:
        assessment += "; hazard elevated — continue approach cautiously"

    return {
        "target_visible": target_visible,
        "landing_site_box": box,
        "hazard_score": float(hazard),
        "progress_assessment": assessment,
        "site_uv": [float(u), float(v)],
        "site_depth_m": float(depth),
        "lateral_miss_m": lateral,
    }


def perception_feature_vector(perception: Optional[Dict[str, Any]]) -> np.ndarray:
    """Pack key perception fields into a short float vector for controllers.

    Args:
        perception: Dict from ``build_perception_stub``, or ``None``.

    Returns:
        Shape ``(4,)`` ``float32`` vector
        ``[visible, box_center_u, box_center_v, hazard_score]``.
    """
    if not perception:
        return np.zeros(4, dtype=np.float32)
    box = perception.get("landing_site_box") or [0, 0, 0, 0]
    visible = 1.0 if perception.get("target_visible") else 0.0
    cu = 0.5 * (float(box[0]) + float(box[2]))
    cv = 0.5 * (float(box[1]) + float(box[3]))
    return np.array(
        [visible, cu, cv, float(perception.get("hazard_score", 1.0))],
        dtype=np.float32,
    )
