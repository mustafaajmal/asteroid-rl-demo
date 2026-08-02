"""Basilisk body-fixed instrument camera helpers (Vizard-rendered).

Sets up ``Basilisk.simulation.camera.Camera`` as a spacecraft-mounted sensor
and reads RGB frames from its ``imageOutMsg``. Images are produced by Vizard
through ``vizInterface`` (same path used by Basilisk OpNav scenarios).
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional, Sequence

import numpy as np

from Basilisk.simulation import camera
from Basilisk.utilities import macros


# Side/belly mount: offset in +y so the FOV is not buried in the thruster exhaust
# plume (thrust is +body-z; Vizard draws the plume along -thrust, i.e. -body-z).
DEFAULT_CAMERA_POS_B = (0.0, 1.5, -0.8)
# ~179 deg about body x (MRP). Exact 180 deg is singular; this aims camera +z ≈ body -z.
DEFAULT_CAMERA_SIGMA_CB = (float(np.tan(np.deg2rad(179.0) / 4.0)), 0.0, 0.0)
DEFAULT_CAMERA_ID = 1
DEFAULT_CAMERA_FOV_DEG = 60.0


def create_instrument_camera(
    *,
    parent_name: str,
    width: int,
    height: int,
    render_rate_sec: float,
    camera_pos_B: Sequence[float] = DEFAULT_CAMERA_POS_B,
    sigma_CB: Sequence[float] = DEFAULT_CAMERA_SIGMA_CB,
    camera_id: int = DEFAULT_CAMERA_ID,
    field_of_view_deg: float = DEFAULT_CAMERA_FOV_DEG,
) -> camera.Camera:
    """Create a body-fixed Basilisk ``Camera`` SysModel (not yet task-linked).

    Args:
        parent_name: Spacecraft / body name Vizard should attach the camera to
            (for this demo, ``\"hub\"``).
        width: Image width in pixels.
        height: Image height in pixels.
        render_rate_sec: Minimum interval between image requests, seconds.
        camera_pos_B: Camera position in the parent body frame, meters.
        sigma_CB: MRP from parent body frame to camera frame.
        camera_id: Integer ID used by Vizard for this instrument.
        field_of_view_deg: Vertical edge-to-edge field of view, degrees.

    Returns:
        Configured ``camera.Camera`` instance (caller adds it to a sim task and
        wires ``imageInMsg`` / ``vizInterface.addCamMsgToModule``).
    """
    cam = camera.Camera()
    cam.ModelTag = "navcam"
    cam.cameraIsOn = 1
    cam.cameraID = int(camera_id)
    cam.saveImages = 0
    cam.parentName = str(parent_name)
    cam.cameraPos_B = list(camera_pos_B)
    cam.sigma_CB = list(sigma_CB)
    cam.resolution = [int(width), int(height)]
    cam.fieldOfView = float(np.deg2rad(field_of_view_deg))
    cam.renderRate = int(macros.sec2nano(max(float(render_rate_sec), 1e-3)))
    cam.skyBox = "black"
    return cam


def image_msg_to_rgb(
    image_msg: Any,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Convert a Basilisk ``CameraImageMsg`` / payload into an RGB ``uint8`` array.

    Args:
        image_msg: Either a ``CameraImageMsg`` with ``.read()``, or a
            ``CameraImageMsgPayload`` instance.
        width: Expected image width in pixels.
        height: Expected image height in pixels.

    Returns:
        Shape ``(height, width, 3)`` ``uint8`` array, or ``None`` if no valid
        image is present / the buffer cannot be decoded.
    """
    payload = image_msg.read() if hasattr(image_msg, "read") else image_msg
    if payload is None or int(getattr(payload, "valid", 0)) != 1:
        return None
    pointer = getattr(payload, "imagePointer", None)
    length = int(getattr(payload, "imageBufferLength", 0))
    channels = int(getattr(payload, "imageType", 0) or 3)
    if pointer is None or length <= 0 or channels <= 0:
        return None

    try:
        address = ctypes.cast(pointer, ctypes.c_void_p).value
    except (TypeError, ValueError, ctypes.ArgumentError):
        try:
            address = int(pointer)
        except (TypeError, ValueError):
            return None
    if not address:
        return None

    try:
        raw = (ctypes.c_uint8 * length).from_address(address)
        buffer = np.frombuffer(raw, dtype=np.uint8).copy()
    except (ValueError, TypeError, OSError):
        return None

    expected = int(width) * int(height) * int(channels)
    if buffer.size == expected:
        image = buffer.reshape((int(height), int(width), int(channels)))
        if channels >= 3:
            # Vizard / OpenCV path is typically BGR; convert to RGB for callers.
            bgr = image[..., :3]
            return bgr[..., ::-1].copy()
        if channels == 1:
            return np.repeat(image, 3, axis=-1)
        return None

    # Encoded buffer (e.g. PNG/JPEG) — optional OpenCV decode.
    try:
        import cv2

        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def read_camera_rgb(
    camera_mod: camera.Camera,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Read the latest RGB frame from a Basilisk instrument camera module.

    Args:
        camera_mod: Live ``camera.Camera`` SysModel wired to Vizard images.
        width: Expected image width in pixels.
        height: Expected image height in pixels.

    Returns:
        RGB ``uint8`` array, or ``None`` if Vizard has not delivered a frame yet.
    """
    return image_msg_to_rgb(camera_mod.imageOutMsg, width=width, height=height)


def launch_vizard_for_camera(
    *,
    port: str = "5556",
    show_gui: bool = False,
    find_app_fn,
    sleep_fn,
    popen_fn,
) -> None:
    """Launch Vizard in OpNav/direct-comm mode for instrument camera images.

    Args:
        port: TCP port published by ``vizInterface``.
        show_gui: If True, use ``-directComm`` (visible window). If False, use
            ``-noDisplay`` headless OpNav rendering.
        find_app_fn: Callable returning a Vizard.app path or ``None``.
        sleep_fn: Sleep callable used after launch.
        popen_fn: Subprocess launcher (``subprocess.Popen``).
    """
    app = find_app_fn()
    address = f"tcp://localhost:{port}"
    mode = "-directComm" if show_gui else "-noDisplay"
    if app is None:
        print(
            "Vizard.app not found.\n"
            f"Open Vizard manually with {mode} and connect to {address}\n"
            "(Basilisk instrument camera images require a Vizard connection.)"
        )
        return

    # Prefer the MacOS binary path used by Basilisk OpNav examples.
    binary = app
    mac_binary = f"{app}/Contents/MacOS/Vizard"
    import os

    if os.path.isfile(mac_binary):
        cmd = [mac_binary, "--args", mode, address]
    else:
        cmd = ["open", app, "--args", mode, address]
    print(f"Launching Vizard for camera: {' '.join(cmd)}")
    popen_fn(cmd)
    sleep_fn(2.5)
