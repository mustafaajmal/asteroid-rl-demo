"""Asteroid surface queries for altitude-above-terrain termination.

Loads a precomputed heightmap of the scaled Itokawa mesh (asteroid body at
``(0, 0, -150)``) and returns hub altitude / landing-site helpers used by the
Gymnasium environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import numpy as np

from asteroid_rl.paths import ASSETS_DIR

_HEIGHTMAP_PATH = os.path.join(ASSETS_DIR, "Itokawa", "surface_heightmap.npz")

# Nominal asteroid body pose matching ``assets/sat_ast_landing.xml``.
ASTEROID_BODY_POSITION = np.array([0.0, 0.0, -150.0], dtype=np.float64)


class SurfaceMap:
    """Grid of maximum surface ``z`` values over the asteroid footprint.

    Attributes:
        H: Shape ``(ny, nx)`` array of surface ``z`` in the inertial frame.
        xmin: World ``x`` of column 0, meters.
        ymin: World ``y`` of row 0, meters.
        res: Grid spacing in meters.
    """

    def __init__(self, path: str = _HEIGHTMAP_PATH):
        """Load a heightmap NPZ from disk.

        Args:
            path: Filesystem path to ``surface_heightmap.npz`` with keys
                ``H``, ``xmin``, ``ymin``, and ``res``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing surface heightmap: {path}. "
                "Regenerate with the project mesh tooling."
            )
        data = np.load(path)
        self.H = np.asarray(data["H"], dtype=np.float64)
        self.xmin = float(data["xmin"])
        self.ymin = float(data["ymin"])
        self.res = float(data["res"])
        self.ny, self.nx = self.H.shape

    def contains_xy(self, x: float, y: float) -> bool:
        """Return True if ``(x, y)`` maps to a finite heightmap cell."""
        ix = int(round((float(x) - self.xmin) / self.res))
        iy = int(round((float(y) - self.ymin) / self.res))
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return False
        return bool(np.isfinite(self.H[iy, ix]))

    def surface_z(self, x: float, y: float) -> float:
        """Return the surface ``z`` under ``(x, y)`` via nearest grid sample.

        Args:
            x: World ``x`` coordinate, meters.
            y: World ``y`` coordinate, meters.

        Returns:
            Surface ``z`` in meters. If the query is off the map or the cell is
            empty, returns a large negative sentinel so altitude stays large.
        """
        ix = int(round((float(x) - self.xmin) / self.res))
        iy = int(round((float(y) - self.ymin) / self.res))
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return -1.0e6
        z = float(self.H[iy, ix])
        if not np.isfinite(z):
            return -1.0e6
        return z

    def radial_altitude(
        self,
        position_N: np.ndarray,
        *,
        com_N: np.ndarray,
        site_N: np.ndarray,
    ) -> float:
        """Altitude above a spherical shell through the landing site.

        Used when the craft is outside the heightmap footprint (orbital phase).

        Args:
            position_N: Hub inertial position, meters.
            com_N: Asteroid COM inertial position, meters.
            site_N: Landing-site inertial position, meters.

        Returns:
            ``||r-com|| - ||site-com||`` in meters.
        """
        r = np.asarray(position_N, dtype=np.float64).reshape(3)
        com = np.asarray(com_N, dtype=np.float64).reshape(3)
        site = np.asarray(site_N, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(r - com) - np.linalg.norm(site - com))

    def altitude(self, position_N: np.ndarray) -> float:
        """Hub altitude above the local surface.

        Args:
            position_N: Length-3 inertial hub position ``[x, y, z]``, meters.

        Returns:
            ``position_z - surface_z(x, y)`` in meters (positive above terrain).
        """
        p = np.asarray(position_N, dtype=np.float64).reshape(3)
        return float(p[2] - self.surface_z(float(p[0]), float(p[1])))

    def landing_site(self, x: float = 0.0, y: float = 0.0) -> np.ndarray:
        """Return the surface point used as the fixed landing site.

        Args:
            x: Desired site ``x`` in the inertial frame, meters.
            y: Desired site ``y`` in the inertial frame, meters.

        Returns:
            Shape ``(3,)`` array ``[x, y, surface_z(x, y)]``.
        """
        return np.array([float(x), float(y), self.surface_z(x, y)], dtype=np.float64)


@lru_cache(maxsize=1)
def get_surface_map() -> SurfaceMap:
    """Return the process-wide cached ``SurfaceMap``.

    Returns:
        Shared ``SurfaceMap`` instance loaded from the default NPZ path.
    """
    return SurfaceMap()


def default_landing_site() -> Tuple[float, float, float]:
    """Fixed-site landing target on the asteroid surface near the approach axis.

    Returns:
        Tuple ``(x, y, z)`` for the surface point under ``(0, 0)``.
    """
    site = get_surface_map().landing_site(0.0, 0.0)
    return float(site[0]), float(site[1]), float(site[2])
