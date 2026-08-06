"""Unit tests for surface altitude helpers."""

from __future__ import annotations

import numpy as np

from asteroid_rl.environment.surface import ASTEROID_BODY_POSITION, get_surface_map


def test_heightmap_contains_origin_site():
    surf = get_surface_map()
    assert surf.contains_xy(0.0, 0.0)
    z = surf.surface_z(0.0, 0.0)
    assert np.isfinite(z)
    assert z > -1.0e5


def test_offmap_not_contained():
    surf = get_surface_map()
    assert not surf.contains_xy(0.0, -500.0)


def test_radial_altitude_positive_outside():
    surf = get_surface_map()
    site = surf.landing_site(0.0, 0.0)
    com = ASTEROID_BODY_POSITION
    # Point farther from COM than the site along +z from COM.
    direction = site - com
    direction = direction / np.linalg.norm(direction)
    pos = com + direction * (np.linalg.norm(site - com) + 40.0)
    alt = surf.radial_altitude(pos, com_N=com, site_N=site)
    assert alt > 30.0
