"""Filesystem anchors shared across the package.

Keeps asset / example fallback resolution independent of how deeply modules are
nested under ``asteroid_rl/``.
"""

from __future__ import annotations

import os

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
EXAMPLES_MUJOCO = os.path.join(REPO_ROOT, "examples", "mujoco")
EXAMPLES_DATA = os.path.join(REPO_ROOT, "examples", "dataForExamples")
