"""Filesystem roots for the Adam-U checkout.

Asset paths used to be computed as ``../..`` from whichever env file needed
them. That breaks as soon as a module moves. Everything that needs the
checkout or the USD tree should use this module.

``REPO_ROOT`` is the repository root, mounted at ``/workspace`` in the
container. ``assets/`` and ``logs/`` live next to this package.
"""

from __future__ import annotations

import os

# genesis_rl/paths.py -> repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAM_U_ASSETS = os.environ.get("ADAM_U_ASSETS", os.path.join(REPO_ROOT, "assets"))
LOGS_ROOT = os.path.join(REPO_ROOT, "logs", "rsl_rl")
