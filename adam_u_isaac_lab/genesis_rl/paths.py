"""Filesystem roots for the Adam-U checkout.

Asset paths used to be computed as ``../..`` from whichever env file needed
them. That breaks as soon as a module moves. Everything that needs the
checkout or the USD tree should use this module.

``REPO_ROOT`` is the directory mounted at ``/workspace`` in the container
(``adam_u_isaac_lab/``), which is where ``assets/`` and ``logs/`` live.
"""

from __future__ import annotations

import os

# genesis_rl/paths.py -> genesis_rl -> adam_u_isaac_lab
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAM_U_ASSETS = os.environ.get("ADAM_U_ASSETS", os.path.join(REPO_ROOT, "assets"))
LOGS_ROOT = os.path.join(REPO_ROOT, "logs", "rsl_rl")
