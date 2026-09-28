"""Copy a moved module's names onto an old import path."""

from __future__ import annotations

import importlib


def reexport(module_name: str, namespace: dict) -> None:
    """Bind every public and private name from ``module_name`` into ``namespace``.

    Old ``from genesis_rl.envs.<file> import _helper`` statements keep working
    after a module moves, including names that ``import *`` would drop.
    """
    source = importlib.import_module(module_name)
    namespace.update({key: value for key, value in vars(source).items() if not key.startswith("__")})
