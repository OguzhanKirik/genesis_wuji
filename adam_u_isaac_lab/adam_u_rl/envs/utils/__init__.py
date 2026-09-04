# Upstream shipped this file as "__init__,py" (comma, not dot), so the package
# only ever worked as an implicit namespace package and this content was dead.
# The `.actions` module it referenced does not exist in the repo.
from .observations import *  # noqa: F401, F403
