"""The cell geometry the safety observation needs: the table and the bin as
axis-aligned boxes in the planning frame (world).

Read from genesis_isaac's workcell JSON, the one file that places the table
and the bin for the simulator and the planning scene (genesis_cell's
cell.yaml holds the bolt task's offsets and topics, not the furniture). The
bin is five boxes: a floor slab and four walls.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .level import Box


@dataclass(frozen=True)
class Cell:
    file: str
    table: Box
    table_top: float
    bin_center_xy: Tuple[float, float]
    bin_boxes: Tuple[Box, ...]

    def obstacles(self, extra=()):
        """The table first, then the bin walls, then any extra boxes."""
        return [self.table, *self.bin_boxes, *extra]


def workcell_path(name='workcell.json'):
    """genesis_isaac's workcell file: this package's vendored copy first, then the installed package."""
    from .model import assets_dir
    vendored = os.path.join(assets_dir(), 'workcell', name)
    if os.path.exists(vendored):
        return vendored
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('genesis_isaac'), 'assets', 'workcell', name)
        if os.path.exists(path):
            return path
    except Exception:  # noqa: BLE001 - no ROS environment, or not installed
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'genesis_isaac', 'assets', 'workcell', name))


def load_workcell(path=None, mirror_y=False):
    """Table and bin boxes from a workcell JSON.

    `mirror_y` flips the bin to the other side of the table (y -> -y), which
    is how adam-rl trained the right arm against a bin that the cell file
    places in front of the left arm. See README "Open questions".
    """
    path = path or workcell_path()
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    table = raw['table']
    centre = np.array(table['center'], dtype=float)
    size = np.array(table['size'], dtype=float)
    table_box = Box(centre - size / 2.0, centre + size / 2.0)
    table_top = float(table_box.hi[2])

    sign = -1.0 if mirror_y else 1.0
    bin_boxes: List[Box] = []
    cx = cy = 0.0
    bin_ = raw.get('bin')
    if bin_:
        cx, cy = float(bin_['center_xy'][0]), sign * float(bin_['center_xy'][1])
        length, width, height = (float(v) for v in bin_['inner'])
        wall = float(bin_['wall'])
        z0, z1 = table_top, table_top + wall + height
        hx, hy = length / 2.0, width / 2.0
        bin_boxes = [
            Box(np.array([cx - hx - wall, cy - hy - wall, z0]), np.array([cx + hx + wall, cy + hy + wall, z0 + wall])),
            Box(np.array([cx - hx - wall, cy - hy - wall, z0]), np.array([cx - hx, cy + hy + wall, z1])),
            Box(np.array([cx + hx, cy - hy - wall, z0]), np.array([cx + hx + wall, cy + hy + wall, z1])),
            Box(np.array([cx - hx - wall, cy - hy - wall, z0]), np.array([cx + hx + wall, cy - hy, z1])),
            Box(np.array([cx - hx - wall, cy + hy, z0]), np.array([cx + hx + wall, cy + hy + wall, z1])),
        ]
    return Cell(file=path, table=table_box, table_top=table_top,
                bin_center_xy=(cx, cy), bin_boxes=tuple(bin_boxes))
