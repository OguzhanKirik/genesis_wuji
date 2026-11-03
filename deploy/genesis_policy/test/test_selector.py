"""The rod selector: the highest exposed reachable rod, standing ones counted lower."""
import math

import numpy as np
import pytest

from genesis_policy import selector as S

TOP = 0.95
R, HALF = 0.0125, 0.0175
CX, CY = 0.5474, -0.1099
YAW0 = 0.0          # the lying rod's axis along +x at rod yaw 0, for these tests


def ws(**kw):
    base = dict(rod_xy_centre=np.array([CX, CY]), xy_core=0.05, xy_edge=0.025, hand_yaw=(-math.radians(60), math.radians(45)),
                hand_yaw_core=(-math.radians(45), math.radians(30)), pinch_angle=math.radians(45), yaw0=YAW0,
                lying_max_deg=12.0, leaning_max_deg=40.0, standing_min_deg=75.0, stand_penalty=0.015, resting_gap=0.003,
                table_top=TOP, rod_radius=R, rod_half_len=HALF)
    base.update(kw)
    return S.Workspace(**base)


def q_axis(axis):
    """xyzw quaternion taking +z onto `axis`."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    v = np.cross([0.0, 0.0, 1.0], a)
    w = 1.0 + a[2]
    if w < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([*v, w])
    return q / np.linalg.norm(q)


LYING_X = q_axis([1, 0, 0])
STANDING = q_axis([0, 0, 1])


def test_empty_bin():
    c = S.select([], [], ws())
    assert c.index == -1 and c.reasons == ()


def test_highest_lying_rod_wins():
    pos = [[CX, CY - 0.03, TOP + 0.012], [CX, CY + 0.03, TOP + 0.037]]      # side by side, 6 cm apart
    c = S.select(pos, [LYING_X, LYING_X], ws())
    assert c.index == 1 and c.kind == S.LYING
    assert c.height == pytest.approx(0.037)
    assert c.reasons == ('lower than the pick', 'picked')


def test_a_rod_with_one_resting_on_it_is_skipped():
    # the second rod lies across the first, one diameter higher: the first is covered, the second picked
    pos = [[CX, CY, TOP + 0.012], [CX, CY, TOP + 0.012 + 2 * R]]
    across = q_axis([0, 1, 0])
    c = S.select(pos, [LYING_X, across], ws())
    assert c.index == 1
    assert c.reasons[0] == 'another rod rests on it'


def test_standing_counted_lower():
    lying = [CX, CY, TOP + 0.030]
    # 10 mm higher than the lying rod: the 15 mm penalty keeps the lying one
    c = S.select([lying, [CX + 0.04, CY + 0.04, TOP + 0.040]], [LYING_X, STANDING], ws())
    assert c.index == 0
    # 20 mm higher: the standing rod wins
    c = S.select([lying, [CX + 0.04, CY + 0.04, TOP + 0.050]], [LYING_X, STANDING], ws())
    assert c.index == 1 and c.kind == S.STANDING


def test_steep_rods_are_not_targets():
    steep = q_axis([math.cos(math.radians(60)), 0, math.sin(math.radians(60))])
    c = S.select([[CX, CY, TOP + 0.04]], [steep], ws())
    assert c.index == -1 and c.reasons == ('too steep to lie or stand',)


def test_outside_the_window():
    c = S.select([[CX + 0.06, CY, TOP + 0.04]], [LYING_X], ws())
    assert c.index == -1 and c.reasons == ('outside the reachable window',)


def test_edge_band_needs_the_whole_yaw_window():
    # 4 cm off centre: inside the core limit (5 cm) only. A lying rod at yaw 82 deg needs a hand yaw of 37..127 deg
    # (mod 180: -143..-53), outside the core window (-45..30); only the whole window reaches it (37..45, -60..-53),
    # where the limit is 2.5 cm: refused
    a = math.radians(82)
    yaw82 = q_axis([math.cos(a), math.sin(a), 0])
    c = S.select([[CX + 0.04, CY, TOP + 0.03]], [yaw82], ws())
    assert c.index == -1
    # 2 cm off centre the edge limit holds: picked, with a hand yaw outside the core window
    c = S.select([[CX + 0.02, CY, TOP + 0.03]], [yaw82], ws())
    assert c.index == 0
    assert not (math.radians(-45) <= c.hand_yaw <= math.radians(30))
    assert math.radians(-60) <= c.hand_yaw <= math.radians(45)


def test_leaning_direction_matters():
    # leaning 25 deg with its raised end toward +x (rod yaw 0): reachable in the core with any pinch near 0
    th = math.radians(25)
    up_x = q_axis([math.cos(th), 0, math.sin(th)])
    c = S.select([[CX, CY, TOP + 0.03]], [up_x], ws())
    assert c.index == 0 and c.kind == S.LEANING
    assert abs(c.hand_yaw) <= math.radians(45) + 1e-9
    # raised end toward -x (rod yaw 180 deg): needs a hand yaw of 135..225 deg, outside -60..45: refused
    up_mx = q_axis([-math.cos(th), 0, math.sin(th)])
    c = S.select([[CX, CY, TOP + 0.03]], [up_mx], ws())
    assert c.index == -1


def test_level_without_workspace_is_refused(level):
    with pytest.raises(S.SelectorError, match='no workspace section'):
        S.Workspace.from_level(level)


def test_parity_with_training():
    """The selector picks the rod adam-rl's level-52 env picked (tools/dump_fill_choices.py), heap for heap.
    GENESIS_POLICY_FILL=<fill_choices.npz> GENESIS_POLICY_FILL_LEVEL=<that level's level.yaml>; skips without them."""
    import os
    from genesis_policy import level as level_mod
    fill, lv = os.environ.get('GENESIS_POLICY_FILL'), os.environ.get('GENESIS_POLICY_FILL_LEVEL')
    if not fill or not lv:
        pytest.skip('set GENESIS_POLICY_FILL and GENESIS_POLICY_FILL_LEVEL (adam-rl tools/dump_fill_choices.py, export_policy.py)')
    w = S.Workspace.from_level(level_mod.load(lv))
    z = np.load(fill)
    bad = []
    for h in range(len(z['count'])):
        n = int(z['count'][h])
        c = S.select(z['pos'][h, :n], [q_axis(a) for a in z['axis'][h, :n]], w)
        if c.index != int(z['chosen'][h]):
            bad.append((h, c.index, int(z['chosen'][h])))
    assert not bad, f'{len(bad)} of {len(z["count"])} heaps differ (heap, selector, training): {bad[:5]}'
