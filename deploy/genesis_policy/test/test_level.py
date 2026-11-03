"""level.yaml loading: the stub parses, and every malformed input is refused
with the key named, before anything could move."""
import copy
import os
import tempfile

import numpy as np
import pytest
import yaml

from genesis_policy import level as lv
from genesis_policy import stub


def stub_dict():
    return yaml.safe_load(stub.LEVEL_TEMPLATE.format(obs_dim=74))


def test_stub_parses_and_is_consistent(level):
    assert level.layout.size == 74
    assert level.layout.offset('safety') == 58
    assert len(level.arm_joints) == 7 and len(level.hand_joints) == 9 and len(level.all_hand_joints) == 20
    assert level.hand_joints[:3] == ('r_thumb_cmc_flex', 'r_thumb_mcp', 'r_thumb_ip')
    assert np.all(level.hand_closed > level.hand_open)
    assert level.phase.pre_lift_steps == 18 and level.phase.lift_start == 28 and level.phase.scripted_steps == 98
    assert level.dt == pytest.approx(1.0 / 30.0)
    assert np.allclose(level.phase.lift_dir, [0, 0, 1])
    assert np.allclose(level.wrist_offset, np.eye(4))
    assert level.safety_dist_clamp == (-0.1, 0.5)


def test_write_level_round_trips():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'level.yaml')
        a = stub.write_level(path)
        b = lv.load(path)
        assert a.layout == b.layout
        assert a.phase.scripted_steps == b.phase.scripted_steps and a.rate_hz == b.rate_hz
        assert np.array_equal(a.hand_open, b.hand_open)


@pytest.mark.parametrize('mutate, needle', [
    (lambda d: d.pop('phase'), 'phase'),
    (lambda d: d['control'].update(rate_hz='fast'), 'control.rate_hz'),
    (lambda d: d.update(schema=99), 'schema'),
    (lambda d: d['hand']['open'].pop('r_thumb_mcp'), 'hand.open'),
    (lambda d: d['hand']['held'].update(r_thumb_mcp=0.0), 'hand.held'),
    (lambda d: d['observation']['layout'].append({'name': 'bogus', 'size': 2}), 'unknown term'),
    (lambda d: d['observation']['layout'][0].update(size=4), 'size'),
    (lambda d: d['observation']['arm_q_limits'].update(hi=[-5.0] * 7), 'arm_q_limits'),
    (lambda d: d['action'].update(wrist_smoothing=1.5), 'wrist_smoothing'),
    (lambda d: d['phase'].update(lift_slow_steps=40), 'lift_slow_steps'),
    (lambda d: d['phase'].update(lift_dir=[0, 0, 0]), 'lift_dir'),
    (lambda d: d['safety']['groups'].update(palm=[]), 'safety.groups.palm'),
    (lambda d: d['safety']['groups']['tips'].append(['r_thumb_distal', [0, 0], 0.01]), 'safety.groups.tips'),
    (lambda d: d['ik']['workspace'].update(z_max=0.0), 'z_max'),
    (lambda d: d['arm'].update(joints=['a'] * 7), 'arm.joints'),
])
def test_malformed_levels_are_refused(mutate, needle):
    d = stub_dict()
    mutate(d)
    with pytest.raises(lv.LevelError) as info:
        lv.parse(copy.deepcopy(d))
    assert needle in str(info.value)


def test_layout_may_be_a_subset_in_any_order():
    d = stub_dict()
    d['observation']['layout'] = [{'name': 'arm_q_norm', 'size': 7}, {'name': 'bolt_pos_rel', 'size': 3}]
    level = lv.parse(d)
    assert level.layout.size == 10 and level.layout.names == ['arm_q_norm', 'bolt_pos_rel']


def test_optional_keys_for_the_transfer_level():
    """training_boxes, the torque latch, sense_est and roll_cap are optional; when present they are read."""
    lvl = lv.parse(stub_dict())
    assert lvl.training_boxes == () and lvl.phase.latch.mode == 'closure'
    assert lvl.phase.sense_est is False and lvl.phase.roll_cap == pytest.approx(np.pi / 2)
    d = stub_dict()
    d['safety']['training_boxes'] = [{'lo': [0.5, -0.3, 0.95], 'hi': [0.51, 0.0, 0.99]}]
    d['phase']['latch'] = {'mode': 'torque', 'steps': 4, 'tau_min_nm': 0.2, 'grasp_region_m': 0.05,
                           'pinch_centre_tol_m': 0.006, 'rel_vel_max_m_s': 0.1}
    d['phase']['sense_est'] = True
    d['phase']['roll_cap'] = 0.5
    lvl = lv.parse(d)
    assert len(lvl.training_boxes) == 1 and np.allclose(lvl.training_boxes[0].hi, [0.51, 0.0, 0.99])
    assert lvl.phase.latch.mode == 'torque' and lvl.phase.latch.tau_min_nm == pytest.approx(0.2)
    assert lvl.phase.sense_est is True and lvl.phase.roll_cap == pytest.approx(0.5)
    d['safety']['training_boxes'][0]['hi'][0] = 0.4
    with pytest.raises(lv.LevelError, match='training_boxes'):
        lv.parse(d)
    d = stub_dict()
    d['phase']['latch'] = {'mode': 'force', 'steps': 4}
    with pytest.raises(lv.LevelError, match='mode'):
        lv.parse(d)
