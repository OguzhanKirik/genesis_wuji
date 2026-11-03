"""The robot model against an independent walk of the URDF: FK of the wrist
and the fingertips, the Jacobian by finite differences, the URDF limits of
the right arm and the joint-name mapping."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from genesis_policy import so3

ARM = ('shoulderPitch_Right', 'shoulderRoll_Right', 'shoulderYaw_Right', 'elbow_Right',
       'wristYaw_Right', 'wristPitch_Right', 'wristRoll_Right')


class UrdfChain:
    """Plain-numpy FK from the URDF XML: each joint is origin * rotation
    about its axis, composed from the root. Written without pinocchio on
    purpose, so the two cannot share a mistake."""

    def __init__(self, urdf_xml):
        root = ET.fromstring(urdf_xml)
        self.joints = {}
        self.parent_joint = {}
        for j in root.iter('joint'):
            origin = j.find('origin')
            xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
            axis = j.find('axis')
            ax = [float(v) for v in (axis.get('xyz', '1 0 0') if axis is not None else '1 0 0').split()]
            limit = j.find('limit')
            lim = (float(limit.get('lower', 0)), float(limit.get('upper', 0))) if limit is not None else None
            self.joints[j.get('name')] = dict(
                type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                T=so3.transform(xyz, so3.rpy(*rpy)), axis=np.array(ax), limit=lim)
            self.parent_joint[j.find('child').get('link')] = j.get('name')

    def chain(self, link):
        out = []
        while link in self.parent_joint:
            name = self.parent_joint[link]
            out.append(name)
            link = self.joints[name]['parent']
        return list(reversed(out))

    def pose(self, link, q):
        T = np.eye(4)
        for name in self.chain(link):
            j = self.joints[name]
            T = T @ j['T']
            if j['type'] in ('revolute', 'continuous'):
                T = T @ so3.transform([0, 0, 0], so3.exp(j['axis'] * q.get(name, 0.0)))
            elif j['type'] == 'prismatic':
                T = T @ so3.transform(j['axis'] * q.get(name, 0.0), np.eye(3))
        return T[:3, 3], T[:3, :3]


@pytest.fixture(scope='module')
def chain(urdf):
    return UrdfChain(urdf)


def random_q(rng, robot, chain):
    q = {}
    for name in ARM + ('waistRoll', 'waistPitch', 'waistYaw', 'r_thumb_cmc_flex', 'r_index_finger_pip'):
        lo, hi = chain.joints[name]['limit']
        q[name] = float(rng.uniform(lo, hi))
    return q


def test_wrist_and_tip_fk_match_the_independent_chain(robot, chain):
    rng = np.random.default_rng(1)
    for _ in range(6):
        q = random_q(rng, robot, chain)
        robot.forward(robot.configuration(q))
        for link in ('wristRollRight', 'r_thumb_distal', 'r_index_finger_distal', 'r_middle_finger_distal',
                     'elbowRight', 'shoulderRollRight'):
            p_ref, R_ref = chain.pose(link, q)
            p, R = robot.frame_pose(link)
            assert np.allclose(p, p_ref, atol=1e-9), link
            assert np.allclose(R, R_ref, atol=1e-9), link
        p, R = robot.wrist()
        p_ref, R_ref = chain.pose('wristRollRight', q)
        assert np.allclose(p, p_ref, atol=1e-9) and np.allclose(R, R_ref, atol=1e-9)


def test_wrist_offset_shifts_the_frame(urdf, level, chain):
    from genesis_policy.model import RobotModel
    offset = so3.transform([0.0, 0.0, -0.05], so3.rpy(0.0, 0.3, 0.0))
    robot = RobotModel(urdf, level.arm_joints, level.wrist_link, offset)
    q = {'shoulderPitch_Right': 0.4, 'elbow_Right': -1.0}
    robot.forward(robot.configuration(q))
    p, R = robot.wrist()
    p_ref, R_ref = chain.pose('wristRollRight', q)
    T = so3.transform(p_ref, R_ref) @ offset
    assert np.allclose(p, T[:3, 3]) and np.allclose(R, T[:3, :3])
    # the Jacobian of the offset point: finite differences of the offset FK
    q_full = robot.configuration(q)
    J = robot.wrist_jacobian(q_full)
    for i, name in enumerate(ARM):
        qp = dict(q)
        qp[name] = q.get(name, 0.0) + 1e-6
        robot.forward(robot.configuration(qp))
        p2, _ = robot.wrist()
        assert np.allclose((p2 - p) / 1e-6, J[:3, i], atol=1e-4), name


def test_jacobian_matches_finite_differences(robot, chain):
    rng = np.random.default_rng(2)
    for _ in range(3):
        q = random_q(rng, robot, chain)
        q_full = robot.configuration(q)
        robot.forward(q_full)
        p0, R0 = robot.wrist()
        J = robot.wrist_jacobian(q_full)
        assert J.shape == (6, 7)
        h = 1e-6
        for i, name in enumerate(ARM):
            qp = dict(q)
            qp[name] += h
            robot.forward(robot.configuration(qp))
            p1, R1 = robot.wrist()
            assert np.allclose((p1 - p0) / h, J[:3, i], atol=1e-4), name
            assert np.allclose(so3.log(R1 @ R0.T) / h, J[3:, i], atol=1e-4), name


def test_arm_limits_are_the_urdf_values(robot, chain):
    lo, hi = robot.limits(ARM)
    for i, name in enumerate(ARM):
        assert (lo[i], hi[i]) == pytest.approx(chain.joints[name]['limit'])
    assert lo[0] == pytest.approx(-3.613, abs=1e-3)   # the wide ros2_control value, not MoveIt's -1.047


def test_configuration_maps_names_to_the_right_slots(robot):
    base = robot.configuration({})
    assert np.allclose(base, 0.0)
    for name in ARM + ('r_pinky_dip', 'waistYaw'):
        q = robot.configuration({name: 0.5, 'not_a_joint': 9.0})
        changed = np.nonzero(q)[0]
        assert len(changed) == 1 and changed[0] == robot.q_index[name]
    assert list(robot.arm_q) == [robot.q_index[n] for n in ARM]


def test_right_arm_reaches_the_bin_side(robot, chain):
    """A sanity check of the frame: at zero the right wrist hangs on y < 0."""
    robot.forward(robot.configuration({}))
    p, _ = robot.wrist()
    assert p[1] < -0.15 and 0.9 < p[2] < 1.1
    assert math.isfinite(p[0])
