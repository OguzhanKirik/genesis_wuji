"""Shared fixtures: the stub level, the URDF (xacro, or GENESIS_POLICY_URDF
on a host without ROS) and the robot model. Each is built once per session;
pinocchio parsing the 83-joint URDF takes a moment."""
import os
import tempfile

import pytest

from genesis_policy import stub


@pytest.fixture(scope='session')
def level():
    with tempfile.TemporaryDirectory() as d:
        return stub.write_level(os.path.join(d, 'level.yaml'))


@pytest.fixture(scope='session')
def urdf():
    from genesis_policy import model as model_mod
    return model_mod.urdf_string('genesis_wuji')


@pytest.fixture(scope='session')
def robot(urdf, level):
    from genesis_policy import model as model_mod
    return model_mod.RobotModel(urdf, level.arm_joints, level.wrist_link, level.wrist_offset)
