"""The robot model the runtime computes with: FK, the wrist Jacobian and the
joint limits of the right arm, from the same xacro every launch file loads.

pinocchio reads the URDF the xacro produces (design:=genesis_wuji,
ros2_control:=false); only the kinematic tree is built, no meshes. The
policy's wrist frame is a link plus a fixed offset from level.yaml, so the
Jacobian is shifted to that point.

Everything is in the URDF's root frame, `world`, which is MoveIt's planning
frame and the frame perception reports in.
"""
import os
from typing import Dict, Mapping, Sequence

import numpy as np
import pinocchio as pin

from . import so3

URDF_ENV = 'GENESIS_POLICY_URDF'


def assets_dir():
    """This package's vendored adam-moveit files (scripts/vendor_from_adam_moveit.sh)."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets'))


def vendored_urdf_path():
    """The pre-generated genesis_wuji URDF, or None when the assets have not been vendored yet."""
    path = os.path.join(assets_dir(), 'genesis_wuji.urdf')
    return path if os.path.exists(path) else None


def xacro_path():
    """genesis_description's top-level xacro, installed or in the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('genesis_description'), 'urdf', 'genesis.urdf.xacro')
        if os.path.exists(path):
            return path
    except Exception:  # noqa: BLE001 - no ROS environment, or not installed
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'genesis_description', 'urdf', 'genesis.urdf.xacro'))


def urdf_string(design='genesis_wuji', path=None):
    """The URDF as text, the way stack.launch.py builds it (xacro with the
    design argument, ros2_control off). Order: GENESIS_POLICY_URDF=<file>, then the vendored
    assets/genesis_wuji.urdf, then xacro on genesis_description (adam-moveit workspace only)."""
    pre = os.environ.get(URDF_ENV) or vendored_urdf_path()
    if pre and os.path.exists(pre):
        with open(pre, encoding='utf-8') as f:
            return f.read()
    import xacro  # ROS tooling, imported here so the module loads without it
    doc = xacro.process_file(path or xacro_path(), mappings={'design': design, 'ros2_control': 'false'})
    return doc.toxml()


class RobotModel:
    """pinocchio model of the whole robot with the right-arm view the policy needs."""

    def __init__(self, urdf_xml: str, arm_joints: Sequence[str], wrist_link: str, wrist_offset=None):
        self.model = pin.buildModelFromXML(urdf_xml)
        self.data = self.model.createData()
        self.q_index: Dict[str, int] = {}
        self.v_index: Dict[str, int] = {}
        for jid in range(1, self.model.njoints):
            joint = self.model.joints[jid]
            if joint.nq == 1 and joint.nv == 1:
                name = self.model.names[jid]
                self.q_index[name] = joint.idx_q
                self.v_index[name] = joint.idx_v
        missing = [j for j in arm_joints if j not in self.q_index]
        if missing:
            raise ValueError(f'arm joints not in the URDF (or not single-DOF): {missing}')
        self.arm_joints = tuple(arm_joints)
        self.arm_q = np.array([self.q_index[j] for j in arm_joints])
        self.arm_v = np.array([self.v_index[j] for j in arm_joints])
        if not self.model.existFrame(wrist_link):
            raise ValueError(f'wrist link {wrist_link!r} is not in the URDF')
        self.wrist_link = wrist_link
        self.wrist_fid = self.model.getFrameId(wrist_link)
        self.wrist_offset = np.eye(4) if wrist_offset is None else np.asarray(wrist_offset, dtype=float)
        self._q = pin.neutral(self.model)

    # ---- joints ----------------------------------------------------------------

    @property
    def nq(self):
        return self.model.nq

    def has_frame(self, name):
        return self.model.existFrame(name)

    def limits(self, joints: Sequence[str]):
        """(lo, hi) position limits of `joints`, from the URDF."""
        idx = [self.q_index[j] for j in joints]
        return (np.array(self.model.lowerPositionLimit)[idx], np.array(self.model.upperPositionLimit)[idx])

    def configuration(self, positions: Mapping[str, float]):
        """A full q from {joint: position}; joints not named stay at zero,
        names that are not joints of this model are ignored."""
        q = pin.neutral(self.model)
        for name, value in positions.items():
            i = self.q_index.get(name)
            if i is not None:
                q[i] = float(value)
        return q

    def arm_of(self, q):
        return np.asarray(q)[self.arm_q]

    # ---- kinematics --------------------------------------------------------------

    def forward(self, q):
        """Run FK; frame_pose / wrist read the result."""
        self._q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.model, self.data, self._q)
        pin.updateFramePlacements(self.model, self.data)

    def frame_pose(self, name):
        """(position, rotation) of frame `name` in world, after forward()."""
        M = self.data.oMf[self.model.getFrameId(name)]
        return np.array(M.translation), np.array(M.rotation)

    def wrist(self):
        """(position, rotation) of the policy's wrist frame, after forward()."""
        M = self.data.oMf[self.wrist_fid]
        T = np.eye(4)
        T[:3, :3] = M.rotation
        T[:3, 3] = M.translation
        T = T @ self.wrist_offset
        return T[:3, 3].copy(), T[:3, :3].copy()

    def wrist_twist(self, q, velocities: Mapping[str, float]):
        """(linear, angular) velocity of the policy wrist point, world-aligned,
        from every joint's reported velocity: the waist and the other arm move
        the wrist too. Isaac Lab's body_link_lin_vel_w / ang_vel_w."""
        q = np.asarray(q, dtype=float)
        v = np.zeros(self.model.nv)
        for name, value in velocities.items():
            i = self.v_index.get(name)
            if i is not None:
                v[i] = value
        J = np.asarray(pin.computeFrameJacobian(self.model, self.data, q, self.wrist_fid, pin.LOCAL_WORLD_ALIGNED))
        d = self.data.oMf[self.wrist_fid].rotation @ self.wrist_offset[:3, 3]
        if np.any(d != 0.0):
            J = J.copy()
            J[:3, :] = J[:3, :] - so3.skew(d) @ J[3:, :]
        tw = J @ v
        return tw[0:3], tw[3:6]

    def wrist_jacobian(self, q):
        """6x7 Jacobian of the policy wrist point, rows [linear; angular] in
        world-aligned axes, columns the arm joints in order. Matches Isaac
        Lab's body_link_jacobian_w shifted to the offset point."""
        q = np.asarray(q, dtype=float)
        J = pin.computeFrameJacobian(self.model, self.data, q, self.wrist_fid, pin.LOCAL_WORLD_ALIGNED)
        J = np.asarray(J)[:, self.arm_v]
        d = self.data.oMf[self.wrist_fid].rotation @ self.wrist_offset[:3, 3]
        if np.any(d != 0.0):
            J = J.copy()
            J[:3, :] = J[:3, :] - so3.skew(d) @ J[3:, :]
        return J
