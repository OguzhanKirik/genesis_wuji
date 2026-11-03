"""The runtime on a scripted actor and a perfect-tracking fake robot: the
observation layout, the IK converging on commanded wrist steps, the
hand-joint mapping into the controller order, the phase machine with the
latch, the grip check and a retry, and determinism."""
import numpy as np
import pytest

from genesis_policy import so3
from genesis_policy.actor import ScriptedActor, ZeroActor
from genesis_policy.cell import load_workcell
from genesis_policy.level import Box
from genesis_policy.runtime import (PHASE_DONE, PHASE_GRASP, PHASE_HOLD, PHASE_LIFT, PHASE_PRELIFT,
                                    PHASE_SETTLE, JointStates, RodPose, Runtime, clearance)

HAND_ORDER = [  # right_hand_controller order (ros2_controllers_genesis_wuji.yaml)
    'r_thumb_cmc_flex', 'r_thumb_cmc_abd', 'r_thumb_mcp', 'r_thumb_ip',
    'r_index_finger_mcp_flex', 'r_index_finger_mcp_abd', 'r_index_finger_pip', 'r_index_finger_dip',
    'r_middle_finger_mcp_flex', 'r_middle_finger_mcp_abd', 'r_middle_finger_pip', 'r_middle_finger_dip',
    'r_ring_finger_mcp_flex', 'r_ring_finger_mcp_abd', 'r_ring_finger_pip', 'r_ring_finger_dip',
    'r_pinky_mcp_flex', 'r_pinky_mcp_abd', 'r_pinky_pip', 'r_pinky_dip',
]
MOVEIT_LO = np.array([-1.047, -2.793, -2.583, -2.496, -2.67, -0.96, -0.96])
MOVEIT_HI = np.array([2.042, 0.628, 2.583, 0.209, 2.67, 0.96, 0.96])
# A right-arm configuration with the wrist above the (mirrored) bin: inside the
# workspace clamp (0.02..0.35 m above the 0.95 m table) and clear of every limit.
START_ARM = {'shoulderPitch_Right': 0.55, 'shoulderRoll_Right': -0.35, 'shoulderYaw_Right': 0.0,
             'elbow_Right': -1.35, 'wristYaw_Right': 0.0, 'wristPitch_Right': 0.0, 'wristRoll_Right': 0.0}


class FakeRobot:
    """Perfect tracking with a one-step delay: what the runtime commands is
    what the next joint state reports. Fingers included."""

    def __init__(self, robot, arm_joints, hand_order, arm_q, hand_q=None):
        self.robot = robot
        self.arm_joints = list(arm_joints)
        self.hand_order = list(hand_order)
        self.pos = {j: float(v) for j, v in zip(arm_joints, arm_q)}
        self.pos.update({j: 0.0 for j in hand_order} if hand_q is None else dict(zip(hand_order, hand_q)))

    def joint_states(self):
        return JointStates(dict(self.pos))

    def apply(self, arm, hand):
        self.pos.update(zip(self.arm_joints, arm))
        self.pos.update(zip(self.hand_order, hand))

    def wrist(self):
        self.robot.forward(self.robot.configuration(self.pos))
        return self.robot.wrist()

    def tips_z(self, level):
        self.robot.forward(self.robot.configuration(self.pos))
        return np.mean([(self.robot.frame_pose(link)[0] + self.robot.frame_pose(link)[1] @ off)[2]
                        for link, off in level.tips.values()])


def make_runtime(level, robot, actor, **kw):
    cell = load_workcell(mirror_y=True)
    return Runtime(level, robot, cell.obstacles(), cell.table_top, actor, HAND_ORDER, MOVEIT_LO, MOVEIT_HI, **kw)


def rod_near(fake, dz=-0.06):
    p, R = fake.wrist()
    return RodPose(np.array([p[0] + 0.0474, p[1] + 0.0901, p[2] + dz]), np.eye(3))


@pytest.fixture
def fake(robot, level):
    return FakeRobot(robot, level.arm_joints, HAND_ORDER, [START_ARM[j] for j in level.arm_joints])


def test_clearance_is_signed_and_points_at_the_box():
    box = Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    d, v = clearance(np.array([0.5, 0.5, 1.3]), 0.1, [box])
    assert d == pytest.approx(0.2) and np.allclose(v, [0, 0, -0.2])
    d, _ = clearance(np.array([0.5, 0.5, 0.5]), 0.1, [box])
    assert d == pytest.approx(-0.1)


def test_observation_layout_and_first_step(level, robot, fake):
    rt = make_runtime(level, robot, ZeroActor(level.layout.size))
    rod = rod_near(fake)
    arm, hand, info = rt.step(fake.joint_states(), rod, 0.0)
    obs = info['obs']
    assert obs.dtype == np.float32 and obs.shape == (74,)
    L = level.layout
    # the rod is 6 cm below the wrist in world: in the wrist frame it is 6 cm away
    assert np.linalg.norm(obs[L.offset('bolt_pos_rel'):L.offset('bolt_pos_rel') + 3]) == pytest.approx(
        np.linalg.norm(rod.position - info['wrist_p']), abs=1e-5)
    # two orthonormal columns
    rot = obs[L.offset('bolt_rot_rel'):L.offset('bolt_rot_rel') + 6]
    assert np.linalg.norm(rot[:3]) == pytest.approx(1.0, abs=1e-5) and abs(rot[:3] @ rot[3:]) < 1e-5
    assert np.all(obs[L.offset('wrist_lin_vel'):L.offset('wrist_ang_vel') + 3] == 0.0)
    assert np.all(obs[L.offset('last_action'):L.offset('last_action') + 8] == 0.0)
    assert obs[L.offset('ep_time')] == 0.0
    norm = obs[L.offset('arm_q_norm'):L.offset('arm_q_norm') + 7]
    assert np.all(np.abs(norm) <= 1.0)
    safety = obs[L.offset('safety'):]
    assert np.all(np.abs(safety[:12]) <= 1.0) and np.all(safety[12:] >= -0.1) and np.all(safety[12:] <= 0.5)
    assert safety[12] < safety[15]   # the tips are closer to the table than the upper arm
    # zero action: the wrist holds (the joints may drift in the null space
    # towards mid-range, as in the simulator); the hand heads for half closure
    fake.apply(arm, hand)
    assert np.linalg.norm(fake.wrist()[0] - info['wrist_p']) < 1e-3
    assert info['phase'] == 'grasp' and info['live'] and not info['latched']
    c = level.action_smoothing * 0.5
    assert hand[HAND_ORDER.index('r_thumb_mcp')] == pytest.approx(0.628 + c * (0.942 - 0.628))
    assert hand[HAND_ORDER.index('r_thumb_cmc_abd')] == -0.85
    assert hand[HAND_ORDER.index('r_pinky_pip')] == 0.0


def test_hand_targets_land_in_the_controller_order(level, robot, fake):
    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [0, 0, 0, 0, 0, 0, 1.0, -1.0]))
    rod = rod_near(fake)
    for k in range(60):
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
    # thumb closure -> 1 (closed), fingers -> 0 (open), held joints fixed
    for j in ('r_thumb_cmc_flex', 'r_thumb_mcp', 'r_thumb_ip'):
        assert hand[HAND_ORDER.index(j)] == pytest.approx(level.hand_closed[level.hand_joints.index(j)], abs=1e-3)
    for j in ('r_index_finger_pip', 'r_middle_finger_dip'):
        assert hand[HAND_ORDER.index(j)] == pytest.approx(level.hand_open[level.hand_joints.index(j)], abs=1e-3)
    assert hand[HAND_ORDER.index('r_thumb_cmc_abd')] == -0.85
    for j in ('r_ring_finger_mcp_flex', 'r_pinky_dip', 'r_index_finger_mcp_abd'):
        assert hand[HAND_ORDER.index(j)] == 0.0
    assert np.allclose(info['closure'], [1.0, 0.0], atol=1e-3)


def test_ik_follows_a_wrist_step_command(level, robot, fake):
    """+x for 30 steps: the wrist moves +x by about 30 effective steps (the
    first-order smoothing eats a few), nothing else, and the IK converges."""
    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [1.0, 0, 0, 0, 0, 0, -1.0, -1.0]))
    rod = rod_near(fake)
    p0, R0 = fake.wrist()
    for k in range(30):
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
        assert not info['ik_failed'] and info['clipped'] == 0
    p1, R1 = fake.wrist()
    expect = 30 * level.effective_translation
    assert expect * 0.85 < p1[0] - p0[0] <= expect + 1e-6
    assert abs(p1[1] - p0[1]) < 2e-3 and abs(p1[2] - p0[2]) < 2e-3
    assert np.linalg.norm(so3.log(R1 @ R0.T)) < 0.02
    assert info['track_err'] < 1.5 * level.effective_translation   # one step of lead on a one-step-delay robot


def test_ik_follows_a_rotation_command(level, robot, fake):
    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [0, 0, 0, 0, 0, 1.0, -1.0, -1.0]))
    rod = rod_near(fake)
    p0, R0 = fake.wrist()
    for k in range(20):
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
    p1, R1 = fake.wrist()
    rotvec = so3.log(R1 @ R0.T)
    expect = 20 * level.effective_rotation * level.yaw_rate_scale
    assert expect * 0.8 < rotvec[2] <= expect + 1e-6     # about world z
    assert np.linalg.norm(rotvec[:2]) < 0.01
    assert np.linalg.norm(p1 - p0) < 3e-3


def test_joint_clip_uses_the_narrow_limits(level, robot, fake):
    """Pushing the wrist up pulls shoulderPitch towards its MoveIt stop at
    -1.047 + margin, never towards the URDF's -3.61."""
    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [0, 0, 1.0, 0, 0, 0, -1.0, -1.0]))
    rod = rod_near(fake)
    lowest = 0.0
    for k in range(400):
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
        lowest = min(lowest, arm[0])
        assert np.all(arm >= MOVEIT_LO + level.joint_margin - 1e-9) and np.all(arm <= MOVEIT_HI - level.joint_margin + 1e-9)
    assert lowest > -1.047 - 1e-9


def test_workspace_clamp_bounds_the_target(level, robot, fake):
    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [0, -1.0, 0, 0, 0, 0, -1.0, -1.0]))
    rod = rod_near(fake)
    centre = rod.position[:2] + level.nominal_wrist_offset_xy
    for k in range(200):
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
        assert np.linalg.norm(info['target_p'][:2] - centre) <= level.workspace_radius_xy + 1e-9
        assert info['target_p'][2] >= rt.table_top + level.workspace_z_min - 1e-9
    assert np.linalg.norm(info['target_p'][:2] - centre) == pytest.approx(level.workspace_radius_xy, abs=1e-6)


def run_episode(level, robot, fake, actor, rod_fn, steps, tip_source=None):
    rt = make_runtime(level, robot, actor, tip_source=tip_source)
    trace = []
    for k in range(steps):
        rod = rod_fn(k, fake)
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
        trace.append(info)
    return rt, trace


def test_phase_machine_times_out_lifts_and_retries(level, robot, fake):
    """No latch (hand never closes), rod never rises: grasp times out at
    grasp_max_steps, settle, an empty pre-lift, a 10 cm lift, then the retry
    reopens and restarts; after retry_max the episode ends in hold."""
    P = level.phase
    rod0 = rod_near(fake)
    actor = ScriptedActor(lambda k, o: [0, 0, 0, 0, 0, 0, -1.0, -1.0])
    rt, trace = run_episode(level, robot, fake, actor, lambda k, f: rod0, P.grasp_max_steps + P.scripted_steps + 5)
    phases = [t['phase'] for t in trace]
    g = P.grasp_max_steps
    assert phases[g - 1] == 'grasp' and phases[g] == 'settle'
    assert phases[g + P.settle_steps] == 'prelift'
    assert phases[g + P.lift_start] == 'lift'
    lift_end = g + P.lift_start + P.lift_steps
    # the lift raised the wrist by lift_height
    z_before = trace[g + P.lift_start]['wrist_p'][2]
    z_after = trace[lift_end]['wrist_p'][2]
    assert z_after - z_before == pytest.approx(P.lift_height_m, abs=0.004)
    # the rod did not come up: a retry fires at the end of the lift
    assert trace[lift_end]['retry_now'] and trace[lift_end]['attempts'] == 1
    assert phases[lift_end] == 'grasp' and trace[lift_end]['grasp_end'] == lift_end + P.grasp_retry_steps
    assert not trace[lift_end - 1]['latched']


def test_latch_grip_check_and_success(level, robot, fake):
    """The hand closes on a rod that then follows the fingertips: the latch
    fires, the test lift passes the grip check, the lift and hold succeed."""
    P = level.phase
    state = {'rod': rod_near(fake, dz=-0.02), 'tips0': None, 'latched_k': None}

    def rod_fn(k, f):
        # once latched, the rod rides with the fingertips
        if state['latched_k'] is not None:
            dz = f.tips_z(level) - state['tips0']
            r = state['rod']
            return RodPose(r.position + np.array([0.0, 0.0, dz]), r.rotation)
        return state['rod']

    rt = make_runtime(level, robot, ScriptedActor(lambda k, o: [0, 0, 0, 0, 0, 0, 1.0, 1.0]))
    trace = []
    for k in range(P.grasp_max_steps + P.scripted_steps):
        rod = rod_fn(k, fake)
        arm, hand, info = rt.step(fake.joint_states(), rod, k * level.dt)
        fake.apply(arm, hand)
        trace.append(info)
        if info['latched_now']:
            state['latched_k'] = k
            state['tips0'] = fake.tips_z(level)
        if info['done']:
            break
    assert state['latched_k'] is not None and state['latched_k'] < P.grasp_max_steps
    g = state['latched_k']
    assert trace[g]['grasp_end'] == g
    assert trace[g + 1]['phase'] == 'settle'
    assert trace[g + P.lift_start]['phase'] == 'lift'
    assert not rt.grip_fired
    assert rt.attempts == 0
    assert trace[-1]['done'] and trace[-1]['success']
    # the hand stayed at the latched closure through the lift
    assert np.allclose(trace[-1]['closure'], trace[g]['closure'])


def test_grip_check_fires_when_the_rod_stays_down(level, robot, fake):
    P = level.phase
    rod0 = rod_near(fake, dz=-0.02)
    rt, trace = run_episode(level, robot, fake, ScriptedActor(lambda k, o: [0, 0, 0, 0, 0, 0, 1.0, 1.0]),
                            lambda k, f: rod0, P.grasp_max_steps)
    latched = [t['step'] for t in trace if t['latched_now']]
    assert latched
    g = latched[0]
    # the grip check fired at the end of the test lift and the retry reopened
    # the hand; the actor keeps closing, so the latch and the check repeat
    # until retry_max is used up
    assert rt.grip_fired and 1 <= rt.attempts <= P.retry_max
    assert trace[g + P.lift_start]['retry_now'] and trace[g + P.lift_start]['attempts'] == 1
    # reset to zero by the retry, then one smoothing step towards the actor's +1
    assert np.allclose(trace[g + P.lift_start]['closure'], level.action_smoothing)
    assert trace[g + P.lift_start]['grasp_end'] == g + P.lift_start + P.grasp_retry_steps


def test_step_is_deterministic(level, robot):
    def episode():
        fake = FakeRobot(robot, level.arm_joints, HAND_ORDER, [START_ARM[j] for j in level.arm_joints])
        rng = np.random.default_rng(7)
        actor = ScriptedActor(lambda k, o: rng.uniform(-1, 1, 8))
        rt = make_runtime(level, robot, actor)
        out = []
        for k in range(50):
            arm, hand, info = rt.step(fake.joint_states(), rod_near(fake), k * level.dt)
            fake.apply(arm, hand)
            out.append(np.concatenate([arm, hand, info['obs']]))
        return np.array(out)
    a, b = episode(), episode()
    assert np.array_equal(a, b)


def test_tip_source_feeds_the_forces_and_contacts(level, robot, fake):
    rt = make_runtime(level, robot, ZeroActor(level.layout.size),
                      tip_source=lambda js: (np.array([4.0, 80.0, -4.0]), np.array([1.0, 0.0, 0.0])))
    _, _, info = rt.step(fake.joint_states(), rod_near(fake), 0.0)
    L = level.layout
    forces = info['obs'][L.offset('tip_forces'):L.offset('tip_forces') + 3]
    assert np.allclose(forces, [0.1, 2.0, -0.1])
    assert np.array_equal(info['obs'][L.offset('contacts'):L.offset('contacts') + 3], [1, 0, 0])


def test_hand_order_must_be_the_controllers(level, robot):
    cell = load_workcell(mirror_y=True)
    with pytest.raises(ValueError):
        Runtime(level, robot, cell.obstacles(), cell.table_top, ZeroActor(74), HAND_ORDER[:-1] + ['x'])


def test_phases_by_step(level, robot, fake):
    rt = make_runtime(level, robot, ZeroActor(74))
    rt.reset(fake.joint_states(), rod_near(fake), 0.0)
    P = level.phase
    g = P.grasp_max_steps
    assert rt.phase(0) == PHASE_GRASP and rt.phase(g - 1) == PHASE_GRASP
    assert rt.phase(g) == PHASE_SETTLE and rt.phase(g + P.settle_steps) == PHASE_PRELIFT
    assert rt.phase(g + P.lift_start) == PHASE_LIFT
    assert rt.phase(g + P.lift_start + P.lift_steps) == PHASE_HOLD
    assert rt.phase(g + P.scripted_steps) == PHASE_DONE


def test_lift_dir_from_rod():
    """The up-normal of a leaning rod, world z for a lying or standing one, either axis sign."""
    import math
    from genesis_policy import so3
    from genesis_policy.runtime import RodPose, lift_dir_from_rod
    lo, hi = math.radians(10), math.radians(60)
    for yaw in (0.0, 0.7, -2.0):
        u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        n = np.cross([0.0, 0.0, 1.0], u)
        for tilt_deg in (18.0, 30.0, 40.0):
            t = math.radians(tilt_deg)
            a = math.cos(t) * u + math.sin(t) * np.array([0.0, 0.0, 1.0])
            R = np.column_stack([n, np.cross(a, n), a])          # z column = the rod axis
            for flip in (1.0, -1.0):
                Rf = R.copy()
                Rf[:, 1:] *= flip                                  # the other end up, still a rotation
                m = lift_dir_from_rod(RodPose(np.zeros(3), Rf), math.pi / 2, lo, hi)
                assert np.allclose(m, math.cos(t) * np.array([0.0, 0.0, 1.0]) - math.sin(t) * u, atol=1e-9)
                assert abs(m @ a) < 1e-9 and m[2] > 0.0
            # roll_cap caps the roll; 0 never rolls
            m = lift_dir_from_rod(RodPose(np.zeros(3), R), math.radians(20), lo, hi)
            c = min(t, math.radians(20))
            assert np.allclose(m, math.cos(c) * np.array([0.0, 0.0, 1.0]) - math.sin(c) * u, atol=1e-9)
            assert np.allclose(lift_dir_from_rod(RodPose(np.zeros(3), R), 0.0, lo, hi), [0.0, 0.0, 1.0])
    for tilt_deg in (0.0, 5.0, 75.0, 90.0):                        # lying or standing: straight up
        t = math.radians(tilt_deg)
        R = so3.rpy(0.0, math.pi / 2 - t, 0.0)
        assert np.allclose(lift_dir_from_rod(RodPose(np.zeros(3), R), math.pi / 2, lo, hi), [0.0, 0.0, 1.0])
