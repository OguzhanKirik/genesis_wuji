"""Replay adam-rl's recorded episodes through the runtime and the ONNX actor
and compare the joint targets. Skips, loudly, until the fixture exists.

The recording (episode.npz, written by adam-rl's tools/export_policy.py
--record N) carries N consecutive episodes of one simulated env, per step:
the joint states (position, velocity, effort = the simulator's applied
torque), the estimated rod pose, the scene the safety observation measured
(boxes and neighbour-rod capsules), the observation, the action and the joint
targets of the first physics substep. Per episode: the IK target seed, the
workspace centre and the lift direction. Everything is in adam-moveit's
planning frame `world`. Three comparisons, each isolating one stage:

  action        the ONNX actor on the recorded observation against the
                recorded action                    (the export)
  observation   the runtime's observation on the recorded state against the
                recorded observation               (FK, frames, layout)
  targets       runtime.step driven by the recorded actions against the
                recorded joint targets             (IK, phase machine, latch)

The URDF must be the xacro's (with the world joint); a pre-generated file
through GENESIS_POLICY_URDF must be one too.

Looked for at $GENESIS_POLICY_EPISODE, else <policy_dir>/episode.npz for
every directory under /handoff/policy.
"""
import glob
import os

import numpy as np
import pytest

KEYS = ('joint_names', 'joint_pos', 'joint_vel', 'joint_effort', 'rod_pos', 'rod_quat_xyzw', 'obs', 'action',
        'arm_targets', 'hand_target_names', 'hand_targets', 'time', 'episode', 'lift_dir',
        'box_lo', 'box_hi', 'cap_p0', 'cap_p1', 'cap_r',
        'target_seed_pos', 'target_seed_quat_xyzw', 'workspace_centre')
OBS_TOL = 1e-3
# The simulator's link velocities are PhysX's, which agree with J(q) qdot of its
# own joint velocities to ~0.1 % (median; under 1 % at worst, float32 solver):
# 2.6e-3 rad/s at 0.9 rad/s on the L32 pick at level 39. Positions match to 1e-6.
OBS_TOL_TERM = {'wrist_lin_vel': 5e-3, 'wrist_ang_vel': 5e-3}
# plus this share of the value for the velocity terms: PhysX's link velocity and J qdot part by ~0.1 % in free motion,
# up to 0.5 % where the hand strikes a rod in a full bin (2 of 4,400 steps of the level-52 recording)
OBS_REL_TOL_TERM = {'wrist_lin_vel': 5e-3, 'wrist_ang_vel': 5e-3}
# The safety vectors point at the closest obstacle point over every proxy
# sphere; when two candidates tie to float32 precision (seen: 6e-8 m, two palm
# spheres to one neighbour rod) the simulator and the runtime may pick
# different ones and the vector jumps. The distances stay exact, so they are
# checked strictly and the vectors may flip on a few steps.
SAFETY_FLIP_MAX = 0.005
ACTION_TOL = 1e-4
TARGET_TOL = 1e-4


def find_episode():
    env = os.environ.get('GENESIS_POLICY_EPISODE')
    if env:
        return env if os.path.exists(env) else None
    for path in sorted(glob.glob('/handoff/policy/*/episode.npz')):
        return path
    return None


EPISODE = find_episode()


@pytest.mark.skipif(EPISODE is None, reason='no episode.npz: adam-rl has not handed over the recorded episode '
                                            '(SIM_TO_REAL.md 3.1); set GENESIS_POLICY_EPISODE=<file> or put it '
                                            'next to actor.onnx under /handoff/policy/<name>/')
def test_replay_matches_the_recording():
    from genesis_policy import level as lv
    from genesis_policy import so3
    from genesis_policy.actor import OnnxActor
    from genesis_policy.level import Box
    from genesis_policy.model import RobotModel, urdf_string
    from genesis_policy.cell import load_workcell
    from genesis_policy.runtime import Capsule, JointStates, RodPose, Runtime

    data = np.load(EPISODE, allow_pickle=False)
    missing = [k for k in KEYS if k not in data]
    assert not missing, f'{EPISODE} lacks {missing}; re-export with adam-rl tools/export_policy.py --record N'
    policy_dir = os.path.dirname(EPISODE)
    level = lv.load(os.path.join(policy_dir, 'level.yaml'))
    actor = OnnxActor(os.path.join(policy_dir, 'actor.onnx'), obs_dim=level.layout.size)
    robot = RobotModel(urdf_string('genesis_wuji'), level.arm_joints, level.wrist_link, level.wrist_offset)
    table_top = load_workcell(mirror_y=True).table_top
    names = [str(n) for n in data['joint_names']]
    hand_order = [str(n) for n in data['hand_target_names']]
    n = len(data['time'])

    # 1. the actor reproduces the recorded actions from the recorded observations
    for k in range(n):
        assert np.allclose(actor(data['obs'][k]), data['action'][k], atol=ACTION_TOL), f'step {k}: action'

    # 2 and 3. the runtime, driven by the recorded actions, reproduces the
    #          observations and the joint targets, one episode at a time
    layout = level.layout
    failures = []
    flips = []
    s0 = layout.offset('safety') if 'safety' in layout.names else None
    for e in np.unique(data['episode']):
        steps = np.nonzero(data['episode'] == e)[0]
        cursor = {'k': int(steps[0])}

        def replay_actor(obs):
            k = cursor['k']
            err = np.abs(obs - data['obs'][k])
            if s0 is not None and err[s0:s0 + 12].max() > OBS_TOL:
                flips.append(k)
                err[s0:s0 + 12] = 0.0
            for nm, size in layout.terms:
                sl = slice(layout.offset(nm), layout.offset(nm) + size)
                tol = OBS_TOL_TERM.get(nm, OBS_TOL) + OBS_REL_TOL_TERM.get(nm, 0.0) * np.abs(data['obs'][k][sl])
                te = err[sl].max()
                if np.any(err[sl] > tol) and len(failures) < 5:
                    failures.append(f'episode {e} step {k}: observation {nm} off by {te:.2e}')
            cursor['k'] += 1
            return data['action'][k]

        rt = Runtime(level, robot, [], table_top, replay_actor, hand_order, level.arm_q_lo, level.arm_q_hi,
                     finger_velocity='reported', wrist_velocity='jacobian')
        seed = (data['target_seed_pos'][e], so3.from_quat_xyzw(data['target_seed_quat_xyzw'][e]))
        for k in steps:
            rt.set_scene([Box(lo, hi) for lo, hi in zip(data['box_lo'][k], data['box_hi'][k])],
                         [Capsule(p0, p1, float(r)) for p0, p1, r in zip(data['cap_p0'][k], data['cap_p1'][k], data['cap_r'][k])])
            js = JointStates(dict(zip(names, data['joint_pos'][k])), dict(zip(names, data['joint_vel'][k])),
                             dict(zip(names, data['joint_effort'][k])))
            rod = RodPose(np.asarray(data['rod_pos'][k], float), so3.from_quat_xyzw(data['rod_quat_xyzw'][k]))
            arm, hand, info = rt.step(js, rod, float(data['time'][k]), lift_dir=data['lift_dir'][k],
                                      target_seed=seed, workspace_centre=data['workspace_centre'][e])
            if not np.allclose(arm, data['arm_targets'][k], atol=TARGET_TOL) and len(failures) < 5:
                failures.append(f'episode {e} step {k}: arm targets off by {np.abs(arm - data["arm_targets"][k]).max():.2e} ({info["phase"]})')
            if not np.allclose(hand, data['hand_targets'][k], atol=TARGET_TOL) and len(failures) < 5:
                failures.append(f'episode {e} step {k}: hand targets off by {np.abs(hand - data["hand_targets"][k]).max():.2e} ({info["phase"]})')
    if len(flips) > max(2, SAFETY_FLIP_MAX * n):
        failures.append(f'safety vectors differ on {len(flips)} of {n} steps (first {flips[:5]}): more than tie flips')
    assert not failures, '\n'.join(failures)
