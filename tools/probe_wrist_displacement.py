#!/usr/bin/env python3
"""How far can the wrist be displaced from the solved pinch and still be valid?

Host-side, numpy + scipy, no Isaac.

The displacement-curriculum design keeps the BOLT fixed and offsets the WRIST's
starting pose, so the policy's job is to find the grasping position and
orientation rather than to track a moving object. The open question is the
magnitude: the solved pinch guarantees every fingertip clears the table by
1.1 cm over the whole synergy stroke, and displacing the wrist -- especially
rotating it -- breaks that guarantee. Too little displacement and there is
nothing to learn; too much and the episode starts in a state no closure can
rescue, or with fingers already through the table.

For each candidate displacement this reports:

    clearance   the lowest fingertip's height above the table at the OPEN
                posture, which is where every episode resets. Negative means
                the reset pose is already inside the table.
    reach       whether the arm can hold the displaced pose, and the minimum
                normalised joint slack if it can
    recover     the Cartesian travel needed to get back, against the budget
                the grasp phase actually has

    python3 tools/probe_wrist_displacement.py
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fp = _load("find_pinch_pose")
pp = _load("solve_pipeline_pregrasp")

# The staged pipeline's solved pose (wuji_pipeline_env_cfg).
WRIST_XY = (0.50, -0.20)
WRIST_Z = 1.0092
WRIST_ROT = R.from_quat([-0.1821, -0.7296, 0.4661, 0.4661])
TABLE_TOP = 0.95
OPEN_CURL, THUMB_ABD = 0.40, -0.85
HELD = ("ring_finger", "pinky")          # flat, not curled
# Per-step limits the wrist actually has (velocity caps bind, see LimitsCfg).
STEP_T, STEP_R = 0.00167, 0.01667
GRASP_STEPS = 90

CLOSED_CURL = 0.60
STROKE_SAMPLES = 9

joints = fp.kin.parse(fp.kin.URDF)
# Fingertip positions in the WRIST frame, sampled across the WHOLE synergy
# stroke -- not just at the open reset posture. Curling the fingers brings
# their tips DOWN in this tilted pose, which is how Experiment 2 logged a
# finger-table collision in every episode before ring and little were
# flattened. Checking only the reset pose would miss exactly that.
STROKE = []
for _c in [OPEN_CURL + (CLOSED_CURL - OPEN_CURL) * i / (STROKE_SAMPLES - 1)
           for i in range(STROKE_SAMPLES)]:
    STROKE.append(fp.tips(joints, fp.posture(_c, _c, THUMB_ABD, held=HELD)))
TIPS = STROKE[0]

arm = pp.Arm(pp._load_solver(), waist_max=1.36)
arm.lo[-1], arm.hi[-1] = 1.05 - 1e-6, 1.05 + 1e-6      # the held lean


def clearance(pos, rot: R) -> tuple[float, str]:
    """Lowest fingertip height above the table over the WHOLE synergy stroke."""
    M = rot.as_matrix()
    best, who = 1e9, ""
    for tips_at_c in STROKE:
        for n, t in tips_at_c.items():
            z = pos[2] + (M @ t)[2]
            if z - TABLE_TOP < best:
                best, who = z - TABLE_TOP, n
    return best, who


def report(label, dpos, drot_deg, axis):
    pos = np.array([WRIST_XY[0], WRIST_XY[1], WRIST_Z]) + np.array(dpos)
    rot = (R.from_rotvec(np.radians(drot_deg) * np.array(axis)) * WRIST_ROT
           if drot_deg else WRIST_ROT)
    c, who = clearance(pos, rot)
    r = arm.solve(pos, rot, seeds=8)
    ok_reach = r["pos_err"] <= 0.002 and np.degrees(r["ori_err"]) <= 1.0
    # Travel back to the solved pose, in steps.
    n_t = np.linalg.norm(dpos) / STEP_T
    n_r = abs(np.radians(drot_deg)) / STEP_R
    steps = max(n_t, n_r)
    verdict = ("OK" if (c > 0.002 and ok_reach and steps < GRASP_STEPS * 0.5) else
               "table" if c <= 0.002 else "reach" if not ok_reach else "slow")
    print(f"  {label:22s} {c * 100:+7.2f} {who:>7s} "
          f"{'yes' if ok_reach else 'NO ':>4s} {r['s_min']:6.3f} {steps:7.0f} {verdict:>7s}")


print(__doc__.split("\n\n")[0])
print(f"\nsolved pose: wrist ({WRIST_XY[0]}, {WRIST_XY[1]}, {WRIST_Z}), table top {TABLE_TOP}")
c0, who0 = clearance(np.array([WRIST_XY[0], WRIST_XY[1], WRIST_Z]), WRIST_ROT)
print(f"undisplaced clearance over the whole stroke: {c0 * 100:.2f} cm (lowest: {who0})")
print(f"grasp-phase budget: {GRASP_STEPS} steps x {STEP_T * 1000:.2f} mm = "
      f"{GRASP_STEPS * STEP_T * 100:.1f} cm, or x {np.degrees(STEP_R):.2f} deg = "
      f"{GRASP_STEPS * np.degrees(STEP_R):.0f} deg")
print(f"\n  {'displacement':22s} {'clear cm':>7s} {'lowest':>7s} {'reach':>4s} {'s_min':>6s} "
      f"{'steps':>7s} {'verdict':>7s}")

print("\n  -- translation --")
for d in (0.01, 0.02, 0.03):
    for ax, nm in ((0, "x"), (1, "y"), (2, "z")):
        for sgn in (+1, -1):
            v = [0.0, 0.0, 0.0]; v[ax] = sgn * d
            report(f"{sgn * d * 100:+.0f} cm {nm}", v, 0.0, (0, 0, 1))

print("\n  -- rotation about world axes --")
for deg in (5, 10, 15, 20):
    for axis, nm in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")):
        for sgn in (+1, -1):
            report(f"{sgn * deg:+d} deg about {nm}", [0, 0, 0], sgn * deg, axis)
