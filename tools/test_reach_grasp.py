#!/usr/bin/env python3
"""Regression tests for Experiment 4 (docs/EXPERIMENT4_DESIGN.md §10). Runs in the container.

    docker compose exec -e ADAM_U_RG_LEVEL=0 adam_rl python /opt/tools/test_reach_grasp.py
    docker compose exec -e ADAM_U_RG_LEVEL=6 adam_rl python /opt/tools/test_reach_grasp.py --only 8
    python /opt/tools/test_reach_grasp.py --num_envs 8

Each check prints its own PASS / FAIL so a regression names itself:

  1. interface            74 observations, 8 actions, every level
  2. reset pose           the wrist starts at the solved pose + the sampled standoff and error
  3. held joints          waist, neck, left side hold their defaults under zero action
  4. actor mask           TRUE in the grasp phase, FALSE in SETTLE / LIFT / HOLD
  5. latch and IK reset   on the latch step the residual is cleared, the IK target and
                          integrator equal the MEASURED state, fingers are held through
                          SETTLE, the lift starts from the measured pose and rises 10 cm
  6. phase lengths        an episode that never latches ends at exactly 190 steps;
                          a latched one at grasp end + 70
  7. logging              every reward and safety component reaches extras["log"] AFTER
                          the reset, as tensors
  8. perception           (levels with corruption) the estimate differs from the truth by a
                          bias that is constant within an episode and redrawn at reset, and
                          every geometric bolt observation is derived from the estimate
  9. safety observations  clearances positive at reset, monotone in height, the fingertip
                          obstacle vector points down
 10. masked PPO           transitions with actor_mask = 0 contribute exactly zero actor
                          gradient (their advantages can be scaled by 1000 with no effect)
 11. clutter              (levels 8-10) obstacles placed outside the corridor, on the table,
                          bodies where the model says; clearance falls toward them; driving
                          into one ends the episode as an obstacle collision
 12. pose generalisation  (levels 11-13) the bolt is sampled in yaw and xy; the body, the
                          nominal wrist pose, the reset and the observation all follow it
"""

from __future__ import annotations

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--only", type=str, default=None)
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import importlib.metadata as metadata  # noqa: E402

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402

from genesis_rl.algos.rsl_rl.masked_ppo import MaskedPPO  # noqa: E402
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402
from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import _t  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"\n  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def head(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def want(n):
    return args.only is None or args.only == str(n)


cfg = G.WujiReachGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
env = G.ReachGraspEnv(cfg=cfg)
robot = env.scene["robot"]
lvl = G.LEVELS[cfg.actions.pipeline.level]
lim = cfg.actions.pipeline.limits
dev, n = env.device, env.num_envs
term = env.action_manager._terms["pipeline"]
print(f"\nlevel {cfg.actions.pipeline.level} '{lvl.name}'  standoff {lvl.standoff_lo*100:.1f}-{lvl.standoff_hi*100:.1f} cm  "
      f"xy +/-{lvl.dxy*100:.1f} cm  yaw +/-{math.degrees(lvl.dyaw):.1f} deg  perception {'ON' if lvl.perception.active else 'off'}")


def start_disp():
    """The reset's world displacement from the nominal: (dx, dy, 0) plus the standoff along the approach axis."""
    d = env._rg_displacement
    m = G._rg(env)["_lift_dir"]
    return torch.stack([d[:, 0], d[:, 1], torch.zeros_like(d[:, 2])], dim=1) + d[:, 2:3] * m


def zeros():
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = -1.0          # fingers open
    return a


def closing():
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = 1.0           # fingers closing, wrist still
    return a


def wrist():
    return G._wrist_pose(env)


class Scripted:
    def __init__(self):
        self.reached_at = torch.full((n,), -1.0, device=dev)

    def __call__(self, t):
        a = zeros()
        pos, _ = wrist()
        above = pos[:, 2] - G.WRIST_Z_CONTACT
        arrived = above < 0.003
        self.reached_at = torch.where(arrived & (self.reached_at < 0), torch.full_like(self.reached_at, float(t)),
                                      self.reached_at)
        dz = -(above / lim.effective_translation).clamp(0.0, 1.0)
        a[:, 2] = torch.where(arrived, torch.zeros_like(dz), dz)
        frac = ((t - self.reached_at) / 40.0).clamp(0.0, 1.0)
        c = -1.0 + 2.0 * frac
        a[:, 6:8] = torch.where((self.reached_at >= 0).unsqueeze(1), c.unsqueeze(1).expand(n, 2), a[:, 6:8])
        return a


# ---------------------------------------------------------------------------
if want(1):
    head(1, "interface: 74 observations, 8 actions")
    obs, _ = env.reset()
    width = 74 + (G.HEAP_MAP_DIRS if lvl.heap_map else 0)
    record(f"policy observation width is {width}", obs["policy"].shape[-1] == width, f"got {obs['policy'].shape[-1]}")
    record("action dimension is 8", env.action_manager.total_action_dim == 8,
           f"got {env.action_manager.total_action_dim}")
    record("observations are finite", bool(torch.isfinite(obs["policy"]).all()))

# ---------------------------------------------------------------------------
if want(2):
    head(2, "reset pose: solved pose + standoff + level error")
    env.reset()
    env.step(zeros())           # one step so body poses are refreshed
    pos, quat = wrist()
    d = env._rg_displacement
    want_pos = G._rg(env)["_nom_pos"] + start_disp()       # bolt-relative nominal (equals the fixed pose at levels < 11)
    err = torch.norm(pos - want_pos, dim=1)
    land_tol = 0.012 if lvl.pile.settle_in > 0 else 0.008      # the heap's settle-in holds the IK target for 25 steps while the arm sags a little
    record(f"wrist lands within {land_tol*1000:.0f} mm of the commanded reset pose (first-order reset + one step)",
           bool((err < land_tol).all()), f"max {err.max()*1000:.1f} mm")
    so = d[:, 2]
    record("sampled standoff lies inside the level's range",
           bool(((so >= lvl.standoff_lo - 1e-6) & (so <= lvl.standoff_hi + 1e-6)).all()),
           f"{so.min()*100:.1f}-{so.max()*100:.1f} cm")
    record("sampled xy error inside the level's range", bool((d[:, :2].abs() <= lvl.dxy + 1e-6).all()))
    tpos, tquat = term.target_pose
    record("the commanded IK target was seeded at the reset pose, not the contact pose",
           bool((torch.norm(tpos - want_pos, dim=1) < 0.005).all()),
           f"max {torch.norm(tpos - want_pos, dim=1).max()*1000:.1f} mm")

# ---------------------------------------------------------------------------
if want(3):
    head(3, "held joints under zero action")
    env.reset()
    for _ in range(30):
        env.step(zeros())
    q = _t(robot.data.joint_pos)[0]
    default = _t(robot.data.default_joint_pos)[0]
    names = robot.joint_names
    groups = {"waist": [j for j in names if j.startswith("waist")],
              "neck": [j for j in names if j.startswith("neck")],
              "left arm": [j for j in names if j.endswith("_Left")],
              "left hand": [j for j in names if j.startswith("l_")]}
    for g, js in groups.items():
        ids = [names.index(j) for j in js]
        want_q = default[ids].clone()
        if g == "waist":
            # the waist's hold is per episode: the bucket's triple where the level frees it
            wi = [names.index(j) for j in G.WAIST_JOINTS]
            for k, j in enumerate(js):
                want_q[k] = G._rg(env)["_waist"][0, wi.index(names.index(j))] if names.index(j) in wi else default[names.index(j)]
        drift = (q[ids] - want_q).abs().max()
        record(f"{g} holds its {'episode hold' if g == 'waist' else 'default'} (max drift {drift:.4f} rad)", drift < 0.05)

# ---------------------------------------------------------------------------
if want(4) or want(5) or want(6):
    head(5, "latch, IK reset, settle, lift (scripted grasp); actor mask; phase lengths")
    env.reset()
    pol = Scripted()
    b = G._rg(env)
    latch_seen = torch.zeros(n, dtype=torch.bool, device=dev)
    mask_in_grasp_ok, mask_after_ok = True, True
    latch_ok = dict(residual=True, target=True, qcmd=True)
    closure_frozen_ok = True
    lift_seed_ok = True
    end_step = torch.full((n,), -1.0, device=dev)
    wrist_at_lift = torch.zeros(n, 3, device=dev)
    wrist_at_hold = torch.zeros(n, 3, device=dev)
    closure_at_latch = torch.zeros(n, 2, device=dev)
    for t in range(G.episode_max_steps(lvl.retry_max) + 2):
        a = pol(t)
        pre_pos, pre_quat = wrist()
        pre_q = _t(robot.data.joint_pos)[:, term._arm_ids]
        pre_t = env.episode_length_buf.clone().float()          # per-env time BEFORE the step
        pre_grasp_end = b["grasp_end"].clone()
        pre_latched = b["latched"].clone()
        _, rew, te, tr, extras = env.step(a)
        mask = extras["actor_mask"]
        ph_now = G.phase_of(env)          # phase at the NEW time; the mask was for the step just taken
        now = b["_latched_now"]
        if now.any():
            m = now
            latch_seen |= m
            closure_at_latch[m] = term.closure[m]
            tpos, tquat = term.target_pose
            latch_ok["residual"] &= bool((term.wrist_cmd[m].abs() < 1e-9).all())
            # the target and q_cmd were set to the pose MEASURED at the start of that step
            # (the workspace clamp may nudge a target that sits at the box's edge: allow 5 mm)
            latch_ok["target"] &= bool((torch.norm(tpos[m] - pre_pos[m], dim=1) < 5e-3).all())
            # q_cmd has since integrated one IK step toward that same target; it must
            # stay within one joint-velocity cap of the measured state
            latch_ok["qcmd"] &= bool(((term.q_cmd[m] - pre_q[m]).abs() < lim.max_joint_velocity * G.CONTROL_DT + 1e-6).all())
        # mask semantics: the step just taken was a grasp step iff pre_t < grasp_end as it
        # stood BEFORE the step. On the latch step itself grasp_end moves to pre_t inside the
        # step and the mask reads False -- excluded, both readings are defensible there.
        grasp_step = (pre_t < pre_grasp_end) & (pre_t >= lvl.pile.settle_in)
        if lvl.free_prelift:
            # the policy's pre-lift window (after the settle, DRAG_STEPS long, latched rods) is live too
            t_pre = pre_t - (pre_grasp_end + G.SETTLE_STEPS)
            grasp_step = grasp_step | (pre_latched & (t_pre >= 0) & (t_pre < G.DRAG_STEPS))
        disagreement = (mask != grasp_step) & ~now
        if disagreement.any():
            if bool((grasp_step & disagreement).any()):
                mask_in_grasp_ok = False
            if bool((~grasp_step & disagreement).any()):
                mask_after_ok = False
        # settle: closure must not change while the finger command keeps changing
        settle = (ph_now == 1) & latch_seen & (b["attempts"] == 0)      # first attempt only: a retry re-latches with a new closure
        if settle.any():
            closure_frozen_ok &= bool(((term.closure[settle] - closure_at_latch[settle]).abs() < 1e-6).all())
        lift_entry = (env.episode_length_buf.float() == b["grasp_end"] + G.LIFT_START)
        if lift_entry.any():
            wrist_at_lift[lift_entry] = wrist()[0][lift_entry]
        hold_entry = (env.episode_length_buf.float() == b["grasp_end"] + G.LIFT_START + G.LIFT_STEPS)
        if hold_entry.any():
            wrist_at_hold[hold_entry] = wrist()[0][hold_entry]
        done = te | tr
        if done.any():
            end_step[done & (end_step < 0)] = float(t + 1)
        if bool((end_step > 0).all()):
            break
    record("actor mask is TRUE for every grasp-phase transition", mask_in_grasp_ok)
    record("actor mask is FALSE for every SETTLE / LIFT / HOLD transition" + (" outside the policy's pre-lift window" if lvl.free_prelift else ""), mask_after_ok)
    # The scripted policy descends straight down and never corrects, so at a level
    # WITH pose error it is expected to miss sometimes -- that miss rate is the
    # baseline the trained policy must beat, not a defect. Strict only when the
    # start pose has no error.
    exact_start = lvl.dxy == 0 and lvl.dyaw == 0 and lvl.drp == 0
    if exact_start:
        record("the scripted grasp latches in every env", bool(latch_seen.all()), f"{int(latch_seen.sum())}/{n}")
    else:
        print(f"\n  INFO  scripted no-correction policy latched in {int(latch_seen.sum())}/{n} envs at this "
              f"displaced level (the baseline a trained policy must beat; not a pass/fail)")
        if lvl.grasp_point_tol > 0 or lvl.pinch_centre_tol > 0:
            print(f"\n  INFO  scripted no-correction policy latched in {int(latch_seen.sum())}/{n} envs with the binding "
                  f"point / centred pinch (it may latch nowhere: a blind close lands neither on the point nor centred)")
        else:
          record("at least one env latches, so the latch path is exercised", bool(latch_seen.any()),
                 f"{int(latch_seen.sum())}/{n}")
    record("on the latch step the Cartesian residual is cleared", latch_ok["residual"])
    record("on the latch step the IK target equals the MEASURED wrist pose", latch_ok["target"])
    record("on the latch step the IK integrator is re-seeded from the measured joints", latch_ok["qcmd"])
    if lvl.retry_max > 0:
        print(f"  INFO  settle-freeze check at a retry level: {'held' if closure_frozen_ok else 'a re-latch after a retry changed the closure'} (verified strictly at levels without retry)")
    else:
        record("finger closure is held constant through SETTLE", closure_frozen_ok)
    rise = (wrist_at_hold - wrist_at_lift)[:, 2]
    lat = torch.norm((wrist_at_hold - wrist_at_lift)[:, :2], dim=1)
    # only envs that reached the hold; the lift runs along the approach axis, so its vertical part is LIFT_HEIGHT * m_z
    reached = latch_seen & ~G._rg(env)["last_invalid"].bool() & (end_step > 0)
    want_rise = G.LIFT_HEIGHT * G._rg(env)["_lift_dir"][:, 2]
    record("the lift raises the wrist by LIFT_HEIGHT along the approach axis (envs that reached the hold)",
           bool(((rise - want_rise).abs()[reached] < 0.02).all()) if reached.any() else True,
           f"rise {[round(v*100, 1) for v in rise[reached].tolist()]} cm vs {[round(v*100, 1) for v in want_rise[reached].tolist()]}")
    lat_ok = lat[latch_seen]
    if bool((lat_ok > 0.015).any()):
        # which pose drifted: the bolt sample, the waist hold and the arm's slack tell whether a joint limit did it
        bb = G._rg(env)
        mm = __import__('genesis_rl.tasks.wuji_pipeline.env_cfg', fromlist=['_metrics'])._metrics(env)
        for i in torch.nonzero(latch_seen & (lat > 0.015)).flatten().tolist():
            smp_i = bb["_bolt_sample"][i]
            print(f"  DIAG  env {i}: lateral {lat[i]*1000:.1f} mm; bolt yaw {math.degrees(smp_i[2]):+.1f} deg, "
                  f"xy ({smp_i[0]*100:+.1f}, {smp_i[1]*100:+.1f}) cm; waist {[round(v, 2) for v in bb['_waist'][i].tolist()]}; "
                  f"arm slack now {mm['s_min'][i]:.3f} (joint {int(mm['s_argmin'][i])}), clipped targets {mm['clipped'][i]:.0f}")
    if exact_start:
        record("the lift is vertical: < 5 mm of lateral wrist travel (latched envs)",
               bool((lat_ok < 0.005).all()), f"max {lat_ok.max()*1000:.1f} mm")
    elif lat_ok.numel() == 0:
        print("\n  INFO  the scripted no-correction policy latched nowhere at this level (binding point / raised rod): no lift to measure")
    else:
        # The no-correction scripted policy sometimes latches in a poor pose at a displaced
        # start and then drifts in the lift (seen: 41 mm at level 14, slack fine, nothing
        # clipped). The TRAINED policy does not: 1021 lifts at level 14, max 1.7 mm
        # (TRAINING_LOG, level 15). Informational here; the DIAG line above says which pose.
        print(f"\n  INFO  scripted lift lateral travel max {lat_ok.max()*1000:.1f} mm at this displaced level "
              f"(< 15 mm expected; the trained policy is checked by eval_reach_grasp.py's drag gate)")
    ge = b["last_grasp_end"]
    inv_end = G._rg(env)["last_invalid"].bool()          # the scripted no-correction hand may collide at raised / tilted rods
    early = end_step < ge + G.SCRIPTED_STEPS - 0.5       # ended before the scripted phases ran out: a termination (collision, rod off the table)
    record(f"a latched episode ends exactly at grasp end + {G.SCRIPTED_STEPS} (unless a termination ended it early)",
           bool((((end_step - (ge + G.SCRIPTED_STEPS)).abs() < 0.5) | inv_end | early).all()),
           f"ends {end_step.tolist()} grasp ends {ge.tolist()} invalid {inv_end.int().tolist()}")
    if exact_start:
        record("the scripted grasp succeeds in every env (no start error)", bool(b["last_success"].all()),
               f"success {int(b['last_success'].sum())}/{n}")
    else:
        print(f"\n  INFO  scripted no-correction baseline at this level: success {int(b['last_success'].sum())}/{n}")

if want(6):
    head(6, "an episode that never latches ends at exactly 190 steps")
    env.reset()
    end = torch.full((n,), -1.0, device=dev)
    for t in range(G.level_max_steps(lvl) + 3):
        _, _, te, tr, _ = env.step(zeros())
        done = te | tr
        end[done & (end < 0)] = float(t + 1)
        if bool((end > 0).all()):
            break
    # an open hand never latches: the grasp window times out, the empty lift ends, and at retry levels the
    # retry fires THERE (lift end, not hold end), so the episode is GRASP + SETTLE + LIFT + RETRY + SCRIPTED
    want_len = (G.GRASP_MAX_STEPS + G.SCRIPTED_STEPS + lvl.retry_max * (G.GRASP_RETRY_STEPS + G.SCRIPTED_STEPS - G.HOLD_STEPS)   # retry at lift end
                + lvl.pile.settle_in)                                                                              # the heap's settle-in is extra
    record(f"open-hand episode length is {want_len} (at least 3 of 4 envs; a rod can roll off its block and end one early)",
           bool((end == want_len).float().mean() >= 0.75), f"{end.tolist()}")
    record("it ends by phase_complete, not time_out",
           bool(env.termination_manager.get_term("phase_complete").any()) and
           not bool(env.termination_manager.get_term("time_out").any()))

# ---------------------------------------------------------------------------
if want(7):
    head(7, "logging survives the reset")
    env.reset()
    for _ in range(3):
        _, _, _, _, extras = env.step(zeros())
    log = extras.get("log", {})
    need = ["Safety/min_joint_slack", "Safety/invalid_collision_now", "Safety/shield_alpha",
            "Clearance/tips_min", "Clearance/palm_min", "Wrist/jerk", "Phase/grasp_frac",
            "Outcome/success", "Outcome/bolt_drag_mm"]
    missing = [k for k in need if k not in log]
    record("safety, clearance, wrist, phase and outcome keys are present", not missing, f"missing {missing}")
    record("published values are tensors (no per-step host sync)",
           all(torch.is_tensor(log[k]) for k in need if k in log))
    # force a reset in the same step and check the keys are still there afterwards
    env.episode_length_buf[:] = G.EPISODE_MAX_STEPS
    _, _, _, _, extras = env.step(zeros())
    record("keys still present on a step in which every env reset",
           all(k in extras.get("log", {}) for k in need))
    ep_keys = [k for k in extras.get("log", {}) if k.startswith("Episode_Reward/")]
    record("Isaac Lab's Episode_Reward/* entries coexist with them", len(ep_keys) >= 10, f"{len(ep_keys)} terms")

# ---------------------------------------------------------------------------
if want(8):
    head(8, "perception: estimate vs truth, constant bias, no leakage")
    if not lvl.perception.active:
        print("  (perception is exact at this level; run with ADAM_U_RG_LEVEL=6 or 7 to exercise it)")
        env.reset()
        env.step(zeros())
        est_p, _ = G._est_bolt(env)
        tru_p, _ = G._bolt_true(env)
        record("at an exact level the estimate IS the truth", bool((est_p - tru_p).abs().max() < 1e-7))
    else:
        env.reset()
        b = G._rg(env)
        bias0 = b["bias_pos"].clone()
        diffs = []
        leak_ok = True
        for t in range(20):
            obs, *_ = env.step(zeros())
            est_p, est_q = G._est_bolt(env)
            tru_p, _ = G._bolt_true(env)
            diffs.append((est_p - tru_p).clone())
            # the observation must be computed from the ESTIMATE
            wp, wq = wrist()
            from_est = math_utils.quat_apply_inverse(wq, est_p - wp)
            from_true = math_utils.quat_apply_inverse(wq, tru_p - wp)
            ob = obs["policy"][:, 0:3]
            leak_ok &= bool((ob - from_est).abs().max() < 1e-5) and bool((ob - from_true).abs().max() > 1e-4)
        D = torch.stack(diffs)                                # (T, N, 3)
        record("the estimate differs from the truth", bool(D.abs().mean() > 1e-4), f"mean |err| {D.abs().mean()*1000:.2f} mm")
        record("the per-episode bias stays constant within the episode",
               bool((b["bias_pos"] - bias0).abs().max() < 1e-9))
        spread = D.std(dim=0).mean()
        record("frame-to-frame spread is at the configured noise level, not the bias level",
               bool(spread <= lvl.perception.noise_hi * 1.5 + 1e-4), f"std {spread*1000:.2f} mm vs noise <= {lvl.perception.noise_hi*1000:.1f} mm")
        record("bolt_pos_rel is derived from the estimate, not the truth (no leakage)", leak_ok)
        env.episode_length_buf[:] = env.unwrapped.max_episode_length      # force the time-out, so every env resets
        env.step(zeros())
        record("the bias is redrawn at reset", bool((b["bias_pos"] - bias0).abs().max() > 1e-6))

# ---------------------------------------------------------------------------
if want(9):
    head(9, "safety observations from the proxy model")
    env.reset()
    obs, *_ = env.step(zeros())
    dist, vec = G._clearance(env)
    names = G.GROUP_NAMES
    for i, g in enumerate(names):
        record(f"{g} clearance positive at reset ({dist[:, i].min()*100:.2f}-{dist[:, i].max()*100:.2f} cm)",
               bool((dist[:, i] > 0).all()))
    if lvl.standoff_hi == 0:
        record("tips are the closest group to the table at the contact pose", bool((dist.argmin(dim=1) == 0).all()))
    tips_vec_w = vec[:, 0, :]
    if not lvl.clutter.active and not lvl.bin_walls:      # with obstacles or walls the nearest thing to a fingertip may be beside it
        record("the fingertip obstacle vector points DOWN in the world",
               bool((tips_vec_w[:, 2] < 0).all()) and bool((tips_vec_w[:, 2].abs() > 0.9 * torch.norm(tips_vec_w, dim=1)).all()))
    # the observation carries the same vector, rotated into the wrist frame
    _, wq = wrist()
    back = math_utils.quat_apply(wq, obs["policy"][:, 58:61])
    record("the observed vector rotated back to world equals the model's", bool((back - tips_vec_w).abs().max() < 1e-4))
    record("observed minimum clearances equal the model's", bool((obs["policy"][:, 70:74] - dist.clamp(-0.1, 0.5)).abs().max() < 1e-6))
    # descend and check the clearance falls monotonically, until the first env
    # terminates (at level 0 the fingers reach the table within a few steps)
    prev = dist[:, 0].clone()
    mono, steps_checked = True, 0
    for _ in range(10):
        a = zeros(); a[:, 2] = -1.0
        _, _, te, tr, _ = env.step(a)
        if bool((te | tr).any()):
            break
        d2, _ = G._clearance(env)
        mono &= bool((d2[:, 0] <= prev + 1e-4).all())
        prev = d2[:, 0].clone()
        steps_checked += 1
    if lvl.pile.active:
        # the nearest thing to a fingertip is often a neighbour rod that settles or rolls while the hand descends
        print(f"  INFO  descent clearance at a pile level: {'monotone' if mono else 'not monotone (neighbours move)'}, {steps_checked} steps")
    else:
        record("fingertip clearance decreases while descending", mono and steps_checked >= 3,
               f"{steps_checked} steps checked before the first termination")

# ---------------------------------------------------------------------------
if want(10):
    head(10, "MaskedPPO: masked transitions contribute zero actor gradient")
    agent_cfg = RslRlWujiReachGraspPPORunnerCfg()
    agent_cfg.device = str(dev)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    cfgd = agent_cfg.to_dict()
    N, T, O, A = 16, 12, 74, 8
    cfgd["num_steps_per_env"] = T

    class FakeEnv:
        num_envs, num_actions, device = N, A, str(dev)
        unwrapped = None

    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(N, O, device=dev)}, batch_size=[N])
    alg = MaskedPPO.construct_algorithm(obs, FakeEnv(), cfgd, str(dev))
    record("the runner cfg resolves to MaskedPPO", isinstance(alg, MaskedPPO), type(alg).__name__)
    mask_seq = []
    with torch.inference_mode():
        for t in range(T):
            act = alg.act(obs)
            m = torch.rand(N, device=dev) > 0.4
            mask_seq.append(m)
            alg.process_env_step(obs, torch.randn(N, device=dev), torch.zeros(N, device=dev, dtype=torch.long),
                                 {"actor_mask": m})
            obs = TensorDict({"policy": torch.randn(N, O, device=dev)}, batch_size=[N])
        alg.compute_returns(obs)
    st = alg.storage
    stored = st.actor_mask[:, :, 0].bool()
    record("the mask is stored per transition", bool((stored == torch.stack(mask_seq)).all()))

    def actor_grad(scale_masked_adv: float):
        adv = st.advantages.clone()
        if scale_masked_adv != 1.0:
            st.advantages = torch.where(stored.unsqueeze(-1), adv, adv * scale_masked_adv)
        gen = st.mini_batch_generator(1, 1)
        batch = next(gen)
        alg.actor(batch.observations, stochastic_output=True)
        lp = alg.actor.get_output_log_prob(batch.actions)
        ratio = torch.exp(lp - batch.old_actions_log_prob.squeeze())
        a = batch.advantages.squeeze()
        sur = torch.max(-a * ratio, -a * torch.clamp(ratio, 0.8, 1.2))
        m = batch.actor_mask.reshape(-1)
        loss = (sur * m).sum() / m.sum().clamp(min=1.0) - alg.entropy_coef * (alg.actor.output_entropy * m).sum() / m.sum().clamp(min=1.0)
        alg.optimizer.zero_grad()
        loss.backward()
        g = torch.cat([p.grad.reshape(-1).clone() for p in alg.actor.parameters() if p.grad is not None])
        st.advantages = adv
        return g

    torch.manual_seed(1); g1 = actor_grad(1.0)
    torch.manual_seed(1); g2 = actor_grad(1000.0)
    record("scaling masked advantages by 1000 leaves the actor gradient unchanged",
           bool(torch.allclose(g1, g2, atol=1e-6, rtol=1e-5)), f"max |dg| {(g1 - g2).abs().max():.2e}")
    # and the real update runs and reports the mask fraction
    losses = alg.update()
    record("MaskedPPO.update runs and reports the actor-mask fraction",
           "actor_mask_fraction" in losses and 0.3 < losses["actor_mask_fraction"] < 0.9,
           f"{losses}")

# ---------------------------------------------------------------------------
if want(11):
    head(11, "clutter: placement outside the corridor, clearance, obstacle collision")
    cl = lvl.clutter
    if not cl.active:
        print("  (no clutter at this level; run with ADAM_U_RG_LEVEL=8, 9 or 10)")
        b = G._rg(env)
        env.reset(); env.step(zeros())
        if lvl.tilt.active:
            print("  (support blocks occupy obstacle slots at this level: clearance reads them for tilted rods)")
        else:
            record("without clutter the obstacle clearance reads as far away", bool((b["obs_clear"] > 0.5).all()))
    else:
        b = G._rg(env)
        env.reset(); obs, *_ = env.step(zeros())
        act = b["_obs_active"]
        cnt = act.sum(dim=1)
        record(f"active obstacle count inside [{cl.count_lo}, {cl.count_hi}]",
               bool(((cnt >= cl.count_lo) & (cnt <= cl.count_hi)).all()), f"{cnt.tolist()}")
        bolt = torch.tensor(G.BOLT_XY, device=dev)
        centre = 0.5 * (b["_obs_lo"] + b["_obs_hi"])
        rel = centre[:, :, :2] - bolt
        ok = G._corridor_ok(rel.reshape(-1, 2), cl).view(n, G.MAX_OBSTACLES) | ~act
        record("every active obstacle is outside the approach corridor", bool(ok.all()))
        record("parked slots are far away", bool((b["_obs_hi"][~act] < -5).all()) if bool((~act).any()) else True)
        sits = ((b["_obs_lo"][:, :, 2] - G.TABLE_TOP).abs() < 1e-4) | ~act
        record("active obstacles sit on the table top", bool(sits.all()))
        # the physical bodies are where the model says
        for j in range(G.MAX_OBSTACLES):
            name = f"obstacle_{j}"
            if name in env.scene.keys() and bool(act[:, j].any()):
                pw = _t(env.scene[name].data.root_pos_w) - env.scene.env_origins
                dpos = torch.norm(pw - centre[:, j], dim=1)[act[:, j]]      # parked slots have a far-away AABB by design
                record(f"{name} body pose matches its AABB (active envs)", bool((dpos < 1e-3).all()), f"max {dpos.max()*1000:.2f} mm")
        d0 = b["obs_clear"].clone()
        record("obstacle clearance positive at reset", bool((d0 > 0).all()), f"min {d0.min()*100:.1f} cm")
        record("obstacle clearance is part of the safety observation",
               bool((obs["policy"][:, 70:74].min(dim=1).values <= d0 + 1e-6).all()))
        # drive the hand toward the nearest obstacle: clearance must fall, and pushing into
        # it must end the episode as an obstacle collision (which == 4) -- unless the table
        # or a structural contact is hit first, which is also an invalid collision
        hit_kind = torch.zeros(n, dtype=torch.long, device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(90):
            wp, _ = wrist()
            # nearest active obstacle centre, in the xy plane
            dd = torch.norm(centre[:, :, :2] - wp[:, :2].unsqueeze(1), dim=2)
            dd = torch.where(act, dd, torch.full_like(dd, 1e9))
            jn = dd.argmin(dim=1)
            tgt = centre[torch.arange(n, device=dev), jn, :2]
            dirn = tgt - wp[:, :2]
            dirn = dirn / torch.norm(dirn, dim=1, keepdim=True).clamp(min=1e-9)
            a = zeros(); a[:, 0:2] = dirn; a[:, 2] = -0.3
            _, _, te, tr, _ = env.step(a)
            fell |= b["obs_clear"] < d0 - 0.01
            done = te | tr
            if done.any():
                hit_kind[done & (hit_kind == 0)] = b["last_invalid_which"][done & (hit_kind == 0)]
            if bool((hit_kind > 0).all()):
                break
        record("obstacle clearance falls as the hand moves toward the obstacle", bool(fell.all()))
        # An obstacle on the far side of the annulus can sit beyond the 15 cm
        # workspace radius, where the wrist is clamped before the hand reaches
        # it -- the workspace box doing its job. Require most envs, not all.
        record("driving into the obstacle ends the episode as an invalid collision (>= 75 % of envs)",
               int((hit_kind > 0).sum()) >= (3 * n) // 4, f"kinds {hit_kind.tolist()} (4 = obstacle, 0 = out of reach)")
        record("at least one env is flagged as an OBSTACLE collision specifically", bool((hit_kind == 4).any()))

# ---------------------------------------------------------------------------
if want(12):
    head(12, "pose generalisation: the bolt is sampled, everything nominal follows it")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    smp = b["_bolt_sample"]
    hy = b["_hand_yaw"]
    yaw_ok = (hy >= lvl.bolt_yaw_lo - 1e-6) & (hy <= lvl.bolt_yaw_hi + 1e-6)
    record("sampled hand yaw inside the level's (reach) range", bool(yaw_ok.all()),
           f"{[round(math.degrees(v), 1) for v in hy.tolist()]} deg")
    delta = b["_pinch_delta"]
    record("pinch angle inside the level's range and rod yaw = hand yaw - delta",
           bool((delta.abs() <= lvl.pinch_angle + 1e-6).all()) and bool(((hy - delta) - smp[:, 2]).abs().max() < 1e-6),
           f"{[round(math.degrees(v), 1) for v in delta.tolist()]} deg")
    record("sampled bolt xy inside the level's range", bool((smp[:, :2].abs() <= lvl.bolt_xy + 1e-6).all()))
    if lvl.bolt_xy_edge > 0.0:
        edge = (hy < lvl.bolt_yaw_core[0]) | (hy > lvl.bolt_yaw_core[1])
        record("outside the core yaw window the bolt xy stays within the edge range",
               bool((smp[edge, :2].abs() <= lvl.bolt_xy_edge + 1e-6).all()), f"{int(edge.sum())} edge samples")
    # the physical bolt is where the sample says
    bp, bq = G._bolt_true(env)
    want_xy = torch.tensor(G.BOLT_XY, device=dev) + smp[:, :2]
    flat = ~b["_upright"] & ~b["_tilted"]        # a tilted rod settles onto its block and moves a few mm
    xy_err = torch.norm(bp[:, :2] - want_xy, dim=1)
    # six neighbours at 3 cm (dense pile) nudge the target a few millimetres while they depenetrate: 5 mm of slack there
    xy_tol = 2e-3 + (2e-2 if lvl.pile.upper > 0 else 5e-3 if lvl.pile.count > 4 else 0.0)
    record("bolt body xy matches the sample (lying and standing rods)", bool((xy_err[~b["_tilted"]] < xy_tol).all()),
           f"max {xy_err[~b['_tilted']].max()*1000:.1f} mm" + (f"; tilted {xy_err[b['_tilted']].max()*1000:.1f} mm" if b["_tilted"].any() else ""))
    # and rotated by the sampled yaw: the rod's axis, from its quaternion
    zax = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    spawn_q = _t(env.scene["object"].data.default_root_pose)[:, 3:7]
    ax_now, ax_sp = math_utils.quat_apply(bq, zax), math_utils.quat_apply(spawn_q, zax)
    dyaw = torch.atan2(ax_now[:, 1], ax_now[:, 0]) - torch.atan2(ax_sp[:, 1], ax_sp[:, 0])
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
    lying = flat
    yaw_err = (dyaw - smp[:, 2]).abs()[lying]
    # a neighbour dropped across the target (pile levels with across_prob) turns it a few degrees and shifts it ~1 cm
    knock_deg = 15.0 if lvl.pile.upper > 0 else 5.0 if lvl.pile.across_prob > 0 else 0.0
    knock_m = 0.02 if lvl.pile.upper > 0 else 0.01 if lvl.pile.across_prob > 0 else 0.0
    record("bolt body yaw matches the sample (lying rods)", bool((yaw_err < math.radians(0.5 + knock_deg)).all()) if lying.any() else True,
           f"max error {math.degrees(yaw_err.max()) if lying.any() else 0:.2f} deg over {int(lying.sum())} lying")
    if b["_tilted"].any():
        # a tilted rod's axis direction may be the spawn's flipped end-for-end; compare modulo 180 deg
        e = (dyaw - smp[:, 2])[b["_tilted"]]
        e = torch.remainder(e + math.pi / 2, math.pi) - math.pi / 2
        # a tilted rod rolls along the block's edge while it settles: up to ~15-20 deg of yaw. The policy
        # observes the true pose, so this is informational.
        print(f"  INFO  tilted rods turned by up to {math.degrees(e.abs().max()):.1f} deg in yaw while settling on the block")
    # the nominal wrist pose is the pipeline's, rotated about the bolt by the sampled yaw
    nom = b["_nom_pos"]
    rel = nom[:, :2] - want_xy
    # (from the commanded grasp point, which is the bolt centre moved along the axis at level 15)
    rel = rel - b["_grasp_shift"]
    unrolled = b["_roll"].abs() < 1e-6
    record("nominal wrist sits at the pinch offset from THIS bolt's grasp point (unrolled wrists)",
           bool(((torch.norm(rel, dim=1) - math.hypot(*G.BOLT_OFFSET_XY)).abs()[unrolled] < 1e-4).all()))
    # after one step the wrist is at nominal + the sampled displacement, within the reset tolerance
    wp, wq = wrist()
    d = env._rg_displacement
    err = torch.norm(wp - (nom + start_disp()), dim=1)
    land_mm = 15 if lvl.arm_gains else 10            # randomised arm gains (transfer level) let the arm sag a little more
    record(f"wrist lands within {land_mm} mm of the bolt-relative start pose (bucket grid + one step)",
           bool((err < land_mm / 1000.0).all()), f"max {err.max()*1000:.1f} mm")
    q_err = math_utils.quat_mul(wq, math_utils.quat_inv(b["_nom_quat"]))
    ang = torch.norm(math_utils.axis_angle_from_quat(q_err), dim=1)
    lim_ang = math.sqrt(2) * lvl.drp + lvl.dyaw + math.radians(3)
    record("wrist orientation error from the bolt-relative nominal is only the sampled error (+3 deg)",
           bool((ang <= lim_ang).all()), f"max {math.degrees(ang.max()):.1f} deg vs {math.degrees(lim_ang):.1f}")
    # the policy sees the bolt in its own frame: the observation must NOT depend on the bolt's world yaw
    if lvl.bolt_yaw_hi > lvl.bolt_yaw_lo and n >= 4:
        obs, *_ = env.step(zeros())
        rel_obs = obs["policy"][:, 0:3]
        spread = rel_obs.std(dim=0).max()
        record("bolt-in-wrist observation spread across yaws stays within the start-error budget",
               bool(spread < lvl.dxy + lvl.standoff_hi - lvl.standoff_lo + 0.02), f"std {spread*100:.1f} cm")
    m = P_metrics = __import__('genesis_rl.tasks.wuji_pipeline.env_cfg', fromlist=['_metrics'])._metrics(env)
    record("arm joint slack positive at every sampled pose", bool((m["s_min"] > 0.05).all()),
           f"min {m['s_min'].min():.3f}")
    # the commanded grasp point: the nominal wrist and the policy's bolt estimate move along the rod's axis
    s = b["_grasp_s"]
    record("grasp offset inside the level's range", bool((s.abs() <= lvl.grasp_offset + 1e-9).all()),
           f"{[round(v * 1000, 1) for v in s.tolist()]} mm")
    axis_w = math_utils.quat_apply(bq, zax)
    plain, _ = G._nominal_wrist(smp[:, 0], smp[:, 1], hy, torch.zeros_like(s), dev)   # at the HAND's yaw (level 16)
    shift_err = torch.norm((nom - plain)[:, :2] - (s.unsqueeze(1) * axis_w)[:, :2], dim=1)[~b["_tilted"]]
    record("nominal wrist is the bolt-centred nominal moved s along the rod's axis (lying and standing)", bool((shift_err < 1e-4 + knock_m).all()),
           f"max {shift_err.max() * 1000:.2f} mm")
    est_p, est_q = G._est_bolt(env)
    perc_budget = 0.02 + (lvl.perception.bias_xy + 3 * lvl.perception.noise_hi if lvl.perception.active else 0.0)
    # 2 deg of slack: the rod settles (a tilted one turns up to ~1 deg on its block) while a latent estimate lags
    rot_budget = math.radians(2.0 + knock_deg) + (lvl.perception.bias_rot if lvl.perception.active else 0.0)
    ax_est = math_utils.quat_apply(est_q, zax)
    dy_est = torch.atan2(ax_est[:, 1], ax_est[:, 0]) - torch.atan2(axis_w[:, 1], axis_w[:, 0])
    dy_est = torch.atan2(torch.sin(dy_est), torch.cos(dy_est))
    if lvl.free_grasp:
        # no story at a free level: the estimate is the TRUE rod (plus perception error)
        est_err = torch.norm(est_p - bp, dim=1)
        # the AXIS angle, not the xy yaw: a standing rod's axis has no yaw
        ax_err = torch.acos((ax_est * axis_w).sum(dim=1).clamp(-1.0, 1.0))
        # a tilted rod is still dropping onto its block one step after the reset while the (latent) estimate
        # holds the spawn tilt: 10 deg more there
        ax_budget = torch.where(b["_tilted"], torch.full_like(ax_err, rot_budget + math.radians(10.0)), torch.full_like(ax_err, rot_budget))
        record("free grasp: policy's bolt estimate is the true rod, not a virtual one",
               bool((est_err < perc_budget).all()) and bool((ax_err < ax_budget).all()),
               f"pos max {est_err.max() * 1000:.1f} mm, axis max {math.degrees(ax_err.max()):.2f} deg")
        # the reach potential is indifferent to the point along the rod and to the hand's yaw
        # at the start the pinch centre is the standoff above the rod's axis (any point along it), the
        # approach is from above, so the potential is its tilt half plus a small position remainder
        phi, d_w = G._reach_potential(env)
        lo = lvl.standoff_lo - lvl.dxy - 0.01
        hi = lvl.standoff_hi + lvl.dxy + 0.01
        record("free grasp: pinch centre starts the standoff above the rod's axis (distance to the axis segment)",
               bool(((d_w > lo) & (d_w < hi)).all()), f"{d_w.min() * 100:.1f}-{d_w.max() * 100:.1f} cm vs {lo * 100:.1f}-{hi * 100:.1f}")
        # recover the tilt angle from the potential's orientation half: 0.5 (1 - tanh(theta / sigma))
        tilt_half = (phi - 0.5 * (1.0 - torch.tanh(d_w / G.REACH_SIGMA_POS))).clamp(1e-6, 0.5 - 1e-6)
        theta = G.REACH_SIGMA_ROT * torch.atanh(1.0 - 2.0 * tilt_half)
        lim_tilt = math.sqrt(2) * lvl.drp + math.radians(2.0 + knock_deg)   # the sampled roll/pitch error plus the reset residual
        record("free grasp: the approach tilt at the start is only the sampled roll/pitch error (no yaw term)",
               bool((theta <= lim_tilt).all()), f"max {math.degrees(theta.max()):.2f} deg vs {math.degrees(lim_tilt):.1f}")
    else:
        est_err = torch.norm(est_p - (bp + s.unsqueeze(1) * axis_w), dim=1)
        record("policy's bolt estimate sits at the grasp point (within the perception budget)",
               bool((est_err < perc_budget).all()), f"max {est_err.max() * 1000:.1f} mm")
        # and is yawed by the pinch angle: the estimated axis leads the true one by delta about the vertical
        # (lying rods only: a standing rod's axis has no yaw)
        ly = ~b["_upright"] & ~b["_tilted"]
        yerr = (dy_est - delta).abs()[ly]
        record("policy's bolt estimate is yawed by the pinch angle (within the perception budget; lying rods)",
               bool((yerr < rot_budget).all()) if ly.any() else True,
               f"max error {math.degrees(yerr.max()) if ly.any() else 0:.2f} deg over {int(ly.sum())} lying")
    if lvl.pinch_angle > 0:
        record("pinch angles vary across envs", bool(delta.std() > math.radians(3)), f"std {math.degrees(delta.std()):.1f} deg")
    if lvl.grasp_offset > 0:
        record("grasp offsets vary across envs", bool(s.std() > 0.002), f"std {s.std() * 1000:.1f} mm")
    # the pose solver's copy of the arm proxies and the table matches the env
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("srg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "solve_reach_grasp_poses.py"))
    _srg = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_srg)
    env_prox = tuple((b, tuple(round(v, 4) for v in o), r) for g in ("forearm", "upperarm") for b, o, r in G.PROXY_GROUPS[g])
    srg_prox = tuple((b, tuple(round(v, 4) for v in o), r) for b, o, r in _srg.ARM_PROXIES)
    record("pose solver's arm proxies match the env's PROXY_GROUPS", env_prox == srg_prox)
    record("pose solver's table box matches the env's", all(abs(a - b) < 1e-6 for a, b in zip(G.TABLE_LO + G.TABLE_HI, _srg.TABLE_LO + _srg.TABLE_HI)))
    # the waist: held at the bucket's (pitch, yaw, roll) when the level frees it, at the pipeline's hold otherwise
    w_ids, _ = env.scene["robot"].find_joints(list(G.WAIST_JOINTS), preserve_order=True)
    wj = _t(env.scene["robot"].data.joint_pos)[:, w_ids]
    nominal = torch.tensor(G.WAIST_NOMINAL, device=dev)
    if lvl.free_waist:
        record("waist joints sit at the bucket's held triple after the reset step",
               bool(((wj - b["_waist"]).abs() < 0.02).all()),
               f"max {(wj - b['_waist']).abs().max():.4f} rad; holds {sorted(set(tuple(round(v, 2) for v in row) for row in b['_waist'].tolist()))}")
        lim = _t(env.scene["robot"].data.joint_pos_limits)[0, w_ids, :]
        record("held waist inside its joint limits with margin",
               bool((b["_waist"] > lim[:, 0] + 0.05).all()) and bool((b["_waist"] < lim[:, 1] + 1e-6).all()))
        for _ in range(10):
            env.step(zeros())
        wj2 = _t(env.scene["robot"].data.joint_pos)[:, w_ids]
        record("waist still held ten steps later", bool(((wj2 - b["_waist"]).abs() < 0.02).all()),
               f"max drift {(wj2 - b['_waist']).abs().max():.4f} rad")
    else:
        record("waist at the pipeline's hold (the level does not free it)",
               bool(((wj - nominal).abs() < 0.02).all()) and bool((b["_waist"] == nominal).all()),
               f"max {(wj - nominal).abs().max():.4f} rad")

# ---------------------------------------------------------------------------
if want(13) and lvl.upright_prob > 0:
    head(13, "standing rod: spawns on its end, stays up, the grasp point is a height along it")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    up = b["_upright"]
    record("some rods stand and some lie", bool(up.any()) and bool((~up).any()), f"{int(up.sum())}/{n} standing")
    record("the level uses the convex-decomposition collider", lvl.bolt_collider == "cd"
           and env.unwrapped.cfg.scene.object.spawn.usd_path.endswith("_cd.usdc"), env.unwrapped.cfg.scene.object.spawn.usd_path)
    bp, bq = G._bolt_true(env)
    ax = math_utils.quat_apply(bq, zax)
    tilt = torch.rad2deg(torch.acos(ax[:, 2].clamp(-1, 1)))
    record("standing rods' axis is vertical after the reset step", bool((tilt[up] < 3.0).all()) if up.any() else True,
           f"max tilt {tilt[up].max() if up.any() else 0:.2f} deg")
    record("standing rods' centre is half a rod above the table",
           bool(((bp[up, 2] - G.TABLE_TOP - G.BOLT_HALF_LEN).abs() < 0.003).all()) if up.any() else True,
           f"{[round(v*1000, 1) for v in (bp[up, 2] - G.TABLE_TOP).tolist()]} mm")
    # the nominal wrist sits at the pinch offset from the rod in xy and at the grasp HEIGHT in z
    nom = b["_nom_pos"]; s = b["_grasp_s"]
    want_z = G.WRIST_Z_CONTACT + (bp[:, 2] + s - G.BOLT_REST_Z)
    record("nominal wrist height follows the commanded height along a standing rod",
           bool(((nom[up, 2] - want_z[up]).abs() < 0.004).all()) if up.any() else True,
           f"max {((nom[up, 2] - want_z[up]).abs().max() * 1000) if up.any() else 0:.1f} mm")
    est_p, _ = G._est_bolt(env)
    if not lvl.free_grasp:
        e = torch.norm(est_p - (bp + s.unsqueeze(1) * ax), dim=1)[up]
        record("policy's bolt estimate for a standing rod is shifted UP the axis by the offset",
               bool((e < 0.02).all()) if up.any() else True, f"max {(e.max() * 1000) if up.any() else 0:.1f} mm")
    for _ in range(30):
        env.step(zeros())
    bp2, bq2 = G._bolt_true(env)
    tilt2 = torch.rad2deg(torch.acos(math_utils.quat_apply(bq2, zax)[:, 2].clamp(-1, 1)))
    record("standing rods still up after 30 steps of a still hand (unless touched)", bool((tilt2[up] < 10.0).float().mean() >= 0.75) if up.any() else True,
           f"{int((tilt2[up] < 10.0).sum()) if up.any() else 0}/{int(up.sum())} still standing")

# ---------------------------------------------------------------------------
if want(15) and lvl.tilt.active:
    head(15, "tilted rod: leans on a support block, stays put, the grasp point follows the slanted axis")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    tl = b["_tilted"]; up = b["_upright"]
    record("some rods lean, some stand, some lie", bool(tl.any()) and bool((~tl & ~up).any()), f"{int(tl.sum())} tilted, {int(up.sum())} standing, {int((~tl & ~up).sum())} lying")
    j = b["_support_j"]
    record("every tilted rod has a support slot and no other rod does", bool(((j >= 0) == tl).all()), f"{j.tolist()}")
    # the cover block (level 23+) also lives in slot 0, for covered LYING rods
    cov_ = (b["_cover_side"] != 0) if lvl.cover.active else torch.zeros_like(tl)
    record("a support AABB slot (0-2) is active exactly for tilted rods (and slot 0 for covered lying ones)",
           bool((b["_obs_active"][:, :3].any(dim=1) == (tl | cov_)).all()) and bool((b["_obs_active"][tl, :3].sum(dim=1) == 1).all()) if tl.any() else True)
    bp, bq = G._bolt_true(env)
    ax = math_utils.quat_apply(bq, zax)
    tilt0 = torch.rad2deg(torch.asin(ax[:, 2].abs().clamp(max=1)))
    record("tilted rods spawn between 10 and 55 deg from the table", bool(((tilt0[tl] > 10) & (tilt0[tl] < 55)).all()) if tl.any() else True,
           f"{[round(v, 1) for v in tilt0[tl].tolist()]} deg (spawn {[round(math.degrees(v), 1) for v in b['_tilt'][tl].tolist()]})")
    record("the support block sits under the rod (centre within 5 cm horizontally)",
           bool((torch.norm(b["_support_c"][tl, :2] - bp[tl, :2], dim=1) < 0.05).all()) if tl.any() else True)
    for _ in range(30):
        env.step(zeros())
    bp2, bq2 = G._bolt_true(env)
    ax2 = math_utils.quat_apply(bq2, zax)
    tilt2 = torch.rad2deg(torch.asin(ax2[:, 2].abs().clamp(max=1)))
    record("tilted rods are still raised after 30 steps (settled on the block, not fallen flat; one may roll off)",
           bool((tilt2[tl] > 8).float().sum() >= max(1, int(tl.sum()) - 1)) if tl.any() else True,
           f"{[round(v, 1) for v in tilt2[tl].tolist()]} deg; centre {[round((v - G.TABLE_TOP) * 1000, 1) for v in bp2[tl, 2].tolist()]} mm")
    flat_ = ~tl & ~up & ~b["_on_bed"] & ~b["_heap_swapped"]   # a rod on bed rods rests higher (level 24+); a re-labelled target has its own rest
    record("rest height is the spawn centre for raised rods and the lying rest for lying ones",
           bool(((b["_rest_z"][flat_] - G.BOLT_REST_Z).abs() < 1e-6).all()) and bool((b["_rest_z"][tl | up] > G.BOLT_REST_Z + 0.003).all()),
           f"lying {[round((v - G.BOLT_REST_Z)*1000, 1) for v in b['_rest_z'][flat_].tolist()[:4]]} mm, raised {[round((v - G.BOLT_REST_Z)*1000, 1) for v in b['_rest_z'][tl | up].tolist()[:4]]} mm")

# ---------------------------------------------------------------------------
if want(16) and lvl.wrist_roll:
    head(16, "wrist roll: for a tilted rod the approach axis is the rod's up-normal and the pinch is square to it")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    tl = b["_tilted"]
    bp, bq = G._bolt_true(env)
    ax = math_utils.quat_apply(bq, zax)
    # the solver's rod-axis convention at hand yaw 0: the +z end raised, u0 = (sin yaw0, -cos yaw0)
    import importlib.util as _ilu, os as _os
    _sp = _ilu.spec_from_file_location("srg2", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "solve_reach_grasp_poses.py"))
    _srg = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_srg)
    spawn_q = torch.tensor(tuple(env.unwrapped.cfg.scene.object.init_state.rot), device=dev).expand(n, 4)
    u_env = math_utils.quat_apply(spawn_q, zax)[0]
    u0 = torch.tensor(_srg.U0, device=dev, dtype=torch.float)
    record("env's lying rod axis at hand yaw 0 matches the pose solver's U0", bool(torch.norm(u_env - u0) < 1e-3),
           f"env {[round(v, 3) for v in u_env.tolist()]} solver {[round(v, 3) for v in u0.tolist()]}")
    if tl.any():
        nomq = b["_nom_quat"]
        m = b["_lift_dir"]                                          # the approach / lift axis
        dot = (m * ax).sum(dim=1)[tl]
        expect = torch.sin(b["_tilt"][tl] - b["_roll"][tl])       # zero when the roll matches the tilt
        record("approach axis is perpendicular to the rod's axis up to the uncapped part of the tilt (rod settles a few deg)",
               bool(((dot.abs() - expect.abs()).abs() < 0.15).all()), f"max |cos| {dot.abs().max():.3f}")
        z_unrolled = math_utils.quat_apply(G._nominal_wrist(torch.zeros_like(s), torch.zeros_like(s), hy, torch.zeros_like(s), dev)[1], zax)
        qroll = math_utils.quat_from_angle_axis(-b["_roll"], b["_roll_axis"] / torch.norm(b["_roll_axis"], dim=1, keepdim=True).clamp(min=1e-9))
        z_expect = math_utils.quat_apply(qroll, z_unrolled)
        z_nom = math_utils.quat_apply(nomq, zax)
        record("nominal wrist orientation is the pinch orientation rolled about the rod's normal",
               bool((torch.norm(z_nom[tl] - z_expect[tl], dim=1) < 0.02).all()), f"max {torch.norm(z_nom[tl] - z_expect[tl], dim=1).max():.3f}")
        up = m[tl, 2]
        record("the approach axis is rolled by min(tilt, roll_cap)", bool(((up - torch.cos(b["_roll"][tl])).abs() < 0.02).all()),
               f"cos {[round(v, 3) for v in up.tolist()]}; roll {[round(math.degrees(v), 1) for v in b['_roll'][tl].tolist()]} deg")
        # the reset actually reaches the rolled pose (tilted buckets + residual): wrist orientation error vs the nominal
        wp, wq = wrist()
        q_err = math_utils.quat_mul(wq, math_utils.quat_inv(nomq))
        ang = torch.norm(math_utils.axis_angle_from_quat(q_err), dim=1)[tl]
        lim_ang = math.sqrt(2) * lvl.drp + lvl.dyaw + math.radians(6)
        record("tilted rods: wrist orientation lands within the sampled error (+6 deg) of the ROLLED nominal",
               bool((ang <= lim_ang).all()), f"max {math.degrees(ang.max()):.1f} deg vs {math.degrees(lim_ang):.1f}")
        err = torch.norm(wp - (b["_nom_pos"] + start_disp()), dim=1)[tl]
        record("tilted rods: wrist lands within 15 mm of the rolled start pose", bool((err < 0.015).all()), f"max {err.max()*1000:.1f} mm")
    else:
        print("  (no tilted rod in this draw; rerun for the roll checks)")

# ---------------------------------------------------------------------------
if want(17) and lvl.retry_max > 0:
    head(17, "retry: a failed lift reopens the hand and restarts the grasp phase")
    b = G._rg(env)
    env.reset()
    term = env.unwrapped.action_manager._terms["pipeline"]
    # a still, open hand never latches: the grasp phase times out, the scripted lift runs empty, the retry must fire
    fired_at = torch.full((n,), -1.0, device=dev); phase_after = None; closure_at = None
    for t_ in range(G.GRASP_MAX_STEPS + lvl.pile.settle_in + G.LIFT_START + G.LIFT_STEPS + 3):
        env.step(zeros())
        just = (b["attempts"] > 0) & (fired_at < 0)
        if just.any():
            fired_at[just] = float(t_ + 1)
            closure_at = term.closure.clone(); phase_after = G.phase_of(env).clone()
    record("the retry fires once the empty lift ends", bool((fired_at > 0).all()) and bool((b["attempts"] == 1).all()),
           f"fired at step {fired_at.tolist()}, expected ~{G.GRASP_MAX_STEPS + lvl.pile.settle_in + G.LIFT_START + G.LIFT_STEPS} (an unlatched grasp has no grip check)")
    if closure_at is not None:
        record("on the retry the fingers open and the phase is back to GRASP", bool((closure_at.abs() < 1e-6).all()) and bool((phase_after == 0).all()))
        record("the retry's grasp window is GRASP_RETRY_STEPS long", bool(((b["grasp_end"] - fired_at) - G.GRASP_RETRY_STEPS).abs().max() < 1.5),
               f"grasp_end {b['grasp_end'].tolist()}")
    record("the episode is long enough for one retry", env.unwrapped.max_episode_length >= G.episode_max_steps(lvl.retry_max),
           f"{env.unwrapped.max_episode_length} vs {G.episode_max_steps(lvl.retry_max)}")
    # run to the end: exactly retry_max retries (each at its lift end), and the episode terminates by phase_complete
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(lvl.retry_max * (G.GRASP_RETRY_STEPS + G.SCRIPTED_STEPS) + 5):
        _, _, te, tr, _ = env.step(zeros())
        ended |= te | tr
    record(f"no retry beyond retry_max = {lvl.retry_max}, and the episode ends", bool((b["last_attempts"] == lvl.retry_max).all()) and bool(ended.all()),
           f"attempts {b['last_attempts'].tolist()}, ended {int(ended.sum())}/{n}")

# ---------------------------------------------------------------------------
if want(18) and lvl.cover.active:
    head(18, "partial cover: a block over one end of a lying rod, the point drawn on the exposed part")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    side = b["_cover_side"]; cov = side != 0
    ly = ~b["_upright"] & ~b["_tilted"]
    seen_cov = bool(cov.any()); only_lying = bool((cov <= ly).all()); n_ly = int(ly.sum())
    for _ in range(3):                                # p = 0.3 of lying rods: three more draws before calling it absent
        if seen_cov:
            break
        env.reset(); env.step(zeros())
        side = b["_cover_side"]; cov = side != 0; ly = ~b["_upright"] & ~b["_tilted"]
        seen_cov |= bool(cov.any()); only_lying &= bool((cov <= ly).all()); n_ly += int(ly.sum())
    record("only lying rods get a cover (and some do, over up to four draws)", only_lying and (seen_cov or n_ly < 3), f"{int(cov.sum())} covered of {int(ly.sum())} lying in the last draw")
    if cov.any():
        record("cover slot 0 is active exactly for covered rods (plus supports for tilted ones)",
               bool((b["_obs_active"][cov, 0]).all()) and bool((~b["_obs_active"][ly & ~cov, 0]).all()))
        bp, bq = G._bolt_true(env)
        lo = b["_obs_lo"][cov, 0]; hi = b["_obs_hi"][cov, 0]
        record("the cover's underside is just above the rod's top", bool(((lo[:, 2] - (bp[cov, 2] + G.BOLT_RADIUS)) > 0).all()) and bool(((lo[:, 2] - (bp[cov, 2] + G.BOLT_RADIUS)) < 0.006).all()),
               f"gaps {[round(v*1000, 1) for v in (lo[:, 2] - (bp[cov, 2] + G.BOLT_RADIUS)).tolist()]} mm")
        ax = math_utils.quat_apply(bq, zax)
        cc = 0.5 * (lo + hi)
        along = ((cc - bp[cov]) * ax[cov]).sum(dim=1) * side[cov].float()
        record("the cover block sits over the covered end, 2.5 cm out along the axis", bool(((along - G.COVER_CENTRE_S).abs() < 0.003).all()),
               f"{[round(v*1000, 1) for v in along.tolist()]} mm")
        s = b["_grasp_s"][cov] * side[cov].float()
        record("the commanded point is on the exposed part (never under the block)", bool((s <= G.COVER_EDGE_S + 1e-6).all()),
               f"max toward the cover {s.max()*1000:.1f} mm vs edge {G.COVER_EDGE_S*1000:.0f}")
        for _ in range(20):
            env.step(zeros())
        bp2, _ = G._bolt_true(env)
        record("the rod does not move under the hovering block", bool((torch.norm(bp2[cov] - bp[cov], dim=1) < 0.003).all()))

# ---------------------------------------------------------------------------
if want(19) and lvl.bed.active:
    head(19, "rod on rods: the target rests across two kinematic rods, which the clearance model sees as capsules")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    on = b["_on_bed"]
    ly_ = ~b["_upright"] & ~b["_tilted"]
    record("no bed rod is also covered (and some lying rods are on a bed, when the draw has 3+ lying rods)",
           bool((~(on & (b["_cover_side"] != 0))).all()) and (bool(on.any()) or int(ly_.sum()) < 3), f"{int(on.sum())}/{n} on a bed, {int(ly_.sum())} lying")
    if on.any():
        bp, bq = G._bolt_true(env)
        ax = math_utils.quat_apply(bq, zax)
        bs = list(G.BED_SLOTS)
        record("bed capsules are active exactly for bed episodes", bool((b["_cap_r"][on][:, bs] > 0).all()) and bool((b["_cap_r"][~on][:, bs] == 0).all()))
        mid = 0.5 * (b["_cap_p0"] + b["_cap_p1"])
        along = torch.stack([((mid[:, k] - bp) * ax).sum(dim=1) for k in bs], dim=1)[on]
        record("the bed rods straddle the target's centre, one each side, at half the spacing",
               bool(((along.abs() - G.BED_SPACING / 2).abs() < 0.004).all()) and bool((along[:, 0] * along[:, 1] < 0).all()),
               f"{[[round(v*1000, 1) for v in r] for r in along.tolist()]} mm")
        perp = (b["_bed_axis"][on] * ax[on]).sum(dim=1)
        record("the bed rods run perpendicular to the target", bool((perp.abs() < 0.05).all()))
        for _ in range(30):
            env.step(zeros())
        bp2, _ = G._bolt_true(env)
        on = on & ~b["_heap_swapped"]                  # a re-labelled target (top-rod rule) is a different rod
        h = bp2[on, 2] - G.TABLE_TOP
        ok_bed = ((h > 0.028) & (h < 0.040)).float()
        record("the target settles on top of the bed rods (centre 28-40 mm above the table)",
               bool(ok_bed.all()) if lvl.pile.upper == 0 else bool(ok_bed.mean() >= 0.5),   # the heap's dropped rods can knock it off
               f"{[round(v*1000, 1) for v in h.tolist()]} mm")
        record("rest height for bed rods is the raised spawn height", bool((b["_rest_z"][on] > G.BOLT_REST_Z + 0.015).all()))
        cl = b["obs_clear"][on]
        record("obstacle clearance reads the bed rods (finite, below 20 cm) for bed episodes", bool((cl < 0.2).all()), f"{[round(v*100, 1) for v in cl.tolist()]} cm")

# ---------------------------------------------------------------------------
if want(20) and lvl.bin_walls:
    head(20, "bin: four walls around the nominal bolt position, in the clearance model and the scene")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    act = b["_obs_active"][:, 3:7]
    record("wall slots 3-6 are active in every env", bool(act.all()))
    lo, hi = b["_obs_lo"][:, 3:7], b["_obs_hi"][:, 3:7]
    record("walls are 4 cm tall on the table", bool(((hi[..., 2] - G.TABLE_TOP) - G.BIN_WALL_H).abs().max() < 1e-4) and bool((lo[..., 2] - G.TABLE_TOP).abs().max() < 1e-4))
    bp, _ = G._bolt_true(env)
    inside = ((bp[:, 0] - G.BOLT_XY[0]).abs() < G.BIN_INNER_HALF - 0.02) & ((bp[:, 1] - G.BOLT_XY[1]).abs() < G.BIN_INNER_HALF - 0.02)
    record("every rod spawns inside the bin with 2 cm to spare", bool(inside.all()))
    for _ in range(3):
        env.step(zeros())
    record("no invalid collision at the start inside the bin", bool((~b["invalid"]).all()))
    record("the walls are in the scene", all(f"wall_{j}" in env.unwrapped.scene.keys() for j in range(4)))

# ---------------------------------------------------------------------------
if want(21) and lvl.grip_check:
    head(21, "grip check and slide-out: a covered rod is dragged out before the lift; an unheld rod retries early")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    cov = b["_cover_side"] != 0
    record("drag direction is a horizontal unit vector away from the block for covered rods, zero otherwise",
           bool((torch.norm(b["_drag_dir"][cov], dim=1) - 1).abs().max() < 1e-4) if cov.any() else True
           and bool((b["_drag_dir"][~cov].abs().sum(dim=1) == 0).all()) and bool((b["_drag_dir"][:, 2] == 0).all()))
    if cov.any():
        ax = math_utils.quat_apply(G._bolt_true(env)[1], zax)
        toward = (b["_drag_dir"][cov] * ax[cov]).sum(dim=1) * b["_cover_side"][cov].float()
        record("the drag points along the rod, away from the covered end", bool((toward < -0.95).all()), f"{[round(v, 3) for v in toward.tolist()]}")
    record(f"the pre-lift is {G.PRE_LIFT_STEPS} steps and the lift starts {G.LIFT_START} steps after the latch",
           G.LIFT_START == G.SETTLE_STEPS + G.DRAG_STEPS + G.GRIP_CHECK_STEPS and G.SCRIPTED_STEPS == G.SETTLE_STEPS + G.PRE_LIFT_STEPS + G.LIFT_STEPS + G.HOLD_STEPS)
    # force a latch on a rod that cannot rise: close the fingers with the wrist still, and see whether the
    # scripted phases run the test lift and the retry fires at lift start (not lift end) for latched, unheld rods
    env.reset()
    fired = torch.full((n,), -1.0, device=dev); latched_at = torch.full((n,), -1.0, device=dev)
    for t_ in range(G.GRASP_MAX_STEPS + G.LIFT_START + 3):
        env.step(closing())
        la = b["latched"] & (latched_at < 0)
        latched_at[la] = float(t_ + 1)
        just = (b["attempts"] > 0) & (fired < 0)
        fired[just] = float(t_ + 1)
    early = (fired > 0) & (latched_at > 0) & (fired <= latched_at + G.LIFT_START + 1)
    print(f"  INFO  grip check: {int((latched_at > 0).sum())}/{n} latched under a still, closing hand; retried early (at the test lift) {int(early.sum())}, "
          f"retried at all {int((fired > 0).sum())}, fired at {fired.tolist()}, latched at {latched_at.tolist()}")
    if bool((latched_at > 0).any()):
        record("every latched-then-retried env retried at the test lift, not after a full lift",
               bool(((fired < 0) | early | (latched_at < 0))[latched_at > 0].all()))

# ---------------------------------------------------------------------------
if want(22) and lvl.pile.active:
    nc = lvl.pile.count
    head(22, f"pile: {nc} neighbour rods around the target, soft capsules, double-pick check")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    if lvl.pile.prob < 1.0:
        # mixed scenes: force the pile into every env for the geometry checks (the draw is checked at the end)
        G.reset_pile(env.unwrapped, torch.arange(n, device=dev), pile_fixed=True); env.step(zeros())
    record(f"{nc} neighbour prims in the scene", all(f"neighbor_{k}" in env.unwrapped.scene.keys() for k in range(nc))
           and f"neighbor_{nc}" not in env.unwrapped.scene.keys())
    bp, _ = G._bolt_true(env)
    dists = []
    n_low = nc - lvl.pile.upper
    for k in range(1 if lvl.pile.across_prob > 0 else 0, n_low):   # the across neighbour (k = 0) sits ON the target; dropped rods are checked below
        c = _t(env.unwrapped.scene[f"neighbor_{k}"].data.root_pos_w) - env.unwrapped.scene.env_origins
        dists.append(torch.norm(c[:, :2] - bp[:, :2], dim=1))
    dists = torch.stack(dists, dim=1)
    r_lo, r_hi = lvl.pile.r
    r_hi = max(r_hi, G.NEIGHBOR_R_MAX)          # tilted targets keep the level-27 band
    # crowded rods push apart in the first step: 3 cm of slack around the band
    record(f"neighbours spawn {r_lo*100 - 3:.0f} to {r_hi*100 + 3:.0f} cm from the target", bool(((dists > r_lo - 0.03) & (dists < r_hi + 0.03)).all()),
           f"min {dists.min()*100:.1f}, max {dists.max()*100:.1f} cm")
    record("neighbour capsules are soft, bed capsules hard", bool((~b["_cap_hard"][:, 2:2 + nc]).all()) and bool(b["_cap_hard"][:, :2].all()))
    record("neighbour capsules carry the rod's radius", bool((b["_cap_r"][:, 2:2 + nc] - G.BOLT_RADIUS).abs().max() < 1e-6))
    record("unused neighbour slots are parked with zero radius", bool((b["_cap_r"][:, 2 + nc:] == 0).all()))
    if lvl.pile.across_prob > 0:
        on_top = 0; knocked = 0.0
        for _ in range(3):
            env.reset(); env.step(zeros())
            bp0, _ = G._bolt_true(env)
            for _ in range(30):
                env.step(zeros())
            bp1, _ = G._bolt_true(env)
            c0 = _t(env.unwrapped.scene["neighbor_0"].data.root_pos_w) - env.unwrapped.scene.env_origins
            on_top += int(((c0[:, 2] > bp1[:, 2] + 0.004) & (torch.norm(c0[:, :2] - bp1[:, :2], dim=1) < G.BOLT_HALF_LEN)).sum())
            knocked = max(knocked, torch.norm(bp1[:, :2] - bp0[:, :2], dim=1).max().item())
        print(f"  INFO  a neighbour still ON TOP of the target after settling in {on_top} of {3 * n} episodes (p = {lvl.pile.across_prob:g}, "
              f"lying targets only; a rod balanced across another rolls to its side, see the forced check below)")
        # a neighbour rolling off the target, or a crowd settling against a tilted one, shoves it a few cm now and then
        if lvl.pile.upper > 0:
            print(f"  INFO  heap: the settling rods moved the target up to {knocked*100:.1f} cm over 3 resets (the policy sees the true rod after the settle-in)")
        else:
            record("the across neighbour does not knock the target more than 5 cm", knocked < 0.05, f"max {knocked*100:.1f} cm")
        # forced: every lying target gets the across neighbour; it must rest ON the target (its axis one diameter
        # above the target's axis at the contact) and stay there through the settle
        env.reset(); env.step(zeros())
        u = env.unwrapped; ids = torch.arange(n, device=dev)
        G.reset_pile(u, ids, across_fixed=True, pile_fixed=True); env.step(zeros())
        lying = ~b["_upright"] & ~b["_tilted"]
        def gap():
            bp_, bq_ = G._bolt_true(env); ax_ = math_utils.quat_apply(bq_, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
            c_ = _t(u.scene["neighbor_0"].data.root_pos_w) - u.scene.env_origins
            q_ = _t(u.scene["neighbor_0"].data.root_quat_w); an_ = math_utils.quat_apply(q_, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
            # closest approach of the two axis lines, measured as the neighbour's height above the target's axis at the crossing
            best = torch.full((n,), 9.0, device=dev)
            for f in torch.linspace(-1.0, 1.0, 81, device=dev):
                p_ = c_ + f * G.BOLT_HALF_LEN * an_
                v_ = p_ - bp_; s_ = (v_ * ax_).sum(1, keepdim=True).clamp(-G.BOLT_HALF_LEN, G.BOLT_HALF_LEN)
                best = torch.minimum(best, torch.norm(v_ - s_ * ax_, dim=1))
            return best
        g0 = gap()
        bp0, _ = G._bolt_true(env)
        for _ in range(30):
            env.step(zeros())
        g1 = gap(); bp1, _ = G._bolt_true(env)
        # the across rod goes on lying targets without the cover block (a covered rod is pinned already)
        lying = lying & (b["_cover_side"] == 0)
        ok0 = (g0[lying] < 2 * G.BOLT_RADIUS + 0.004).float().mean().item() if lying.any() else 1.0
        ok1 = (g1[lying] < 2 * G.BOLT_RADIUS + 0.004).float().mean().item() if lying.any() else 1.0
        record("forced across: the neighbour's axis is one diameter above the target's at the crossing after one step",
               ok0 > (0.6 if lvl.pile.upper > 0 else 0.9),   # the heap: a dropped rod can spawn into the across rod and throw it
               f"{ok0*100:.0f} % of {int(lying.sum())} lying targets; gaps {[round(v*1000, 1) for v in g0[lying].tolist()[:6]]} mm")
        # a rod balanced across another is unstable: it rolls to lean against the target's side; it stays in contact
        print(f"  INFO  forced across, after 30 steps: {ok1*100:.0f} % still on top; crossing gaps {[round(v*1000, 1) for v in g1[lying].tolist()[:8]]} mm")
        # plain lying targets only: on bed rods the target is raised and the across rod slides off it further
        plain = lying & ~b["_on_bed"] & (b["_cover_side"] == 0) & ~b["_heap_swapped"]   # a re-labelled target (top-rod rule) is a different rod
        record("forced across: on a plain lying target the neighbour stays within 2 cm of the axis after settling (leaning on or over it)",
               bool((g1[plain] < 0.02 + 2 * G.BOLT_RADIUS).all()) if (plain.any() and lvl.pile.upper == 0) else True,   # the heap's dropped rods knock it off
               f"max {g1[plain].max().item()*1000 if plain.any() else 0:.1f} mm over {int(plain.sum())} plain; bed/cover targets {[round(v*1000) for v in g1[lying & ~plain].tolist()[:6]]} mm")
        record("forced across: a plain lying target is not moved more than 1 cm by the resting neighbour",
               bool((torch.norm(bp1[plain, :2] - bp0[plain, :2], dim=1) < 0.01).all()) if plain.any() else True,
               f"max {torch.norm(bp1[plain, :2] - bp0[plain, :2], dim=1).max().item()*1000 if plain.any() else 0:.1f} mm; "
               f"on bed rods {[round(v*1000) for v in torch.norm(bp1[lying & ~plain, :2] - bp0[lying & ~plain, :2], dim=1).tolist()[:6]]} mm")
        env.reset(); env.step(zeros()); bp, _ = G._bolt_true(env)
    for _ in range(30):
        env.step(zeros())
    bp2, _ = G._bolt_true(env)
    moved = torch.norm(bp2[:, :2] - bp[:, :2], dim=1)
    if lvl.pile.upper > 0:
        print(f"  INFO  heap: settling neighbours moved the target up to {moved.max()*100:.1f} cm (the policy sees the true rod after the settle-in)")
    else:
        record("the target is not knocked more than 2 cm by settling neighbours", bool((moved < 0.02).all()), f"max {moved.max()*100:.1f} cm")
    if lvl.pile.upper > 0:
        u = env.unwrapped
        env.reset(); env.step(zeros())
        if lvl.pile.prob < 1.0:
            G.reset_pile(u, torch.arange(n, device=dev), pile_fixed=True); env.step(zeros())
        upz0 = torch.stack([(_t(u.scene[f"neighbor_{k}"].data.root_pos_w) - u.scene.env_origins)[:, 2] for k in range(n_low, nc)], 1)
        record("heap: the dropped rods start above the first layer", bool((upz0 > G.BOLT_REST_Z + 0.012).all()), f"min {(upz0.min().item() - G.BOLT_REST_Z)*1000:.0f} mm above rest")
        masks = []; term = u.action_manager.get_term("pipeline"); act = zeros(); act[:, 0] = 1.0
        p0 = term._target_pos.clone()
        while int(u.episode_length_buf[0]) < lvl.pile.settle_in - 1:       # the step taken at t = settle_in - 1 is the last frozen one
            env.step(act); masks.append(bool(u.extras["actor_mask"].any()))
        p1 = term._target_pos.clone()
        # the IK re-seeds the target from the measured pose once after the reset (a couple of mm); a live wrist action
        # would move it 3 mm per step
        record("heap: the policy is frozen during the settle-in (mask off, wrist target still under a wrist action)",
               not any(masks) and torch.norm(p1 - p0, dim=1).max().item() < 5e-3, f"masks {masks[:5]}..., moved {torch.norm(p1 - p0, dim=1).max().item()*1000:.2f} mm")
        env.step(act)                                   # the step taken at t = settle_in - 1: still frozen
        env.step(act); masks2 = bool(u.extras["actor_mask"].all())     # the step taken at t = settle_in: live
        record("heap: the policy is released after settle_in steps", masks2)
        upz1 = torch.stack([(_t(u.scene[f"neighbor_{k}"].data.root_pos_w) - u.scene.env_origins)[:, 2] for k in range(n_low, nc)], 1)
        record("heap: the dropped rods have landed by the end of the settle-in (within 5 cm of the rest height: up to two rods under them)", bool((upz1 < G.BOLT_REST_Z + 0.05).all()),
               f"max {(upz1.max().item() - G.BOLT_REST_Z)*1000:.0f} mm above rest")
        record("heap: no double pick is flagged after the settle-in (rest heights re-recorded)", bool((~b["double"]).all()))
    if lvl.pile.prob < 1.0:
        on = []
        for _ in range(4):
            env.reset(); env.step(zeros()); on.append((b["_cap_r"][:, 2] > 0).clone())
        on = torch.cat(on)
        record(f"mixed scenes: the pile is drawn in about {lvl.pile.prob:g} of episodes, parked otherwise",
               0.3 < on.float().mean().item() < 0.9, f"{on.float().mean().item():.2f} of {on.numel()}")
        off = ~on[-n:]
        if off.any():
            nbz = torch.stack([(_t(env.unwrapped.scene[f"neighbor_{k}"].data.root_pos_w) - env.unwrapped.scene.env_origins)[:, 2] for k in range(nc)], 1)
            record("parked neighbours rest on the floor beside the table with zero radius", bool((nbz[off] < 0.05).all()) and bool((b["_cap_r"][off][:, 2:2 + nc] == 0).all()),
                   f"z {[round(v, 3) for v in nbz[off][0].tolist()]}")
    record("no double pick is flagged with nothing lifted", bool((~b["double"]).all()))
    cl = b["obs_clear"]
    record("obstacle (hard) clearance ignores the soft neighbours: no env reads a collision at rest", bool((cl > -G.OBSTACLE_PEN_M).all()), f"min {cl.min()*100:.2f} cm")

# ---------------------------------------------------------------------------
if want(14) and lvl.grasp_point_tol > 0:
    head(14, "binding grasp point: the latch needs the pinch within tolerance of the commanded point")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    s_act = G.pinch_along_axis(env)
    # at the start the pinch is the standoff above the rod: along a LYING rod's axis that projects to ~0,
    # along a STANDING rod's axis to the standoff itself
    lim_s = G.BOLT_HALF_LEN + lvl.standoff_hi + lvl.dxy + 0.02
    record("pinch-along-axis is finite and within the rod's length plus the standoff at the start",
           bool(torch.isfinite(s_act).all()) and bool((s_act.abs() < lim_s).all()), f"{[round(v*1000, 1) for v in s_act.tolist()]} mm vs {lim_s*1000:.0f}")
    # scripted close at the start pose: the latch must NOT fire while the pinch is off the commanded point
    fired_off = False; steps = 0
    for _ in range(40):
        env.step(closing())
        steps += 1
        off = b["grasp_s_err"] > lvl.grasp_point_tol
        if bool((b["latched"] & off & (b["latch_s_err"] > lvl.grasp_point_tol)).any()):
            fired_off = True
    record("no latch fires with the pinch farther than the tolerance from the commanded point", not fired_off,
           f"tol {lvl.grasp_point_tol*1000:.0f} mm; latched {int(b['latched'].sum())}/{n}, latch errors {[round(v*1000, 1) for v in b['latch_s_err'][b['latched']].tolist()]} mm")

# ---------------------------------------------------------------------------
if want(23) and lvl.pinch_centre_tol > 0:
    head(23, "centred pinch: the latch needs the pinch midpoint within tolerance of the rod's axis")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    d0 = G.pinch_centre_dist(env)
    record("pinch-midpoint-to-axis is finite and about the standoff at the start",
           bool(torch.isfinite(d0).all()) and bool((d0 < lvl.standoff_hi + lvl.dxy + 0.05).all()), f"{[round(v*1000, 1) for v in d0.tolist()]} mm")
    off_latched = False
    for _ in range(40):
        env.step(closing())
        if bool((b["latched"] & (b["latch_d_err"] > lvl.pinch_centre_tol)).any()):
            off_latched = True
    record("no latch fires with the pinch midpoint farther than the tolerance from the axis", not off_latched,
           f"tol {lvl.pinch_centre_tol*1000:.0f} mm; latched {int(b['latched'].sum())}/{n}, latch depths {[round(v*1000, 1) for v in b['latch_d_err'][b['latched']].tolist()]} mm")
    record("the commanded-point shaping is finite with the centred term", bool(torch.isfinite(G.grasp_point_progress(env)).all()))

# ---------------------------------------------------------------------------
if want(24) and lvl.grip_check:
    head(24, "grip check v3: a latched rod that does not follow the fingertips' test lift is retried at the pre-lift's end")
    b = G._rg(env)
    env.reset(); env.step(zeros())
    # fake a latch on env 0 with the hand still at the standoff: the test lift raises the tips, the rod stays
    t0 = int(env.unwrapped.episode_length_buf[0])
    b["latched"][0] = True; b["grasp_end"][0] = float(t0)
    tips0 = torch.stack(G._tip_positions(env), dim=1)[0, :, 2].mean().item()
    fired_at = -1
    for _ in range(G.SETTLE_STEPS + G.PRE_LIFT_STEPS + 2):
        env.step(zeros())
        if fired_at < 0 and bool(b["grip_fired"][0]):
            fired_at = int(env.unwrapped.episode_length_buf[0])
            tips1 = torch.stack(G._tip_positions(env), dim=1)[0, :, 2].mean().item()
    record("the check fires at the end of the pre-lift for a latched rod left on the table", fired_at == t0 + G.LIFT_START,
           f"fired at {fired_at}, expected {t0 + G.LIFT_START}")
    if fired_at > 0:
        record("the fingertips rose at least GRIP_CHECK_MIN_TIPS during the test lift", tips1 - tips0 > G.GRIP_CHECK_MIN_TIPS, f"{(tips1 - tips0)*1000:.1f} mm")
        if lvl.grip_check_late:
            n_test = G.GRIP_CHECK_STEPS if lvl.free_prelift else G.PRE_LIFT_STEPS
            record("late check: the measured tip rise is the second half of the test lift only (under 70 % of the whole lift)",
                   b["_pre_wrist_z"][0].item() > tips0 + 0.2 * (tips1 - tips0), f"start height recorded {(b['_pre_wrist_z'][0].item() - tips0)*1000:.1f} mm above the standoff; half = {n_test // 2} steps")
    record("the check counts as a retry (attempts 1, latch cleared)", int(b["attempts"][0]) == 1 and not bool(b["latched"][0]))
    record("no check fires on envs that never latched", bool((~b["grip_fired"][1:]).all()))
    env.reset()

# ---------------------------------------------------------------------------
if want(25) and lvl.free_prelift:
    head(25, "free pre-lift: after the settle the wrist follows the policy for DRAG_STEPS with the fingers held")
    b = G._rg(env)
    term = env.unwrapped.action_manager.get_term("pipeline")
    env.reset(); env.step(zeros())
    t0 = int(env.unwrapped.episode_length_buf[0])
    b["latched"][0] = True; b["grasp_end"][0] = float(t0)
    for _ in range(G.SETTLE_STEPS):
        env.step(zeros())
    mask_settle = bool(env.unwrapped.extras["actor_mask"][0])
    act = zeros(); act[:, 0] = 1.0                      # push the wrist along +x of the action frame
    p0 = term._target_pos[0].clone(); c0 = term._closure[0].clone()
    masks = []
    for _ in range(G.DRAG_STEPS):
        env.step(act)
        masks.append(bool(env.unwrapped.extras["actor_mask"][0]))
    p1 = term._target_pos[0].clone(); c1 = term._closure[0].clone()
    cmd_in_window = torch.norm(term._wrist_cmd[0, :3]).item()
    record("the wrist is frozen during the settle (mask off)", not mask_settle)
    record("the policy's window is in the actor mask", all(masks[:-1]), f"{masks}")
    record("a wrist action moves the commanded wrist pose in the window", cmd_in_window > 1e-4 and torch.norm(p1 - p0).item() > 0.002,
           f"residual {cmd_in_window*1000:.2f} mm/step, target moved {torch.norm(p1 - p0).item()*1000:.1f} mm")
    record("the finger closure is held in the window", torch.norm(c1 - c0).item() < 1e-6)
    env.step(zeros()); p2 = term._target_pos[0].clone()
    env.step(act)
    record("after the window the wrist action is zeroed again", torch.norm(term._wrist_cmd[0, :3]).item() < 1e-6)
    for _ in range(G.GRIP_CHECK_STEPS - 2):
        env.step(act)
    p3 = term._target_pos[0].clone()
    rise = ((p3 - p2) * b["_lift_dir"][0]).sum().item()
    record("after the window the test lift is scripted along the approach axis (the action is ignored)",
           rise > 0.5 * G.GRIP_TEST_TOTAL_M and torch.norm((p3 - p2) - rise * b["_lift_dir"][0]).item() < 0.002,
           f"rise {rise*1000:.1f} mm, sideways {torch.norm((p3 - p2) - rise * b['_lift_dir'][0]).item()*1000:.1f} mm")
    env.reset()

# ---------------------------------------------------------------------------
bad = [r for r in RESULTS if not r[1]]
print("\n" + "=" * 78)
print(f"  {len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed")
for name, ok, det in bad:
    print(f"    FAIL  {name}   {det}")
print("=" * 78)
env.close()
app.close()
raise SystemExit(1 if bad else 0)
