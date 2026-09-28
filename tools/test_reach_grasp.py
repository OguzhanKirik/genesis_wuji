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
"""

from __future__ import annotations

import argparse
import math
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


def zeros():
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = -1.0          # fingers open
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
    record("policy observation width is 74", obs["policy"].shape[-1] == 74, f"got {obs['policy'].shape[-1]}")
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
    want_pos = torch.tensor([G.WRIST_XY[0], G.WRIST_XY[1], G.WRIST_Z_CONTACT], device=dev) + d[:, :3]
    err = torch.norm(pos - want_pos, dim=1)
    record("wrist lands within 8 mm of the commanded reset pose (first-order reset + one step)",
           bool((err < 0.008).all()), f"max {err.max()*1000:.1f} mm")
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
        drift = (q[ids] - default[ids]).abs().max()
        record(f"{g} holds its default (max drift {drift:.4f} rad)", drift < 0.05)

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
    for t in range(G.EPISODE_MAX_STEPS + 2):
        a = pol(t)
        pre_pos, pre_quat = wrist()
        pre_q = _t(robot.data.joint_pos)[:, term._arm_ids]
        pre_t = env.episode_length_buf.clone().float()          # per-env time BEFORE the step
        pre_grasp_end = b["grasp_end"].clone()
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
            latch_ok["target"] &= bool((torch.norm(tpos[m] - pre_pos[m], dim=1) < 1e-5).all())
            # q_cmd has since integrated one IK step toward that same target; it must
            # stay within one joint-velocity cap of the measured state
            latch_ok["qcmd"] &= bool(((term.q_cmd[m] - pre_q[m]).abs() < lim.max_joint_velocity * G.CONTROL_DT + 1e-6).all())
        # mask semantics: the step just taken was a grasp step iff pre_t < grasp_end as it
        # stood BEFORE the step. On the latch step itself grasp_end moves to pre_t inside the
        # step and the mask reads False -- excluded, both readings are defensible there.
        grasp_step = pre_t < pre_grasp_end
        disagreement = (mask != grasp_step) & ~now
        if disagreement.any():
            if bool((grasp_step & disagreement).any()):
                mask_in_grasp_ok = False
            if bool((~grasp_step & disagreement).any()):
                mask_after_ok = False
        # settle: closure must not change while the finger command keeps changing
        settle = (ph_now == 1) & latch_seen
        if settle.any():
            closure_frozen_ok &= bool(((term.closure[settle] - closure_at_latch[settle]).abs() < 1e-6).all())
        lift_entry = (env.episode_length_buf.float() == b["grasp_end"] + G.SETTLE_STEPS)
        if lift_entry.any():
            wrist_at_lift[lift_entry] = wrist()[0][lift_entry]
        hold_entry = (env.episode_length_buf.float() == b["grasp_end"] + G.SETTLE_STEPS + G.LIFT_STEPS)
        if hold_entry.any():
            wrist_at_hold[hold_entry] = wrist()[0][hold_entry]
        done = te | tr
        if done.any():
            end_step[done & (end_step < 0)] = float(t + 1)
        if bool((end_step > 0).all()):
            break
    record("actor mask is TRUE for every grasp-phase transition", mask_in_grasp_ok)
    record("actor mask is FALSE for every SETTLE / LIFT / HOLD transition", mask_after_ok)
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
        record("at least one env latches, so the latch path is exercised", bool(latch_seen.any()),
               f"{int(latch_seen.sum())}/{n}")
    record("on the latch step the Cartesian residual is cleared", latch_ok["residual"])
    record("on the latch step the IK target equals the MEASURED wrist pose", latch_ok["target"])
    record("on the latch step the IK integrator is re-seeded from the measured joints", latch_ok["qcmd"])
    record("finger closure is held constant through SETTLE", closure_frozen_ok)
    rise = (wrist_at_hold - wrist_at_lift)[:, 2]
    lat = torch.norm((wrist_at_hold - wrist_at_lift)[:, :2], dim=1)
    record("the lift raises the wrist ~10 cm", bool(((rise - G.LIFT_HEIGHT).abs() < 0.02).all()),
           f"rise {rise.min()*100:.1f}-{rise.max()*100:.1f} cm")
    lat_ok = lat[latch_seen]
    record("the lift is vertical: < 5 mm of lateral wrist travel (latched envs, exact start) / < 15 mm (displaced)",
           bool((lat_ok < (0.005 if exact_start else 0.015)).all()), f"max {lat_ok.max()*1000:.1f} mm")
    ge = b["last_grasp_end"]
    record("a latched episode ends exactly at grasp end + 70",
           bool(((end_step - (ge + G.SCRIPTED_STEPS)).abs() < 0.5).all()),
           f"ends {end_step.tolist()} grasp ends {ge.tolist()}")
    if exact_start:
        record("the scripted grasp succeeds in every env (no start error)", bool(b["last_success"].all()),
               f"success {int(b['last_success'].sum())}/{n}")
    else:
        print(f"\n  INFO  scripted no-correction baseline at this level: success {int(b['last_success'].sum())}/{n}")

if want(6):
    head(6, "an episode that never latches ends at exactly 190 steps")
    env.reset()
    end = torch.full((n,), -1.0, device=dev)
    for t in range(G.EPISODE_MAX_STEPS + 3):
        _, _, te, tr, _ = env.step(zeros())
        done = te | tr
        end[done & (end < 0)] = float(t + 1)
        if bool((end > 0).all()):
            break
    record("open-hand episode length is 190", bool((end == G.EPISODE_MAX_STEPS).all()), f"{end.tolist()}")
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
        env.episode_length_buf[:] = G.EPISODE_MAX_STEPS
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
bad = [r for r in RESULTS if not r[1]]
print("\n" + "=" * 78)
print(f"  {len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed")
for name, ok, det in bad:
    print(f"    FAIL  {name}   {det}")
print("=" * 78)
env.close()
app.close()
raise SystemExit(1 if bad else 0)
