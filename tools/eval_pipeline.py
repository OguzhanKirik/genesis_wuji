#!/usr/bin/env python3
"""Evaluate a staged-pipeline checkpoint per EPISODE, not per step.

    docker compose exec adam_rl python /opt/tools/eval_pipeline.py \
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_pipeline_s0/<run>/model_999.pt \
        --num_envs 256 --episodes 512

    # watch it, slowed down, with the Isaac Lab viewport open
    docker compose exec adam_rl python /opt/tools/eval_pipeline.py --viz kit \
        --num_envs 16 --episodes 16 --real-time --speed 0.25 --start-delay 15 \
        --checkpoint_path .../model_999.pt

    # record videos instead (offscreen, no display needed)
    ... --video --video_envs 4 --episodes 8

The stage comes from ADAM_U_STAGE, exactly as in training, and it MUST match
the stage the checkpoint was trained on -- the action mask and the reset height
differ, so a stage-0 policy evaluated at stage 1 is measuring nothing. The
stage is printed; check it.

Reports the same per-episode criteria as Experiment 2, so the numbers are
directly comparable with the 100 % in TRAINING_LOG.md, plus what is new here:

  stable-grasp success        bolt >= SUCCESS_LIFT for the whole hold phase, never
                              touching the table during it, no finger-table
                              collision and no fingertip force over F_MAX anywhere
  lifted / held / dropped     the grasp itself, without the disqualifiers
  minimum arm joint slack     how close the arm came to a stop (1.0 = mid-range)
  IK failures / clipped       the deterministic safety path, per episode
  wrist tracking error        how far the arm lagged the commanded pose

Episode outcomes come from the env's `last_*` buffers, written at the final
step and deliberately NOT cleared by the reset -- Isaac Lab resets inside the
same step() that ends an episode, so anything read afterwards belongs to the
next one. That detail once produced a confident 0 % report for a policy that
was succeeding 99 % of the time.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--video", action="store_true", help="record videos of the first envs")
parser.add_argument("--video_envs", type=int, default=4)
parser.add_argument("--video_dir", type=str, default=None, help="default: <checkpoint dir>/videos")
parser.add_argument("--real-time", action="store_true", default=False,
                    help="with --viz kit: throttle to wall-clock so the grasps are watchable")
parser.add_argument("--speed", type=float, default=1.0,
                    help="playback speed with --real-time: 0.25 = quarter speed")
parser.add_argument("--start-delay", type=float, default=0.0,
                    help="seconds to render the reset pose before the policy acts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiPipelinePPORunnerCfg  # noqa: E402
from adam_u_rl.envs import wuji_pipeline_env_cfg as P  # noqa: E402
from adam_u_rl.envs import wuji_synergy_grasp_env_cfg as C  # noqa: E402

device = "cuda:0"
cfg = P.WujiPipelineEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = device
stage = P.LEVELS[cfg.actions.pipeline.level]

gym.register(id="EvalPipeline", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": P.WujiPipelineEnvCfg})
env = gym.make("EvalPipeline", cfg=cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    video_dir = args.video_dir or os.path.join(os.path.dirname(args.checkpoint_path), "videos")
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, step_trigger=lambda s: s % P.EPISODE_STEPS == 0,
        video_length=P.EPISODE_STEPS, disable_logger=True, name_prefix=f"pipeline_L{P.LEVEL}",
    )
    print(f"[eval] recording videos to {video_dir}")
u = env.unwrapped

agent_cfg = RslRlWujiPipelinePPORunnerCfg()
agent_cfg.device = device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=device)

n = u.num_envs
tot = dict(episodes=0, success=0, held=0, dropped=0, table=0, excess=0, lifted=0, dirty=0,
           thumb=0.0, finger=0.0, clo_t=0.0, clo_f=0.0)
# Safety metrics are per-step, so track the worst and the mean over the run.
s_min_worst = 1.0
s_min_sum, s_min_steps = 0.0, 0
ik_fail_steps, clip_steps = 0, 0
track_worst, track_sum = 0.0, 0.0

if args.start_delay > 0:
    # Reset FIRST -- see the note in play.py. The scene spawns at the USD's own
    # pose (robot upright, arm at its side); only the reset event writes the
    # pre-grasp pose, so rendering before it shows the wrong robot and then
    # snaps, which reads as the policy teleporting into the grasp.
    wenv.reset()
    print(f"[eval] holding the reset pose for {args.start_delay:g} s")
    _end = time.time() + args.start_delay
    while time.time() < _end:
        u.sim.render()
        time.sleep(0.05)

dt = u.step_dt
obs = wenv.get_observations()
term = u.action_manager._terms["pipeline"]
with torch.inference_mode():
    while tot["episodes"] < args.episodes:
        _t0 = time.time()
        act = policy(obs)
        obs, rew, dones, extras = wenv.step(act)
        if args.real_time:
            _left = dt / args.speed - (time.time() - _t0)
            if _left > 0:
                time.sleep(_left)

        m = P._metrics(u)
        sm = float(m["s_min"].min())
        s_min_worst = min(s_min_worst, sm)
        s_min_sum += float(m["s_min"].mean()); s_min_steps += 1
        ik_fail_steps += int((m["ik_failed"] > 0.5).sum())
        clip_steps += int((m["clipped"] > 0).sum())
        tw = float(m["wrist_track_err"].max())
        track_worst = max(track_worst, tw)
        track_sum += float(m["wrist_track_err"].mean())

        done = dones.bool()
        if done.any():
            b = C._buffers(u)
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            tot["episodes"] += len(ids)
            tot["success"] += int(b["last_success"][ids].sum())
            tot["held"] += int(b["last_held"][ids].sum())
            tot["dropped"] += int(b["last_drop"][ids].sum())
            tot["table"] += int(b["last_table_hit"][ids].sum())
            tot["excess"] += int(b["last_excess"][ids].sum())
            tot["lifted"] += int(b["last_was_up"][ids].sum())
            tot["dirty"] += int((~b["last_verify_clean"][ids]).sum())
            tot["thumb"] += float(b["last_thumb_frac"][ids].sum())
            tot["finger"] += float(b["last_finger_frac"][ids].sum())
            # From the last_* buffers, NOT from the live action term: Isaac
            # Lab resets inside the same step() that ends the episode and the
            # term's reset() zeroes the closure, so reading it live reports
            # 0.00 for every episode.
            tot["clo_t"] += float(b["last_closure"][ids, 0].sum())
            tot["clo_f"] += float(b["last_closure"][ids, 1].sum())

E = max(1, tot["episodes"])
S = max(1, s_min_steps)
print("=" * 70)
print(f"checkpoint {args.checkpoint_path}")
print(f"level {cfg.actions.pipeline.level} '{stage.name}'   "
      f"displacement xy +/-{stage.dxy*100:.1f} cm, z {stage.dz_lo*100:+.1f}/{stage.dz_hi*100:+.1f} cm")
print(f"episodes {tot['episodes']}   (num_envs {n}, policy mean action)")
print()
print("  GRASP")
print(f"    stable-grasp success rate        {tot['success'] / E * 100:5.1f} %")
print(f"    held >= {C.SUCCESS_LIFT*100:.0f} cm through the hold     {tot['held'] / E * 100:5.1f} %"
      f"   (ignoring collision/force rules)")
print(f"    lifted at all (> {C.UP_THRESHOLD*100:.0f} cm)           {tot['lifted'] / E * 100:5.1f} %")
print(f"    bolt-drop rate                   {tot['dropped'] / E * 100:5.1f} %")
print(f"    bolt touched the table in verify {tot['dirty'] / E * 100:5.1f} %")
print(f"    finger-table collision rate      {tot['table'] / E * 100:5.1f} %")
print(f"    excessive-force rate (> {C.F_MAX:.0f} N)   {tot['excess'] / E * 100:5.1f} %")
print(f"    thumb contact                    {tot['thumb'] / E * 100:5.1f} % of grasp-phase steps")
print(f"    opposing-finger contact          {tot['finger'] / E * 100:5.1f} % of grasp-phase steps")
print(f"    final synergy (thumb, fingers)   ({tot['clo_t'] / E:.2f}, {tot['clo_f'] / E:.2f})")
print()
print("  ARM AND IK")
print(f"    minimum arm joint slack, worst   {s_min_worst:5.3f}   (1.0 = every joint mid-range)")
print(f"    minimum arm joint slack, mean    {s_min_sum / S:5.3f}")
print(f"    env-steps with an IK failure     {ik_fail_steps}")
print(f"    env-steps with a clipped target  {clip_steps}")
print(f"    wrist tracking error, mean       {track_sum / S * 1000:5.2f} mm")
print(f"    wrist tracking error, worst      {track_worst * 1000:5.2f} mm")
print("=" * 70)
env.close()
app.close()
