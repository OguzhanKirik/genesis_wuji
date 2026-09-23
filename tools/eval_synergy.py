#!/usr/bin/env python3
"""Evaluate an Experiment 2 (finger-only synergy grasp) checkpoint per episode.

    docker compose run --rm adam_rl python /opt/tools/eval_synergy.py \\
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/<run>/model_499.pt --num_envs 256 --episodes 512
    # with videos (rendered offscreen; needs --enable_cameras, slower):
    ... --video --video_envs 4 --episodes 8

    # with the Isaac Lab viewport open, slowed down so the grasps are watchable:
    ... --viz kit --num_envs 16 --episodes 16 --real-time --speed 0.25 --start-delay 15

Reports what the spec asks for:
  stable-grasp success rate      bolt >= 5 cm above rest for the whole 1 s hold, no
                                 finger-table contact, no fingertip force > F_MAX
  bolt-drop rate                 lifted (> 2 cm after the grasp phase) and lost again
  finger-table collision rate    any fingertip touched the table (force > TABLE_HIT_N)
  excessive-force rate           any fingertip net force > F_MAX
  average thumb contact          fraction of grasp-phase steps with thumb-bolt contact
  average opposing contact       same for index/middle
  final synergy values           mean closure (thumb, fingers) at the end of the grasp phase
  held-ignoring-rules rate       the grasp itself, without the collision/force disqualifiers

Episode outcomes are read from the env's `last_*` buffers, which the reward
bookkeeping writes at the final step and the reset does not clear (Isaac Lab
resets inside the same step(), so anything else would be post-reset state).
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
parser.add_argument("--video", action="store_true", help="record videos of the first --video_envs envs")
parser.add_argument("--video_envs", type=int, default=4)
parser.add_argument("--video_dir", type=str, default=None, help="default: <checkpoint dir>/videos")
parser.add_argument("--real-time", action="store_true", default=False,
                    help="with --viz kit: throttle to wall-clock so the grasps are watchable")
parser.add_argument("--speed", type=float, default=1.0,
                    help="playback speed with --real-time: 0.25 = quarter speed")
parser.add_argument("--start-delay", type=float, default=0.0,
                    help="seconds to render the reset pose before the policy acts, so the "
                         "window is up before the first grasp")
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

from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiSynergyPPORunnerCfg  # noqa: E402
from adam_u_rl.envs import wuji_synergy_grasp_env_cfg as C  # noqa: E402

device = "cuda:0"
cfg = C.WujiSynergyGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = device
gym.register(id="EvalSyn", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": C.WujiSynergyGraspEnvCfg})
env = gym.make("EvalSyn", cfg=cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    video_dir = args.video_dir or os.path.join(os.path.dirname(args.checkpoint_path), "videos")
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, step_trigger=lambda step: step % C.EPISODE_STEPS == 0,
        video_length=C.EPISODE_STEPS, disable_logger=True, name_prefix="synergy",
    )
    print(f"[eval] recording videos to {video_dir}")
u = env.unwrapped

agent_cfg = RslRlWujiSynergyPPORunnerCfg()
agent_cfg.device = device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=device)

n = u.num_envs
tot = dict(episodes=0, success=0, held=0, dropped=0, table=0, excess=0, lifted=0, dirty=0,
           thumb=0.0, finger=0.0, clo_t=0.0, clo_f=0.0)
if args.start_delay > 0:
    print(f"[eval] holding the reset pose for {args.start_delay:g} s")
    _end = time.time() + args.start_delay
    while time.time() < _end:
        u.sim.render()
        time.sleep(0.05)

dt = u.step_dt
obs = wenv.get_observations()
with torch.inference_mode():
    while tot["episodes"] < args.episodes:
        _t0 = time.time()
        act = policy(obs)
        obs, rew, dones, extras = wenv.step(act)
        if args.real_time:
            _left = dt / args.speed - (time.time() - _t0)
            if _left > 0:
                time.sleep(_left)
        done = dones.bool()
        if done.any():
            b = C._buffers(u)
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            k = len(ids)
            tot["episodes"] += k
            tot["success"] += int(b["last_success"][ids].sum())
            tot["held"] += int(b["last_held"][ids].sum())
            tot["dropped"] += int(b["last_drop"][ids].sum())
            tot["table"] += int(b["last_table_hit"][ids].sum())
            tot["excess"] += int(b["last_excess"][ids].sum())
            tot["lifted"] += int(b["last_was_up"][ids].sum())
            tot["dirty"] += int((~b["last_verify_clean"][ids]).sum())
            tot["thumb"] += float(b["last_thumb_frac"][ids].sum())
            tot["finger"] += float(b["last_finger_frac"][ids].sum())
            tot["clo_t"] += float(b["last_closure"][ids, 0].sum())
            tot["clo_f"] += float(b["last_closure"][ids, 1].sum())

E = max(1, tot["episodes"])
print("=" * 64)
print(f"checkpoint {args.checkpoint_path}")
print(f"episodes {tot['episodes']}   (num_envs {n}, policy mean action, {C.NUM_FINGERS} fingers)")
print(f"  stable-grasp success rate          {tot['success'] / E * 100:5.1f} %")
print(f"  held >= 5 cm through the hold      {tot['held'] / E * 100:5.1f} %   (ignoring collision/force rules)")
print(f"  lifted at all (> 2 cm)             {tot['lifted'] / E * 100:5.1f} %")
print(f"  bolt-drop rate                     {tot['dropped'] / E * 100:5.1f} %")
print(f"  bolt touched the table in verify   {tot['dirty'] / E * 100:5.1f} %")
print(f"  finger-table collision rate        {tot['table'] / E * 100:5.1f} %")
print(f"  excessive-force rate (> {C.F_MAX:.0f} N)     {tot['excess'] / E * 100:5.1f} %")
print(f"  average thumb contact              {tot['thumb'] / E * 100:5.1f} % of grasp-phase steps")
print(f"  average opposing-finger contact    {tot['finger'] / E * 100:5.1f} % of grasp-phase steps")
print(f"  final synergy values (thumb, fingers)  ({tot['clo_t'] / E:.2f}, {tot['clo_f'] / E:.2f})")
print("=" * 64)
env.close()
app.close()
