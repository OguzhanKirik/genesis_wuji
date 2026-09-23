#!/usr/bin/env python3
"""Per-episode evaluation of a grasp-only Wuji checkpoint.

The training console reports per-STEP averages (fraction of timesteps with the
bolt lifted). This scores EPISODES, which is what "did it grasp the bolt"
means:

  lifted     the bolt got more than 2 cm above its rest height at some point
  held       it was still more than 2 cm up at the END of the episode
  held_1s    and had been up for the whole last second (30 steps)
  dropped    it was lifted at some point and ended below +2 cm
  off_table  the episode ended early because the bolt fell off

Also reports the max lift height per episode and the final finger closure.

    docker compose run --rm adam_rl python /opt/tools/eval_grasponly.py \\
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasponly/<run>/model_599.pt --num_envs 256 --episodes 512
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="stop after this many completed episodes")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg  # noqa: E402
from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import LAYOUT, RIGHT_HAND_FLEXION, _t  # noqa: E402
from adam_u_rl.envs.adam_u_wuji_grasponly_env_cfg import AdamUWujiGraspOnlyEnvCfg  # noqa: E402

device = "cuda:0"
cfg = AdamUWujiGraspOnlyEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = device
gym.register(id="EvalGrasp", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": AdamUWujiGraspOnlyEnvCfg})
env = gym.make("EvalGrasp", cfg=cfg)
u = env.unwrapped

agent_cfg = RslRlAdamUGraspPPORunnerCfg()
agent_cfg.device = device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=device)

bolt = u.scene["object"]
robot = u.scene["robot"]
rest_z = LAYOUT["bolt_rest_z"]
LIFT = 0.02
n = u.num_envs
flex_ids, _ = robot.find_joints(RIGHT_HAND_FLEXION)
upper = _t(robot.data.joint_pos_limits)[0, flex_ids, 1]

max_lift = torch.zeros(n, device=device)
up_streak = torch.zeros(n, device=device)          # consecutive steps with the bolt up
ever_lifted = torch.zeros(n, dtype=torch.bool, device=device)
steps = torch.zeros(n, device=device)

stats = dict(episodes=0, lifted=0, held=0, held_1s=0, dropped=0, off_table=0,
             max_lift_sum=0.0, closure_sum=0.0, len_sum=0.0)

obs = wenv.get_observations()
prev_closure = torch.zeros(n, device=device)
with torch.inference_mode():
    while stats["episodes"] < args.episodes:
        act = policy(obs)
        # Isaac Lab resets finished envs INSIDE step(), so after a done the
        # buffers already show the new episode. Score a finished episode from
        # the state tracked up to the previous step.
        streak_before = up_streak.clone()
        lifted_before = ever_lifted.clone()
        maxlift_before = max_lift.clone()
        closure_before = prev_closure.clone()
        obs, rew, dones, extras = wenv.step(act)
        steps += 1
        done = dones.bool()
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in ids.tolist():
                stats["episodes"] += 1
                lifted = bool(lifted_before[i])
                held = bool(streak_before[i] >= 1)
                held_1s = bool(streak_before[i] >= 30)
                early = bool(steps[i] < u.max_episode_length - 1)
                stats["lifted"] += lifted
                stats["held"] += held
                stats["held_1s"] += held_1s
                stats["dropped"] += lifted and not held
                stats["off_table"] += early
                stats["max_lift_sum"] += float(maxlift_before[i])
                stats["closure_sum"] += float(closure_before[i])
                stats["len_sum"] += float(steps[i])
            max_lift[ids] = 0.0
            up_streak[ids] = 0.0
            ever_lifted[ids] = False
            steps[ids] = 0.0
        z = _t(bolt.data.root_pos_w)[:, 2] - rest_z
        up = z > LIFT
        max_lift = torch.maximum(max_lift, z)
        ever_lifted |= up
        up_streak = torch.where(up, up_streak + 1, torch.zeros_like(up_streak))
        q = _t(robot.data.joint_pos)[:, flex_ids]
        prev_closure = (q / upper).clamp(0, 1).mean(dim=1)

E = stats["episodes"]
print("=" * 64)
print(f"checkpoint {args.checkpoint_path}")
print(f"episodes {E}   (num_envs {n}, inference policy = distribution mean)")
print(f"  lifted   (>2 cm at any point)          {stats['lifted'] / E * 100:5.1f} %")
print(f"  held     (>2 cm at episode end)         {stats['held'] / E * 100:5.1f} %")
print(f"  held_1s  (up for the whole last second) {stats['held_1s'] / E * 100:5.1f} %")
print(f"  dropped  (lifted, then lost)            {stats['dropped'] / E * 100:5.1f} %")
print(f"  off_table (early termination)           {stats['off_table'] / E * 100:5.1f} %")
print(f"  mean max lift height                    {stats['max_lift_sum'] / E * 100:5.1f} cm")
print(f"  mean final flexion closure              {stats['closure_sum'] / E:5.2f}")
print(f"  mean episode length                     {stats['len_sum'] / E:5.1f} / {u.max_episode_length} steps")
print("=" * 64)
env.close()
app.close()
