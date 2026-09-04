import argparse
import os
import sys
import time
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

import configs.rsl_rl_cli_args as rsl_rl_cli_args

# -------------------------
# CLI 参数设置
# -------------------------
parser = argparse.ArgumentParser(description="Play a trained policy for Adam-U grasp task.")

parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoint.")
parser.add_argument("--video", action="store_true", default=True, help="Record video of the rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for evaluation.")

# 追加 RSL-RL + Omniverse args
rsl_rl_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 hydra 参数
sys.argv = [sys.argv[0]]

# 启动 Omniverse 仿真
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# -------------------------
# 环境 & 算法配置
# -------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"


from envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg
from algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

env_cfg = AdamUGraspEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = device

agent_cfg = RslRlAdamUGraspPPORunnerCfg()
agent_cfg.device = device

# -------------------------
# 创建环境
# -------------------------
gym.register(
    id="Isaac-Adam-U-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AdamUGraspEnvCfg,
        "rsl_rl_cfg_entry_point": RslRlAdamUGraspPPORunnerCfg,
    },
)

env = gym.make("Isaac-Adam-U-Grasp-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

if args_cli.video:
    video_path = os.path.join(os.path.dirname(args_cli.checkpoint_path), "videos", "play")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_path,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )

env = RslRlVecEnvWrapper(env)

# -------------------------
# 加载训练好的策略
# -------------------------
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load(args_cli.checkpoint_path)

try:
    policy_nn = runner.alg.policy
except AttributeError:
    policy_nn = runner.alg.actor_critic

policy = runner.get_inference_policy(device=device)

# 导出策略
export_dir = os.path.join(os.path.dirname(args_cli.checkpoint_path), "exported")
export_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_dir, filename="policy.pt")
export_policy_as_onnx(policy_nn, normalizer=runner.obs_normalizer, path=export_dir, filename="policy.onnx")

# -------------------------
# 推理执行回放
# -------------------------
obs, _ = env.get_observations()
timestep = 0
dt = env.unwrapped.step_dt

print("[INFO] Starting policy playback...")

while simulation_app.is_running():
    start_time = time.time()

    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    timestep += 1
    if timestep == args_cli.video_length:
        break

    if args_cli.real_time:
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

# -------------------------
# 清理资源
# -------------------------
env.close()
simulation_app.close()
print("[INFO] Playback finished.")
