# adam_u_rl/scripts/train.py

"""训练Adam-U机器人抓取任务的脚本"""

import argparse
import sys
import os
import torch

from isaaclab.app import AppLauncher

# local imports
import configs.rsl_rl_cli_args as rsl_rl_cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train Adam-U grasping task with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1500, help="RL Policy training iterations.")
parser.add_argument(
    "--env",
    type=str,
    default="AdamUGrasp",
    help="Which environment to train (e.g. AdamUGrasp, FrankaPick, etc.)"
)

rsl_rl_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 现在导入模块
import gymnasium as gym
from envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg
from algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg

ENV_REGISTRY = {
    "AdamUGrasp": (AdamUGraspEnvCfg, RslRlAdamUGraspPPORunnerCfg),
    # "FrankaPick": (FrankaPickEnvCfg, RslRlFrankaPickPPORunnerCfg),
}

def main():
    """主训练函数"""
    
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 环境配置
    if args_cli.env not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{args_cli.env}'. Available: {list(ENV_REGISTRY.keys())}")

    EnvCfgClass, RunnerCfgClass = ENV_REGISTRY[args_cli.env]

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    
    # RSL-RL配置
    agent_cfg = RunnerCfgClass()
    # 注册环境
    env_id = f"Isaac-{args_cli.env}"

    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": EnvCfgClass,
            "rsl_rl_cfg_entry_point": RunnerCfgClass,
        },
    )

    agent_cfg.device = device
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    
    # 创建环境
    env = gym.make(f"Isaac-{args_cli.env}", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 设置日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    
    print(f"[INFO] 训练日志保存到: {log_dir}")
    
    # 视频录制设置
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 启用视频录制")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # 包装环境用于RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # 创建RSL-RL训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 开始训练
    print(f"[INFO] 开始训练Adam-U抓取任务...")
    print(f"[INFO] 环境数量: {args_cli.num_envs}")
    print(f"[INFO] 最大迭代数: {args_cli.max_iterations}")
    print(f"[INFO] 设备: {device}")
    
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # 关闭环境
    env.close()
    print("[INFO] 训练完成!")

if __name__ == "__main__":
    main()
    # 关闭仿真应用
    simulation_app.close()