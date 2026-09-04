# adam_u_rl/__init__.py

"""Adam-U RL项目 - 机器人抓取任务"""

import gymnasium as gym
from .envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg
from .algos.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg

# 注册环境
gym.register(
    id="Isaac-Adam-U-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AdamUGraspEnvCfg,
        "rsl_rl_cfg_entry_point": RslRlAdamUGraspPPORunnerCfg,
    },
)