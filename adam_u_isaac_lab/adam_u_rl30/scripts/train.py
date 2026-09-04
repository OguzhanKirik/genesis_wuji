"""Train the Adam-U grasping task — Isaac Sim 6.0 / Isaac Lab 3.0.

Same task and hyperparameters as adam_u_rl/scripts/train.py, on the rsl-rl 5.x
schema. Use this if you want to train *and* watch in one environment.

Production training belongs on the 5.1 image (`adam_u_rl/scripts/train.py`):
Isaac Lab 3.0 is beta and its release notes warn of continuing breaking
changes. This exists so the visualization environment is not read-only, and so
a policy can be trained natively here when a native checkpoint is wanted.

Checkpoints written here are NOT loadable by the 5.1 image, and vice versa.

Isaac Lab 3.0 replaced --headless with --viz: omit it for headless, or pass a
CSV of backends from kit,newton,rerun,viser.

    python adam_u_rl30/scripts/train.py --num_envs 16 --max_iterations 3
    python adam_u_rl30/scripts/train.py --num_envs 4096
    python adam_u_rl30/scripts/train.py --viz kit --num_envs 16   # watch it learn
"""

import argparse
import os
import sys
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Adam-U grasping with RSL-RL (Isaac Lab 3.0).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=1500, help="RL policy training iterations.")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from --checkpoint_path.")
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Checkpoint to resume from. Requires --resume.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Hydra is not used here; keep it from seeing our flags.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from adam_u_rl30.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg  # noqa: E402
from adam_u_rl30.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = AdamUGraspEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed

    agent_cfg = RslRlAdamUGraspPPORunnerCfg()
    agent_cfg.device = device
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations

    gym.register(
        id="Isaac-AdamUGrasp30",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": AdamUGraspEnvCfg,
            "rsl_rl_cfg_entry_point": RslRlAdamUGraspPPORunnerCfg,
        },
    )
    env = gym.make("Isaac-AdamUGrasp30", cfg=env_cfg)

    # Anchored to the repo root, not the working directory, so checkpoints land
    # in the bind-mounted ./logs wherever this was launched from.
    log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[train30] logging to {log_dir}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    # Unlike the 5.1 script, --resume here actually does something.
    if args_cli.resume:
        if not args_cli.checkpoint_path:
            print("[train30] ERROR: --resume needs --checkpoint_path", file=sys.stderr)
            return 1
        print(f"[train30] resuming from {args_cli.checkpoint_path}")
        runner.load(args_cli.checkpoint_path)

    print(f"[train30] {args_cli.num_envs} envs, {agent_cfg.max_iterations} iterations, {device}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print("[train30] done")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
