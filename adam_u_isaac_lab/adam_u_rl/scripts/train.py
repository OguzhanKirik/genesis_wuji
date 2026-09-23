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

    python adam_u_rl/scripts/train.py --num_envs 16 --max_iterations 3
    python adam_u_rl/scripts/train.py --num_envs 4096
    python adam_u_rl/scripts/train.py --viz kit --num_envs 16   # watch it learn

    # Wuji Hand 2 body, bolt in adam-moveit's workcell (see adam_u_wuji_grasp_env_cfg.py)
    python adam_u_rl/scripts/train.py --robot wuji --num_envs 4096
    ADAM_U_WORKCELL=bin python adam_u_rl/scripts/train.py --robot wuji --num_envs 4096
    # grasp-only: hand starts over the bolt, no reach phase
    python adam_u_rl/scripts/train.py --robot wuji --scenario grasp --num_envs 4096
    # Experiment 2: hand only, fixed wrist, two finger synergies, scripted lift
    python adam_u_rl/scripts/train.py --robot wuji --scenario synergy --num_envs 1024
    # the staged pipeline: whole robot, IK wrist + synergies, 55 obs / 8 actions
    ADAM_U_STAGE=0 python adam_u_rl/scripts/train.py --robot wuji --scenario pipeline --num_envs 2048
"""

import argparse
import importlib.metadata as metadata
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
parser.add_argument("--reset-std", type=float, default=None, metavar="STD",
                    help="After --resume, reset the actor's exploration std to this value. "
                         "A curriculum should inherit the POLICY, not its exploration noise: "
                         "the std inflates within every run and each resume carries the inflated "
                         "value forward, so it ratchets up the ladder (0.98 -> 1.15 -> 2.25 over "
                         "levels 0-2) until the noise is larger than the displacement being "
                         "corrected.")
parser.add_argument("--robot", choices=["pnd", "wuji"], default="pnd",
                    help="pnd: Inspire hands, cube on a table (adam_u_grasp_env_cfg). "
                         "wuji: Wuji Hand 2, bolt in adam-moveit's workcell (adam_u_wuji_grasp_env_cfg).")
parser.add_argument("--scenario", choices=["reach", "grasp", "synergy", "pipeline"], default="reach",
                    help="wuji only. reach: arm starts beside the torso and must reach the bolt. "
                         "grasp: arm starts hovering over the bolt, fingers do the work "
                         "(adam_u_wuji_grasponly_env_cfg). synergy: Experiment 2, hand only, fixed "
                         "wrist, two finger synergies, scripted lift (wuji_synergy_grasp_env_cfg). "
                         "pipeline: the staged pipeline, whole robot, 6 Cartesian wrist actions "
                         "through differential IK + 2 synergies, 55 obs / 8 actions fixed across "
                         "stages (wuji_pipeline_env_cfg). ADAM_U_STAGE picks the stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Hydra is not used here; keep it from seeing our flags.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg  # noqa: E402

# The two robots are different articulations with different action/observation
# widths, so they are different tasks with separate experiment directories.
# A checkpoint from one cannot run on the other.
AgentCfg = RslRlAdamUGraspPPORunnerCfg
if args_cli.robot == "wuji" and args_cli.scenario == "pipeline":
    from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiPipelinePPORunnerCfg as AgentCfg  # noqa: E402
    from adam_u_rl.envs.wuji_pipeline_env_cfg import WujiPipelineEnvCfg as EnvCfg  # noqa: E402
    # The stage changes the action mask and the randomisation, not the
    # interface -- but it changes what the numbers mean, so runs get their own
    # directory per stage.
    TASK_ID = "Isaac-WujiPipeline30"
    EXPERIMENT = f"adam_u30_wuji_pipeline_L{__import__('os').environ.get('ADAM_U_LEVEL', '0')}"
elif args_cli.robot == "wuji" and args_cli.scenario == "synergy":
    from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiSynergyPPORunnerCfg as AgentCfg  # noqa: E402
    from adam_u_rl.envs.wuji_synergy_grasp_env_cfg import WujiSynergyGraspEnvCfg as EnvCfg  # noqa: E402
    # The finger count changes the robot asset, the action semantics and the
    # observation width, so runs must not share a directory.
    TASK_ID = "Isaac-WujiSynergyGrasp30"
    EXPERIMENT = f"adam_u30_wuji_synergy_f{__import__('os').environ.get('ADAM_U_FINGERS', '3')}"
elif args_cli.robot == "wuji" and args_cli.scenario == "grasp":
    from adam_u_rl.envs.adam_u_wuji_grasponly_env_cfg import AdamUWujiGraspOnlyEnvCfg as EnvCfg  # noqa: E402
    TASK_ID, EXPERIMENT = "Isaac-AdamUWujiGraspOnly30", "adam_u30_wuji_grasponly"
elif args_cli.robot == "wuji":
    from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import AdamUWujiGraspEnvCfg as EnvCfg  # noqa: E402
    TASK_ID, EXPERIMENT = "Isaac-AdamUWujiGrasp30", "adam_u30_wuji_grasp"
else:
    from adam_u_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg as EnvCfg  # noqa: E402
    TASK_ID, EXPERIMENT = "Isaac-AdamUGrasp30", "adam_u30_grasp"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed

    agent_cfg = AgentCfg()
    agent_cfg.device = device
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.experiment_name = EXPERIMENT

    # RslRlMLPModelCfg still carries the pre-5.0 `stochastic` and
    # `init_noise_std` fields, both MISSING by default. Isaac Lab expects this
    # helper to strip or migrate them; without it they reach the model as
    # kwargs and rsl-rl 5.0.1 dies with
    #   TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'
    # Isaac Lab 3.0's own train script calls this; a hand-written one must too.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": EnvCfg,
            "rsl_rl_cfg_entry_point": AgentCfg,
        },
    )
    env = gym.make(TASK_ID, cfg=env_cfg)

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
        if args_cli.reset_std is not None:
            actor = runner.alg.get_policy()
            found = False
            for name, param in actor.named_parameters():
                if "std" in name.lower():
                    with torch.no_grad():
                        param.fill_(args_cli.reset_std)
                    print(f"[train30] reset {name} to {args_cli.reset_std}")
                    found = True
            if not found:
                print("[train30] WARNING: --reset-std found no std parameter on the actor; "
                      "the exploration noise was NOT reset.", file=sys.stderr)

    print(f"[train30] robot={args_cli.robot} task={TASK_ID} {args_cli.num_envs} envs, "
          f"{agent_cfg.max_iterations} iterations, {device}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print("[train30] done")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
