"""Watch an Adam-U grasping policy — Isaac Sim 6.0 / Isaac Lab 3.0.

This is the reason the 6.0 environment exists: Isaac Sim 5.1's RTX renderer
segfaults on an RTX PRO 6000 Blackwell, so the 5.1 training image cannot draw
anything. 6.0 renders on the same card.

Two ways to load a policy:

  --policy  <exported/policy.pt>       TorchScript, from the 5.1 side  [default path]
  --checkpoint_path <model_N.pt>       native rsl-rl 5.0.1 checkpoint

Prefer --policy. rsl-rl 4.0 split the `policy` model config into separate
`actor` and `critic` configs, so a checkpoint written by the 5.1 image
(rsl-rl 3.0.1) will NOT load here. The TorchScript export sidesteps that
entirely: it bakes the observation normalizer and the actor MLP into one
module whose forward is just `actor(normalizer(obs))`, with no rsl-rl
involvement on either side. `scripts/play.py` on the 5.1 image writes it to
`<checkpoint_dir>/exported/policy.pt`.

Rendering backends come from Isaac Lab 3.0's --viz flag (CSV):

    --viz kit        RTX viewport      (verified working on this card)
    --viz newton     Newton viewer
    --viz rerun      Rerun            (bypasses RTX)
    --viz viser      browser-based     (bypasses RTX; works from another machine)
    (omit)           headless

Examples:

    # watch a 5.1-trained policy in the RTX viewport
    python adam_u_rl/scripts/play.py --viz kit --num_envs 4 --real-time \
        --policy logs/rsl_rl/adam_u_grasp/<run>/exported/policy.pt

    # same, in the browser, no RTX involved
    python adam_u_rl/scripts/play.py --viz viser --num_envs 4 \
        --policy logs/rsl_rl/adam_u_grasp/<run>/exported/policy.pt
"""

import argparse
import importlib.metadata as metadata
import os
import sys
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Watch an Adam-U grasping policy (Isaac Lab 3.0).")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--policy", type=str, default=None,
                   help="TorchScript policy exported by the 5.1 image (exported/policy.pt).")
group.add_argument("--checkpoint_path", type=str, default=None,
                   help="Native rsl-rl 5.0.1 checkpoint, i.e. trained by THIS environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to show.")
parser.add_argument("--steps", type=int, default=100000, help="Steps to run before exiting.")
parser.add_argument("--real-time", action="store_true", default=False,
                    help="Throttle to wall-clock, so the motion is watchable.")
parser.add_argument("--skip-dim-check", action="store_true", default=False,
                    help="Run even if the policy's input width disagrees with the env.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Hydra is not used here; keep it from seeing our flags.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from adam_u_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg  # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def actor_input_width(module: torch.nn.Module) -> int | None:
    """Width of the first Linear in a (possibly scripted) actor MLP."""
    for p in module.parameters():
        if p.dim() == 2:
            return p.shape[1]
    return None


def main() -> int:
    env_cfg = AdamUGraspEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    gym.register(
        id="Isaac-AdamUGrasp30",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": AdamUGraspEnvCfg},
    )
    env = gym.make("Isaac-AdamUGrasp30", cfg=env_cfg)

    # The two load paths differ in how observations are fetched and how many
    # values step() returns, so keep them apart rather than sniffing at
    # runtime. `native` means a checkpoint trained by THIS environment.
    native = args_cli.checkpoint_path is not None

    if not native:
        # --- TorchScript from the 5.1 side --------------------------------
        path = args_cli.policy
        if not os.path.isfile(path):
            print(
                f"[play30] ERROR: no TorchScript policy at {path}\n"
                f"         Export one from the 5.1 image:\n"
                f"           docker compose run --rm adam_rl python adam_u_rl/scripts/play.py \\\n"
                f"               --headless --checkpoint_path logs/rsl_rl/adam_u_grasp/<run>/model_N.pt\n"
                f"         which writes <run>/exported/policy.pt",
                file=sys.stderr,
            )
            return 1
        module = torch.jit.load(path, map_location=device).to(device).eval()
        print(f"[play30] loaded TorchScript policy from {path}")
        policy = module
        expected = actor_input_width(module)

        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
    else:
        # --- native rsl-rl 5.0.1 checkpoint -------------------------------
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from adam_u_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlAdamUGraspPPORunnerCfg

        agent_cfg = RslRlAdamUGraspPPORunnerCfg()
        agent_cfg.device = device
        # Strip the deprecated pre-5.0 model fields -- see train.py.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        try:
            runner.load(args_cli.checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[play30] ERROR loading {args_cli.checkpoint_path}: "
                f"{type(exc).__name__}: {exc}\n"
                f"         A checkpoint from the 5.1 image (rsl-rl 3.0.1) cannot load here --\n"
                f"         rsl-rl 4.0 split the model config into separate actor and critic.\n"
                f"         Use --policy with the TorchScript export instead.",
                file=sys.stderr,
            )
            return 1
        policy = runner.get_inference_policy(device=device)
        expected = actor_input_width(runner.alg.policy)

        # rsl-rl 5.x policies take the whole observation TensorDict and pick
        # their groups internally, so pass it through unmodified.
        obs = env.get_observations()

    # --- observation width has to agree ------------------------------------
    probe = obs["policy"] if not torch.is_tensor(obs) else obs
    actual = probe.shape[-1]
    if expected is not None and expected != actual:
        msg = (
            f"[play30] observation width mismatch: the policy expects {expected}, "
            f"this env produces {actual}.\n"
            f"         The 3.0 env config has drifted from the 5.1 one -- compare the\n"
            f"         observation terms in adam_u_rl/envs/adam_u_grasp_env_cfg.py\n"
            f"         against adam_u_rl/envs/adam_u_grasp_env_cfg.py."
        )
        if not args_cli.skip_dim_check:
            print(msg + "\n         Pass --skip-dim-check to run anyway.", file=sys.stderr)
            return 1
        print(msg + "\n         --skip-dim-check given; continuing.", file=sys.stderr)
    else:
        print(f"[play30] observation width {actual}, matches the policy")

    dt = env.unwrapped.step_dt
    print(
        f"[play30] running {args_cli.num_envs} envs, dt={dt:.4f}s"
        f"{', real-time' if args_cli.real_time else ''}"
    )

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            stepped = env.step(actions)
            obs = stepped[0]  # 5-tuple raw env, 4-tuple rsl-rl wrapper; obs is first either way
            if not native:
                obs = obs["policy"]
        step += 1
        if args_cli.real_time:
            remaining = dt - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)

    env.close()
    print(f"[play30] stopped after {step} steps")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
