"""PPO runner config for Adam-U grasping, rsl-rl 5.x schema (Isaac Lab 3.0).

rsl-rl >= 4.0 replaced the single `policy` model config with separate `actor`
and `critic` configs (`RslRlMLPModelCfg`), and folded `init_noise_std` into a
distribution config on the actor. Hyperparameters are otherwise identical to
adam_u_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py.

Because the model config schema differs, **checkpoints are NOT interchangeable
between the two environments**. To watch a policy trained on the 5.1 side, use
the TorchScript export instead -- see scripts/play.py.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAdamUGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "adam_u30_grasp"

    # This env publishes only a "policy" observation group, so the critic
    # shares it. Required since 2.3 / rsl-rl 3.x.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        # Matches actor_obs_normalization=True on the 5.1 side. The observation
        # vector mixes metre-scale positions with 1e-2 joint velocities.
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
