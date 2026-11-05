"""PPO runner config for Adam-U grasping, rsl-rl 5.x schema (Isaac Lab 3.0).

rsl-rl >= 4.0 replaced the single `policy` model config with separate `actor`
and `critic` configs (`RslRlMLPModelCfg`), and folded `init_noise_std` into a
distribution config on the actor. Hyperparameters are otherwise identical to
genesis_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py.

Because the model config schema differs, **checkpoints are NOT interchangeable
between the two environments**. To watch a policy trained on the 5.1 side, use
the TorchScript export instead -- see scripts/play.py.
"""

import os
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
    # "actor" is named explicitly: rsl-rl 5.x falls back to the group called
    # "policy" when it is missing, but warns that the fallback will be removed.
    obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy"]}

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


@configclass
class RslRlWujiSynergyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Experiment 2 (finger-only synergy grasp): the spec's initial PPO settings.

    Small actor/critic (128-128), fixed 3e-4 learning rate, entropy 0.01. The
    episode is 120 steps, so 24 steps/env x 1024 envs = 24.6k samples per
    iteration, ~5 episodes' worth per env over an iteration cycle.
    """

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "adam_u30_wuji_synergy"
    # "actor" is named explicitly: rsl-rl 5.x falls back to the group called
    # "policy" when it is missing, but warns that the fallback will be removed.
    obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128], activation="elu", obs_normalization=True)

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.003, not the spec's 0.01. Measured: at 0.01 the action std inflates
        # steadily within a run AND is inherited across curriculum levels, so it
        # ratchets -- 0.98 at the end of level 0, 1.15 at level 1, 2.25 at level
        # 2. It stopped being cosmetic at level 2, where the last 148 iterations
        # took success from 99.8 % down to 96.1 % and finger-table collisions
        # from 0.2 % to 3.9 %: wider exploration on the Cartesian wrist channels
        # drives the fingers into the table. Evaluation uses the mean action, so
        # this hides in the success curve until it does not.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlWujiPipelinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Staged pipeline (wuji_pipeline_env_cfg): 55 observations, 8 actions.

    The architecture is FIXED across every stage and across behaviour cloning:
    `[256, 256, 128]` elu, because a stage-k checkpoint has to load into stage
    k+1 and a cloned actor has to load into PPO with no surgery. That is also
    why this is a new config rather than an edit to the Experiment 1
    (`[512,256,128]`) or Experiment 2 (`[128,128]`) ones -- changing those
    would invalidate their logged runs.

    Hyperparameters are the spec's: lr 3e-4 fixed, entropy 0.01, gamma 0.99,
    lambda 0.95, clip 0.2. The episode is 150 steps, so 24 steps/env x 2048
    envs is ~49k samples per iteration.

    The auxiliary behaviour-cloning loss

        actor_loss = PPO_loss + bc_coefficient * BC_loss,   1.0 -> 0.0

    is NOT expressible here: rsl-rl 5.0.1's PPO has no auxiliary-loss hook, so
    it needs a subclass that holds a demonstration batch and adds the term in
    `update()`. Until that exists this config trains PPO-only, which is the
    baseline the BC+PPO comparison is measured against anyway.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "adam_u30_wuji_pipeline"
    obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # Well under Experiment 1's 1.0: a large initial std on the Cartesian
        # wrist channels walks the hand off the pre-grasp pose before the
        # fingers have done anything, and once behaviour cloning is in the loop
        # it would throw the initialisation away.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.003, not the spec's 0.01. Measured: at 0.01 the action std inflates
        # steadily within a run AND is inherited across curriculum levels, so it
        # ratchets -- 0.98 at the end of level 0, 1.15 at level 1, 2.25 at level
        # 2. It stopped being cosmetic at level 2, where the last 148 iterations
        # took success from 99.8 % down to 96.1 % and finger-table collisions
        # from 0.2 % to 3.9 %: wider exploration on the Cartesian wrist channels
        # drives the fingers into the table. Evaluation uses the mean action, so
        # this hides in the success curve until it does not.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlWujiReachGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Experiment 4 (wuji_reach_grasp_env_cfg): 74 observations, 8 actions.

    The pipeline's settings, unchanged: 256-256-128 elu, obs normalisation,
    init std 0.3, lr 3e-4 fixed, entropy 0.003, 24 steps/env. The one
    difference is the algorithm: `MaskedPPO` weights the actor loss by the
    env's `actor_mask`, so the 70 scripted SETTLE / LIFT / HOLD steps of every
    episode never train the actor (docs/EXPERIMENT4_DESIGN.md §4).
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "adam_u30_wuji_reach_grasp"
    obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy"]}
    if os.environ.get("ADAM_U_RG_PRIV_CRITIC", "0") == "1":   # asymmetric critic (env_cfg.CriticPrivCfg)
        obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy", "critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="genesis_rl.algos.rsl_rl.masked_ppo:MaskedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
