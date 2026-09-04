from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass 
class RslRlAdamUGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "adam_u_grasp"

    # Required since Isaac Lab 2.3 / rsl-rl 3.x: maps algorithm observation sets
    # onto the groups the env publishes. This env exposes only "policy", so the
    # critic shares it. Without this the field stays MISSING and OnPolicyRunner
    # fails in resolve_obs_groups() before the first step.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Also new in rsl-rl 3.x, and MISSING by default. Empirical observation
        # normalization matters here: object positions are ~1 m while joint
        # velocities are ~1e-2, so an unnormalized critic is badly conditioned.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
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