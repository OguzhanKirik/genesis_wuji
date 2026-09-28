"""PPO with a per-transition ACTOR mask (rsl-rl 5.0.1).

Experiment 4's episode has 70 scripted steps (SETTLE, LIFT, HOLD) in which the
environment ignores the policy's actions. Left in the rollout, those
transitions still enter the surrogate loss: the ratio of the new to the old
log-probability of an action that had no effect, times an advantage the action
could not have influenced. Its expectation is zero and its variance is not,
and it is 37 % of every rollout. The pipeline lived with it ("LIFT and HOLD
still train the actor" in TRAINING_LOG.md); this class removes it.

The env publishes `extras["actor_mask"]`, (num_envs,) bool, TRUE for the
transitions whose action the environment actually used. `MaskedPPO` stores it
alongside each transition and weights the surrogate loss, the entropy bonus
and the KL estimate by it. The critic is trained on every transition: the
value of a scripted state is still a value.

If the env publishes no mask, every transition counts and this is exactly PPO.

Pinned to rsl-rl 5.0.1's `PPO.update()`; `tools/test_reach_grasp.py` checks
that masked transitions contribute zero actor gradient.
"""

from __future__ import annotations

from collections.abc import Generator

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config


class MaskedRolloutStorage(RolloutStorage):
    """RolloutStorage plus an (T, N, 1) actor mask, yielded on each mini-batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_mask = torch.ones(self.num_transitions_per_env, self.num_envs, 1, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        mask = getattr(transition, "actor_mask", None)
        step = self.step
        super().add_transition(transition)
        if mask is None:
            self.actor_mask[step] = 1.0
        else:
            self.actor_mask[step].copy_(mask.view(-1, 1).float())

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        actor_mask = self.actor_mask.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
                batch.actor_mask = actor_mask[batch_idx]
                yield batch


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of x over the entries where m is 1. x: (B,) or (B, 1); m: (B, 1)."""
    x = x.reshape(-1)
    m = m.reshape(-1)
    return (x * m).sum() / m.sum().clamp(min=1.0)


class MaskedPPO(PPO):
    """rsl-rl 5.0.1 PPO with the actor loss weighted by the env's actor mask."""

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor,
                         extras: dict) -> None:
        mask = extras.get("actor_mask") if isinstance(extras, dict) else None
        self.transition.actor_mask = None if mask is None else mask.to(self.device)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("MaskedPPO supports feedforward actor and critic only")
        if self.rnd or self.symmetry:
            raise NotImplementedError("MaskedPPO does not combine with RND or symmetry augmentation")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_mask = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            mask = getattr(batch, "actor_mask", None)
            if mask is None:
                mask = torch.ones_like(batch.values)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = _masked_mean(kl, mask)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss, masked.
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            adv = torch.squeeze(batch.advantages)
            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = _masked_mean(torch.max(surrogate, surrogate_clipped), mask)

            # Value loss, unmasked.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            entropy_m = _masked_mean(entropy, mask)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_m

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_m.item()
            mean_mask += mask.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "actor_mask_fraction": mean_mask / num_updates,
        }

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MaskedPPO":
        """PPO.construct_algorithm with a MaskedRolloutStorage."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = MaskedRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg.get("multi_gpu"))
