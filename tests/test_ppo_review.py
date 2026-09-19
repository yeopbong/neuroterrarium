"""Independent scalar-sequence checks of the batched PPO objective."""

import copy
import math

import pytest
import torch

from neuroterrarium.controllers import Policy
from neuroterrarium.training import PPOTrainer
from test_training import small_config


@pytest.mark.parametrize("architecture", ["feedforward", "recurrent", "hybrid"])
def test_padded_update_matches_unpadded_scalar_sequences(architecture):
    config = small_config()
    config.update(sequence_length=3, minibatch_sequences=16, episode_steps=4)
    trainer = PPOTrainer(config, architecture, 101)
    rollout = trainer.collect(7)
    reference = copy.deepcopy(trainer.policy)
    # Scalar processing deliberately has no padded tensor or validity mask.
    logps, values, entropies, old_logps, returns, advantages = [], [], [], [], [], []
    mean = rollout["advantage"].mean()
    scale = rollout["advantage"].std(unbiased=False) + 1e-8
    for environment in range(2):
        for start in range(0, 7, 3):
            hidden = rollout["hidden"][start, environment][None].detach()
            for step in range(start, min(start + 3, 7)):
                distribution, value, hidden = reference(
                    rollout["observations"][step, environment][None], hidden,
                    rollout["starts"][step, environment][None])
                raw = rollout["raw_action"][step, environment][None]
                residual = (raw - distribution.mean) / distribution.stddev
                logps.append((-0.5 * residual.square() - distribution.stddev.log()
                              - 0.5 * math.log(2 * math.pi)).sum())
                entropies.append((distribution.stddev.log()
                                  + 0.5 * math.log(2 * math.pi * math.e)).sum())
                values.append(value.squeeze())
                old_logps.append(rollout["log_probability"][step, environment])
                returns.append(rollout["return"][step, environment])
                advantages.append((rollout["advantage"][step, environment] - mean) / scale)
    ratio = torch.exp(torch.stack(logps) - torch.stack(old_logps))
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)
    advantage = torch.stack(advantages)
    actor = -torch.minimum(ratio * advantage,
                           ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage).mean()
    critic = (torch.stack(values) - torch.stack(returns)).square().mean()
    entropy = torch.stack(entropies).mean()
    loss = actor + config["value_coefficient"] * critic - config["entropy_coefficient"] * entropy
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), config["max_grad_norm"])
    captured = {}
    trainer.optimizer.step = lambda: captured.update({
        name: parameter.grad.detach().clone() for name, parameter in trainer.policy.named_parameters()})
    trainer.update(rollout)
    for name, parameter in reference.named_parameters():
        # Float32 batch-vs-scalar reduction changes final rounding only.
        torch.testing.assert_close(captured[name], parameter.grad, atol=2e-6, rtol=2e-5)


def test_recurrent_chunk_start_is_not_an_episode_reset():
    policy = Policy("recurrent", 303, 16)
    generator = torch.Generator().manual_seed(481)
    observations = torch.randn(9, 1, 59, generator=generator)
    hidden = policy.initial_state(1)
    history, output = [], []
    for step in range(9):
        history.append(hidden.detach().clone())
        distribution, _, hidden = policy(observations[step], hidden, torch.tensor([step == 0]))
        output.append(distribution.mean.detach())
    hidden = history[4]
    for step in range(4, 9):
        distribution, _, hidden = policy(observations[step], hidden, torch.tensor([False]))
        torch.testing.assert_close(distribution.mean, output[step], atol=0, rtol=0)
    reset_distribution, _, _ = policy(observations[4], history[4], torch.tensor([True]))
    assert not torch.equal(reset_distribution.mean, output[4])
