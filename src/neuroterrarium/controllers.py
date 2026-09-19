"""Small learned controllers with one shared, inspectable action pipeline."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .interface import OBSERVATION_SIZE
from .world import ACTION_SCHEMA, OBSERVATION_SCHEMA

ARCHITECTURES = ("feedforward", "recurrent", "hybrid")
POLICY_SCHEMA = "gaussian-latent-ppo-v1"


def named_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)


def weights_digest(policy: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ActionResult:
    action: torch.Tensor
    base_action: torch.Tensor
    reflex_triggered: torch.Tensor
    correction_l1: torch.Tensor


def apply_action(raw: torch.Tensor, observation: torch.Tensor, *, hybrid: bool = False,
                 reflex_enabled: bool = True) -> ActionResult:
    """Transform latent samples identically during training and inference.

    PPO likelihoods belong to ``raw``, before sigmoid/tanh and the optional
    deterministic engineering reflex. The reflex only reads shared proximity
    and projection-change channels. It is not a second connectome model.
    """
    if raw.shape[-1:] != (4,) or observation.shape[-1:] != (OBSERVATION_SIZE,):
        raise ValueError("action or observation schema mismatch")
    if raw.shape[:-1] != observation.shape[:-1]:
        raise ValueError("action and observation batch dimensions disagree")
    if not torch.isfinite(raw).all() or not torch.isfinite(observation).all():
        raise ValueError("nonfinite controller input")
    base = torch.stack((raw[..., 0].sigmoid(), raw[..., 1].tanh(),
                        raw[..., 2].sigmoid(), raw[..., 3].sigmoid()), dim=-1)
    action = base.clone()
    if hybrid and reflex_enabled:
        front = observation[..., [15, 0, 1]].amax(dim=-1).clamp(0, 1)
        obstacle = ((front - 0.75) / 0.25).clamp(0, 1)
        expansion = observation[..., 32:48].clamp(0, 1).amax(dim=-1)
        escape = ((expansion - 0.25) / 0.75).clamp(0, 1)
        left = observation[..., 1:5].mean(dim=-1)
        right = observation[..., 12:16].mean(dim=-1)
        action[..., 0] = base[..., 0] * (1 - obstacle)
        action[..., 1] = (base[..., 1] + 0.8 * obstacle * (right-left)).clamp(-1, 1)
        action[..., 2] = torch.maximum(base[..., 2], escape)
    correction = (action - base).abs().sum(dim=-1)
    return ActionResult(action, base, correction > 0, correction)


class Policy(nn.Module):
    def __init__(self, architecture: str, seed: int, hidden_size: int = 64):
        super().__init__()
        if architecture not in ARCHITECTURES or type(hidden_size) is not int or not 8 <= hidden_size <= 256:
            raise ValueError("unsupported policy architecture or hidden size")
        self.architecture, self.seed, self.hidden_size = architecture, seed, hidden_size
        self.initialization_seed = named_seed(seed, f"initialization/{architecture}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.embedding = nn.Linear(OBSERVATION_SIZE, hidden_size)
            self.memory = (nn.GRUCell(hidden_size, hidden_size) if self.recurrent
                           else nn.Linear(hidden_size, hidden_size))
            self.actor = nn.Linear(hidden_size, 4)
            self.critic = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch.full((4,), -0.5))
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)
            # Start with modest escape drive; this is a common initialization,
            # shared by all architectures, not a post-training intervention.
            with torch.no_grad():
                self.actor.bias[2] = -2.0

    @property
    def recurrent(self) -> bool:
        return self.architecture != "feedforward"

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor,
                episode_start: torch.Tensor | None = None):
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_SIZE:
            raise ValueError("policy observation must be batch by 59")
        if hidden.shape != (observation.shape[0], self.hidden_size):
            raise ValueError("policy hidden state shape mismatch")
        if episode_start is not None:
            hidden = torch.where(episode_start[:, None], torch.zeros_like(hidden), hidden)
        embedded = torch.tanh(self.embedding(observation))
        if self.recurrent:
            features = self.memory(embedded, hidden)
            next_hidden = features
        else:
            features = torch.tanh(self.memory(embedded))
            next_hidden = torch.zeros_like(hidden)
        distribution = Normal(self.actor(features), self.log_std.clamp(-5, 2).exp())
        return distribution, self.critic(features).squeeze(-1), next_hidden

    def sample(self, observation: torch.Tensor, hidden: torch.Tensor,
               episode_start: torch.Tensor, generator: torch.Generator,
               *, deterministic: bool = False, reflex_enabled: bool = True) -> dict:
        distribution, value, next_hidden = self(observation, hidden, episode_start)
        raw = (distribution.mean if deterministic else distribution.mean +
               distribution.stddev * torch.randn(distribution.mean.shape, generator=generator))
        applied = apply_action(raw, observation, hybrid=self.architecture == "hybrid",
                               reflex_enabled=reflex_enabled)
        return {"raw_action": raw, "log_probability": distribution.log_prob(raw).sum(-1),
                "value": value, "hidden": next_hidden, "applied_action": applied.action,
                "base_action": applied.base_action, "reflex_triggered": applied.reflex_triggered,
                "correction_l1": applied.correction_l1}


class LearnedController:
    """Fixed weights with evolving recurrent state and a private sample stream."""
    def __init__(self, policy: Policy, sampling_seed: int, *, deterministic: bool = False):
        self.policy = policy.eval()
        self.weights_sha256 = weights_digest(policy)
        self.hidden = policy.initial_state(1)
        self.generator = torch.Generator().manual_seed(named_seed(sampling_seed, "policy-sampling"))
        self.episode_start = True
        self.deterministic = deterministic
        self.reflex_enabled = True
        self.last: dict = {}

    def reset(self) -> None:
        self.hidden.zero_()
        self.episode_start = True

    def act(self, observation: np.ndarray) -> np.ndarray:
        observed = np.asarray(observation, dtype=np.float32)
        if observed.shape != (OBSERVATION_SIZE,) or not np.isfinite(observed).all():
            raise ValueError("invalid shared observation")
        with torch.no_grad():
            result = self.policy.sample(torch.from_numpy(observed[None]), self.hidden,
                                        torch.tensor([self.episode_start]), self.generator,
                                        deterministic=self.deterministic,
                                        reflex_enabled=self.reflex_enabled)
        self.hidden = result["hidden"].clone()
        self.episode_start = False
        self.last = {key: value[0].detach().cpu().tolist() for key, value in result.items() if key != "hidden"}
        return result["applied_action"][0].numpy().copy()

    def snapshot(self) -> dict:
        return {"schema": POLICY_SCHEMA, "architecture": self.policy.architecture,
                "weights_sha256": self.weights_sha256,
                "hidden_size": self.policy.hidden_size,
                "hidden": self.hidden.tolist(), "random_state": self.generator.get_state().tolist(),
                "episode_start": self.episode_start, "deterministic": self.deterministic,
                "reflex_enabled": self.reflex_enabled, "last": copy.deepcopy(self.last)}

    def restore(self, state: dict) -> None:
        if state.get("schema") != POLICY_SCHEMA or state.get("architecture") != self.policy.architecture or state.get("hidden_size") != self.policy.hidden_size or state.get("weights_sha256") != self.weights_sha256:
            raise ValueError("controller snapshot architecture mismatch")
        hidden = torch.as_tensor(state["hidden"], dtype=torch.float32)
        if hidden.shape != self.hidden.shape or not torch.isfinite(hidden).all():
            raise ValueError("invalid controller hidden state")
        for key in ("episode_start", "deterministic", "reflex_enabled"):
            if type(state[key]) is not bool:
                raise ValueError("invalid controller flag")
        random = state["random_state"]
        if not isinstance(random, list) or len(random) != self.generator.get_state().numel() or any(type(x) is not int or not 0 <= x <= 255 for x in random):
            raise ValueError("invalid controller random state")
        generator = torch.Generator()
        generator.set_state(torch.tensor(random, dtype=torch.uint8))
        self.hidden, self.generator = hidden.clone(), generator
        self.episode_start = state["episode_start"]
        self.deterministic = state["deterministic"]
        self.reflex_enabled = state["reflex_enabled"]
        self.last = copy.deepcopy(state["last"])

    def clone(self) -> "LearnedController":
        """Share fixed weights while copying every mutable inference state."""
        result = LearnedController(self.policy, 0, deterministic=self.deterministic)
        result.restore(self.snapshot())
        return result


def policy_metadata(policy: Policy) -> dict:
    return {"schema": POLICY_SCHEMA, "architecture": policy.architecture,
            "seed": policy.seed, "initialization_seed": policy.initialization_seed,
            "hidden_size": policy.hidden_size, "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA, "observation_size": OBSERVATION_SIZE,
            "normalization": "fixed world schema scales; no fitted running normalization",
            "action_distribution": "four independent Gaussian latent variables",
            "action_transform": "sigmoid drive/escape/feed; tanh turn; shared optional hybrid reflex"}
