"""Original RE-SAC equations with multi-line categorical inputs and actions."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

SOURCE = Path(__file__).resolve().parent.parent / "legacy_bus/runtime/BAPR/bus_experiments/frozen_policy_bank_core.py"
names = {"TrainConfig", "ReplayBuffer", "EmbeddingLayer", "VectorizedLinear", "BusCritic", "BusPolicy", "BusSAC"}
tree = ast.parse(SOURCE.read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"
             or isinstance(node, ast.ClassDef) and node.name in names]
namespace = dict(__name__=__name__, torch=torch, nn=nn, np=np, F=F, Normal=Normal, math=math,
                 deepcopy=deepcopy, dataclass=dataclass, Any=Any)
exec(compile(tree, str(SOURCE), "exec"), namespace)
core = SimpleNamespace(**namespace)


def encode(state):
    value = np.asarray(state, dtype=np.float32).copy()
    value[..., :5] = np.rint(value[..., :5] * np.array([12., 25., 55., 24., 1.], np.float32))
    return value


class Replay(core.ReplayBuffer):
    def push(self, state, action, reward, next_state, done):
        super().push(encode(state), 30.*(np.asarray(action)+1.), reward, encode(next_state), done)


class Agent:
    def __init__(self, env, arm, mode, seed, protocol):
        dimension = 2 if mode == "dual" else 1
        config = core.TrainConfig(seed=seed, max_episodes=protocol["episodes"], role="specialist", fixed_mode="normal",
            hidden_dim=protocol["hidden_dim"], ensemble_size=protocol["ensemble_size"],
            batch_size=protocol["batch_size"], learning_rate=protocol["learning_rate"],
            beta=0. if arm in ("actor_off", "ensemble_sac_control") else -2.,
            beta_ood=0. if arm in ("critic_off", "ensemble_sac_control") else .01)
        # max_agent_num caps simultaneous buses, not the lifetime bus-ID range.
        categories = (len(env.line_map), max(line.effective_trip_num for line in env.line_map.values())+1,
                      max(len(line.stations) for line in env.line_map.values())+1, 25, 2)
        learner = core.BusSAC.__new__(core.BusSAC)
        learner.config, learner.device, learner.action_dim = config, torch.device("cpu"), dimension
        learner.critic = core.BusCritic(15, dimension, config.hidden_dim, config.ensemble_size, categories)
        learner.target_critic = deepcopy(learner.critic)
        learner.policy = core.BusPolicy(15, dimension, config.hidden_dim, categories, 60.)
        learner.log_alpha = torch.zeros(1, requires_grad=True)
        learner.alpha = 1.
        learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=config.learning_rate)
        learner.policy_optimizer = torch.optim.Adam(learner.policy.parameters(), lr=config.learning_rate)
        learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=config.learning_rate)
        self.learner, self.config, self.actor = learner, config, learner.policy
        self.replay_buffer = Replay(config.replay_capacity)
        self.updates = 0

    def select_action(self, state, deterministic=False):
        return self.learner.select_action(encode(state), deterministic=deterministic)/30.-1.

    def update(self, recent_rollout=None):
        self.updates += 1
        metrics = self.learner.update(self.replay_buffer, self.updates)
        # The inherited runner intentionally does not update the actor every critic step.
        if self.updates % self.config.critic_actor_ratio:
            del metrics["policy_loss"]
        return metrics
