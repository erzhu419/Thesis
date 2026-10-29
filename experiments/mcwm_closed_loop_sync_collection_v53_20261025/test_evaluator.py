"""Catch live-actor leakage, target arithmetic/RNG drift and F-input mismatch."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters
from experiments.mcwm_closed_loop_sync_collection_v53_20261025.evaluator import FrozenIssueEvaluator


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Module()
        self.network.network = torch.nn.ModuleList([torch.nn.Linear(1, 3)])
        with torch.no_grad():
            self.network.network[-1].weight.zero_()
            self.network.network[-1].bias.copy_(torch.tensor([.3, .7, -.2]))

    @property
    def gate(self):
        return self.network.network[-1].bias[0]

    @property
    def mean(self):
        return self.network.network[-1].bias[1]

    def distribution_parameters(self, states):
        shape = (len(states),)
        return self.gate.expand(shape), self.mean.expand(shape), states.new_full(shape, -.2)

    def soft_action_components(self, states):
        gate, mean, logstd = self.distribution_parameters(states)
        hp = gate.sigmoid()
        latent = mean+logstd.exp()*torch.randn_like(mean)
        fraction = latent.sigmoid().clamp(torch.finfo(states.dtype).eps*.5,
                                         1.-torch.finfo(states.dtype).eps*.5)
        conditional = -.5*((latent-mean)/logstd.exp()).square()-logstd
        return dict(zero_probability=1.-hp, hold_probability=hp,
            zero_action=states.new_full((len(states), 1), -1.),
            positive_action=(2.*fraction-1.).reshape(-1, 1),
            zero_log_prob=torch.log1p(-hp), positive_log_prob=hp.log()+conditional,
            positive_conditional_log_prob=conditional)


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(2.))

    def encode_states(self, states):
        assert states.shape[1] == 313
        return states

    def forward_features_subset(self, features, actions, chosen):
        base = self.level+actions[:, 0]*3.+features[:, 229]*.5
        return torch.stack((base, base+.4)).index_select(0, chosen)


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(1.7))

    def forward(self, states):
        assert states.shape[1] == 229
        return self.level.expand(len(states))


class Learner:
    def __init__(self):
        self.actor, self.target_critic, self.value = Actor(), Critic(), Value()
        self.device, self.total_updates = torch.device('cpu'), 50024
        self.alpha = torch.tensor(.2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.target_critic.parameters())
        self.value_optimizer = torch.optim.Adam(self.value.parameters())
        self.alpha_optimizer = torch.optim.Adam([self.alpha.requires_grad_()])
        self.action_transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))
        self.config = SimpleNamespace(mixed_backup_lambda=.1, ensemble_size=2,
            target_subset_size=2, target_subset_replacement=False, online_backup='h2oplus',
            reward_scale=.01, backup_entropy=True)

    def _sample_target_subset(self):
        return torch.randperm(2)

    def _target(self, batch, *, backup):
        states = batch['next_observations'][:, :229]
        parts = self.actor.soft_action_components(states)
        actions = torch.cat((self.action_transform(states, parts['zero_action']),
                             self.action_transform(states, parts['positive_action'])))
        chosen = self._sample_target_subset()
        features = self.target_critic.encode_states(batch['next_observations'])
        qs = self.target_critic.forward_features_subset(torch.cat((features, features)), actions, chosen)
        zero, positive = qs.split(len(states), dim=1)
        zero, positive = zero.min(0).values, positive.min(0).values
        zero -= self.alpha.detach()*parts['zero_log_prob']
        positive -= self.alpha.detach()*parts['positive_log_prob']
        sac = parts['zero_probability']*zero+parts['hold_probability']*positive
        bootstrap = .1*self.value(states)+.9*sac
        return self.config.reward_scale*batch['rewards']+batch['discounts']*(1.-batch['terminals'])*bootstrap


def row():
    state = np.zeros(229, np.float32)
    state[0] = .4
    memory = dict(vector=[.2]*84, observed_at=9001.)
    return SimpleNamespace(ns=state, r=-10., terminal=False, memory_ns=memory,
        completed_at=9001.)


def test_native_M1_lambda_zero_M8_and_torch_RNG():
    learner = Learner()
    evaluator = FrozenIssueEvaluator(learner, 1800.)
    before = torch.get_rng_state().clone()
    first = evaluator.score(learner, row())
    second = evaluator.score(learner, row())
    assert first['all_pass'] and first == second
    assert torch.equal(before, torch.get_rng_state())
    assert first['m1']['native_target'] == first['m1']['lambda_zero_target']
    assert first['m8']['native_target'] == first['m8']['lambda_zero_target']
    assert all(not p.requires_grad for p in evaluator.actor.parameters())


def test_frozen_actor_survives_private_live_actor_perturbation_but_V_Q_alpha_remain_live():
    learner = Learner()
    evaluator = FrozenIssueEvaluator(learner, 1800.)
    first = evaluator.score(learner, row())
    private = deepcopy(learner)
    with torch.no_grad():
        private.actor.mean.add_(2.)
        private.actor.gate.sub_(3.)
    changed = evaluator.score(private, row())
    assert changed['m1'] == first['m1'] and changed['m8'] == first['m8']
    assert changed['current_actor_M1_target'] != first['current_actor_M1_target']
    for name in ('value', 'target_critic'):
        altered = deepcopy(learner)
        with torch.no_grad():
            getattr(altered, name).level.add_(1.)
        assert evaluator.score(altered, row())['m1']['native_target'] != first['m1']['native_target']
    altered = deepcopy(learner)
    with torch.no_grad():
        altered.alpha += .7
    assert evaluator.score(altered, row())['m1']['native_target'] != first['m1']['native_target']
    assert evaluator.unchanged()


def test_issue_actual_F_and_whole_overlap_are_computed():
    learner = Learner()
    evaluator = FrozenIssueEvaluator(learner, 1800.)
    state = row().ns
    proposal = np.asarray([.3], np.float32)
    executed = learner.action_transform.numpy(state, proposal)
    behavior = dict(policy_input=state.tolist(), execution_input=state.tolist(),
        proposal=proposal.tolist(), executed_action=executed.tolist(),
        nominal_executed_action=executed.tolist(), parameters=actor_parameters(learner, state),
        intervention=False)
    actual = evaluator.issue(behavior, 9000.)
    assert actual['all_pass'] and actual['ratio']['c'] == 1.
    assert abs(actual['overlap']['total_overlap']-1.) < 5e-12
    wrong = deepcopy(behavior)
    wrong['parameters']['mean'] += 1.
    assert not evaluator.issue(wrong, 9000.)['all_pass']
    wrong = deepcopy(behavior)
    wrong['executed_action'] = [-1.]
    assert not evaluator.issue(wrong, 9000.)['checks']['actual_F_exact']


def test_matching_Q_requires_actual_successor_memory_and_does_not_reapply_F():
    learner = Learner()
    evaluator = FrozenIssueEvaluator(learner, 1800.)
    current = row()
    following = dict(started_at=current.completed_at, s=current.ns, a=[.5], memory_s=current.memory_ns)
    result = evaluator.score(learner, current, following)
    assert result['recorded_next_target_q'] == pytest.approx(3.6)
    bad = deepcopy(following)
    bad['memory_s']['vector'][0] += 1.
    with pytest.raises(RuntimeError, match='genuine contiguous'):
        evaluator.score(learner, current, bad)


def test_snapshot_must_be_post24_and_issue_pre():
    learner = Learner()
    learner.total_updates = 50016
    with pytest.raises(ValueError, match='actual issue-pre'):
        FrozenIssueEvaluator(learner, 1800.)
    learner.total_updates = 50024
    with pytest.raises(ValueError, match='actual issue-pre'):
        FrozenIssueEvaluator(learner, 9000.)


def test_native_wiring_challenge_uses_only_private_Hurdle_head_and_keeps_Adam_RNG():
    learner = Learner()
    evaluator = FrozenIssueEvaluator(learner, 1800.)
    actor_values = deepcopy(learner.actor.state_dict())
    rng = torch.get_rng_state().clone()
    proof = evaluator.check_wiring(learner, row())
    assert proof['all_pass']
    assert proof['original_live_M1_target'] != proof['perturbed_live_M1_target']
    assert all(torch.equal(value, actor_values[key]) for key, value in learner.actor.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(not optimizer.state for optimizer in (learner.actor_optimizer, learner.critic_optimizer,
        learner.value_optimizer, learner.alpha_optimizer))
