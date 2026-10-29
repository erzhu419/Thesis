"""Executed-action gradients, isolation, exact zero-coefficient and real counts."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import learning as v
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_replay, table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def trainer(arm='fresh', *, coefficient=.2, vanilla=False):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    cls = v.FreshValueTrainer if vanilla else v.ExecutedCQLTrainer
    result = cls(make_replay(arm), config=config, device='cpu', seed=42,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    if not vanilla:
        result.activate_executed_cql(coefficient)
    return result


def covariates(learner):
    learner.replay_buffer.sample_mixed(learner.config.batch_size, .5, rng=learner.rng)
    return learner.replay_buffer.last_covariates


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_zero_coefficient_exactly_reproduces_v64_parameters_adam_and_main_rng(arm):
    vanilla, audited = trainer(arm, vanilla=True), trainer(arm, coefficient=0.)
    for _ in range(2):
        before_packet = torch.get_rng_state().clone()
        audited.begin_actor_packet()
        assert torch.equal(before_packet, torch.get_rng_state())
        for step in range(8):
            before = torch.get_rng_state().clone()
            expected = vanilla.continue_step(collect_diagnostics=step == 0)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = audited.continue_step(collect_diagnostics=step == 0)
            if step == 0:
                diagnostic = actual['td_diagnostic']
                assert diagnostic.pop('gradient_scope') == 'TD only; CQL excluded'
                diagnostic['critic_change_cosine_negative_total_gradient'] = diagnostic.pop(
                    'critic_change_cosine_negative_td_gradient')
            exact(expected, actual)
            assert torch.equal(after, torch.get_rng_state())
            assert vanilla.rng.bit_generator.state == audited.rng.bit_generator.state
            assert vanilla.replay_buffer.last_sampling == audited.replay_buffer.last_sampling
        for name in ('actor', 'critic', 'target_critic', 'value'):
            exact(getattr(vanilla, name).state_dict(), getattr(audited, name).state_dict())
        for name in ('actor', 'critic', 'value', 'alpha'):
            exact(getattr(vanilla, name+'_optimizer').state_dict(),
                  getattr(audited, name+'_optimizer').state_dict())
        exact(vanilla.log_alpha, audited.log_alpha)
    assert audited.cql_audit()['penalty_calls'] == 0
    assert audited.actor_schedule_audit()['schedule'] == v.SCHEDULE


def test_floor_aliases_are_exactly_deduplicated_before_all_head_logmeanexp():
    learner = trainer()
    values = covariates(learner)
    states = values['observations']
    states[:, 0] = .3
    slots = states[:, 5:].reshape(-1, 16, 14)
    slots[:, 0, 11:13] = torch.tensor([1., 0.])
    slots[:, 0, 8:10] = torch.tensor([.2, .1])
    actual = learner._cql_penalty(object(), object())
    with torch.no_grad():
        executed = learner.action_transform(states, torch.full_like(values['actions'], -1.))
        assert torch.all(learner.action_transform.hold_seconds(executed) == 60.)
        expected = (learner.critic(states, executed)-learner.critic(states, values['actions'])).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    audit = learner.cql_audit()
    assert audit['candidate_rows_before_dedup'] == len(states)*22
    assert audit['candidate_rows_after_dedup'] == len(states)
    assert audit['unique_candidates_per_state'] == dict(min=1, max=1, mean=1.)
    assert len(audit['by_head']) == 2 and audit['all_heads']


def test_cql_reads_only_common_covariates_and_never_transforms_factual_actions_twice():
    learner = trainer('no_fresh')
    values = covariates(learner)
    values['actions'].fill_(.9)

    class CovariatesOnly(dict):
        def __getitem__(self, key):
            assert key in ('observations', 'actions'), f'outcome access: {key}'
            return super().__getitem__(key)

    learner.replay_buffer.last_covariates = CovariatesOnly(values)
    transform = learner.action_transform
    transforms, evaluated = [], []

    def transform_spy(states, proposal):
        transforms.append((states.detach().clone(), proposal.detach().clone()))
        return transform(states, proposal)

    learner.action_transform = transform_spy
    forward = learner.critic.forward_features

    def critic(features, actions):
        evaluated.append(actions.detach().clone())
        return forward(features, actions)

    learner.critic.forward_features = critic
    learner._cql_penalty(object(), object())
    assert len(transforms) == 1 and len(transforms[0][0]) == len(values['actions'])*22
    exact(evaluated[0], values['actions'])
    repeated = values['observations'][:, None].expand(-1, 22, -1).reshape(-1, 229)
    exact(transforms[0][0], repeated)
    audit = learner.cql_audit()
    assert not audit['outcome_fields_read'] and not audit['factual_actions_transformed']


def test_candidate_randomness_is_isolated_and_no_actor_gradient_enters_cql():
    learner = trainer()
    values = covariates(learner)
    before_torch = torch.get_rng_state().clone()
    before_numpy = deepcopy(np.random.get_state())
    before_sampler = deepcopy(learner.rng.bit_generator.state)
    first = learner._executed_candidates(values['observations'])
    second = learner._executed_candidates(values['observations'])
    exact(first, second)
    penalty = learner._cql_penalty(object(), object())
    penalty.backward()
    assert all(parameter.grad is None for parameter in learner.actor.parameters())
    assert any(parameter.grad is not None for parameter in learner.critic.parameters())
    assert torch.equal(before_torch, torch.get_rng_state())
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert before_sampler == learner.rng.bit_generator.state
    assert learner.cql_audit()['main_rng_preserved']


def test_nonzero_penalty_reduces_unobserved_action_advantage_in_each_critic_head():
    learner = trainer()
    values = covariates(learner)
    values['actions'].fill_(-1.)

    class LinearCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slopes = torch.nn.Parameter(torch.tensor([1., 2.]))
            self.offset = 0.

        def encode_states(self, states):
            return states

        def forward_features(self, features, actions):
            return self.slopes[:, None]*actions.reshape(1, -1)+self.offset

    learner.critic = LinearCritic()
    optimizer = torch.optim.SGD(learner.critic.parameters(), lr=.1)
    penalty = learner._cql_penalty(object(), object())
    learner.critic.offset = 20.
    torch.testing.assert_close(learner._cql_penalty(object(), object()), penalty,
                               rtol=1e-5, atol=1e-6)
    learner.critic.offset = 0.
    before = learner.critic.slopes.detach().clone()
    (.2*penalty).backward()
    assert torch.all(learner.critic.slopes.grad > 0.)
    optimizer.step()
    assert torch.all(learner.critic.slopes < before)
    assert learner._cql_penalty(object(), object()) < penalty


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test57_packets_keep_original456_optimizer_updates_and_cql_factual_scope(arm):
    learner = trainer(arm)
    for _ in range(57):
        learner.begin_actor_packet()
        for _ in range(8):
            learner.continue_step()
    audit = learner.actor_schedule_audit()
    assert audit['packets'] == 57
    for family in v.FAMILIES:
        assert audit[family+'_updates'] == 456
        assert audit['packet_'+family+'_steps'] == [8]*57
        optimizer = getattr(learner, family+'_optimizer')
        assert {int(state['step']) for state in optimizer.state.values() if 'step' in state} == {456}
    assert audit['optimizer_order_exact'] and audit['schedule'] == v.SCHEDULE
    assert audit['additional_random_draws'] == 0 and not audit['kl_penalty']
    cql = learner.cql_audit()
    assert all(v.cql_checks(cql, 456, rows_per_call=4).values())
    assert cql['candidate_rows_after_dedup'] <= 456*4*22
    assert cql['penalty']['min'] <= cql['penalty']['mean'] <= cql['penalty']['max']
    assert audit['isolated_positive_proposals'] == audit['isolated_uniform_proposals'] == 456*4*10
    assert learner.total_updates == learner.online_critic_updates == 456
    targets = learner.target_audit()
    assert targets['parent_mixed_target_calls'] == 456
    assert targets['fresh_value_replacement_calls'] == (456 if arm == 'fresh' else 0)
    assert targets['no_fresh_unchanged_returns'] == (456 if arm == 'no_fresh' else 0)
    json.dumps(cql, allow_nan=False)


def test_fixed_regularizer_changes_only_online_cql_config_and_probe_is_passive():
    learner = trainer(coefficient=0.)
    before = asdict(learner.config)
    learner.activate_executed_cql()
    changed = {key for key, value in asdict(learner.config).items() if value != before[key]}
    assert changed == {'online_cql_alpha'}
    assert learner.config.online_cql_alpha == learner.config.offline_cql_alpha == .2
    learner.begin_actor_packet()
    random_state = torch.get_rng_state().clone()
    learner.actor_comparison(torch.as_tensor(table(.1)['s']))
    assert torch.equal(random_state, torch.get_rng_state())
    assert not any(parameter.requires_grad for parameter in learner._packet_actor.parameters())
    with pytest.raises(RuntimeError, match='preceding'):
        learner.begin_actor_packet()
    for _ in range(8):
        learner.continue_step()
    assert learner.packet_actor_audit()['complete']
    with pytest.raises(RuntimeError, match='Begin'):
        learner.continue_step()
