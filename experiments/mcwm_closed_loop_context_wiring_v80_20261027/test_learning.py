"""Zero-update tests of native replay, Bellman, actor/V and executed-CQL wiring."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import learning
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def initial(arm='fresh'):
    torch.set_num_threads(1)
    native = learning.native
    contract = TransitMDPConfigV7()
    replay = native.DecoupledReplay(table(.1, 40), table(.4, 40), contract,
        np.full(12, 40, np.float32), arm=arm, outcome_seed=600042)
    config = native.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    learner = native.ExecutedCQLTrainer(replay, config=config, device='cpu', seed=42,
        action_transform=native.ExecutionAction(contract))
    learner.activate_executed_cql()
    return learner


def packet(start=0, *, missing=False):
    data = table(.72, count=5)
    result = []
    for index in range(5):
        fields = {field: data[field][index] for field in learning.native.FIELDS.values()}
        fields.update(sequence=start+index,
            ctx_s=dict(observed=not missing, z=None if missing else (start+index+.1)/100),
            ctx_ns=dict(observed=not missing, z=None if missing else -(start+index+.2)/100))
        result.append(SimpleNamespace(**fields))
    return result


def sampled_pair(arm='fresh', missing=False):
    native = initial(arm)
    candidate = learning.attach_context(deepcopy(native))
    for index in range(7):
        covariates, outcomes = packet(index*10, missing=missing), packet(index*10+5, missing=missing)
        native.replay_buffer.add_target(covariates, outcomes)
        candidate.replay_buffer.add_target(covariates, outcomes)
    left, right = np.random.default_rng(91), np.random.default_rng(91)
    old = native.replay_buffer.sample_mixed(8, .5, rng=left)
    new = candidate.replay_buffer.sample_mixed(8, .5, rng=right)
    return native, candidate, old, new, left, right


def original_actor_loss(learner, states):
    components = learner.actor.soft_action_components(states)
    actions = torch.cat((learner.action_transform(states, components['zero_action']),
                         learner.action_transform(states, components['positive_action'])))
    features = learner.critic.encode_states(states)
    values = learner._reduce_actor_q(learner.critic.forward_features(torch.cat((features, features)), actions))
    zero, positive = values.split(len(states))
    return (components['zero_probability']*(learner.alpha.detach()*components['zero_log_prob']-zero)
        + components['hold_probability']*(learner.alpha.detach()*components['positive_log_prob']-positive)).mean()


def test_attachment_retains_old_objects_Adam_and_RNG_without_optimizer_step():
    learner = initial()
    actor, value, transform, optimizer = learner.actor, learner.value, learner.action_transform, learner.critic_optimizer
    parameters = tuple(learner.critic.parameters())
    optimizer.state[parameters[0]] = {'step': torch.tensor(3.),
        'exp_avg': torch.zeros_like(parameters[0]), 'exp_avg_sq': torch.ones_like(parameters[0])}
    old_state = optimizer.state[parameters[0]]
    rng = _rng()
    learning.attach_context(learner)
    assert _equal(rng, _rng())
    assert learner.actor is actor and learner.value is value and learner.action_transform is transform
    assert learner.critic_optimizer is optimizer and optimizer.state[parameters[0]] is old_state
    assert all(a is b for a, b in zip(parameters, learner.critic.base.parameters()))
    assert optimizer.param_groups[0]['params'][-1] is learner.critic.action_weight
    assert learner.critic.action_weight not in optimizer.state
    assert not learner.target_critic.action_weight.requires_grad
    assert learner.actor_schedule_audit()['critic_updates'] == learner.actor_schedule_audit()['actor_updates'] == 0


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_native_five_draws_and_independent_covariate_outcome_row_alignment(arm):
    native, candidate, old, new, left, right = sampled_pair(arm)
    replay = candidate.replay_buffer
    assert _equal(left.bit_generator.state, right.bit_generator.state)
    assert _equal(native.replay_buffer.outcome_rng.bit_generator.state, replay.outcome_rng.bit_generator.state)
    assert native.replay_buffer.last_sampling == replay.last_sampling
    assert native.replay_buffer.last_outcome_sampling == replay.last_outcome_sampling
    for before, after in zip(old, new):
        for key in before:
            actual = after[key][:, :229] if key in ('observations', 'next_observations') else after[key]
            assert torch.equal(before[key], actual)
    for table_ in new:
        assert torch.isnan(table_['observations'][:2, 229]).all()
        assert not table_['observations'][:2, 230].any()
    cov = replay.last_covariates
    assert cov['observations'].shape == (4, 229)
    expected = torch.tensor(np.asarray(replay.context_covariates)[replay.last_sampling['recent']])
    assert torch.equal(cov['context'][2:], expected)
    if arm == 'fresh':
        ids = replay.last_outcome_sampling['local_indices']
        tail = np.asarray(list(replay.context_outcomes)[-30:])[ids]
        assert torch.equal(new[0]['observations'][2:, 229:], torch.tensor(tail[:, 0]))
        assert torch.equal(new[0]['next_observations'][2:, 229:], torch.tensor(tail[:, 1]))
        assert not torch.equal(new[0]['observations'][2:, 229:], cov['context'][2:])
    else:
        assert not new[0]['observations'][:, 230].any()
        assert not new[0]['next_observations'][:, 230].any()
        assert not replay.context_outcomes
    assert all(set(row) == {'s', 'a'} for row in replay.recent)
    assert replay.recent.maxlen == replay.context_covariates.maxlen == 285
    assert candidate.config.online_cql_alpha == .2


def test_no_fresh_never_reads_outcome_or_outcome_context():
    candidate = learning.attach_context(initial('no_fresh'))
    class Outcome:
        def __init__(self, sequence):
            self.sequence = sequence
        def __getattr__(self, name):
            raise AssertionError('No-fresh accessed '+name)
    candidate.replay_buffer.add_target(packet(), [Outcome(i+5) for i in range(5)])
    assert len(candidate.replay_buffer.context_covariates) == 5
    assert not candidate.replay_buffer.fresh_outcomes and not candidate.replay_buffer.context_outcomes


def test_new_records_require_explicit_missing_context_instead_of_silent_collector_disconnect():
    for role, field in (('covariates', 'ctx_s'), ('outcomes', 'ctx_s'), ('outcomes', 'ctx_ns')):
        learner = learning.attach_context(initial())
        covariates, outcomes = packet(missing=True), packet(5, missing=True)
        delattr((covariates if role == 'covariates' else outcomes)[0], field)
        with pytest.raises(ValueError, match='explicit ctx_'):
            learner.replay_buffer.add_target(covariates, outcomes)
        assert learner.replay_buffer.target_received == 0
    learner = learning.attach_context(initial())
    learner.replay_buffer.add_target(packet(missing=True), packet(5, missing=True))
    assert learner.replay_buffer.context_summary()['observed_covariates'] == 0


def test_zero_initialization_full_subset_feature_and_action_gradient_parity():
    native, candidate, old, new, _, _ = sampled_pair()
    states, actions = old[0]['observations'], old[0]['actions']
    packed = new[0]['observations']
    features = candidate.critic.encode_states(packed)
    assert torch.equal(features[:, :256], native.critic.encode_states(states))
    indices = torch.tensor([1, 0])
    for subset in (None, indices):
        a, b = actions.clone().requires_grad_(), actions.clone().requires_grad_()
        before = native.critic(states, a) if subset is None else native.critic.forward_subset(states, a, subset)
        after = candidate.critic(packed, b) if subset is None else candidate.critic.forward_subset(packed, b, subset)
        from_features = candidate.critic.forward_features(features, b) if subset is None else candidate.critic.forward_features_subset(features, b, subset)
        assert torch.equal(before, after) and torch.equal(before, from_features)
        assert torch.equal(torch.autograd.grad(before.min(0).values.sum(), a)[0],
                           torch.autograd.grad(after.min(0).values.sum(), b)[0])


@pytest.mark.parametrize('missing', [False, True])
def test_native_mixed_target_V_actor_grad_CQL_and_RNG_parity_without_update(missing):
    native, candidate, old, new, _, _ = sampled_pair(missing=missing)
    old_batch = {key: torch.cat((old[0][key], old[1][key])) for key in old[0]}
    new_batch = {key: torch.cat((new[0][key], new[1][key])) for key in new[0]}
    rng = _rng()
    target = native._target(old_batch, backup='h2oplus')
    after = _rng()
    _restore(rng)
    assert torch.equal(target, candidate._target(new_batch, backup='h2oplus'))
    assert _equal(after, _rng())
    assert _equal(native._iql_value_terms(old[0]), candidate._iql_value_terms(new[0]))
    actor_states = candidate._actor_states(*new)
    canonical_actor = native._actor_states(*old)
    assert torch.equal(actor_states[:, :229], canonical_actor)
    assert torch.equal(candidate._temperature_states(*new), native._temperature_states(*old))
    rng = _rng()
    loss = original_actor_loss(native, canonical_actor)
    before_gradient = torch.autograd.grad(loss, tuple(native.actor.parameters()))
    after = _rng()
    _restore(rng)
    new_loss = candidate.context_actor_terms(actor_states)[0]
    new_gradient = torch.autograd.grad(new_loss, tuple(candidate.actor.parameters()))
    assert torch.equal(loss, new_loss) and _equal(before_gradient, new_gradient)
    assert _equal(after, _rng())
    current = candidate.replay_buffer.last_covariates
    rng = _rng()
    penalty = native._cql_penalty(old[0], native.critic(old[0]['observations'], old[0]['actions']))
    after = _rng()
    _restore(rng)
    new_penalty = candidate._cql_penalty(new[0], candidate.critic(new[0]['observations'], new[0]['actions']))
    assert torch.equal(penalty, new_penalty) and _equal(after, _rng())
    assert candidate.replay_buffer.last_covariates is current
    before_audit, new_audit = native.cql_audit(), candidate.cql_audit()
    before_audit.pop('regularizer_wall_seconds'); new_audit.pop('regularizer_wall_seconds')
    assert _equal(before_audit, new_audit)
    assert candidate.actor_schedule_audit()['critic_updates'] == candidate.actor_schedule_audit()['actor_updates'] == 0


def test_nonzero_interaction_keeps_equal_native_states_with_distinct_contexts_separate():
    learner = learning.attach_context(initial())
    state = torch.tensor(table(.7, 1)['s']).expand(3, -1)
    context = torch.tensor([[1., 1.], [-1., 1.], [float('nan'), 0.]])
    packed = learning.pack_states(state, context)
    low, high = torch.full((3, 1), -1.), torch.full((3, 1), -.9375)
    with torch.no_grad():
        learner.critic.action_weight.copy_(torch.tensor([4., -2.]))
    native_delta = learner.critic.base(state, high)-learner.critic.base(state, low)
    delta = learner.critic(packed, high)-learner.critic(packed, low)
    torch.testing.assert_close(delta-native_delta, torch.tensor([[.125, -.125, 0.], [-.0625, .0625, 0.]]), atol=1e-7, rtol=0)
    assert torch.equal(learner.critic(packed, high)[:, 2], learner.critic.base(state, high)[:, 2])


def test_nonzero_context_reaches_target_actor_value_query_and_CQL_without_changing_native_state():
    _, candidate, _, batches, _, _ = sampled_pair()
    offline, source = batches
    batch = {key: torch.cat((offline[key], source[key])) for key in offline}
    with torch.no_grad():
        candidate.critic.action_weight.fill_(8.)
        candidate.target_critic.action_weight.fill_(8.)
    altered = dict(batch, next_observations=batch['next_observations'].clone())
    altered['next_observations'][:, 229] *= -1
    rng = _rng()
    first = candidate._target(batch, backup='h2oplus')
    _restore(rng)
    second = candidate._target(altered, backup='h2oplus')
    observed = batch['next_observations'][:, 230].bool()
    assert torch.equal(first[~observed], second[~observed]) and torch.any(first[observed] != second[observed])
    current = candidate.replay_buffer.last_covariates
    original_context = current['context'].clone()
    original_states = current['observations'].clone()
    rng = _rng()
    actor_first = candidate.context_actor_terms(candidate._actor_states(*batches))[0]
    value_first = candidate._iql_value_terms(offline)[1]
    cql_first = candidate._cql_penalty(offline, None)
    current['context'][:, 0] *= -1
    _restore(rng)
    actor_second = candidate.context_actor_terms(candidate._actor_states(*batches))[0]
    value_second = candidate._iql_value_terms(offline)[1]
    cql_second = candidate._cql_penalty(offline, None)
    assert not torch.equal(actor_first, actor_second)
    assert torch.any(value_first != value_second)
    assert not torch.equal(cql_first, cql_second)
    assert torch.equal(current['observations'], original_states)
    current['context'] = original_context
    assert candidate.actor_schedule_audit()['critic_updates'] == candidate.actor_schedule_audit()['actor_updates'] == 0
