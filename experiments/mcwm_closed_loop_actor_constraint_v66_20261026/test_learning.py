"""Full-policy KL, fixed packet reference, unchanged draws and original Adam."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_actor_constraint_v66_20261026 import learning as v
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_replay, table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def trainer(arm, *, coefficient=1., parent=False):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    cls = v.FreshValueTrainer if parent else v.ActorConstrainedTrainer
    extra = {} if parent else dict(actor_kl_coefficient=coefficient)
    return cls(make_replay(arm), config=config, device='cpu', seed=42,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()), **extra)


def test_full_hurdle_kl_matches_bernoulli_and_weighted_conditional_normal_without_rng():
    current = tuple(torch.tensor(values, dtype=torch.float64, requires_grad=True)
        for values in ([.2, -.6, 1.], [.3, -.5, .1], [-.4, -.1, .3]))
    reference = tuple(torch.tensor(values, dtype=torch.float64, requires_grad=True)
        for values in ([-.3, .4, .8], [-.2, .3, -.4], [.2, -.5, .1]))
    before = torch.get_rng_state().clone()
    actual = v.hurdle_kl(current, reference)
    expected_gate = torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(logits=current[0]),
        torch.distributions.Bernoulli(logits=reference[0]))
    expected_positive = torch.sigmoid(current[0])*torch.distributions.kl_divergence(
        torch.distributions.Normal(current[1], current[2].exp()),
        torch.distributions.Normal(reference[1], reference[2].exp()))
    torch.testing.assert_close(actual['gate_kl'], expected_gate)
    torch.testing.assert_close(actual['weighted_positive_kl'], expected_positive)
    torch.testing.assert_close(actual['total_kl'], expected_gate+expected_positive)
    assert torch.equal(before, torch.get_rng_state())
    actual['total_kl'].sum().backward()
    assert all(value.grad is not None and torch.count_nonzero(value.grad) == 3 for value in current)
    assert all(value.grad is None for value in reference)


def test_identical_full_distribution_has_exact_zero_kl():
    values = (torch.tensor([-.62, .7]), torch.tensor([-1.4, 2.]), torch.tensor([-.5, -2.]))
    result = v.hurdle_kl(values, tuple(value.clone() for value in values))
    assert all(torch.equal(value, torch.zeros_like(value)) for value in result.values())


def test_production_hook_first_update_equals_v64_when_packet_reference_is_identical():
    old, new = trainer('fresh', parent=True), trainer('fresh')
    new.begin_actor_packet()
    before = torch.get_rng_state().clone()
    left = old.continue_step(collect_diagnostics=True)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    right = new.continue_step(collect_diagnostics=True)
    exact(left, {key: right[key] for key in left})
    assert torch.equal(after, torch.get_rng_state())
    for name in ('actor', 'critic', 'target_critic', 'value'):
        exact(getattr(old, name).state_dict(), getattr(new, name).state_dict())
    for name in ('actor', 'critic', 'value', 'alpha'):
        exact(getattr(old, name+'_optimizer').state_dict(), getattr(new, name+'_optimizer').state_dict())
    exact(old.log_alpha, new.log_alpha)


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_zero_coefficient_reproduces_v64_eight_updates_all_parameters_adam_and_rng(arm):
    old, new = trainer(arm, parent=True), trainer(arm, coefficient=0.)
    before_snapshot = torch.get_rng_state().clone()
    new.begin_actor_packet()
    assert torch.equal(before_snapshot, torch.get_rng_state())
    for step in range(8):
        before = torch.get_rng_state().clone()
        left = old.continue_step(collect_diagnostics=step == 0)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        right = new.continue_step(collect_diagnostics=step == 0)
        exact(left, right)
        assert torch.equal(after, torch.get_rng_state())
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert old.replay_buffer.last_sampling == new.replay_buffer.last_sampling
    for name in ('actor', 'critic', 'target_critic', 'value'):
        exact(getattr(old, name).state_dict(), getattr(new, name).state_dict())
    for name in ('actor', 'critic', 'value', 'alpha'):
        exact(getattr(old, name+'_optimizer').state_dict(), getattr(new, name+'_optimizer').state_dict())
    exact(old.log_alpha, new.log_alpha)
    exact(old.target_audit(), new.target_audit())
    assert new.actor_constraint_audit()['updates'] == 8
    assert new.actor_constraint_audit()['alpha_updates'] == 8


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_positive_constraint_keeps_draw_schedule_and_reference_fixed_for_eight_updates(arm):
    old, new = trainer(arm, parent=True), trainer(arm)
    new.begin_actor_packet()
    snapshot = {key: value.clone() for key, value in new._packet_actor.state_dict().items()}
    assert not any(parameter.requires_grad for parameter in new._packet_actor.parameters())
    for step in range(8):
        before = torch.get_rng_state().clone()
        old.continue_step(collect_diagnostics=step == 0)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        metrics = new.continue_step(collect_diagnostics=step == 0)
        assert torch.equal(after, torch.get_rng_state())
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert old.replay_buffer.last_sampling == new.replay_buffer.last_sampling
        exact(snapshot, new._packet_actor.state_dict())
        assert 'train/actor_packet_total_kl' in metrics
        assert metrics['train/actor_updated'] == 1.
    assert any(not torch.equal(value, new.actor.state_dict()[key]) for key, value in snapshot.items())
    assert new.packet_actor_audit()['actor_updates'] == 8
    assert len(new.packet_actor_audit()['step_kl']) == 8
    first_audit = new.actor_constraint_audit()
    assert first_audit['packet_step_counts'] == [8]
    assert first_audit['updates'] == first_audit['alpha_updates'] == 8
    assert new.total_updates == new.online_critic_updates == 8
    assert all(parameter.requires_grad for parameter in new.critic.parameters())
    new.begin_actor_packet()
    exact(new.actor.state_dict(), new._packet_actor.state_dict())
    assert new.actor_constraint_audit()['reference_refreshes'] == 2
    assert new.actor_constraint_audit()['packet_step_counts'] == [8, 0]


def test_actor_loss_adds_only_fixed_beta_times_full_policy_kl_and_preserves_temperature():
    constrained, original = trainer('fresh'), trainer('fresh', coefficient=0.)
    constrained.begin_actor_packet()
    with torch.no_grad():
        constrained.actor.network.network[-1].bias.add_(torch.tensor([.4, .2, .3]))
    original.actor.load_state_dict(constrained.actor.state_dict())
    original.begin_actor_packet()
    states = torch.as_tensor(table(.1)['s'])
    expected = constrained.actor_comparison(states)['total_kl'].mean()
    before = torch.get_rng_state().clone()
    base_loss, base_alpha, _ = original._update_sac_actor(states)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    new_loss, new_alpha, _ = constrained._update_sac_actor(states)
    torch.testing.assert_close(new_loss, base_loss+expected)
    torch.testing.assert_close(base_alpha, new_alpha, rtol=0, atol=0)
    assert expected > 0.
    assert torch.equal(after, torch.get_rng_state())
    exact(original.alpha_optimizer.state_dict(), constrained.alpha_optimizer.state_dict())


def test_small_gate_kl_can_have_large_deterministic_F_action_jump_and_audit_reports_it():
    learner = trainer('fresh')
    output = learner.actor.network.network[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(torch.tensor([torch.logit(torch.tensor(.499)), -1.4, -.5]))
    learner.begin_actor_packet()
    with torch.no_grad():
        output.bias[0] = torch.logit(torch.tensor(.501))
    states = torch.as_tensor(table(.1)['s'])
    before = torch.get_rng_state().clone()
    audit = learner.actor_comparison(states)
    assert torch.equal(before, torch.get_rng_state())
    assert audit['gate_flip'].all()
    assert audit['total_kl'].max() < 1e-4
    assert torch.equal(audit['reference_hold_seconds'], torch.zeros(len(states)))
    assert torch.all(audit['current_hold_seconds'] > 1.)
    exact(audit['holding_delta_seconds'], audit['current_hold_seconds'])


def test_reference_is_required_and_cannot_be_refreshed_inside_a_packet():
    learner = trainer('fresh')
    with pytest.raises(RuntimeError, match='Begin the paid packet'):
        learner.continue_step()
    learner.begin_actor_packet()
    learner.continue_step()
    with pytest.raises(RuntimeError, match='eight updates'):
        learner.begin_actor_packet()


@pytest.mark.parametrize('changed', [dict(total_updates=49999), dict(online_critic_updates=1),
    dict(extra=dict(source_training_seed=6202))])
def test_seed_initializer_validation_is_same_as_v64(tmp_path, monkeypatch, changed):
    (tmp_path/'history_metadata.json').write_text(json.dumps(dict(
        contract=asdict(TransitMDPConfigV7()), station_max_by_line=[40]*12)))
    np.savez(tmp_path/'history.npz', **table(.1))
    monkeypatch.setattr(v.resident, 'INPUTS', tmp_path)
    monkeypatch.setattr(v, 'load_table', lambda *args: table(.4))
    payload = dict(total_updates=50000, online_critic_updates=0, extra=dict(source_training_seed=6201))
    payload.update(changed)
    monkeypatch.setattr(v.torch, 'load', lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match='Wrong read-only initializer'):
        v.load_learner(6201, arm='fresh')
