"""Production networks: row scope, discounts, native draws and no-fresh Adam."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_fresh_value_v64_20261025 import learning as v
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_replay, table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def trainers(arm, *, batch_size=256):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=batch_size,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    result = []
    for cls in (v.FreshOutcomeTrainer, v.FreshValueTrainer):
        result.append(cls(make_replay(arm), config=config, device='cpu', seed=42,
            action_transform=v.ExecutionAction(TransitMDPConfigV7())))
    return result


def batch(learner):
    factual, source = learner.replay_buffer.sample_mixed(
        learner.config.batch_size, .5, rng=learner.rng)
    return learner._concatenate_batches(factual, source)


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right


def test_formal256_only_fresh64_128_changes_with_original_discount_terminal_reward():
    old, new = trainers('fresh')
    left, right = batch(old), batch(new)
    exact(left, right)
    for value in (left, right):
        value['rewards'] = torch.linspace(-2000., 2000., 256)
        value['discounts'][65] = 0.
        value['discounts'][67] = .25
        value['terminals'][64] = value['terminals'][66] = 1.
        value['discounts'][66] = .5
        value['delta_t_seconds'] = torch.linspace(1., 1200., 256)
    before = torch.get_rng_state().clone()
    original = old._target(left, backup='h2oplus')
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    modified = new._target(right, backup='h2oplus')
    assert torch.equal(after, torch.get_rng_state())
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    torch.testing.assert_close(modified[:64], original[:64], rtol=0, atol=0)
    torch.testing.assert_close(modified[128:], original[128:], rtol=0, atol=0)
    with torch.no_grad():
        values = new.value(right['next_observations'])
        expected = (new.config.reward_scale*right['rewards'].reshape(-1)
            + right['discounts'].reshape(-1)*(1.-right['terminals'].reshape(-1))*values)
    torch.testing.assert_close(modified[64:128], expected[64:128], rtol=0, atol=0)
    assert not torch.equal(modified[64:128], original[64:128])
    for index in (64, 65, 66):
        assert modified[index] == new.config.reward_scale*right['rewards'][index]
    audit = new.target_audit()
    assert audit['parent_mixed_target_calls'] == 1 and audit['parent_mixed_target_rows'] == 256
    assert audit['fresh_value_replacement_calls'] == 1 and audit['fresh_value_replacement_rows'] == 64
    assert audit['no_fresh_unchanged_returns'] == 0
    assert not modified.requires_grad


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_full_parent_actor_and_head_draws_are_real_and_preserved(arm):
    old, new = trainers(arm)
    left, right = batch(old), batch(new)
    calls, heads = [], []
    components, subset = new.actor.soft_action_components, new._sample_target_subset

    def actor(states):
        calls.append(len(states))
        return components(states)

    def target_heads():
        heads.append(True)
        return subset()

    new.actor.soft_action_components = actor
    new._sample_target_subset = target_heads
    before = torch.get_rng_state().clone()
    original = old._target(left, backup='h2oplus')
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    modified = new._target(right, backup='h2oplus')
    assert calls == [256] and len(heads) == 1
    assert torch.equal(after, torch.get_rng_state())
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    if arm == 'no_fresh':
        exact(original, modified)
        assert new.target_audit()['fresh_value_replacement_calls'] == 0
        assert new.target_audit()['no_fresh_unchanged_returns'] == 1


def test_fresh_target_uses_supplied_discount_without_inventing_dt_discount():
    _, learner = trainers('fresh')
    values = batch(learner)
    state, random_state = torch.get_rng_state().clone(), learner.rng.bit_generator.state
    first = learner._target(values, backup='h2oplus')
    torch.set_rng_state(state)
    learner.rng.bit_generator.state = random_state
    values['delta_t_seconds'] *= 100.
    second = learner._target(values, backup='h2oplus')
    exact(first, second)


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_eight_real_updates_preserve_rng_and_no_fresh_parameters_adam_exact(arm):
    old, new = trainers(arm, batch_size=8)
    for step in range(8):
        before = torch.get_rng_state().clone()
        left = old.continue_step(collect_diagnostics=step == 0)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        right = new.continue_step(collect_diagnostics=step == 0)
        assert torch.equal(after, torch.get_rng_state())
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert old.replay_buffer.last_sampling == new.replay_buffer.last_sampling
        if arm == 'no_fresh':
            exact(left, right)
    assert old.total_updates == new.total_updates == 8
    audit = new.target_audit()
    assert audit['parent_mixed_target_calls'] == 8
    assert audit['fresh_value_replacement_calls'] == (8 if arm == 'fresh' else 0)
    if arm == 'fresh':
        assert any(not torch.equal(value, new.critic.state_dict()[name])
            for name, value in old.critic.state_dict().items())
    else:
        for name in ('actor', 'value', 'critic', 'target_critic'):
            exact(getattr(old, name).state_dict(), getattr(new, name).state_dict())
        for name in ('actor', 'value', 'critic', 'alpha'):
            exact(getattr(old, name+'_optimizer').state_dict(), getattr(new, name+'_optimizer').state_dict())
        exact(old.log_alpha, new.log_alpha)
    assert type(new.replay_buffer) is v.FreshOutcomeReplay
    assert new.replay_buffer.recent.maxlen == 285
    assert not hasattr(new.replay_buffer, 'outcome_rng')


def test_first_step_td_log_reports_target_actually_used_by_critic():
    _, learner = trainers('fresh')
    target = learner._target
    captured = {}

    def record(batch, *, backup):
        result = target(batch, backup=backup)
        captured['target'] = result.detach().clone()
        return result

    learner._target = record
    metrics = learner.continue_step(collect_diagnostics=True)
    diagnostic = metrics['td_diagnostic']
    assert diagnostic['real_batches'][0]['target_q']['mean'] == float(captured['target'].mean())
    assert diagnostic['real_batches'][0]['target_q']['n'] == 256
    assert diagnostic['real_scale'] == 1. and diagnostic['model_scale'] == 0.


@pytest.mark.parametrize('changed', [dict(total_updates=49999), dict(online_critic_updates=1),
    dict(extra=dict(source_training_seed=6201))])
def test_seed_initializer_validation_remains_strict(tmp_path, monkeypatch, changed):
    (tmp_path/'history_metadata.json').write_text(json.dumps(dict(
        contract=asdict(TransitMDPConfigV7()), station_max_by_line=[40]*12)))
    np.savez(tmp_path/'history.npz', **table(.1))
    monkeypatch.setattr(v.resident, 'INPUTS', tmp_path)
    monkeypatch.setattr(v, 'load_table', lambda *args: table(.4))
    payload = dict(total_updates=50000, online_critic_updates=0, extra=dict(source_training_seed=6203))
    payload.update(changed)
    requested = []

    def load(path, **kwargs):
        requested.append(path.name)
        return payload

    monkeypatch.setattr(v.torch, 'load', load)
    with pytest.raises(ValueError, match='Wrong read-only initializer'):
        v.load_learner(6203, arm='fresh')
    assert requested == ['h2oplus_seed6203.pt']
