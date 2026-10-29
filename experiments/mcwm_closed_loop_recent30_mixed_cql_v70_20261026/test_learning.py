"""Actual mixed target, complete no-fresh replay, Adam and RNG preservation."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as v
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026 import learning as old
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.test_learning import rows, replay, SEED
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def trainer(arm, *, original=False, batch_size=8):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=batch_size,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    cls = old.ExecutedCQLTrainer if original else v.ExecutedCQLTrainer
    learner = cls(replay(arm), config=config, device='cpu', seed=SEED,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    learner.activate_executed_cql()
    return learner


def batch(learner):
    factual, source = learner.replay_buffer.sample_mixed(
        learner.config.batch_size, .5, rng=learner.rng)
    return learner._concatenate_batches(factual, source)


def test_fresh_only_quarter_returns_original_mixed_target_with_native_actor_and_head_draws():
    original, mixed = trainer('fresh', original=True, batch_size=256), trainer('fresh', batch_size=256)
    for learner in (original, mixed):
        learner.replay_buffer.add_target(rows(table(.7, 35)))
    left, right = batch(original), batch(mixed)
    exact(left, right)
    for data in (left, right):
        data['rewards'] = torch.linspace(-2000., 2000., 256)
        data['discounts'][65] = 0.
        data['discounts'][67] = .25
        data['terminals'][64] = data['terminals'][66] = 1.
    positive_calls, head_calls = [], []
    actor, subset = mixed.actor.soft_action_components, mixed._sample_target_subset

    def record_actor(states):
        positive_calls.append(len(states))
        return actor(states)

    def record_subset():
        head_calls.append(True)
        return subset()

    mixed.actor.soft_action_components = record_actor
    mixed._sample_target_subset = record_subset
    before = torch.get_rng_state().clone()
    value_only = original._target(left, backup='h2oplus')
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    native_before = deepcopy(mixed.rng.bit_generator.state)
    original_mixed = v.FreshOutcomeTrainer._target(mixed, right, backup='h2oplus')
    torch.set_rng_state(before)
    mixed.rng.bit_generator.state = native_before
    actual = mixed._target(right, backup='h2oplus')
    exact(actual, original_mixed)
    exact(actual[:64], value_only[:64])
    exact(actual[128:], value_only[128:])
    assert not torch.equal(actual[64:128], value_only[64:128])
    assert positive_calls == [256, 256] and len(head_calls) == 2
    assert torch.equal(after, torch.get_rng_state())
    assert original.rng.bit_generator.state == mixed.rng.bit_generator.state
    for index in (64, 65, 66):
        assert actual[index] == mixed.config.reward_scale*right['rewards'][index]
    assert all(v.target_checks(mixed.target_audit(), 'fresh', 1).values())
    assert not actual.requires_grad


def test_mixed_formula_keeps_supplied_discounts_and_terminals_on_every_source_quarter():
    learner = trainer('fresh', batch_size=256)
    learner.replay_buffer.add_target(rows(table(.7, 35)))
    data = batch(learner)
    data['discounts'] = torch.linspace(0., 1., 256)
    data['terminals'][::17] = 1.
    data['rewards'] = torch.linspace(-3000., 500., 256)
    expected_sac = torch.linspace(-7., 23., 256)
    learner._sac_bootstrap = lambda incoming: expected_sac
    with torch.no_grad():
        values = learner.value(data['next_observations'])
        expected = (learner.config.reward_scale*data['rewards']
            + data['discounts']*(1.-data['terminals'])*(.1*values+.9*expected_sac))
    actual = learner._target(data, backup='h2oplus')
    exact(actual, expected)
    data['delta_t_seconds'] *= 100.
    exact(actual, learner._target(data, backup='h2oplus'))


def test_no_fresh_all57_packets_exact_v69_parameters_adam_targets_and_all_random_streams():
    original, mixed = trainer('no_fresh', original=True), trainer('no_fresh')
    data = table(.7, 285)
    for packet in range(57):
        for learner in (original, mixed):
            learner.replay_buffer.add_target(rows(data, packet*5, (packet+1)*5))
            learner.begin_actor_packet()
        for step in range(8):
            before = torch.get_rng_state().clone()
            expected = original.continue_step(collect_diagnostics=step == 0)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = mixed.continue_step(collect_diagnostics=step == 0)
            exact(actual, expected)
            assert torch.equal(after, torch.get_rng_state())
            assert original.rng.bit_generator.state == mixed.rng.bit_generator.state
            assert original.replay_buffer.outcome_rng.bit_generator.state == mixed.replay_buffer.outcome_rng.bit_generator.state
            assert original.replay_buffer.last_sampling == mixed.replay_buffer.last_sampling
            exact(original.target_audit(), mixed.target_audit())
        for name in ('actor', 'critic', 'target_critic', 'value'):
            exact(getattr(original, name).state_dict(), getattr(mixed, name).state_dict())
        for name in v.FAMILIES:
            exact(getattr(original, name+'_optimizer').state_dict(),
                getattr(mixed, name+'_optimizer').state_dict())
        exact(original.log_alpha, mixed.log_alpha)
    assert all(v.target_checks(mixed.target_audit(), 'no_fresh', 456, batch_size=8).values())
    assert all(v.recency_checks(mixed.replay_buffer.summary(), 'no_fresh', 456, SEED).values())
    assert all(v.cql_checks(mixed.cql_audit(), 456, rows_per_call=4).values())
    assert all(set(row) == {'s', 'a'} for row in mixed.replay_buffer.recent)


def test_fresh285_slots456_updates_keep_recent30_cql_and_log_actual_mixed_target():
    learner = trainer('fresh')
    captured = {}
    original_target = learner._target

    def record(data, *, backup):
        value = original_target(data, backup=backup)
        captured['target'] = value.clone()
        return value

    learner._target = record
    data = table(.7, 285)
    for packet in range(57):
        learner.replay_buffer.add_target(rows(data, packet*5, (packet+1)*5))
        learner.begin_actor_packet()
        for step in range(8):
            metrics = learner.continue_step(collect_diagnostics=step == 0)
            if step == 0:
                logged = metrics['td_diagnostic']['real_batches'][0]['target_q']
                assert logged['mean'] == float(captured['target'].mean())
                assert logged['n'] == 8
        assert all(v.recency_checks(learner.replay_buffer.summary(), 'fresh', (packet+1)*8, SEED).values())
    audit = learner.actor_schedule_audit()
    assert learner.replay_buffer.target_received == len(learner.replay_buffer.recent) == 285
    assert audit['packets'] == 57 and audit['optimizer_order_exact']
    for family in v.FAMILIES:
        assert audit[family+'_updates'] == 456
        assert audit['packet_'+family+'_steps'] == [8]*57
    assert all(v.target_checks(learner.target_audit(), 'fresh', 456, batch_size=8).values())
    assert all(v.cql_checks(learner.cql_audit(), 456, rows_per_call=4).values())
    assert learner._cql_seed == SEED+68000000
    assert learner.replay_buffer.outcome_seed == SEED+600000


def test_loader_restores_initializer_and_adam_before_unchanged_cql(tmp_path, monkeypatch):
    contract = TransitMDPConfigV7()
    (tmp_path/'history_metadata.json').write_text(json.dumps(dict(
        contract=asdict(contract), station_max_by_line=[40]*12)))
    np.savez(tmp_path/'history.npz', **table(.1))
    monkeypatch.setattr(v.resident, 'INPUTS', tmp_path)
    monkeypatch.setattr(v, 'load_table', lambda *args: table(.4))
    config = v.TransitTrainerConfig(batch_size=256, cql_n_actions=10,
        online_backup='h2oplus', mixed_backup_lambda=.1)
    payload = dict(total_updates=50000, online_critic_updates=0,
        extra=dict(source_training_seed=SEED), source_config=asdict(config))
    monkeypatch.setattr(v.torch, 'load', lambda *args, **kwargs: payload)
    events = []

    class RestoredTrainer:
        def __init__(self, replay_buffer, **kwargs):
            self.replay_buffer, self.config = replay_buffer, kwargs['config']
            assert kwargs['seed'] == SEED
            events.append('construct')

        def load_offline_initializer(self, path):
            assert path == tmp_path/f'h2oplus_seed{SEED}.pt'
            events.append('restore_initializer_and_adam')

        def activate_executed_cql(self):
            events.append('activate_cql')

    monkeypatch.setattr(v, 'ExecutedCQLTrainer', RestoredTrainer)
    monkeypatch.setattr(v.resident.formal, 'set_cpu_optimizer_runtime',
        lambda learner: events.append('cpu_runtime'))
    learner, data, _, _, _ = v.load_learner(SEED, arm='fresh')
    assert events == ['construct', 'restore_initializer_and_adam', 'cpu_runtime', 'activate_cql']
    assert data.recent.maxlen == 285 and data.outcome_seed == SEED+600000
    assert learner.config.mixed_backup_lambda == .1
