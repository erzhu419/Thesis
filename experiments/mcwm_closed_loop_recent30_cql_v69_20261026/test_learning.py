"""Composition checks: recent30 outcomes, unchanged v68 controls and RNGs."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_recent30_cql_v69_20261026 import learning as v
from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import learning as old
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control import RecentOutcomeReplay
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


SEED = 42


def rows(data, start=0, stop=None):
    return [SimpleNamespace(**{key: data[key][index] for key in v.FIELDS.values()})
            for index in range(start, len(data['s']) if stop is None else stop)]


def replay(arm, *, original=False, count=0):
    cls = old.FreshOutcomeReplay if original else v.FreshOutcomeReplay
    kwargs = dict(window=285) if original else dict(outcome_seed=SEED+600000)
    result = cls(table(.1), table(.4), TransitMDPConfigV7(),
        np.full(12, 40, np.float32), arm=arm, **kwargs)
    if count:
        result.add_target(rows(table(.7, count)))
    return result


def trainer(arm, *, original=False):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    learner = v.ExecutedCQLTrainer(replay(arm, original=original),
        config=config, device='cpu', seed=SEED,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    learner.activate_executed_cql()
    return learner


@pytest.mark.parametrize('count', [5, 30, 35, 285])
def test_only_complete_fresh_quarter_changes_at_window_boundaries(count):
    baseline, recent = replay('fresh', original=True, count=count), replay('fresh', count=count)
    baseline_rng, recent_rng = np.random.default_rng(904), np.random.default_rng(904)
    expected, expected_source = baseline.sample_mixed(256, .5, rng=baseline_rng)
    actual, source = recent.sample_mixed(256, .5, rng=recent_rng)
    exact(source, expected_source)
    exact(recent.last_covariates, baseline.last_covariates)
    assert recent.last_sampling == baseline.last_sampling
    assert recent_rng.bit_generator.state == baseline_rng.bit_generator.state
    for key in actual:
        exact(actual[key][:64], expected[key][:64])
    private = np.random.default_rng(SEED+600000)
    local = private.integers(min(30, count), size=64)
    ids = count-min(30, count)+local
    data = table(.7, count)
    for batch_key, table_key in v.FIELDS.items():
        values = torch.as_tensor(data[table_key][ids], dtype=torch.float32)
        exact(actual[batch_key][64:], values.reshape_as(actual[batch_key][64:]))
    assert recent.outcome_rng.bit_generator.state == private.bit_generator.state
    assert len(recent.recent) == count and recent.recent.maxlen == 285
    if count == 285:
        assert min(recent.last_sampling['recent']) < 255
        assert np.all(ids >= 255)


def test_no_fresh_reproduces_v68_all57_packets_parameters_adam_and_main_rng():
    baseline, recent = trainer('no_fresh', original=True), trainer('no_fresh')
    data = table(.7, 285)
    for packet in range(57):
        incoming = rows(data, 5*packet, 5*(packet+1))
        for learner in (baseline, recent):
            learner.replay_buffer.add_target(incoming)
            learner.begin_actor_packet()
        for step in range(8):
            before = torch.get_rng_state().clone()
            expected = baseline.continue_step(collect_diagnostics=step == 0)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = recent.continue_step(collect_diagnostics=step == 0)
            exact(expected, actual)
            assert torch.equal(after, torch.get_rng_state())
            assert baseline.rng.bit_generator.state == recent.rng.bit_generator.state
            assert baseline.replay_buffer.last_sampling == recent.replay_buffer.last_sampling
            exact(baseline.replay_buffer.last_covariates, recent.replay_buffer.last_covariates)
        for name in ('actor', 'critic', 'target_critic', 'value'):
            exact(getattr(baseline, name).state_dict(), getattr(recent, name).state_dict())
        for name in v.FAMILIES:
            exact(getattr(baseline, name+'_optimizer').state_dict(),
                  getattr(recent, name+'_optimizer').state_dict())
        exact(baseline.log_alpha, recent.log_alpha)
    audit = recent.actor_schedule_audit()
    assert audit['packets'] == 57 and audit['optimizer_order_exact']
    for family in v.FAMILIES:
        assert audit[family+'_updates'] == 456
        assert audit['packet_'+family+'_steps'] == [8]*57
    assert all(v.recency_checks(recent.replay_buffer.summary(), 'no_fresh', 456, SEED).values())
    assert all(v.cql_checks(recent.cql_audit(), 456, rows_per_call=4).values())
    assert recent.target_audit()['no_fresh_unchanged_returns'] == 456
    assert recent.target_audit()['fresh_value_replacement_calls'] == 0
    assert all(set(row) == {'s', 'a'} for row in recent.replay_buffer.recent)


def test_fresh_combination_keeps285_slots456_updates_and_replaces_only_its_target_quarter():
    learner = trainer('fresh')
    data = table(.7, 285)
    for packet in range(57):
        learner.replay_buffer.add_target(rows(data, 5*packet, 5*(packet+1)))
        learner.begin_actor_packet()
        for _ in range(8):
            learner.continue_step()
        assert all(v.recency_checks(learner.replay_buffer.summary(),
            'fresh', (packet+1)*8, SEED).values())
    assert learner.replay_buffer.target_received == len(learner.replay_buffer.recent) == 285
    audit = learner.actor_schedule_audit()
    assert audit['packets'] == 57 and audit['optimizer_order_exact']
    for family in v.FAMILIES:
        assert audit[family+'_updates'] == 456
        assert audit['packet_'+family+'_steps'] == [8]*57
    assert all(v.cql_checks(learner.cql_audit(), 456, rows_per_call=4).values())
    target = learner.target_audit()
    assert target['parent_mixed_target_calls'] == target['fresh_value_replacement_calls'] == 456
    assert target['fresh_value_replacement_rows'] == 456*2
    assert target['no_fresh_unchanged_returns'] == 0


def test_outcome_and_cql_private_streams_do_not_touch_each_other_or_main_rng():
    learner = trainer('fresh')
    replay_buffer = learner.replay_buffer
    replay_buffer.add_target(rows(table(.7, 35)))
    replay_buffer.sample_mixed(8, .5, rng=learner.rng)
    states = replay_buffer.last_covariates['observations']
    outcome_before = deepcopy(replay_buffer.outcome_rng.bit_generator.state)
    native_before = deepcopy(learner.rng.bit_generator.state)
    global_numpy = deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    first = learner._executed_candidates(states)
    assert replay_buffer.outcome_rng.bit_generator.state == outcome_before
    assert learner.rng.bit_generator.state == native_before
    assert torch.equal(torch_before, torch.get_rng_state())
    replay_buffer.sample_mixed(8, .5, rng=learner.rng)
    assert replay_buffer.outcome_rng.bit_generator.state != outcome_before
    exact(first, learner._executed_candidates(states))
    numpy_after = np.random.get_state()
    assert global_numpy[0] == numpy_after[0] and global_numpy[2:] == numpy_after[2:]
    np.testing.assert_array_equal(global_numpy[1], numpy_after[1])
    assert learner._cql_seed == SEED+68000000
    assert replay_buffer.outcome_seed == SEED+600000


def test_loader_reuses_original_restore_order_before_cql_activation(tmp_path, monkeypatch):
    contract = TransitMDPConfigV7()
    (tmp_path/'history_metadata.json').write_text(json.dumps(dict(
        contract=asdict(contract), station_max_by_line=[40]*12)))
    np.savez(tmp_path/'history.npz', **table(.1))
    monkeypatch.setattr(v.resident, 'INPUTS', tmp_path)
    monkeypatch.setattr(v, 'load_table', lambda *args: table(.4))
    events = []
    config = v.TransitTrainerConfig(batch_size=256, cql_n_actions=10)
    payload = dict(total_updates=50000, online_critic_updates=0,
        extra=dict(source_training_seed=SEED), source_config=asdict(config))
    monkeypatch.setattr(v.torch, 'load', lambda *args, **kwargs: payload)

    class RestoredTrainer:
        def __init__(self, replay_buffer, **kwargs):
            self.replay_buffer, self.config = replay_buffer, kwargs['config']
            assert kwargs['seed'] == SEED and isinstance(kwargs['action_transform'], v.ExecutionAction)
            events.append('construct')

        def load_offline_initializer(self, path):
            assert path == tmp_path/f'h2oplus_seed{SEED}.pt'
            events.append('restore_initializer_and_adam')

        def activate_executed_cql(self):
            events.append('activate_cql')

    monkeypatch.setattr(v, 'ExecutedCQLTrainer', RestoredTrainer)
    monkeypatch.setattr(v.resident.formal, 'set_cpu_optimizer_runtime', lambda learner: events.append('cpu_runtime'))
    learner, data, restored_contract, stations, initializer = v.load_learner(SEED, arm='fresh')
    assert events == ['construct', 'restore_initializer_and_adam', 'cpu_runtime', 'activate_cql']
    assert type(data) is RecentOutcomeReplay
    assert data.outcome_seed == SEED+600000 and data.recent.maxlen == 285
    assert learner.replay_buffer is data
    assert asdict(restored_contract) == asdict(contract)
    np.testing.assert_array_equal(stations, [40]*12)
