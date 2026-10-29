"""Detect covariate/outcome coupling, hybrid rows, and no-fresh optimizer drift."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026 import learning as v
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as old
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7

SEED = 42


def rows(data, start=0, stop=None, *, sequence_offset=0):
    return [SimpleNamespace(sequence=i+sequence_offset,
        **{key: data[key][i] for key in v.FIELDS.values()})
        for i in range(start, len(data['s']) if stop is None else stop)]


def replay(arm, *, original=False):
    cls = old.FreshOutcomeReplay if original else v.DecoupledReplay
    return cls(table(.1), table(.4), TransitMDPConfigV7(),
        np.full(12, 40, np.float32), arm=arm, outcome_seed=SEED+600000)


def trainer(data, *, batch_size=8):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=batch_size,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    learner = v.ExecutedCQLTrainer(data, config=config, device='cpu', seed=SEED,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    learner.activate_executed_cql()
    return learner


def test_no_fresh_all456_updates_exact_v70_parameters_adam_targets_and_rng():
    original, decoupled = trainer(replay('no_fresh', original=True)), trainer(replay('no_fresh'))
    data = table(.7, 285)

    class UnreadOutcomes:
        def __init__(self, seq):
            self.sequence = seq

        def __getattr__(self, key):
            raise AssertionError(f'no_fresh learner read outcome row field {key}')

    for packet in range(57):
        cov = rows(data, packet*5, (packet+1)*5)
        original.replay_buffer.add_target(cov)
        decoupled.replay_buffer.add_target(cov,
            [UnreadOutcomes(row.sequence+10000) for row in cov])
        for learner in (original, decoupled):
            learner.begin_actor_packet()
        for step in range(8):
            before = torch.get_rng_state().clone()
            expected = original.continue_step(collect_diagnostics=step == 0)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = decoupled.continue_step(collect_diagnostics=step == 0)
            exact(actual, expected)
            assert torch.equal(after, torch.get_rng_state())
            assert original.rng.bit_generator.state == decoupled.rng.bit_generator.state
            assert original.replay_buffer.outcome_rng.bit_generator.state == decoupled.replay_buffer.outcome_rng.bit_generator.state
            assert original.replay_buffer.last_sampling == decoupled.replay_buffer.last_sampling
            exact(original.replay_buffer.last_covariates, decoupled.replay_buffer.last_covariates)
            exact(original.target_audit(), decoupled.target_audit())
        for name in ('actor', 'critic', 'target_critic', 'value'):
            exact(getattr(original, name).state_dict(), getattr(decoupled, name).state_dict())
        for name in v.FAMILIES:
            exact(getattr(original, name+'_optimizer').state_dict(),
                getattr(decoupled, name+'_optimizer').state_dict())
        exact(original.log_alpha, decoupled.log_alpha)
    audit = decoupled.replay_buffer.summary()
    assert audit['distinct_target_records_used'] == audit['covariate_unique_records'] == 285
    assert audit['outcome_labels_received'] == audit['fresh_outcomes_stored'] == 0
    assert all(v.recency_checks(audit, 'no_fresh', 456, SEED).values())
    assert all(v.target_checks(decoupled.target_audit(), 'no_fresh', 456, batch_size=8).values())
    assert all(v.cql_checks(decoupled.cql_audit(), 456, rows_per_call=4).values())


@pytest.mark.parametrize('packets', [1, 6, 7, 57])
def test_complete_outcome_rows_are_independent_of_covariates_and_sample_recent30(packets):
    baseline, split = replay('fresh', original=True), replay('fresh')
    cov_data, outcome_data = table(.7, 285), table(.9, 285)
    for packet in range(packets):
        cov = rows(cov_data, packet*5, (packet+1)*5)
        outcome = rows(outcome_data, packet*5, (packet+1)*5, sequence_offset=10000)
        baseline.add_target(cov)
        split.add_target(cov, outcome)
    left_rng, right_rng = np.random.default_rng(904), np.random.default_rng(904)
    expected, expected_source = baseline.sample_mixed(256, .5, rng=left_rng)
    actual, actual_source = split.sample_mixed(256, .5, rng=right_rng)
    exact(actual_source, expected_source)
    exact(split.last_covariates, baseline.last_covariates)
    assert split.last_sampling == baseline.last_sampling
    assert left_rng.bit_generator.state == right_rng.bit_generator.state
    local = np.random.default_rng(SEED+600000).integers(min(30, 5*packets), size=64)
    ids = 5*packets-min(30, 5*packets)+local
    for key, field in v.FIELDS.items():
        exact(actual[key][:64], expected[key][:64])
        wanted = torch.as_tensor(outcome_data[field][ids], dtype=torch.float32)
        exact(actual[key][64:], wanted.reshape_as(actual[key][64:]))
    assert all(set(row) == {'s', 'a'} for row in split.recent)
    assert all(set(row) == set(v.FIELDS.values()) for row in split.fresh_outcomes)
    # Admitted rows are copied; later stream-object mutation cannot alter either pool.
    cov[-1].s[:] = -999
    outcome[-1].s[:] = -888
    assert not np.any(split.recent[-1]['s'] == -999)
    assert not np.any(split.fresh_outcomes[-1]['s'] == -888)


def test_identical_selections_reproduce_v70_fresh_first_packet_exactly():
    original, split = trainer(replay('fresh', original=True)), trainer(replay('fresh'))
    packet = rows(table(.7, 5))
    original.replay_buffer.add_target(packet)
    split.replay_buffer.add_target(packet, packet)
    for learner in (original, split):
        learner.begin_actor_packet()
    for step in range(8):
        before = torch.get_rng_state().clone()
        expected = original.continue_step(collect_diagnostics=step == 0)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        exact(expected, split.continue_step(collect_diagnostics=step == 0))
        assert torch.equal(after, torch.get_rng_state())
    for name in ('actor', 'critic', 'target_critic', 'value'):
        exact(getattr(original, name).state_dict(), getattr(split, name).state_dict())
    for name in v.FAMILIES:
        exact(getattr(original, name+'_optimizer').state_dict(),
            getattr(split, name+'_optimizer').state_dict())


def test_native_first_packet_admits_distinct_outcomes_with_common_covariate_receipt():
    from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import DecoupledPacketBudget
    from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.preflight import native_preflight
    from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.test_selection import config
    incoming = rows(table(.7, 10))
    for row in incoming:
        row.available_at, row.line_id = 1000+row.sequence, '7X'
    incoming[1].s[5:].reshape(16, 14)[0, 7:9] = (.2, 1.)
    budget = DecoupledPacketBudget(config())
    budget.ingest(incoming)
    cov, outcomes = budget.take(1200.)
    assert [row.sequence for row in cov] != [row.sequence for row in outcomes]
    learner = trainer(replay('fresh'), batch_size=256)
    learner.total_updates = 50000
    learner.replay_buffer.add_target(cov, outcomes)
    learner.begin_actor_packet()
    for step in range(8):
        learner.continue_step(collect_diagnostics=step == 0)
    result = native_preflight(learner, learner.replay_buffer, TransitMDPConfigV7(),
        '/resident/initializer.pt', cov, budget.receipts[0],
        dict(actor_real_clock=True), dict(shared_rule_exact=True),
        seed=SEED, scenario='stable', arm='fresh', runtime_contract={},
        prefix_before_first_update=dict(rows=45, clock=1200., initial_updates=50000),
        outcome_packet=outcomes)
    assert result['technical_gate_pass'], result['checks']
    assert result['first_feedback']['sequence_ids'] == [5, 6, 7, 8, 9]
    assert result['first_outcome_feedback']['sequence_ids'] == [1, 6, 7, 8, 9]
    assert result['capacity_selection']['target_record_union'] == 6
