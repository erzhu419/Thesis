"""Exercise selected feedback through the original real eight-step learner."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026 import control
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import CapacityPacketBudget
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.test_selection import config
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as v
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import control as original
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_capacity_packet_native_preflight_keeps_all_eight_original_updates(arm):
    torch.set_num_threads(1)
    contract = TransitMDPConfigV7()
    replay = v.FreshOutcomeReplay(table(.1), table(.4), contract,
        np.full(12, 40, np.float32), arm=arm, outcome_seed=6201+600000)
    trainer_config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=256,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    learner = v.ExecutedCQLTrainer(replay, config=trainer_config, device='cpu', seed=42,
        action_transform=v.ExecutionAction(contract))
    learner.total_updates = 50000
    learner.activate_executed_cql()
    data = table(.2, count=10)
    rows = [SimpleNamespace(**{key: data[key][i].copy() for key in v.FIELDS.values()},
        sequence=i, available_at=1000+i, line_id='7X') for i in range(10)]
    for i, row in enumerate(rows):
        row.s[2] = 0
        row.s[5:].reshape(16, 14)[0, 7:9] = (.2, 1. if i == 1 else .8)
    budget = CapacityPacketBudget(config())
    budget.ingest(rows)
    packet, _ = budget.take(1205.)
    assert [row.sequence for row in packet] == [1, 6, 7, 8, 9]
    replay.add_target(packet)
    learner.begin_actor_packet()
    for step in range(8):
        learner.continue_step(collect_diagnostics=step == 0)
    prefix = dict(trace_path='/resident/prefix_trace.jsonl', rows=123, clock=1205., initial_updates=50000)
    row = native_preflight(learner, replay, contract, '/same/seed6201.pt', packet,
        budget.receipts[0], dict(actor_real_clock=True), dict(shared_rule_exact=True),
        seed=6201, scenario='stable', arm=arm, runtime_contract=dict(F='same'),
        prefix_before_first_update=prefix)
    assert row['technical_gate_pass'], row['checks']
    assert row['packet_schedule'] == [8, 8, 8, 8]
    assert row['capacity_selection']['replacements'] == 1
    assert row['reused_full_native_day'] is True
    assert row['prefix_before_first_update'] == prefix
    assert row['target_audit']['fresh_value_replacement_calls'] == 0
    assert row['cql_audit']['penalty_calls'] == 8
    assert (arm == 'fresh' or all(set(item) == {'s', 'a'} for item in replay.recent))
    bad_prefix = dict(prefix, initial_updates=50008)
    bad = native_preflight(learner, replay, contract, '/same/seed6201.pt', packet,
        budget.receipts[0], dict(actor_real_clock=True), dict(shared_rule_exact=True),
        seed=6201, scenario='stable', arm=arm, runtime_contract=dict(F='same'),
        prefix_before_first_update=bad_prefix)
    assert not bad['technical_gate_pass']


def test_learner_and_runtime_are_direct_imports_of_the_fixed_v70_recipe():
    assert control.load_learner is original.load_learner
    assert control.FreshOutcomeReplay is original.FreshOutcomeReplay
    assert control.create_target_env is original.create_target_env
    assert control.completion_safety_action is original.completion_safety_action
    current, old = control.protocol(), original.protocol()
    assert current['runtime_contract'] == old['runtime_contract']
    for key in ('horizon_seconds', 'initializer_updates', 'feedback_delay_seconds',
                'feedback_first_deadline', 'feedback_last_deadline', 'feedback_interval_seconds',
                'feedback_records_per_packet', 'critic_update_budget', 'policy_update_budget',
                'actor_update_budget', 'alpha_update_budget', 'original_replay_window',
                'critic_fresh_outcome_window', 'outcome_seed_offset'):
        assert current[key] == old[key]
    assert current['feedback_selection'] != old.get('feedback_selection')
