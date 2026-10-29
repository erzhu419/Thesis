"""Real eight-step packet to preflight receipt; diagnostic state/RNG isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_actor_constraint_v66_20261026 import learning as v
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.launch import preflight
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import (
    PacketPolicyProbe, native_preflight)
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.test_learning import trainer, table


def first_packet(arm):
    learner = trainer(arm)
    learner.config = replace(learner.config, batch_size=256)
    learner.total_updates = 50000
    replay = learner.replay_buffer
    replay.recent.clear()
    replay.target_received = 0
    data = table(.2, count=5)
    packet = [SimpleNamespace(**{key:data[key][i] for key in v.FIELDS.values()},
        line_id=('7X', '102X', '705X')[i%3]) for i in range(5)]
    replay.add_target(packet)
    learner.begin_actor_packet()
    for step in range(8):
        learner.continue_step(collect_diagnostics=step == 0)
    return learner, replay, packet


def test_actual_first_eight_updates_generate_shared_accepted_native_receipts():
    rows = []
    for arm in ('fresh', 'no_fresh'):
        learner, replay, packet = first_packet(arm)
        row = native_preflight(learner, replay, None, '/same/seed6201.pt', packet,
            dict(processed_at=1203., sequence_ids=list(range(5))),
            dict(actor_real_clock=True), dict(shared_rule_exact=True),
            seed=6201, scenario='stable', arm=arm, runtime_contract=dict(F='same'))
        assert row['technical_gate_pass']
        assert row['initializer']['initial_updates'] == 50000
        rows.append(row)
    assert rows[0]['target_audit']['fresh_value_replacement_calls'] == 8
    assert rows[1]['target_audit']['fresh_value_replacement_calls'] == 0
    result = preflight(rows, dict(native_preflight=dict(seed=6201,scenario='stable',
        feedback_slots=5,policy_updates=8)))
    assert result['all_pass']
    rows[1]['execution_contract']['F'] = 'different'
    assert not preflight(rows, dict(native_preflight=dict(seed=6201,scenario='stable',
        feedback_slots=5,policy_updates=8)))['all_pass']


def test_paid_state_probe_covers_lines_and_windows_without_mutating_states_or_rng():
    learner, _, packet = first_packet('fresh')
    probe = PacketPolicyProbe()
    before_states = [row.s.copy() for row in packet]
    before_rng = torch.get_rng_state().clone()
    before_sampler = learner.rng.bit_generator.state
    for now in (8999.,9000.,12000.):
        row = probe.record(learner, packet, processed_at=now)
        assert row['states'] == 5 and sum(item['count'] for item in row['groups']) == 5
    summary = probe.summary()
    assert summary['states'] == 15 and summary['packets'] == 3
    assert {row['window'] for row in summary['groups']} == {'pre_early','early','late_training'}
    assert {row['line'] for row in summary['groups']} == {'7X','102X','705X'}
    assert sum(row['count'] for row in summary['groups']) == 15
    assert torch.equal(before_rng, torch.get_rng_state())
    assert learner.rng.bit_generator.state == before_sampler
    for before, row in zip(before_states, packet):
        np.testing.assert_array_equal(before, row.s)


@pytest.mark.parametrize('field', ['actor_real_clock', 'shared_rule_exact'])
def test_native_preflight_rejects_execution_or_real_clock_mismatch(field):
    learner, replay, packet = first_packet('no_fresh')
    clock, action = dict(actor_real_clock=True), dict(shared_rule_exact=True)
    (clock if field == 'actor_real_clock' else action)[field] = False
    row = native_preflight(learner,replay,None,'/same/init',packet,
        dict(processed_at=1203.,sequence_ids=list(range(5))),clock,action,
        seed=6201,scenario='stable',arm='no_fresh',runtime_contract=dict(F='same'))
    assert row['technical_gate_pass'] is False
