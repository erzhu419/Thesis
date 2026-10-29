"""Exercise real CQL updates through the native first-packet admission."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import learning as v
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.launch import preflight
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_replay, table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def first_packet(arm):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=256,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    learner = v.ExecutedCQLTrainer(make_replay(arm), config=config, device='cpu', seed=42,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    learner.total_updates = 50000
    learner.activate_executed_cql()
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


def receipt(learner, replay, packet, arm, *, clock=True, shared=True):
    return native_preflight(learner, replay, None, '/same/seed6201.pt', packet,
        dict(processed_at=1203., sequence_ids=list(range(5))),
        dict(actor_real_clock=clock), dict(shared_rule_exact=shared),
        seed=6201, scenario='stable', arm=arm, runtime_contract=dict(F='same'))


def test_real_first_packet_and_matched_native_gate():
    rows=[]
    for arm in ('fresh','no_fresh'):
        learner,replay,packet=first_packet(arm)
        row=receipt(learner,replay,packet,arm)
        assert row['technical_gate_pass'], row['checks']
        assert row['packet_schedule']==[8,8,8,8]
        assert row['initializer']['initial_updates']==50000
        assert row['cql_audit']['penalty_calls']==8
        assert row['cql_audit']['factual_rows']==8*128
        assert row['target_audit']['fresh_value_replacement_calls']==(8 if arm=='fresh' else 0)
        rows.append(row)
    declared=dict(native_preflight=dict(seed=6201,scenario='stable',feedback_slots=5,
        policy_updates=8,critic_updates=8,value_updates=8,actor_updates=8,alpha_updates=8))
    assert preflight(rows,declared)['all_pass']
    rows[1]['execution_contract']['F']='different'
    assert not preflight(rows,declared)['all_pass']


def test_paid_policy_probe_is_passive_after_cql_updates():
    learner,_,packet=first_packet('fresh')
    probe=PacketPolicyProbe()
    before=[row.s.copy() for row in packet]
    main=torch.get_rng_state().clone()
    sampler=learner.rng.bit_generator.state
    for now in (8999.,9000.,12000.):
        result=probe.record(learner,packet,processed_at=now)
        assert result['states']==5
    summary=probe.summary()
    assert summary['states']==15 and summary['packets']==3
    assert {x['line'] for x in summary['groups']}=={'7X','102X','705X'}
    assert {x['window'] for x in summary['groups']}=={'pre_early','early','late_training'}
    assert torch.equal(main,torch.get_rng_state())
    assert sampler==learner.rng.bit_generator.state
    for expected,row in zip(before,packet):
        np.testing.assert_array_equal(expected,row.s)


@pytest.mark.parametrize('bad', ['clock','shared'])
def test_native_preflight_rejects_bad_clock_or_execution(bad):
    learner,replay,packet=first_packet('no_fresh')
    assert not receipt(learner,replay,packet,'no_fresh',**{bad:False})['technical_gate_pass']


def test_control_keeps_original_runtime_and_all456_updates():
    from experiments.mcwm_closed_loop_executed_cql_v68_20261026.control import protocol
    from experiments.mcwm_closed_loop_fresh_value_v64_20261025.control import protocol as old
    current,original=protocol(),old()
    assert current['runtime_contract']==original['runtime_contract']
    assert current['policy_update_budget']==current['critic_update_budget']==456
    assert current['actor_update_budget']==current['alpha_update_budget']==456
    assert current['policy_updates_per_packet']==current['critic_updates_per_packet']==8
    assert current['candidate']=='executed_cql'
