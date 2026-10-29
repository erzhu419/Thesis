"""Run the resumed recent30+CQL combination through real first-packet admission."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as v
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.launch import preflight
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


@pytest.fixture(scope='module')
def first_packets():
    torch.set_num_threads(1)
    result={}
    for arm in ('fresh','no_fresh'):
        contract=TransitMDPConfigV7()
        replay=v.FreshOutcomeReplay(table(.1),table(.4),contract,
            np.full(12,40,np.float32),arm=arm,outcome_seed=6201+600000)
        config=v.TransitTrainerConfig(hidden_dim=8,ensemble_size=2,batch_size=256,
            target_subset_size=2,use_ratio_correction=False,reward_scale=1e-4,
            bootstrap_probability=1.,offline_objective='iql',online_backup='h2oplus',
            mixed_backup_lambda=.1,online_fraction=.5,online_update_order='actor-before-critic')
        learner=v.ExecutedCQLTrainer(replay,config=config,device='cpu',seed=42,
            action_transform=v.ExecutionAction(contract))
        learner.total_updates=50000
        learner.activate_executed_cql()
        data=table(.2,count=5)
        packet=[SimpleNamespace(**{key:data[key][i] for key in v.FIELDS.values()},
            line_id=('7X','102X','705X')[i%3]) for i in range(5)]
        replay.add_target(packet)
        learner.begin_actor_packet()
        for step in range(8):
            learner.continue_step(collect_diagnostics=step==0)
        result[arm]=(learner,replay,packet)
    return result


def receipt(values,arm,*,clock=True,shared=True):
    learner,replay,packet=values
    return native_preflight(learner,replay,None,'/same/seed6201.pt',packet,
        dict(processed_at=1205.,sequence_ids=list(range(37,42))),
        dict(actor_real_clock=clock),dict(shared_rule_exact=shared),
        seed=6201,scenario='stable',arm=arm,runtime_contract=dict(F='same'))


def test_real_first_packet_and_reused_pair_gate(first_packets):
    rows=[]
    for arm,values in first_packets.items():
        row=receipt(values,arm)
        assert row['technical_gate_pass'],row['checks']
        assert row['packet_schedule']==[8,8,8,8]
        assert row['initializer']['initial_updates']==50000
        assert row['reused_full_native_day'] is True
        assert row['recency']['recent_covariates']==5
        assert row['recency']['independent_outcome_sampler_calls']==8
        assert row['recency']['critic_fresh_outcome_window']==(30 if arm=='fresh' else 0)
        assert row['cql_audit']['penalty_calls']==8
        assert row['target_audit']['fresh_value_replacement_calls']==0
        if arm=='fresh':
            assert row['target_audit']['fresh_mixed_target_calls']==8
        rows.append(row)
    declared=dict(native_preflight=dict(seed=6201,scenario='stable',feedback_slots=5,
        policy_updates=8,critic_updates=8,value_updates=8,actor_updates=8,alpha_updates=8))
    assert preflight(rows,declared)['all_pass']
    rows[1]['execution_contract']['F']='different'
    assert not preflight(rows,declared)['all_pass']


def test_wrong_outcome_stream_blocks_native_admission(first_packets,monkeypatch):
    values=first_packets['fresh']
    monkeypatch.setattr(values[1],'outcome_seed',6201+600001)
    row=receipt(values,'fresh')
    assert not row['technical_gate_pass']
    assert row['checks']['recency/private_sampler_seed'] is False


def test_paid_probe_preserves_main_and_private_streams(first_packets):
    learner,replay,packet=first_packets['fresh']
    original_states=[row.s.copy() for row in packet]
    torch_state=torch.get_rng_state().clone()
    main_rng=deepcopy(learner.rng.bit_generator.state)
    private_rng=deepcopy(replay.outcome_rng.bit_generator.state)
    before_cql=learner.cql_audit()
    before_recency=replay.summary()
    probe=PacketPolicyProbe()
    for now in (8999.,9000.,12000.):
        assert probe.record(learner,packet,processed_at=now)['states']==5
    view=probe.summary()
    assert view['states']==15 and view['packets']==3
    assert {x['window'] for x in view['groups']}=={'pre_early','early','late_training'}
    assert {x['line'] for x in view['groups']}=={'7X','102X','705X'}
    assert torch.equal(torch_state,torch.get_rng_state())
    assert learner.rng.bit_generator.state==main_rng
    assert replay.outcome_rng.bit_generator.state==private_rng
    assert replay.summary()==before_recency and learner.cql_audit()==before_cql
    for original,row in zip(original_states,packet):
        np.testing.assert_array_equal(original,row.s)


@pytest.mark.parametrize('bad',['clock','shared'])
def test_clock_and_execution_mismatch_still_block(first_packets,bad):
    assert not receipt(first_packets['no_fresh'],'no_fresh',**{bad:False})['technical_gate_pass']


def test_native_protocol_keeps_v68_budget_F_and_update_schedule():
    from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.control import protocol
    from experiments.mcwm_closed_loop_executed_cql_v68_20261026.control import protocol as old
    current,baseline=protocol(),old()
    assert current['runtime_contract']==baseline['runtime_contract']
    for key in ('policy_update_budget','critic_update_budget','value_update_budget',
                'actor_update_budget','alpha_update_budget'):
        assert current[key]==baseline[key]==456
    assert current['policy_updates_per_packet']==current['critic_updates_per_packet']==8
    assert current['critic_fresh_outcome_window']==30
    assert current['original_replay_window']==285


def test_old_value_only_fresh_target_is_rejected(first_packets,monkeypatch):
    learner,_,_=first_packets['fresh']
    legacy=learner.target_audit()
    legacy.update(fresh_value_replacement_calls=8,fresh_value_replacement_rows=8*64,
        mixed_target_generated_before_replacement=True)
    legacy.pop('fresh_mixed_target_calls')
    legacy.pop('fresh_mixed_target_rows')
    monkeypatch.setattr(learner,'target_audit',lambda:legacy)
    value=receipt(first_packets['fresh'],'fresh')
    assert value['technical_gate_pass'] is False
    assert value['checks']['target/no_value_only_replacement'] is False
    assert value['checks']['target/fresh_mixed_scope'] is False
