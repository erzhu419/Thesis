"""F inversion, causal child endpoints and the exact two-slot pair budget."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('THESIS_H2O_ROOT',str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.collect import (
    ContextForkStream,action_choices,cutoff_boundary,transition_dict,_decode_transition,first_pair_gate,
)
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.test_memory import MemoryNativeEnv
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import policy
from experiments.mcwm_closed_loop_state_response_v18_20261022.budget import PairPacketBudget
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7,Transition


def test_actual1_875_uses_existing_F_in_both_direct_and_one_eighth_regimes():
    transform=ExecutionAction(TransitMDPConfigV7())
    tail=table(.8,count=1)['s'][0]
    choices=action_choices(transform,tail)
    assert choices['low']['hold']==0. and choices['high']['hold']==1.875
    assert choices['high']['proposal'].tolist()==[-.5]  #15s latent proposal.
    early=table(.4,count=1)['s'][0]
    slots=early[5:].reshape(16,14)
    slots[0,4]=2.  # Positive reserve makes the direct proposal executable.
    choices=action_choices(transform,early)
    assert choices['high']['proposal'].tolist()==[-.9375]
    assert choices['high']['hold']==1.875
    slots[0,12]=0.  # The real completion floor makes 0 unreachable.
    assert action_choices(transform,early) is None


@pytest.mark.parametrize('event_at_cutoff',[False,True])
def test_child_every_issue_closure_memory_and_exact_boundary_have_no_fake_state(event_at_cutoff):
    batches=[(10,[('A',0),('B',0)]),(20,[('C',0)]),(30,[('D',0)]),
        (310 if event_at_cutoff else 1000,[('E',0)]),(2000,[('F',0)])]
    env=MemoryNativeEnv(batches)
    issues=[]
    stream=ContextForkStream(env,[3],TransitMDPConfigV7(horizon_seconds=24000),
        selector=lambda event:[-1.],on_issue=issues.append,execute_transform=False,
        fork_duration_seconds=300.,delay_seconds=30.)
    stream.reset();stream.memory=ExecutionMemory(env._bridge)
    rows=[]
    while not stream.cutoff_reached:rows.extend(stream.step(policy))
    row=stream.fork_transition
    assert row.dt==0. and row.r==0.
    assert row.memory_s['observed_at']==row.memory_ns['observed_at']==10.
    assert row.memory_s['vector']==row.memory_ns['vector']==[0.]*7
    assert issues[0]['memory_s']['vector']==issues[1]['memory_s']['vector']==[0.]*7
    next_row=next(row for row in rows if row.sequence==1)
    assert next_row.memory_ns['lines'][0]['recent_vehicle_count']==2
    assert next_row.memory_ns['lines'][0]['departure_observed_count']==2
    boundary=cutoff_boundary(stream)
    assert boundary['clock']==boundary['ledger']['time_seconds']==310.
    assert boundary['memory']['observed_at']==310.
    assert boundary['native_state_status']==('actual_control_available' if event_at_cutoff else 'no_control_at_exact_clock')
    assert (boundary['actual_control_state'] is not None)==event_at_cutoff
    assert stream.now==310. and not stream.done
    stream.memory.close()


def native_row(sequence,issue,completed,action):
    s=np.zeros(229,np.float32);ns=s.copy()
    raw=np.zeros(245,np.float32)
    row=Transition(s,np.asarray([action],np.float32),ns,-3.*(completed-issue),completed-issue,False,
        0,'target',issue,completed,completed+30.,'fleet','7X','station','other',sequence,raw,raw.copy(),'trip')
    row.memory_s=dict(observed_at=issue,vector=[0.]*84)
    row.memory_ns=dict(observed_at=completed,vector=[.1]*84)
    return row


def test_transition_roundtrip_retains_actual_zero_time_and_both_causal_contexts_only():
    original=native_row(44,9600.,9600.,-.9375)
    original.future_truth=dict(cost300=999.)
    record=transition_dict(original)
    assert 'future_truth' not in record
    restored=_decode_transition(record)
    assert restored.dt==0. and restored.terminal is False
    for field in ('s','a','ns','raw_s','raw_ns'):
        np.testing.assert_array_equal(getattr(original,field),getattr(restored,field))
    assert restored.memory_s==original.memory_s and restored.memory_ns==original.memory_ns


def test_nine_atomic_pairs_replace18_labels_and_keep_zero_dt_with285_total():
    protocol=dict(feedback_first_deadline=1200,feedback_last_deadline=18000,
        feedback_interval_seconds=300,feedback_records_per_packet=5,target_record_budget=285,
        feedback_delay_seconds=30.)
    budget=PairPacketBudget(protocol)
    pair_packets={30,31,32,40,41,42,50,51,52}
    paired_count=0;admitted=[]
    for index,deadline in enumerate(budget.deadlines):
        factual=[]
        if index in pair_packets:
            source=100000+paired_count
            budget.reserve(source,f'pair{paired_count}')
            issue=deadline-340.
            pair=dict(id=f'pair{paired_count}',index=paired_count,role='train',source_sequence=source,
                issued_at=issue,line_id='7X',features=[1.,0.,0.,.2],
                outcomes={arm:dict(truth=dict(status='complete',dt=300.,wait=10.,ride=20.,cost=40.),
                    transition=native_row(source,issue,issue if arm=='low' else issue+2.,action))
                    for arm,action in [('low',-1.),('high',-.9375)]})
            budget.add_pair(pair);paired_count+=1
            factual.append(native_row(source,deadline-60.,deadline-40.,0.))
        factual.extend(native_row(index*5+j,deadline-60.,deadline-40.,0.) for j in range(5))
        budget.ingest(factual)
        rows,_=budget.take(deadline)
        assert len(rows)==5
        assert sum(row.sequence<0 for row in rows) in (0,2)
        admitted.extend(rows)
    summary=budget.summary()
    assert summary['target_labels']==285 and summary['ordinary_labels']==267
    assert summary['training_pairs']==9 and summary['paired_action_labels']==18
    assert len({row.sequence for row in admitted})==285
    assert sum(row.sequence<0 and row.dt==0 for row in admitted)==9
    assert len(summary['excluded_factual_sequences'])==9
    assert all(hasattr(row,'memory_s') and hasattr(row,'memory_ns') for row in admitted)
