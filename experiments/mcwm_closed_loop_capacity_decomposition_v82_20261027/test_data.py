"""Native endpoint joins cannot splice another record's state and outcome."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.data import (
    EI, TransitMDPConfigV7, chronological_outcome,load_run,exact_outcome_exposures)


def issue(sequence,clock,vehicle,wait,ride):
    event = [0.]*20
    event[EI['sim_time_seconds']] = clock-1
    binding=dict(sequence=sequence,episode_id=0,trip_id=vehicle,vehicle_id=vehicle,
        line_id='L',station_id='S',issue_clock=clock)
    context=dict(observed=True,z=-.2,reasons=[],binding=binding,
        clocks=dict(issue=clock,event=clock-1,obs_start=clock,obs_end=clock))
    return dict(sequence=sequence,episode_id=0,trip_id=vehicle,vehicle_id=vehicle,
        line_id='L',station_id='S',started_at=clock,event=event,
        issue_ledger=dict(time_seconds=clock,waiting_passengers=2,in_vehicle_passengers=3,
            accrued_wait_seconds=wait,accrued_in_vehicle_seconds=ride),
        s=np.full(229,sequence,np.float32),raw_s=np.full(230,sequence,np.float32),
        a=np.array([.5],np.float32),proposed_a=np.array([.4],np.float32),ctx_s=context)


def test_chronological_other_bus_reward_is_exact_native_ledger_difference():
    a,b=issue(0,10.,'busA',100.,200.),issue(1,11.,'busB',102.,203.)
    selected=dict(ctx_s=a['ctx_s'],ctx_ns=b['ctx_s'])
    row=chronological_outcome(a,b,selected,TransitMDPConfigV7())
    assert row['dt']==1 and row['r']==-7 and row['available_at']==41
    assert row['next_vehicle_id']=='busB' and not row['terminal']
    np.testing.assert_array_equal(row['ns'],b['s'])


def test_no_skipped_successor_or_context_hybrid_is_accepted():
    a,b=issue(0,10.,'busA',100.,200.),issue(1,11.,'busB',102.,203.)
    wrong=issue(2,11.,'busC',102.,203.)
    with pytest.raises(ValueError,match='actual next'):
        chronological_outcome(a,wrong,dict(ctx_s=a['ctx_s'],ctx_ns=wrong['ctx_s']),TransitMDPConfigV7())
    with pytest.raises(ValueError,match='same-record'):
        chronological_outcome(a,b,dict(ctx_s=wrong['ctx_s'],ctx_ns=b['ctx_s']),TransitMDPConfigV7())


@pytest.mark.parametrize('arm',['fresh','no_fresh'])
def test_loader_keeps_same_state_records_separate_and_respects_outcome_rights(tmp_path,arm):
    rows=[issue(0,10.,'busA',100.,200.),issue(1,11.,'busB',102.,203.)]
    # Distinct issue identities may have exactly equal canonical values.
    rows[1]['s']=rows[0]['s'].copy()
    def plain(row):
        return {key:(value.tolist() if isinstance(value,np.ndarray) else value)
                for key,value in row.items() if key!='ctx_s'}
    (tmp_path/'prefix_trace.jsonl').write_text(''.join(json.dumps(plain(r))+'\n' for r in rows))
    (tmp_path/'context_issues.jsonl').write_text(''.join(json.dumps(dict(sequence=r['sequence'],ctx_s=r['ctx_s']))+'\n' for r in rows))
    selected=dict(packet=1,processed_at=50.,covariates=[dict(sequence=0,ctx_s=rows[0]['ctx_s'])],
        outcomes=[dict(sequence=0,ctx_s=rows[0]['ctx_s'],ctx_ns=rows[1]['ctx_s'])] if arm=='fresh' else [])
    (tmp_path/'context_selected.jsonl').write_text(json.dumps(selected)+'\n')
    (tmp_path/'summary.json').write_text(json.dumps(dict(cell=dict(seed=6208,scenario='stable',arm=arm),
        receipts=[dict(covariate_sequence_ids=[0],outcome_sequence_ids=[0],deadline=45.)])))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(protocol=dict(horizon_seconds=24000))))
    result=load_run(tmp_path,TransitMDPConfigV7(),expected_packets=1,records_per_packet=1)
    assert len(result['issues'])==2 and len(result['covariates'])==1
    assert len(result['outcomes'])==(1 if arm=='fresh' else 0)
    assert result['covariates'][0]['arm']==arm
    assert result['counts']['cross_vehicle_outcomes']==(1 if arm=='fresh' else 0)


def test_private_draw_counts_match_the_actual_sampler_window_and_order():
    rows=[dict(sequence=index+100,packet=index//5+1) for index in range(285)]
    state_before=np.random.get_state()
    value=exact_outcome_exposures(rows,6208)
    assert value['calls']==456 and value['total_draws']==29184
    assert np.all(value['weights']>0)
    reference=np.zeros(285,np.int64)
    rng=np.random.default_rng(606208)
    for stop in range(5,286,5):
        for _ in range(8):
            for local in rng.integers(min(30,stop),size=64):
                reference[stop-min(30,stop)+local]+=1
    np.testing.assert_array_equal(value['weights'],reference)
    assert value['final_rng_state']==rng.bit_generator.state
    # Isolated draw-count reconstruction cannot move NumPy's native global RNG.
    current=np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(state_before,current))
