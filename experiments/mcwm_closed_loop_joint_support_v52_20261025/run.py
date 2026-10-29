"""Joint historical coverage/overlap precheck; no learner, Q or simulator."""
from pathlib import Path
import json
import math
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_physical_feedback_v49_20261025.run import read,write,projected_behavior


def issue_inputs(path, first, last):
    result={}
    with path.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['sequence']<first: continue
            if row['sequence']>last: break
            # No passenger ledger, reward or future endpoint is forwarded.
            result[row['sequence']]={key:row[key] for key in ('sequence','started_at','line_id','s','a')}
    return result


def old_current_actor_drift(path):
    result={}
    with path.open() as handle:
        for line in handle:
            entry=json.loads(line)
            if entry['updates'] not in (24,32,456): continue
            rows=[]
            for chain in entry['chains']:
                scores=chain['rows']
                rows.append(dict(line=chain['line_id'],arm=chain['arm'],links=len(scores),
                    actual_c_product=math.prod(row['c'] for row in scores),
                    actual_log_c=math.fsum(row['log_c'] for row in scores),
                    conditional_overlap_mean=math.fsum(row['overlap'] for row in scores)/len(scores)))
            result[str(entry['updates'])]=rows
    return dict(scope='existing six45-event branches only; not long-flow current-pi overlap',checkpoints=result)


def audit_overlap(issues,contexts,inputs,transform):
    import numpy as np
    from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import execution_measure,ratio_from_measures
    from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import conditional_overlap
    lookup={row['sequence']:row['behavior_s'] for row in contexts}
    rows=[]
    for issue in issues[:271]:
        sequence=issue['sequence'];context=lookup.get(sequence);actual=inputs.get(sequence)
        if context is None or actual is None:
            rows.append(dict(sequence=sequence,supported=False,reason='missing_issue_or_behavior'))
            continue
        state=np.asarray(actual['s'],np.float32)
        policy_same=np.array_equal(state,np.asarray(context['policy_input'],np.float32))
        execution_same=np.array_equal(state,np.asarray(context['execution_input'],np.float32))
        action=np.asarray(actual['a'],np.float32)
        action_same=np.array_equal(action,np.asarray(context['executed_action'],np.float32))
        native_same=np.array_equal(transform.numpy(context['execution_input'],context['proposal']),action)
        if not (policy_same and execution_same and action_same and native_same):
            rows.append(dict(sequence=sequence,supported=False,reason='actual_issue_inputs_or_F_differ',
                actor_input_equal=bool(policy_same),F_input_equal=bool(execution_same),
                logged_action_equal=bool(action_same),native_F_exact=bool(native_same)))
            continue
        measure=execution_measure(transform,state,action,context['parameters'])
        ratio=ratio_from_measures(measure,measure)
        overlap=conditional_overlap(transform,state,context['parameters'],context['parameters'])
        rows.append(dict(sequence=sequence,supported=ratio['status']=='supported' and overlap['status']=='supported',
            native_F_exact=True,actor_input_equal=True,F_input_equal=True,
            measure_kind=measure['measure_kind'],c=ratio['c'],log_c=ratio['log_c'],
            conditional_overlap=overlap['total_overlap'],
            expected_log_c=overlap['expected_log_c_under_behavior'],
            normalization_error=overlap['normalization_error']))
    supported=[row for row in rows if row['supported']]
    full=len(supported)==271
    result=dict(rows=271,supported=len(supported),unresolved=[row for row in rows if not row['supported']],
        native_F_exact=sum(row.get('native_F_exact',False) for row in rows),
        atom_actions=sum(row.get('measure_kind')=='atom' for row in supported),
        continuous_actions=sum(row.get('measure_kind')=='continuous' for row in supported),
        maximum_normalization_error=max((row['normalization_error'] for row in supported),default=None),
        same_snapshot_c_product=math.prod(row['c'] for row in supported) if full else None,
        conditional_overlap_range=[min(row['conditional_overlap'] for row in supported),max(row['conditional_overlap'] for row in supported)] if supported else None,
        scope='old recorded mu24 parameters and actual issue inputs; fixed-evaluator identity, not current learned-pi overlap',
        snapshot_weights_loaded=False,new14_label_snapshot_available=False)
    return result,rows


def main():
    import torch
    from experiments.mcwm_closed_loop_joint_support_v52_20261025.plan import build_plan,precheck_issues
    from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
    torch.set_num_threads(1)
    started=time.monotonic();protocol=read(HERE/'protocol.json');plan=build_plan()
    write(HERE/'budget_plan.json',plan)
    contract=read(HERE.parent/'mcwm_closed_loop_formal_20261020/inputs/history_metadata.json')['contract']
    transform=ExecutionAction(contract);cases={}
    for case in protocol['scenarios']:
        root=HERE.parent/protocol['source_version']/'runs'/case
        pilot=read(root/'pilot_summary.json');updates=read(root/'learning/update_summary.json')
        snapshot=read(root/'behavior_snapshot.json')
        if not (pilot['status']=='complete' and pilot['integrity']['all_pass'] and updates['integrity']['all_pass']
            and updates['budget']['received']==285 and updates['budget']['actual_updates']==456
            and snapshot['updates']==24 and snapshot['clock']<9000):
            raise RuntimeError('Old source or actual pre-issue snapshot audit failed')
        projected=read(HERE.parent/protocol['timing_source']/(case+'_detail.json'))['projected_issues']
        selected=projected[:272]
        contexts=projected_behavior(root/'behavior_prefix.jsonl',selected[0]['sequence'],selected[-1]['sequence'])
        geometry=precheck_issues(projected,contexts)
        inputs=issue_inputs(root/'prefix_trace.jsonl',selected[0]['sequence'],selected[-1]['sequence'])
        overlap,rows=audit_overlap(selected,contexts,inputs,transform)
        coverage={}
        for line,entry in geometry['root_coverage'].items():
            index=entry['root_index'];end=entry['first_real_300_endpoint_index']
            tail=rows[index+1:end] if index is not None and end is not None else []
            supported=bool(tail) and all(row['supported'] for row in tail)
            cumulative=math.prod(row['c'] for row in tail) if supported else None
            coverage[line]=dict(real_300_endpoint=entry['first_real_300_endpoint_clock'],
                actual_successor_links=len(tail),same_snapshot_c_product=cumulative,
                current_learner_pi_c_product=None,
                current_pi_status='not_scored_on_this_long_flow',
                bounded_physical_coefficient_at300=None if not supported else (9/270)*math.exp(
                    -(selected[end-1]['started_at']-selected[index]['started_at'])/300)*cumulative)
        old_last=read(root/'warmup_packet_3.json')['rows'][4]
        write(HERE/(case+'_detail.json'),dict(geometry=geometry,overlap=overlap,overlap_rows=rows))
        cases[case]=dict(historical_geometry=geometry,historical_same_snapshot_overlap=overlap,
            joint_root_support=coverage,old_snapshot=dict(clock=snapshot['clock'],feedback_slots=snapshot['feedback_slots'],updates=snapshot['updates']),
            replaced_old_entitlement_example=dict(packet=3,row_in_packet=5,
                feedback_id=old_last['feedback_id'],sequence=old_last['sequence'],
                scope='identity example only; future14-label run never requests this fifteenth entitlement'),
            old_current_actor_drift=old_current_actor_drift(root/'learning/credit_packets.jsonl'))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(case=case,geometry=geometry['all_pass'],same_snapshot_supported=overlap['supported'])),flush=True)
    result=dict(status='complete',priority_requested='high',
        integrity=dict(all_pass=True,source285_456_and_real_snapshot_verified=True,
            original_files_read_only=True,no_models_or_outcomes_scored=True),
        plan=plan,cases=cases,
        future_joint_gate=dict(status='not_yet_demonstrated',new14_label_snapshot_available=False,
            new_seed_native272_issue_chain_available=False,full_long_TD_Q_memory_arrays_available=False,
            sync_lambda0_numeric_parity_verified=False,collection_started=False,shadow_training_started=False),
        source_contract=dict(initializer_seed=6202,initializer_updates=50000,feedback_slots=285,
            optimizer_updates=456,horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        budget=dict(new_training_updates=0,new_feedback_slots=0,new_native_prefixes=0,new_native_branches=0,
            new_native_seconds=0,new_target_draws=0,checkpoints_loaded=0),elapsed_seconds=time.monotonic()-started)
    write(HERE/'resident_summary.json',result)
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import compact
    encoded=json.dumps(compact(result),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>protocol['summary_max_bytes']:
        raise RuntimeError('Reduce resident summary before downloading')
    with (HERE/'summary.json').open('x') as handle: handle.write(encoded)
    print(json.dumps(dict(status='complete',bytes=len(encoded.encode()),seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
