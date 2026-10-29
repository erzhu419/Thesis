"""One initializer-controlled prefix, budgeted pairs and untouched heldout truth."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from .bootstrap import HERE, write
else:
    from bootstrap import HERE, write

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze, optimizer_steps
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, selected_state, plain
from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_state_response_v18_20261022.budget import PairPacketBudget, paired_label_ids
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.difference import issue_features
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import create_target_env, pack_event_batch_v7


ARMS=('low','high')
HOLDS=(0.,1.875)


def transition_dict(row):
    if row is None:
        return None
    result=plain(asdict(row))
    result.update(memory_s=plain(row.memory_s),memory_ns=plain(row.memory_ns))
    return result


def action_choices(transform,state):
    """Invert the existing F with its direct and one-eighth proposal regimes."""
    choices={}
    for arm,target in zip(ARMS,HOLDS):
        for proposal_hold in (0.,1.875,15.):
            proposal=np.asarray([2.*proposal_hold/transform.max_hold-1.],np.float32)
            action=np.asarray(transform.numpy(state,proposal),np.float32).reshape(1)
            hold=float(transform.hold_seconds(torch.as_tensor(action))[0])
            if hold==target:
                choices[arm]=dict(proposal=proposal,action=action,hold=hold)
                break
        if arm not in choices:
            return None
    return choices


class ContextForkStream(ForkEventStream):
    """Causal memory at every issue and closure, including child suffix events."""
    def __init__(self,*args,selector,on_issue=None,**kwargs):
        self.memory=None
        self._user_selector,self._user_issue=selector,on_issue
        self._contexts={}
        super().__init__(*args,selector=self._select,on_issue=self._issue,**kwargs)

    def _select(self,event):
        event['memory_s']=self.memory.snapshot(event['started_at'])
        return self._user_selector(event)

    def _issue(self,event):
        if 'memory_s' not in event:
            event['memory_s']=self.memory.snapshot(event['started_at'])
        self._contexts[event['sequence']]=event['memory_s']
        if self._user_issue is not None:self._user_issue(event)

    def _close(self,raw_ns,snapshot,next_vehicle_id='',terminal=False):
        pending=self._pending
        sequence=pending['sequence'] if pending is not None else None
        memory_ns=self.memory.snapshot(snapshot.time_seconds) if pending is not None else None
        queued=len(self._queue)
        super()._close(raw_ns,snapshot,next_vehicle_id,terminal)
        if sequence is not None:
            memory_s=self._contexts.pop(sequence)
            if len(self._queue)>queued:
                row=self._queue[-1]
                row.memory_s,row.memory_ns=memory_s,memory_ns


def _ledger(snapshot):
    return {key:float(getattr(snapshot,key)) for key in ('time_seconds','waiting_passengers',
        'in_vehicle_passengers','accrued_wait_seconds','accrued_in_vehicle_seconds')}


def cutoff_boundary(stream):
    """An exact-time ledger/memory is not a fabricated native decision state."""
    snapshot=stream.cutoff_snapshot
    controls=[event for event in stream._batch() if not event['terminal']]
    actual=None
    if controls:
        local=np.stack([event['event'] for event in controls])
        raw=pack_event_batch_v7(local,0,np.full(len(controls),-1.,np.float32),stream.contract)
        actual=dict(s=stream._canonical(raw).tolist(),raw_s=raw.tolist(),
            vehicle_id=controls[0]['vehicle_id'],trip_id=controls[0]['trip_id'],
            station_id=controls[0]['station_id'])
    pending=stream._pending
    last_issue=None if pending is None else {key:plain(pending[key]) for key in
        ('sequence','started_at','line_id','trip_id','station_id','s','raw_s','a','memory_s')}
    return dict(clock=snapshot.time_seconds,ledger=_ledger(snapshot),
        memory=stream.memory.snapshot(snapshot.time_seconds),
        native_state_status='actual_control_available' if actual is not None else 'no_control_at_exact_clock',
        actual_control_state=actual,last_issued_control=last_issue)


class CollectedPair(NativeTruthPair):
    def build_summary(self,stream):
        result=super().build_summary(stream)
        row=stream.fork_transition
        result['global_transition']=transition_dict(row)
        valid=row is None or (row.memory_s['observed_at']==row.started_at and
            row.memory_ns['observed_at']==row.completed_at and
            len(row.memory_s['vector'])==len(row.memory_ns['vector'])==84)
        result['integrity']['checks']['next_event_memory_causal']=valid
        result['integrity']['all_pass']=all(result['integrity']['checks'].values())
        result['status']='complete' if result['integrity']['all_pass'] else 'integrity_failed'
        result['boundary']=cutoff_boundary(stream)
        return result


def plans(protocol,preflight=False):
    result=[];index=0
    for line in protocol['focus_lines']:
        for name,bounds in protocol['train_windows'].items():
            result.append(dict(id=f'train/{line}/{name}',role='train',line_id=line,
                window_name=name,bounds=bounds,index=index,selected=None,reasons=Counter(),attempts=0))
            index+=1
        for name,bounds in protocol['heldout_windows'].items():
            result.append(dict(id=f'heldout/{line}/{name}',role='heldout',line_id=line,
                window_name=name,bounds=bounds,index=None,selected=None,reasons=Counter(),attempts=0))
    return result[:1] if preflight else result


def _decode_transition(value):
    from experiments.mcwm_closed_loop_v1.stream import Transition
    context={key:value[key] for key in ('memory_s','memory_ns')}
    native={key:item for key,item in value.items() if key not in context}
    for key in ('s','a','ns','raw_s','raw_ns'):
        if native[key] is not None:native[key]=np.asarray(native[key],np.float32)
    row=Transition(**native)
    row.memory_s,row.memory_ns=context['memory_s'],context['memory_ns']
    return row


def encode_pair(record):
    result={key:plain(value) for key,value in record.items() if key!='outcomes'}
    result['outcomes']={arm:dict(**{key:plain(value) for key,value in outcome.items() if key!='transition'},
        transition=transition_dict(outcome['transition'])) for arm,outcome in record['outcomes'].items()}
    return result


def first_pair_gate(record):
    values=record['outcomes']
    checks=dict(paused_parent_and_parallel=record['pair_receipt']['all_pass'],
        two_real_next_event_rows=all(value['transition'] is not None for value in values.values()),
        exact300=all(value['truth']['status']=='complete' and value['truth']['dt']==300. for value in values.values()),
        applied_actions=all(abs(values[arm]['execution']['native_applied_hold_seconds']-hold)<=1e-5
            for arm,hold in zip(ARMS,HOLDS)),
        causal_memory=all(value['transition'] is not None and
            value['transition'].memory_s['observed_at']==value['transition'].started_at and
            value['transition'].memory_ns['observed_at']==value['transition'].completed_at for value in values.values()))
    return dict(all_pass=all(checks.values()),checks=checks,id=record['id'],
        sequence=record['source_sequence'],issue=record['issued_at'],holds=list(HOLDS),
        next_dt={arm:value['transition'].dt if value['transition'] is not None else None for arm,value in values.items()},
        parallel_overlap_seconds=record['pair_receipt']['overlap_seconds'],
        physical_status=record['comparison']['physical_separation']['status'],suffix_seconds=600,new_optimizer_updates=0)


def collect(args,protocol):
    p=protocol
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    env=stream=memory=trace=hook=None
    pair=None
    started=time.time()
    try:
        learner,replay,contract,stations,initializer=load_learner(args.seed,arm='fresh')
        if learner.total_updates!=p['initializer_updates'] or replay.target_received:
            raise RuntimeError('Collector requires the unchanged50000 initializer and empty fresh replay')
        freeze(learner)
        frozen_steps=optimizer_steps(learner)
        budget=PairPacketBudget(p)
        schedule=plans(p,args.preflight)
        records=[];packets=[];pre_recorded=set();label_origins={}
        last_proposal=None
        first_gate=None

        def counts():
            return dict(total_updates=learner.total_updates,native_feedback=replay.target_received,
                admitted_slots=len(budget.selected_sequences),packets=budget.index,world_updates=0,trust_updates=0)

        def policy(state):
            nonlocal last_proposal
            last_proposal=learner.act(state[None],deterministic=True)[0]
            return learner.action_transform.numpy(state,last_proposal)

        def selector(event):
            nonlocal pair,first_gate
            if pair is not None and pair.in_child:return None
            plan=next((value for value in schedule if value['selected'] is None and
                value['line_id']==event['line_id'] and value['bounds'][0]<=event['started_at']<value['bounds'][1]),None)
            if plan is None:return None
            plan['attempts']+=1
            choices=action_choices(learner.action_transform,event['s'])
            if choices is None:
                plan['reasons']['F_floor_or_unreachable_action']+=1;return None
            api=importlib.import_module(env._bridge.__class__.__module__).traci
            try:
                stopped=bool(api.vehicle.isAtBusStop(event['trip_id']))
                native_step=float(api.simulation.getDeltaT())
                action_step=float(api.vehicle.getActionStepLength(event['trip_id']))
            except Exception:
                plan['reasons']['native_execution_observation_missing']+=1;return None
            if not stopped or HOLDS[1]-HOLDS[0]<p['minimum_gap_steps']*max(native_step,action_step):
                plan['reasons']['native_stop_or_action_gap_unsupported']+=1;return None
            plan['selected']=dict(sequence=event['sequence'],issue=event['started_at'])
            if plan['role']=='train':budget.reserve(event['sequence'],plan['id'])
            event['proposed_a']=last_proposal.copy()
            trace.observe(event,stream._snapshot());pre_recorded.add(event['sequence'])
            selected=selected_state(event,stream._snapshot())
            actions={arm:value['action'].tolist() for arm,value in choices.items()}
            holds={arm:value['hold'] for arm,value in choices.items()}
            pair=CollectedPair(out/'branches'/plan['id'].replace('/','_'),
                dict(selected=selected,actions=actions,holds=holds),counts,p['native_external_seed'])
            replacement=pair.fork(event,stream,trace)
            if pair.in_child:return replacement
            outcomes={arm:dict(truth=value['truth']['fixed300'],next_event=value['truth']['next_event'],
                transition=_decode_transition(value['global_transition']) if value['global_transition'] is not None else None,
                execution=value['execution'],physical=value['physical'],boundary=value['boundary'])
                for arm,value in pair.values.items()}
            record=dict(id=plan['id'],role=plan['role'],index=plan['index'],line_id=plan['line_id'],
                window_name=plan['window_name'],selected=selected,actions=actions,holds=[holds[arm] for arm in ARMS],
                source_sequence=event['sequence'],issued_at=event['started_at'],features=issue_features(selected),
                status='collected',outcomes=outcomes,pair_receipt=pair.receipt,comparison=pair.comparison,
                selected_for_training=False,available_at=max(event['started_at']+300.,
                    *(value['transition'].completed_at if value['transition'] is not None else event['started_at']+300.
                      for value in outcomes.values()))+p['feedback_delay_seconds'])
            if first_gate is None:
                first_gate=first_pair_gate(record)
                write(out/'preflight_gate.json',first_gate)
                if not first_gate['all_pass']:raise RuntimeError('First native pair preflight failed; no reselection')
            if plan['role']=='train':
                if all(value['transition'] is not None for value in outcomes.values()):
                    record=budget.add_pair(record)
                    for arm,label in paired_label_ids(plan['index']).items():
                        label_origins[label]=dict(origin='paired',pair_id=plan['id'],arm=arm,
                            source_sequence=event['sequence'])
                else:record['status']='missing_next_event'
            records.append(record)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='collect_native_pair',
                current=stream.now,total=p['horizon_seconds'],unit='sim_seconds',id=plan['id'],
                selected_pairs=len(records))),flush=True)
            return None

        def issue(event):
            event['proposed_a']=last_proposal.copy()
            if pair is not None and pair.in_child:
                pair.observe_issue(event,stream);return
            if event['sequence'] not in pre_recorded:trace.observe(event,stream._snapshot())

        env=create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'],seed=p['native_external_seed'])
        stream=ContextForkStream(env,stations,contract,domain='target',selector=selector,on_issue=issue,
            delay_seconds=p['feedback_delay_seconds'],execute_transform=False,fork_duration_seconds=300.)
        stream.reset();memory=ExecutionMemory(env._bridge);stream.memory=memory
        trace=NativePrefixTrace(out/'prefix_trace.jsonl')
        hook=install_scenario(env,args.scenario,shock_at=p['shock_at_seconds'],factor=p['speed_factor'],
            metric_every=p['metric_interval_seconds'])
        native_started=time.time();last_progress=time.monotonic()
        while not stream.done:
            released=stream.step(policy);stream.release_intervals()
            if pair is not None and pair.in_child:
                pair.observe_progress(stream)
                if stream.cutoff_reached:pair.finish_child(stream)
                continue
            if budget.index<len(budget.deadlines):
                budget.ingest(released)
                while budget.due(stream.now):
                    rows,_=budget.take(stream.now)
                    receipt=budget.receipts[-1]
                    receipt['slots']=[dict(sequence=row.sequence,**label_origins.get(row.sequence,
                        dict(origin='ordinary',source_sequence=row.sequence))) for row in rows]
                    packets.append(dict(index=budget.index,receipt=plain(receipt),
                        processed_before_sequence=stream.issued_count,rows=[transition_dict(row) for row in rows]))
            if args.preflight and first_gate is not None:break
            if time.monotonic()-last_progress>=45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='collect_common_prefix',
                    current=stream.now,total=p['horizon_seconds'],unit='sim_seconds',
                    packets=budget.index,selected_pairs=len(records))),flush=True)
                last_progress=time.monotonic()
        native_finished=time.time()
        pair_records=[encode_pair(record) for record in records]
        write(out/'pair_truth.json',pair_records)
        admitted=[encode_pair(record) for record in budget.admitted]
        write(out/'admitted_pair_records.json',admitted)
        heldout=[dict(slot=record['id'].removeprefix('heldout/'),selected=record['selected'],
            actions=record['actions'],holds=dict(zip(ARMS,record['holds'])),
            source_path=str(out/'pair_truth.json'),paired_transitions={arm:transition_dict(
                record['outcomes'][arm]['transition']) for arm in ARMS})
            for record in records if record['role']=='heldout']
        if not args.preflight:write(out/'heldout_prediction_inputs.json',heldout)
        selection={value['id']:dict(role=value['role'],line_id=value['line_id'],bounds=value['bounds'],
            selected=value['selected'],attempts=value['attempts'],reasons=dict(value['reasons']),
            status='selected' if value['selected'] is not None else 'missing') for value in schedule}
        b=budget.summary()
        checks=dict(initializer_50000=learner.total_updates==p['initializer_updates'],
            no_native_model_updates=optimizer_steps(learner)==frozen_steps and replay.target_received==0,
            native_model_frozen=all(not parameter.requires_grad for name in
                ('actor','critic','target_critic','value','ratio') for parameter in getattr(learner,name).parameters()),
            exact_initial_pair_gate=first_gate is not None and first_gate['all_pass'],
            causal_issue_context=memory.summary()['integrity']['all_pass'],
            all_selected_pairs_parallel=all(record['pair_receipt']['all_pass'] for record in records),
            full_prefix_trace=trace.observed_issues==stream.issued_count,
            all_declared_slots_selected=all(value['selected'] is not None for value in schedule))
        if not args.preflight:
            checks.update(full_target_day=stream.done and stream.now==p['horizon_seconds'],
                total285=b['target_labels']==285,packets57=len(packets)==57,
                ordinary267=b['ordinary_labels']==267,paired18=b['paired_action_labels']==18,
                train9=b['training_pairs']==9,heldout6=len(heldout)==6,
                heldout_legal_next_event=all(all(row is not None for row in value['paired_transitions'].values()) for value in heldout),
                heldout_never_admitted=all(record['role']=='train' for record in budget.admitted),
                originalfive_slots=all(len(packet['rows'])==5 for packet in packets),
                pair_atomic_two_slots=all(receipt['paired_records'] in (0,2) for receipt in budget.receipts))
        proof=dict(all_pass=all(checks.values()),checks=checks)
        metadata=dict(initialization_seed=args.seed,scenario=args.scenario,native_external_seed=p['native_external_seed'],
            initializer=str(initializer),initializer_updates=p['initializer_updates'],
            target_record_budget=b['target_labels'],ordinary_rows=b['ordinary_labels'],paired_rows=b['paired_action_labels'],
            feedback_slots=b['target_labels'],packets=len(packets),target_horizon=p['horizon_seconds'],
            new_native_model_updates=0,training_pairs=len(admitted),heldout_states=len(heldout),
            collector_policy='Frozen50000 initializer zero-memory actor under unchanged F')
        if not args.preflight:
            write(out/'dataset.json',dict(proof=proof,metadata=metadata,packets=packets))
        summary=dict(status=('preflight_complete' if args.preflight else 'complete') if proof['all_pass'] else 'integrity_failed',
            integrity=proof,initialization_seed=args.seed,native_external_seed=p['native_external_seed'],
            scenario=args.scenario,selection=selection,receipts=budget.receipts,dataset_proof=proof,
            budget=dict(**b,prefix_native_seconds=stream.now,suffixes=2*len(records),
                suffix_native_seconds=600*len(records),native_model_optimizer_updates=0,
                offline_updates_per_arm=456 if not args.preflight else None,truth_labels_enter_heldout_training=0),
            context=memory.summary(),full_prefix_trace=dict(rows=trace.observed_issues,issued=stream.issued_count),
            paths=dict(dataset=str(out/'dataset.json') if not args.preflight else None,
                heldout=str(out/'heldout_prediction_inputs.json') if not args.preflight else None,
                pair_truth=str(out/'pair_truth.json'),admitted_pairs=str(out/'admitted_pair_records.json')),
            wall=dict(native_started=native_started,native_finished=native_finished,total_seconds=time.time()-started))
        write(out/'summary.json',summary)
        if not proof['all_pass']:raise RuntimeError('Native pair collection incomplete; missing slots retained without reselection')
        return summary
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),native_time=stream.now if stream is not None else None)
        if pair is not None and pair.in_child:
            write(pair.out/pair.arm/'failure.json',failure)
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        write(out/'failure.json',failure)
        raise
    finally:
        if pair is not None and pair.in_child:os._exit(1)
        if memory is not None:memory.close()
        if trace is not None:trace.close()
        if hook is not None:hook.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=6202)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();protocol=json.loads((HERE/'protocol.json').read_text())
    if args.seed not in protocol['training_seeds']:raise ValueError('Seed outside declared v37 collector')
    collect(args,protocol)
