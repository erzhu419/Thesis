"""New native heldouts for frozen stale/fresh response models and one-intervention policies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run import (
    formal, snapshot, fork_children, NativePrefixTrace, install_scenario,
    _equal, create_target_env, selected_state, contract, physical)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference
from experiments.mcwm_closed_loop_fresh_response_v86_20261027.model import predict
import numpy as np
import torch

LINES = ('7X','102X','705X')
ROLES = {'low':'reference','high':'candidate'}


def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        handle.write('\n')


class CollectionFinished(Exception):
    pass


def run(scenario,out):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root_out = out
    artifact = json.loads((HERE/'model.json').read_text())
    if not artifact['gate']['all_pass']:
        raise RuntimeError('Fixed training gate held; no native evaluation allowed')
    model_before = deepcopy(artifact)
    seed, external_seed = 6207,316207
    env = hook = stream = trace = probe = controller = None
    root = choice = pair_path = native_info = prediction = None
    branch = None
    root_actions = 0
    pairs = []
    selection = {line:dict(selected=None,attempted=0,reasons={},status='missing') for line in LINES}
    clock_exact = continuation_exact = True
    prediction_rng_exact = True
    native_started_wall = None
    started = time.monotonic()

    def counts():
        return dict(actor_updates=0,critic_updates=0,value_updates=0,alpha_updates=0,
                    D_refits=0,world_updates=0,trust_updates=0,admitted_labels=0)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,current=stream.now if stream else 0.,
            total=24000 if branch else 12600,unit='sim_seconds',scenario=scenario,
            pairs_complete=len(pairs),elapsed_seconds=time.monotonic()-started)),flush=True)

    def policy(state):
        nonlocal clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0] == np.float32(np.float32(event_clock)/24000))
        return controller(state)

    def selector(event):
        nonlocal branch,out,trace,probe,root,choice,pair_path,native_info,prediction
        nonlocal native_started_wall,root_actions,prediction_rng_exact
        line = event['line_id']
        if branch is not None or line not in LINES or not 12000 <= event['started_at'] < 12600:
            return None
        slot = selection[line]
        if slot['selected'] is not None:
            return None
        slot['attempted'] += 1
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        try:
            step = float(api.simulation.getDeltaT())
            action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            at_stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
        except Exception:
            step = action_step = None
            at_stop = False
        if not at_stop or step is None or action_step is None:
            reason = 'native_stop_or_clock_unavailable'
        else:
            choice = choose_pair(controller.action_transform,event['s'],controller.last_proposal,
                                 simulation_step=step,action_step=action_step)
            reason = None if choice['supported'] else 'singleton_or_insufficient_gap'
        if reason:
            slot['reasons'][reason] = slot['reasons'].get(reason,0)+1
            return None
        if not np.array_equal(event['a'],np.asarray(choice['reference_action'],np.float32)):
            raise RuntimeError('Reference root differs from the frozen actor/F')
        root = selected_state(event,stream._snapshot())
        root['reference_proposal'] = controller.last_proposal.tolist()
        native_info = dict(sim_step_seconds=step,action_step_seconds=action_step,at_stop=at_stop)
        rng_before = _rng()
        prediction = predict(artifact,line,event['s'],choice['reference_hold_seconds'],choice['candidate_hold_seconds'])
        prediction_rng_exact &= _equal(rng_before,_rng())
        if not prediction_rng_exact:
            raise RuntimeError('Readout scoring changed the native random stream')
        slot['selected'] = {key:root[key] for key in ('started_at','sequence','trip_id','station_id')}
        slot['status'] = 'selected'
        pair_path = root_out/'pairs'/line
        write(pair_path/'root.json',dict(root=root,reference=reference,choice=choice,
            prediction=prediction,prediction_recorded_at=root['started_at'],before_suffixes=True,**native_info))

        def prepare_child(key):
            nonlocal branch,out,trace,probe,native_started_wall
            branch,out = key,pair_path/key
            old_rows = trace.observed_issues
            trace.close()
            trace = NativePrefixTrace(out/'trace.jsonl')
            trace.observed_issues = old_rows
            stream.enter_suffix()
            probe = physical.install(env._bridge,root)
            native_started_wall = time.time()

        receipt = fork_children(pair_path,trace,lambda:stream.now,counts,prepare_child,timeout_seconds=3600)
        if isinstance(receipt,str):
            root_actions += 1
            return np.asarray(choice[ROLES[branch]+'_action'],np.float32)
        arms = {ROLES[key]:json.loads((pair_path/key/'summary.json').read_text()) for key in ROLES}
        ref,cand = arms['reference'],arms['candidate']
        departures = [arms[role]['physical']['departure']['clock_seconds'] for role in arms]
        departure_observed = all(value is not None for value in departures)
        checks = dict(same_complete_root=ref['root']==cand['root'],
            same_fixed_reference=ref['reference']==cand['reference']==reference,
            prior_reference_available=reference['prior_run_reference'] and reference['available_before_heldout_episode'],
            same_external_seed=ref['external_seed']==cand['external_seed']==external_seed,
            original_actor_root_exact=ref['executed_action']==choice['reference_action'],
            predeclared_candidate_exact=cand['executed_action']==choice['candidate_action'],
            command_support=choice['supported'],actual_departures_observed=departure_observed,
            physical_departure_separation=bool(departure_observed and abs(departures[1]-departures[0])>=step),
            actual_next_arrivals=all(value['physical']['coverage']['next_arrival_observed'] for value in arms.values()),
            both_real_terminal=all(value['final_clock']==24000 for value in arms.values()),
            nonzero_duration=ref['remaining']['dt']==cand['remaining']['dt']>0,
            label_maturity=all(value['remaining']['available_at']==24030 for value in arms.values()),
            zero_suffix_training=all(value['suffix_updates']==0 for value in arms.values()),
            zero_truth_feedback=all(value['suffix_admitted_labels']==0 for value in arms.values()),
            native_fork=receipt['all_pass'],actual_parallel=receipt['overlap_seconds']>0,
            parent_actor_frozen=_equal(actor_before,controller.snapshot_state()),
            models_frozen=_equal(model_before,artifact),predictions_leave_rng=prediction_rng_exact,
            prediction_before_truth=True,no_replacement_root=True)
        initial = ref['root']['issue_ledger']
        full_wait = initial['accrued_wait_seconds']+ref['remaining']['wait']
        full_ride = initial['accrued_in_vehicle_seconds']+ref['remaining']['ride']
        record = dict(status='complete',all_pass=all(checks.values()),checks=checks,line_id=line,
            root={key:root[key] for key in ('started_at','sequence','trip_id','station_id')},
            reference_hold_seconds=choice['reference_hold_seconds'],candidate_hold_seconds=choice['candidate_hold_seconds'],
            prediction=prediction,prediction_recorded_at=root['started_at'],before_suffixes=True,
            overlap_seconds=receipt['overlap_seconds'],remaining={role:value['remaining'] for role,value in arms.items()},
            delta={key:cand['remaining'][key]-ref['remaining'][key] for key in ('wait','ride','cost')},
            physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds'],
                missing=value['physical']['missing']) for role,value in arms.items()},
            reference_full_endpoint=dict(wait=full_wait,ride=full_ride,cost=2*full_wait+full_ride))
        write(pair_path/'pair_summary.json',record)
        pairs.append(record)
        progress('heldout_pair_complete_parent_resumes')
        if len(pairs)==3:
            raise CollectionFinished()
        return None

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = controller.last_proposal.copy()
        if branch is not None and event['sequence']!=root['sequence']:
            continuation_exact &= bool(np.array_equal(event['a'],controller.action_transform.numpy(event['s'],controller.last_proposal)))
        trace.observe(event,stream._snapshot())

    def spent_budget():
        rows = []
        for line in LINES:
            for key in ROLES:
                directory = root_out/'pairs'/line/key
                if not directory.exists(): continue
                if (directory/'summary.json').exists():
                    value=json.loads((directory/'summary.json').read_text())
                    rows.append(dict(line=line,role=ROLES[key],status=value['status'],seconds=value['remaining']['dt']))
                elif (directory/'failure.json').exists():
                    value=json.loads((directory/'failure.json').read_text())
                    rows.append(dict(line=line,role=ROLES[key],status='failed',seconds=value.get('suffix_native_seconds')))
                else:
                    rows.append(dict(line=line,role=ROLES[key],status='unresolved',seconds=None))
        prefix=stream.now if stream is not None else 0.
        suffix=sum(row['seconds'] for row in rows if row['seconds'] is not None)
        return dict(prefixes=1,suffixes=len(rows),native_prefix_seconds=prefix,suffix_native_seconds=suffix,
            total_native_seconds=prefix+suffix,unknown_suffix_costs=sum(row['seconds'] is None for row in rows),
            labels_observed=sum(row['status']=='complete' for row in rows),new_initialization_updates=0,
            prefix_training_updates=0,suffix_updates=0,labels_fed_back=0,details=rows)

    def finish_parent():
        spent=spent_budget()
        checks=dict(all_three_lines=len(pairs)==3,all_pairs_integrity=all(value['all_pass'] for value in pairs),
            actor_frozen=_equal(actor_before,controller.snapshot_state()),models_frozen=_equal(model_before,artifact),
            fixed_reference=reference['prior_run_reference'] and reference['available_before_heldout_episode'],
            same_training_reference_path=reference['source_actor_path']==artifact['source_reference_paths'][scenario],
            old_reference_actions_reproduced=all(reference_reproduction.values()),
            model_available_before_new_episode=artifact['fitted_at_wall']<=episode_started_wall,
            fixed_new_external_seed=external_seed==316207,prefix_within_budget=stream.now<=12600,
            native_budget=spent['total_native_seconds']<=84600,no_unknown_spend=spent['unknown_suffix_costs']==0,
            zero_new_training=True,zero_truth_feedback=True)
        value=dict(status='complete' if len(pairs)==3 else 'incomplete',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=scenario,seed=seed,
            prefix_id='v86_frozen_prior_reference/'+scenario+'/external316207',
            external_seed=external_seed,departed_passengers=15079,reference_snapshot=reference,
            reference_reproduction=reference_reproduction,
            episode_started_wall=episode_started_wall,model_fitted_wall=artifact['fitted_at_wall'],
            budget=spent,pairs=pairs,selection=selection,training_started=False)
        write(root_out/'collector_summary.json',value)
        progress('heldout_complete_no_control_takeover')
        return value

    try:
        torch.set_num_threads(1)
        controller,mdp,stations,reference=load_reference(scenario)
        reference['heldout_external_seed']=external_seed
        actor_before=controller.snapshot_state()
        reference_reproduction={}
        for version in ('mcwm_closed_loop_reference_collection_v84_20261027',
                        'mcwm_closed_loop_cost_advantage_v85_20261027'):
            old=HERE.parent/version/'runs'/scenario/'pairs'
            for line in LINES:
                declaration=json.loads((old/line/'root.json').read_text())
                actual=controller(np.asarray(declaration['root']['s'],np.float32))
                reference_reproduction[version+'/'+line]=bool(
                    np.array_equal(actual,np.asarray(declaration['choice']['reference_action'],np.float32))
                    and np.array_equal(controller.last_proposal,np.asarray(declaration['root']['reference_proposal'],np.float32)))
        if not all(reference_reproduction.values()):
            raise RuntimeError('Frozen actor/F does not exactly reproduce both prior source cohorts')
        controller.last_proposal=None
        trace=NativePrefixTrace(root_out/'prefix_trace.jsonl')
        env=create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000,seed=external_seed)
        stream=CollectionStream(env,stations,mdp,domain='target',selector=selector,
            delay_seconds=30,execute_transform=False,on_issue=issue)
        stream.reset()
        stream.enable_prefix_limit()
        hook=install_scenario(env,scenario,shock_at=9000,factor=.65,metric_every=300)
        episode_started_wall=time.time()
        last_progress=time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy)
            stream.release_intervals()
            if time.monotonic()-last_progress>=45:
                progress('frozen_suffix' if branch else 'new_frozen_prefix')
                last_progress=time.monotonic()
        if branch is None: return finish_parent()
        remaining=contract.remaining_cost(root['issue_ledger'],snapshot(stream.final_snapshot))
        observation=probe.summary(endpoint_reached=True)
        applied=observation['native_action']
        role=ROLES[branch]
        hold=choice[role+'_hold_seconds']
        checks=dict(real_terminal=stream.now==24000 and stream.done,
            root_action_once=applied is not None and root_actions==1,
            actual_command_exact=applied is not None and abs(applied['applied_hold_seconds']-hold)<=1e-5,
            own_frozen_continuation=continuation_exact,actor_real_clock=clock_exact,
            actor_frozen=_equal(actor_before,controller.snapshot_state()),models_frozen=_equal(model_before,artifact),
            zero_native_fits=counts()['D_refits']==0,zero_admission=counts()['admitted_labels']==0,
            native_nonzero_duration=remaining['dt']>0,label_not_released=not contract.label_available(remaining['available_at'],stream.now),
            all_issues_logged=trace.observed_issues==stream.issued_count)
        value=dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),root=root,reference=reference,role=role,
            external_seed=external_seed,executed_action=choice[role+'_action'],executed_hold_seconds=hold,
            **native_info,final_clock=stream.now,suffix_updates=0,suffix_admitted_labels=0,
            remaining=remaining,physical=observation,
            execution_wall=dict(pid=os.getpid(),native_started_wall=native_started_wall,native_finished_wall=time.time()))
        write(out/'summary.json',value)
        if not all(checks.values()): raise RuntimeError('Native heldout integrity failed')
        progress('complete')
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    except CollectionFinished:
        return finish_parent()
    except Exception as error:
        value=dict(type=type(error).__name__,message=str(error),scenario=scenario,role=ROLES.get(branch),
            native_time=stream.now if stream is not None else None,labels_fed_back=0)
        if branch is None: value['consumed_budget']=spent_budget()
        else: value['suffix_native_seconds']=max(0.,stream.now-root['started_at'])
        write(out/'failure.json',value)
        if branch is not None:
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.scenario,args.out)
