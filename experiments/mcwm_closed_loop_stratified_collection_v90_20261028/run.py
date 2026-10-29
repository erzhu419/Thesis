"""Frozen native load strata: formal paired collection, without fitting."""
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
    snapshot, fork_children, NativePrefixTrace, install_scenario, _equal,
    create_target_env, selected_state, contract, physical)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.stream import NativeContextReader, issue_binding
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import (
    LINES, action_support, service_state)
import numpy as np
import torch

ROLES = {'low':'reference', 'high':'candidate'}
LOAD_BINS = ('lighter','heavier')
SLOTS = tuple(line+'/'+load for line in LINES for load in LOAD_BINS)
ROOT_KEYS = ('started_at','sequence','trip_id','station_id')
V88 = HERE.parent/'mcwm_closed_loop_service_response_v88_20261028'
REUSE = V88/'runs/preflight/pairs/low/7X'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_name(path.name+'.writing')
    with pending.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        handle.write('\n')
    pending.rename(path)


def run(out, scenario):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root_out = out
    p = json.loads((HERE/'protocol.json').read_text())
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    if not (audit['all_pass'] and all(audit['checks'].values())
            and audit['new_fits']==audit['new_neural_updates']==audit['v87_truth_used']==0
            and p['source_initializer']==audit['source_initializer']==6207
            and p['external_seed']==audit['external_seed']==716207
            and p['thresholds']=={'7X':.13,'102X':.5,'705X':.5}
            and scenario in p['scenarios']):
        raise RuntimeError('Frozen actor, reuse audit and strata must agree')
    seed, external_seed = 6207,716207
    gate_path = HERE/'first_pair_gate.json'
    if scenario=='local_speed_drop':
        deadline = time.monotonic()+7200
        while not gate_path.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Stable first new pair gate not available; no shock simulation started')
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_for_stable_physical_gate',
                current=0,total=1,unit='technical_gate')),flush=True)
            time.sleep(30)
        if not json.loads(gate_path.read_text())['all_pass']:
            write(out/'not_started.json',dict(reason='first new stable pair failed physical gate',native_seconds=0))
            raise RuntimeError('First new stable pair failed; no shock prefix started')
    env = hook = stream = trace = probe = controller = context_reader = None
    branch = root = choice = pair_path = native_info = service_row = None
    pairs, coverage = [], {}
    selection = {key:dict(selected=None,attempted=0,reasons={},status='missing') for key in SLOTS}
    skip_reasons, observation_counts = {}, dict(reads=0,observed=0,unsupported=0)
    observation_checks = dict(clock_unchanged=True,ledger_unchanged=True,rng_unchanged=True,
        issue_clock_exact=True,event_clock_exact=True,context_before_action=True)
    clock_exact = continuation_exact = True
    root_actions = 0
    native_started_wall = None
    reuse_checks = {}
    started = time.monotonic()

    def counts():
        return dict(actor_updates=0,critic_updates=0,value_updates=0,alpha_updates=0,
            D_refits=0,world_updates=0,trust_updates=0,admitted_labels=0)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream else 0.,total=24000 if branch else 12600,
            unit='sim_seconds',external_seed=external_seed,scenario=scenario,
            coverage_slots=len(coverage),pairs_complete=len(pairs),
            elapsed_seconds=time.monotonic()-started)),flush=True)

    def skip(line, reason):
        key = line+'/'+reason
        skip_reasons[key] = skip_reasons.get(key,0)+1
        for load in LOAD_BINS:
            slot = selection[line+'/'+load]
            if slot['selected'] is None:
                slot['reasons'][reason] = slot['reasons'].get(reason,0)+1

    def complete():
        return len(coverage)==len(pairs)==6

    def policy(state):
        nonlocal clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0]==np.float32(np.float32(event_clock)/24000))
        return controller(state)

    def selector(event):
        nonlocal branch,out,trace,probe,root,choice,pair_path,native_info,service_row
        nonlocal root_actions,native_started_wall
        line = event['line_id']
        if branch is not None or line not in LINES or not 12000<=event['started_at']<12600:
            return None
        if all(selection[line+'/'+load]['selected'] is not None for load in LOAD_BINS):
            return None
        for load in LOAD_BINS:
            if selection[line+'/'+load]['selected'] is None:
                selection[line+'/'+load]['attempted'] += 1
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        try:
            step = float(api.simulation.getDeltaT())
            action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            at_stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
        except Exception:
            step = action_step = None
            at_stop = False
        if not at_stop or step is None or action_step is None:
            skip(line,'native_stop_or_clock_unavailable')
            return None
        proposed = choose_pair(controller.action_transform,event['s'],controller.last_proposal,
            simulation_step=step,action_step=action_step)
        support = action_support(proposed['reference_hold_seconds'],proposed['candidate_hold_seconds'],step,action_step)
        if not support['supported']:
            skip(line,support['reason'])
            return None
        if not np.array_equal(event['a'],np.asarray(proposed['reference_action'],np.float32)):
            raise RuntimeError('Original actor/F reference differs from issued action')
        before = dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        ctx = context_reader(event)
        state = service_state(ctx)
        binding = issue_binding(event)
        after = dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        checked = dict(clock_unchanged=before['clock']==after['clock'],
            ledger_unchanged=before['ledger']==after['ledger'],rng_unchanged=_equal(before['rng'],after['rng']),
            issue_clock_exact=before['clock']==after['clock']==event['started_at'],
            event_clock_exact=state['event_clock']==binding['event_clock'],context_before_action=True)
        for key,value in checked.items(): observation_checks[key] &= value
        observation_counts['reads'] += 1
        if not all(checked.values()):
            raise RuntimeError('Issue context sampling changed native clock, ledger or RNG')
        if not state['observed']:
            observation_counts['unsupported'] += 1
            for reason in state['reasons']: skip(line,reason)
            return None
        observation_counts['observed'] += 1
        load_bin = 'lighter' if state['features'][0]<p['thresholds'][line] else 'heavier'
        key = line+'/'+load_bin
        slot = selection[key]
        if slot['selected'] is not None: return None
        identity = {name:event[name] for name in ROOT_KEYS}
        row = dict(line_id=line,load_bin=load_bin,root=identity,context=ctx,service=state,
            binding=binding,observation_checks=checked,ref_hold=proposed['reference_hold_seconds'],
            cand_hold=proposed['candidate_hold_seconds'],sim_step_seconds=step,action_step_seconds=action_step)
        coverage[key] = row
        slot.update(selected=identity,status='selected')
        choice, service_row = proposed, deepcopy(row)
        root = selected_state(event,stream._snapshot())
        root['reference_proposal'] = controller.last_proposal.tolist()
        root['service_binding'] = binding
        root['service'] = state
        native_info = dict(sim_step_seconds=step,action_step_seconds=action_step,at_stop=at_stop)
        pair_path = root_out/'pairs'/load_bin/line
        declaration = dict(root=root,reference=reference,choice=choice,
            context=ctx,observation_checks=checked,load_bin=load_bin,before_suffixes=True,**native_info)
        write(pair_path/'root.json',declaration)
        if scenario=='stable' and key=='7X/heavier':
            original = json.loads((REUSE/'root.json').read_text())
            for field in ('root','reference','choice','context','observation_checks',
                          'before_suffixes','sim_step_seconds','action_step_seconds','at_stop'):
                reuse_checks[field] = _equal(declaration[field],original[field])
            reuse_checks['old_pair_audited'] = audit['all_pass']
            if not all(reuse_checks.values()):
                raise RuntimeError('Current prefix differs from frozen v88 reuse root; no replacement queried')
            record = json.loads((REUSE/'pair_summary.json').read_text())
            record.update(load_bin=load_bin,scenario=scenario,external_seed=external_seed,reused=True,
                source_load_bin=record['load_bin'],source_pair_path=str(REUSE/'pair_summary.json'),
                new_stratum_threshold=p['thresholds'][line],reuse_checks=dict(reuse_checks))
            write(pair_path/'pair_summary.json',record)
            pairs.append(record)
            progress('old_pair_reused_without_queries')
            return None

        def prepare_child(role):
            nonlocal branch,out,trace,probe,native_started_wall
            branch,out = role,pair_path/role
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
        arms = {ROLES[role]:json.loads((pair_path/role/'summary.json').read_text()) for role in ROLES}
        ref,cand = arms['reference'],arms['candidate']
        departures = [value['physical']['departure']['clock_seconds'] for value in arms.values()]
        observed = all(value is not None for value in departures)
        checks = dict(same_complete_root=ref['root']==cand['root']==root,
            same_fixed_reference=ref['reference']==cand['reference']==reference,
            same_external_seed=ref['external_seed']==cand['external_seed']==external_seed,
            original_actor_root_exact=ref['executed_action']==choice['reference_action'],
            predeclared_candidate_exact=cand['executed_action']==choice['candidate_action'],
            command_support=support['supported'],actual_departures_observed=observed,
            physical_departure_separation=bool(observed and abs(departures[1]-departures[0])>=step),
            actual_next_arrivals=all(value['physical']['coverage']['next_arrival_observed'] for value in arms.values()),
            both_real_terminal=all(value['final_clock']==24000 for value in arms.values()),
            nonzero_duration=ref['remaining']['dt']==cand['remaining']['dt']>0,
            label_maturity=all(value['remaining']['available_at']==24030 for value in arms.values()),
            zero_suffix_training=all(value['suffix_updates']==0 for value in arms.values()),
            zero_truth_feedback=all(value['suffix_admitted_labels']==0 for value in arms.values()),
            native_fork=receipt['all_pass'],actual_parallel=receipt['overlap_seconds']>0,
            parent_actor_frozen=_equal(actor_before,controller.snapshot_state()),
            service_context_before_action=all(service_row['observation_checks'].values()),
            same_issue_context=all(value['root']['service']==service_row['service'] for value in arms.values()),
            no_replacement_root=True)
        initial = ref['root']['issue_ledger']
        full_wait = initial['accrued_wait_seconds']+ref['remaining']['wait']
        full_ride = initial['accrued_in_vehicle_seconds']+ref['remaining']['ride']
        record = dict(status='complete',all_pass=all(checks.values()),checks=checks,line_id=line,
            scenario=scenario,external_seed=external_seed,reused=False,source_pair_path=None,
            new_stratum_threshold=p['thresholds'][line],
            load_bin=load_bin,root={name:root[name] for name in ROOT_KEYS},service=service_row['service'],
            reference_hold_seconds=choice['reference_hold_seconds'],candidate_hold_seconds=choice['candidate_hold_seconds'],
            observation_checks=service_row['observation_checks'],before_suffixes=True,**native_info,
            overlap_seconds=receipt['overlap_seconds'],remaining={role:value['remaining'] for role,value in arms.items()},
            delta={name:cand['remaining'][name]-ref['remaining'][name] for name in ('wait','ride','cost')},
            physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds'],
                missing=value['physical']['missing']) for role,value in arms.items()},
            reference_full_endpoint=dict(wait=full_wait,ride=full_ride,cost=2*full_wait+full_ride))
        write(pair_path/'pair_summary.json',record)
        pairs.append(record)
        if scenario=='stable' and not gate_path.exists():
            write(gate_path,dict(status='passed' if record['all_pass'] else 'failed',
                all_pass=record['all_pass'],checks=checks,scenario=scenario,slot=key,
                root=record['root'],pair_path=str(pair_path),labels_used_for_gate=False,
                cost_sign_used_for_gate=False,new_pair_in_formal_cohort=True))
            if not record['all_pass']:
                raise RuntimeError('First new formal pair failed physical gate; no replacement')
        progress('formal_pair_complete_parent_resumes')
        return None

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = controller.last_proposal.copy()
        if branch is not None and event['sequence']!=root['sequence']:
            continuation_exact &= bool(np.array_equal(event['a'],controller.action_transform.numpy(event['s'],controller.last_proposal)))
        trace.observe(event,stream._snapshot())

    def spent_budget():
        rows = []
        for load in LOAD_BINS:
            for line in LINES:
                for role in ROLES:
                    directory = root_out/'pairs'/load/line/role
                    if not directory.exists(): continue
                    if (directory/'summary.json').exists():
                        value = json.loads((directory/'summary.json').read_text())
                        rows.append(dict(line_id=line,load_bin=load,role=ROLES[role],status=value['status'],seconds=value['remaining']['dt']))
                    elif (directory/'failure.json').exists():
                        value = json.loads((directory/'failure.json').read_text())
                        rows.append(dict(line_id=line,load_bin=load,role=ROLES[role],status='failed',seconds=value.get('suffix_native_seconds')))
                    else:
                        rows.append(dict(line_id=line,load_bin=load,role=ROLES[role],status='unresolved',seconds=None))
        prefix = stream.now if stream is not None else 0.
        suffix = sum(row['seconds'] for row in rows if row['seconds'] is not None)
        return dict(prefixes=1,suffixes=len(rows),native_prefix_seconds=prefix,suffix_native_seconds=suffix,
            total_native_seconds=prefix+suffix,unknown_suffix_costs=sum(row['seconds'] is None for row in rows),
            labels_observed=sum(row['status']=='complete' for row in rows),new_initialization_updates=0,
            reused_pairs=sum(row['reused'] for row in pairs),reused_labels=2*sum(row['reused'] for row in pairs),
            prefix_training_updates=0,suffix_updates=0,labels_fed_back=0,new_fits=0,new_neural_updates=0,
            v87_truth_used=0,details=rows)

    def finish_parent():
        spent = spent_budget()
        replay_checks = dict(comparison_exact=trace.comparison_exact,
            all_issues_logged=trace.observed_issues==stream.issued_count>0)
        if trace.reference is not None:
            replay_checks.update(reference_exhausted=trace.reference.readline()=='',
                all_issues_match=trace.observed_issues==trace.matched_issues)
        prefix_reproduction = dict(all_pass=all(replay_checks.values()),checks=replay_checks,
            observed_issues=trace.observed_issues,matched_issues=trace.matched_issues,
            reference_trace_path=str(V88/'runs/preflight/prefix_trace.jsonl') if scenario=='stable' else None)
        checks = dict(six_covariate_slots=len(coverage)==6,six_predeclared_pairs=len(pairs)==6,
            pair_integrity=all(row['all_pass'] for row in pairs),actor_frozen=_equal(actor_before,controller.snapshot_state()),
            original_reference_actions_reproduced=all(reference_reproduction.values()),
            same_reference_path=reference['source_actor_path']==audit['reference_source_paths'][scenario],
            freeze_audit_pass=audit['all_pass'],issue_observation_unchanged=all(observation_checks.values()),
            parent_prefix_limit=stream.now==12600,native_budget=spent['total_native_seconds']<=p['budget']['scenario_caps'][scenario],
            exact_reuse_root=all(reuse_checks.values()) and (scenario!='stable' or bool(reuse_checks)),
            prefix_reproduction=prefix_reproduction['all_pass'],
            actor_real_clock=clock_exact,
            first_new_pair_gate=json.loads(gate_path.read_text())['all_pass'],
            known_spend=spent['unknown_suffix_costs']==0,zero_training_and_admission=True)
        value = dict(status='complete' if complete() else 'incomplete',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=scenario,seed=seed,
            external_seed=external_seed,prefix_id='v90/'+scenario+'/external716207',
            departed_passengers=15079,reference_snapshot=reference,reference_reproduction=reference_reproduction,
            coverage=coverage,selection=selection,skip_reasons=skip_reasons,observation_counts=observation_counts,
            observation_checks=observation_checks,budget=spent,pairs=pairs,training_started=False,
            models_fitted=False,predictions_used_for_selection=False,labels_used_for_selection=False,
            prefix_reproduction=prefix_reproduction,reuse_checks=reuse_checks,
            first_pair_gate=json.loads(gate_path.read_text()))
        write(root_out/'collector_summary.json',value)
        progress('formal_collection_finished_no_fit')
        return value

    try:
        torch.set_num_threads(1)
        controller,mdp,stations,reference = load_reference(scenario)
        reference['heldout_external_seed'] = external_seed
        actor_before = controller.snapshot_state()
        reference_reproduction = {}
        for version in ('mcwm_closed_loop_reference_collection_v84_20261027',
                        'mcwm_closed_loop_cost_advantage_v85_20261027'):
            for line in LINES:
                old = HERE.parent/version/'runs'/scenario/'pairs'/line/'root.json'
                declaration = json.loads(old.read_text())
                action = controller(np.asarray(declaration['root']['s'],np.float32))
                reference_reproduction[version+'/'+line] = bool(
                    np.array_equal(action,np.asarray(declaration['choice']['reference_action'],np.float32))
                    and np.array_equal(controller.last_proposal,np.asarray(declaration['root']['reference_proposal'],np.float32)))
        if not all(reference_reproduction.values()): raise RuntimeError('Original actor/F reference mismatch')
        controller.last_proposal = None
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl',
            reference=V88/'runs/preflight/prefix_trace.jsonl' if scenario=='stable' else None)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000,seed=external_seed)
        stream = CollectionStream(env,stations,mdp,domain='target',selector=selector,
            delay_seconds=30,execute_transform=False,on_issue=issue)
        stream.reset()
        stream.enable_prefix_limit()
        context_reader = NativeContextReader(env)
        hook = install_scenario(env,scenario,shock_at=9000,factor=.65,metric_every=300)
        last_progress = time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy)
            stream.release_intervals()
            if time.monotonic()-last_progress>=45:
                progress('frozen_suffix' if branch else 'new_frozen_prefix')
                last_progress = time.monotonic()
        if branch is None: return finish_parent()
        remaining = contract.remaining_cost(root['issue_ledger'],snapshot(stream.final_snapshot))
        observation = probe.summary(endpoint_reached=True)
        applied = observation['native_action']
        role = ROLES[branch]
        hold = choice[role+'_hold_seconds']
        checks = dict(real_terminal=stream.now==24000 and stream.done,root_action_once=applied is not None and root_actions==1,
            actual_command_exact=applied is not None and abs(applied['applied_hold_seconds']-hold)<=1e-5,
            own_frozen_continuation=continuation_exact,actor_real_clock=clock_exact,
            actor_frozen=_equal(actor_before,controller.snapshot_state()),zero_native_fits=counts()['D_refits']==0,
            zero_admission=counts()['admitted_labels']==0,native_nonzero_duration=remaining['dt']>0,
            issue_context_saved_before_action=all(service_row['observation_checks'].values()),
            label_not_released=not contract.label_available(remaining['available_at'],stream.now),
            all_issues_logged=trace.observed_issues==stream.issued_count)
        value = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),root=root,reference=reference,role=role,
            external_seed=external_seed,executed_action=choice[role+'_action'],executed_hold_seconds=hold,
            **native_info,final_clock=stream.now,suffix_updates=0,suffix_admitted_labels=0,
            remaining=remaining,physical=observation,
            execution_wall=dict(pid=os.getpid(),native_started_wall=native_started_wall,native_finished_wall=time.time()))
        write(out/'summary.json',value)
        if not all(checks.values()): raise RuntimeError('Native preflight child integrity failed')
        progress('complete')
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    except Exception as error:
        value = dict(type=type(error).__name__,message=str(error),role=ROLES.get(branch),
            native_time=stream.now if stream is not None else None,labels_fed_back=0)
        if branch is None: value['consumed_budget'] = spent_budget()
        else: value['suffix_native_seconds'] = max(0.,stream.now-root['started_at'])
        write(out/'failure.json',value)
        if branch is None and scenario=='stable' and not gate_path.exists():
            write(gate_path,dict(status='failed',all_pass=False,checks={'native_completion':False},
                scenario=scenario,reason=str(error),labels_used_for_gate=False,cost_sign_used_for_gate=False,
                new_pair_in_formal_cohort=True))
        if branch is not None:
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    args = parser.parse_args();run(args.out,args.scenario)
