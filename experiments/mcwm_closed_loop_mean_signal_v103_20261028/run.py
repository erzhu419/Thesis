"""Complete frozen T panel, six new lane-RNG futures per root, no learning."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
ROLES = {'low':'reference','high':'candidate'}
IDENTITY = ('started_at','sequence','trip_id','station_id')


def declared_roots(protocol, seed, scenario):
    if seed not in protocol['external_seeds'] or scenario not in protocol['scenarios']:
        raise ValueError('Only the frozen complete T panel is declared')
    rows = sorted((row for row in protocol['roots'] if row['external_seed'] == seed
        and row['scenario'] == scenario),key=lambda row:row['root']['sequence'])
    if (len(rows) != 3 or {row['line_id'] for row in rows} != set(protocol['lines'])
            or any(len(row['futures']) != protocol['replicates_per_root']
                   or any(future['seed'] is None for future in row['futures']) for row in rows)):
        raise ValueError('Three fixed roots with six fresh futures are required')
    return rows


def compact_pair(record):
    from copy import deepcopy
    result = deepcopy(record)
    for role in ROLES.values():
        rng = result['rng'][role]
        changed = rng.pop('changed_lane_pools')
        rng['changed_lane_pool_count'] = len(changed)
        rng['all_declared_lane_pools_changed'] = changed == list(range(rng['lane_count']))
        rng.pop('before_lane_counts')
        rng.pop('after_lane_counts')
        rng.pop('loader_symbol',None)
        result['rng_usage'][role].pop('draws_per_pool')
    return result


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_name(path.name+'.writing')
    with pending.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        handle.write('\n')
    if path.exists():
        raise FileExistsError(path)
    pending.rename(path)


def audit_trace(path, first, stop):
    """Read a closed persisted trace; stop is the exclusive global issue count."""
    rows = 0
    first_seen = last_seen = None
    mismatch = None
    with Path(path).open() as handle:
        for raw in handle:
            row = json.loads(raw)
            sequence = row['sequence']
            if first_seen is None:
                first_seen = sequence
            if sequence != first+rows and mismatch is None:
                mismatch = dict(row=rows,expected=first+rows,observed=sequence)
            last_seen = sequence
            rows += 1
    checks = dict(expected_rows=rows == stop-first,continuous_sequences=mismatch is None,
        first_sequence=(first_seen == first if rows else first == stop),
        last_sequence=(last_seen == stop-1 if rows else first == stop))
    return dict(all_pass=all(checks.values()),checks=checks,rows=rows,expected_rows=stop-first,
        first_sequence=first_seen,last_sequence=last_seen,first_mismatch=mismatch)


def finalize_prefix(trace, reference_path):
    unread = sum(1 for _ in trace.reference)
    prefix = dict(all_pass=trace.comparison_exact and trace.matched_issues == trace.observed_issues
        and unread == 0,reference_path=str(reference_path),observed_issues=trace.observed_issues,
        matched_issues=trace.matched_issues,
        source_rows=trace.matched_issues+unread if trace.comparison_exact else None,
        reference_exhausted=unread == 0)
    trace.close()
    return prefix,audit_trace(trace.path,0,trace.observed_issues)


def pair_integrity(checks):
    # Complete departure no-ops retain measured labels; do not filter by effect.
    return all(value for name,value in checks.items() if name != 'physical_departure_separation')


def paired_native_checks(parent, before, after):
    return dict(reference_before_equals_parent=before['reference'] == parent,
        candidate_before_equals_parent=before['candidate'] == parent,
        paired_after_physical=after['reference']['physical'] == after['candidate']['physical'],
        paired_after_lane_rng=after['reference']['lane'] == after['candidate']['lane'],
        paired_after_other_rng=after['reference']['other_rng'] == after['candidate']['other_rng'])


def run(external_seed, scenario, out, libsumo_library, adapter_path=None, sumo_cfg=None):
    # Runtime imports stay here so synthetic tests never initialize SUMO/Torch.
    from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run import (
        snapshot,fork_children,NativePrefixTrace,install_scenario,_equal,
        create_target_env,selected_state,contract,physical)
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
    from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
    from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
    from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference
    from experiments.mcwm_closed_loop_context_wiring_v80_20261027.stream import NativeContextReader,issue_binding
    from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support,service_state
    from experiments.mcwm_closed_loop_issue_conditions_v91_20261028.conditions import CorridorHistory
    from experiments.mcwm_closed_loop_collection_domain_v97_20261028.selection import (
        context_reason,frozen_predictions,model_support)
    from experiments.mcwm_closed_loop_v1.stream import EI
    from experiments.mcwm_closed_loop_future_rng_v102_20261028.rng import LaneRNG,build_adapter
    from experiments.mcwm_closed_loop_future_rng_v102_20261028 import native_probe
    import numpy as np
    import torch

    protocol = read(HERE/'protocol.json')
    declarations = declared_roots(protocol,external_seed,scenario)
    native_cap = protocol['prefix_stop']+sum(2*len(row['futures'])*
        (protocol['horizon']-row['root']['started_at']) for row in declarations)
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root_out = out
    source = HERE.parent/protocol['source']
    old_run = source/'runs'/str(external_seed)/scenario
    expected_roots = {row['line_id']:read(old_run/'pairs'/row['line_id']/'root.json')
        for row in declarations}
    for row in declarations:
        expected = expected_roots[row['line_id']]
        if ({key:expected['root'][key] for key in IDENTITY} != row['root']
                or expected['root']['line_id'] != row['line_id']
                or any(expected['choice'][key] != row[key] for key in (
                    'reference_hold_seconds','candidate_hold_seconds'))):
            raise RuntimeError('Source roots/actions differ from the frozen complete panel')
    model = read(source/'source_model.json')
    adapter_path = Path(adapter_path).resolve() if adapter_path else build_adapter(out/'native_rng.so')
    if libsumo_library is None:
        raise ValueError('Pass the actual already-loaded libsumo shared-object path')
    adapter = LaneRNG(adapter_path,libsumo_library,lane_count=64)
    base_cfg = Path(sumo_cfg) if sumo_cfg else Path(os.environ['THESIS_SUMO_ROOT'])/'online_control'/'control_sim_traci_period.sumocfg'
    isolated_cfg = out/'native.sumocfg'
    native_probe.isolated_config(base_cfg,isolated_cfg)
    env = hook = stream = trace = probe = controller = history = None
    branch = root = choice = future_id = future_seed = rng_receipt = None
    native_started_wall = None
    api = None
    root_actions = 0
    clock_exact = continuation_exact = True
    pairs = []
    root_results = []
    active_declaration = None
    root_checks = {}
    prefix_reproduction = None
    started = time.monotonic()

    def counts():
        return dict(actor_updates=0,critic_updates=0,value_updates=0,alpha_updates=0,
            D_refits=0,world_updates=0,trust_updates=0,admitted_labels=0)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,current=stream.now if stream else 0.,
            total=protocol['horizon'] if branch else protocol['prefix_stop'],unit='sim_seconds',
            external_seed=external_seed,scenario=scenario,line_id=root['line_id'] if root else None,
            future_id=future_id,pairs_complete=sum(len(value['pairs']) for value in root_results),elapsed_seconds=time.monotonic()-started)),flush=True)

    def policy(state):
        nonlocal clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0] == np.float32(np.float32(event_clock)/24000))
        return controller(state)

    def selector(event):
        nonlocal branch,out,trace,probe,root,choice,future_id,future_seed,rng_receipt
        nonlocal root_actions,native_started_wall,root_checks,prefix_reproduction,api,pairs,active_declaration
        if branch is not None or len(root_results) == len(declarations):
            return None
        active_declaration = declarations[len(root_results)]
        expected_identity = active_declaration['root']
        if event['sequence'] < expected_identity['sequence']:
            return None
        if ({key:event[key] for key in IDENTITY} != expected_identity
                or event['line_id'] != active_declaration['line_id']):
            raise RuntimeError('Exact frozen root missing or changed; no later replacement')
        line = active_declaration['line_id']
        expected = expected_roots[line]
        root_directory = root_out/'roots'/line
        pairs = []
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        step = float(api.simulation.getDeltaT())
        action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
        at_stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
        choice = choose_pair(controller.action_transform,event['s'],controller.last_proposal,
            simulation_step=step,action_step=action_step)
        support = action_support(choice['reference_hold_seconds'],choice['candidate_hold_seconds'],step,action_step)
        before = dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        ctx = context_reader(event)
        service = service_state(ctx)
        condition = history.snapshot(event['started_at'])
        binding = issue_binding(event)
        after = dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        reason = context_reason(service,condition,line,model,event['started_at'])
        predictions = None if reason else frozen_predictions(model,line,service,condition['value'],
            choice['reference_hold_seconds'],choice['candidate_hold_seconds'])
        root = selected_state(event,stream._snapshot())
        root.update(reference_proposal=controller.last_proposal.tolist(),service_binding=binding,
            service=service,history_condition=condition,predictions=predictions,
            training_support=model_support(model,line,condition['value']))
        prefix_reproduction = dict(all_pass=trace.comparison_exact and trace.matched_issues ==
            trace.observed_issues == expected_identity['sequence'],reference_path=str(old_run/'prefix_trace.jsonl'),
            observed_issues=trace.observed_issues,matched_issues=trace.matched_issues)
        root_checks = dict(exact_root=root == expected['root'],exact_context=ctx == expected['context'],
            exact_choice=choice == expected['choice'],exact_reference=reference == expected['reference'],
            exact_prefix=prefix_reproduction['all_pass'],native_stop=at_stop,
            exact_steps=step == expected['sim_step_seconds'] and action_step == expected['action_step_seconds'],
            supported=reason is None and support['supported'],
            original_action=np.array_equal(event['a'],np.asarray(choice['reference_action'],np.float32)),
            atomic_observation=before['clock'] == after['clock'] == event['started_at']
                and before['ledger'] == after['ledger'] and _equal(before['rng'],after['rng']),
            frozen_actor=_equal(actor_before,controller.snapshot_state()))
        root_checks = {key:bool(value) for key,value in root_checks.items()}
        write(root_directory/'root.json',dict(root=root,context=ctx,choice=choice,reference=reference,
            checks=root_checks,prefix_reproduction=prefix_reproduction,
            source_root=str(old_run/'pairs'/line/'root.json')))
        root_results.append(dict(id=active_declaration['id'],index=active_declaration['index'],line_id=line,
            root=dict(expected_identity),root_ledger=root['issue_ledger'],checks=root_checks,
            frozen_predictions=predictions,prefix_reproduction=prefix_reproduction,pairs=pairs))
        if not all(root_checks.values()):
            raise RuntimeError('Frozen root/prefix/command/context reproduction failed')
        parent_native = native_probe.save_snapshot(api,root_directory/'root_native.xml')
        for future in active_declaration['futures']:
            future_id,future_seed = future['id'],future['seed']
            pair_path = root_directory/'futures'/future_id

            def prepare_child(role):
                nonlocal branch,out,trace,probe,native_started_wall,rng_receipt
                branch,out = role,pair_path/role
                old_rows = trace.observed_issues
                trace.close()
                trace = NativePrefixTrace(out/'trace.jsonl')
                trace.observed_issues = old_rows
                stream.enter_suffix()
                rng_receipt = native_probe.apply_checked(api,adapter,future_seed,out)
                if not rng_receipt['all_pass']:
                    raise RuntimeError('Native RNG-only intervention check failed')
                probe = physical.install(env._bridge,root)
                native_started_wall = time.time()

            receipt = fork_children(pair_path,trace,lambda:stream.now,counts,prepare_child,timeout_seconds=3600)
            if isinstance(receipt,str):
                root_actions += 1
                return np.asarray(choice[ROLES[branch]+'_action'],np.float32)
            arms = {role:read(pair_path/key/'summary.json') for key,role in ROLES.items()}
            ref,cand = arms['reference'],arms['candidate']
            before = {role:native_probe.read_snapshot(pair_path/key/'native_before.xml')
                for key,role in ROLES.items()}
            after = {role:native_probe.read_snapshot(pair_path/key/'native_after.xml')
                for key,role in ROLES.items()}
            native_pair_checks = paired_native_checks(parent_native,before,after)
            departures = [arms[role]['physical']['departure']['clock_seconds'] for role in ('reference','candidate')]
            checks = dict(same_complete_root=ref['root'] == cand['root'] == root,
                same_fixed_reference=ref['reference'] == cand['reference'] == reference,
                same_future=ref['future_id'] == cand['future_id'] == future_id
                    and ref['future_seed'] == cand['future_seed'] == future_seed,
                original_actor_root_exact=ref['executed_action'] == choice['reference_action'],
                predeclared_candidate_exact=cand['executed_action'] == choice['candidate_action'],
                actual_departures_observed=all(value is not None for value in departures),
                physical_departure_separation=all(value is not None for value in departures)
                    and abs(departures[1]-departures[0]) >= step,
                actual_next_arrivals=all(value['physical']['coverage']['next_arrival_observed'] for value in arms.values()),
                both_real_terminal=all(value['final_clock'] == 24000 for value in arms.values()),
                native_fork=receipt['all_pass'],actual_parallel=receipt['overlap_seconds'] > 0,
                parent_actor_frozen=_equal(actor_before,controller.snapshot_state()),
                parent_at_fixed_root=stream.now == expected_identity['started_at'],
                trace_persisted=all(value['trace_audit']['all_pass'] for value in arms.values()),
                rng_only_change=all(value['rng']['all_pass'] for value in arms.values()),
                rng_used=all(value['rng_usage']['all_pass'] for value in arms.values()))
            checks.update(native_pair_checks)
            record = dict(future_id=future_id,future_seed=future_seed,all_pass=pair_integrity(checks),checks=checks,
                action={role:dict(executed_action=value['executed_action'],hold_seconds=value['executed_hold_seconds'])
                    for role,value in arms.items()},
                physical_noop=checks['actual_departures_observed'] and not checks['physical_departure_separation'],
                remaining={role:value['remaining'] for role,value in arms.items()},
                delta={key:cand['remaining'][key]-ref['remaining'][key] for key in ('wait','ride','cost')},
                physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                    next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds']) for role,value in arms.items()},
                rng={role:value['rng'] for role,value in arms.items()},
                rng_usage={role:value['rng_usage'] for role,value in arms.items()},
                trace_audit={role:value['trace_audit'] for role,value in arms.items()},
                overlap_seconds=receipt['overlap_seconds'])
            write(pair_path/'pair_summary.json',record)
            pairs.append(compact_pair(record))
            if not record['all_pass']:
                raise RuntimeError('Formal future pair failed; no replacement or additional future')
            progress('formal_future_complete_parent_still_paused')
        progress('fixed_root_complete_parent_resumes_reference')
        return None

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = controller.last_proposal.copy()
        if branch is not None and event['sequence'] != root['sequence']:
            continuation_exact &= bool(np.array_equal(event['a'],controller.action_transform.numpy(event['s'],controller.last_proposal)))
        trace.observe(event,stream._snapshot())
        history.observe(event)

    def spent_budget():
        rows = []
        for declared in declarations:
            for future in declared['futures']:
                for key,role in ROLES.items():
                    directory = root_out/'roots'/declared['line_id']/'futures'/future['id']/key
                    if not directory.exists():
                        continue
                    if (directory/'summary.json').exists():
                        value = read(directory/'summary.json')
                        seconds,status = value['remaining']['dt'],value['status']
                    elif (directory/'failure.json').exists():
                        value = read(directory/'failure.json')
                        seconds,status = value.get('suffix_native_seconds'),'failed'
                    else:
                        seconds,status = None,'unresolved'
                    rows.append(dict(line_id=declared['line_id'],future_id=future['id'],role=role,status=status,seconds=seconds))
        prefix = stream.now if stream is not None else 0.
        suffix = sum(row['seconds'] for row in rows if row['seconds'] is not None)
        unknown = sum(row['seconds'] is None for row in rows)
        return dict(prefixes=int(stream is not None),suffixes=len(rows),native_prefix_seconds=prefix,
            suffix_native_seconds=suffix,total_native_seconds=prefix+suffix,
            total_native_seconds_is_lower_bound=unknown>0,unknown_suffix_costs=unknown,
            labels_observed=sum(row['status']=='complete' for row in rows),details=rows,
            new_initialization_updates=0,prefix_training_updates=0,suffix_updates=0,
            labels_fed_back=0,new_fits=0,new_neural_updates=0,native_cap=native_cap)

    def finish_parent(error=None):
        trace_audit = None
        prefix = None
        if trace is not None:
            prefix,trace_audit = finalize_prefix(trace,old_run/'prefix_trace.jsonl')
        spent = spent_budget()
        checks = dict(three_fixed_roots=[row['id'] for row in root_results] == [row['id'] for row in declarations],
            six_fresh_futures_per_root=all([pair['future_id'] for pair in row['pairs']] ==
                [future['id'] for future in declaration['futures']]
                for row,declaration in zip(root_results,declarations)),
            pairs_complete=all(pair['all_pass'] for row in root_results for pair in row['pairs']),
            exact_roots=bool(root_results) and all(all(row['checks'].values()) for row in root_results),
            exact_complete_prefix=prefix is not None and prefix['all_pass'],
            prefix_persisted=trace_audit is not None and trace_audit['all_pass'],
            parent_at_prefix_boundary=stream is not None and stream.now == protocol['prefix_stop'] and stream.prefix_limit_reached,
            parent_actor_frozen=controller is not None and _equal(actor_before,controller.snapshot_state()),
            actor_real_clock=clock_exact,
            native_budget=spent['total_native_seconds'] <= native_cap,
            known_spend=spent['unknown_suffix_costs'] == 0,error_free=error is None)
        value = dict(status='complete' if all(checks.values()) else 'failed',
            external_seed=external_seed,scenario=scenario,
            integrity=dict(all_pass=all(checks.values()),checks=checks),
            prefix_reproduction=prefix,trace_audit=trace_audit,roots=root_results,budget=spent,error=error,
            interpretation='Full frozen T panel; six new future lane-driving RNG samples per root. No learning, fit, F0 repeats or V release.')
        if len(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())+1 > protocol['retrieval']['per_collector_bytes']:
            write(root_out/'summary_oversize_resident.json',value)
            raise RuntimeError('Compact summary exceeds frozen retrieval limit; resident full result retained')
        write(root_out/'summary.json',value)
        return value

    try:
        torch.set_num_threads(1)
        controller,mdp,stations,reference = load_reference(scenario)
        reference['heldout_external_seed'] = external_seed
        actor_before = controller.snapshot_state()
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl',old_run/'prefix_trace.jsonl')
        history = CorridorHistory(EI)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000,seed=external_seed,sumo_cfg=isolated_cfg)
        stream = CollectionStream(env,stations,mdp,domain='target',selector=selector,
            delay_seconds=30,execute_transform=False,on_issue=issue,prefix_limit_seconds=protocol['prefix_stop'])
        stream.reset(); stream.enable_prefix_limit()
        context_reader = NativeContextReader(env)
        hook = install_scenario(env,scenario,shock_at=9000,factor=.65,metric_every=300)
        last_progress = time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy); stream.release_intervals()
            if time.monotonic()-last_progress >= 45:
                progress('frozen_suffix' if branch else 'frozen_prefix_reproduction')
                last_progress = time.monotonic()
        if branch is None:
            return finish_parent()
        remaining = contract.remaining_cost(root['issue_ledger'],snapshot(stream.final_snapshot))
        observation = probe.summary(endpoint_reached=True)
        rng_usage = native_probe.suffix_rng_usage(api,out)
        applied = observation['native_action']
        role = ROLES[branch]
        hold = choice[role+'_hold_seconds']
        trace.close()
        trace_audit = audit_trace(trace.path,root['sequence'],stream.issued_count)
        checks = dict(real_terminal=stream.now == 24000 and stream.done,
            root_action_once=applied is not None and root_actions == 1,
            actual_command_exact=applied is not None and abs(applied['applied_hold_seconds']-hold) <= 1e-5,
            own_frozen_continuation=continuation_exact,actor_real_clock=clock_exact,
            actor_frozen=_equal(actor_before,controller.snapshot_state()),
            original_root=all(root_checks.values()),rng_only_change=rng_receipt['all_pass'],
            rng_used=rng_usage['all_pass'],native_nonzero_duration=remaining['dt']>0,
            zero_training_and_admission=True,label_not_released=not contract.label_available(remaining['available_at'],stream.now),
            all_issues_logged=trace.observed_issues == stream.issued_count,
            persisted_rows=trace_audit['all_pass'])
        value = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),root=root,reference=reference,
            external_seed=external_seed,scenario=scenario,root_id=active_declaration['id'],
            role=role,future_id=future_id,future_seed=future_seed,
            executed_action=choice[role+'_action'],executed_hold_seconds=hold,final_clock=stream.now,
            suffix_updates=0,suffix_admitted_labels=0,remaining=remaining,physical=observation,
            rng=rng_receipt,rng_usage=rng_usage,trace_audit=trace_audit,
            tail_boundary=next((row for row in hook.metrics if row['time_seconds']==18300.),None),
            execution_wall=dict(pid=os.getpid(),native_started_wall=native_started_wall,native_finished_wall=time.time()))
        write(out/'summary.json',value)
        if not all(checks.values()):
            raise RuntimeError('Native future child integrity failed')
        progress('complete'); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except Exception as error:
        value = dict(type=type(error).__name__,message=str(error),role=ROLES.get(branch),
            future_id=future_id,native_time=stream.now if stream else None,labels_fed_back=0)
        if branch is not None and trace is not None:
            trace.close()
        if branch is not None:
            value['suffix_native_seconds'] = max(0.,stream.now-root['started_at'])
        else:
            value['consumed_budget'] = spent_budget()
        write(out/'failure.json',value)
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        if not (root_out/'summary.json').exists() and not (root_out/'summary_oversize_resident.json').exists():
            finish_parent(error=value)
        raise
    finally:
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-seed',type=int,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--libsumo-library',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--sumo-cfg',type=Path)
    args = parser.parse_args()
    run(args.external_seed,args.scenario,args.out,args.libsumo_library,args.adapter,args.sumo_cfg)
