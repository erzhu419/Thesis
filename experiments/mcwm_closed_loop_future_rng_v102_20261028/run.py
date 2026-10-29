"""Two fixed native roots, three predeclared lane-RNG futures, no learning."""
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
CASES = {
    'T':dict(seed=1476154684,time=12003.,sequence=8128,cap=83985.,
        source='mcwm_closed_loop_collection_domain_v97_20261028'),
    'C':dict(seed=1119764361,time=12006.,sequence=8125,cap=83970.,
        source='mcwm_closed_loop_confidence_calibration_v99_20261028'),
}
FUTURES = (('F0',None),('F1',102001),('F2',202001))
ROLES = {'low':'reference','high':'candidate'}
IDENTITY = ('started_at','sequence','line_id','trip_id','station_id')


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


def original_reproduction(arms, original):
    checks = {role:{key:arms[role]['remaining'][key] == original[role][key]
        for key in ('wait','ride')} for role in ('reference','candidate')}
    return dict(all_pass=all(value for arm in checks.values() for value in arm.values()),checks=checks)


def pair_integrity(checks):
    # Complete departure no-ops retain measured labels; do not filter by effect.
    return all(value for name,value in checks.items() if name != 'physical_departure_separation')


def paired_native_checks(parent, before, after):
    return dict(reference_before_equals_parent=before['reference'] == parent,
        candidate_before_equals_parent=before['candidate'] == parent,
        paired_after_physical=after['reference']['physical'] == after['candidate']['physical'],
        paired_after_lane_rng=after['reference']['lane'] == after['candidate']['lane'],
        paired_after_other_rng=after['reference']['other_rng'] == after['candidate']['other_rng'])


class ParentFinished(Exception):
    pass


def run(case, out, adapter_path=None, libsumo_library=None, sumo_cfg=None):
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

    declaration = CASES[case]
    external_seed = declaration['seed']
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root_out = out
    source = HERE.parent/declaration['source']
    old_run = source/'runs'/str(external_seed)/'stable'
    old_pair = old_run/'pairs'/'7X'
    expected = read(old_pair/'root.json')
    old_children = {role:read(old_pair/key/'summary.json') for key,role in ROLES.items()}
    frozen_original = {role:value['remaining'] for role,value in old_children.items()}
    model = read(source/'source_model.json')
    expected_identity = dict(started_at=declaration['time'],sequence=declaration['sequence'],
        line_id='7X',trip_id='7X_32',station_id='7X05_102X18')
    if {key:expected['root'][key] for key in IDENTITY} != expected_identity:
        raise RuntimeError('Source root differs from the predeclared fixed root')
    if not all(value['status'] == 'complete' and value['integrity']['all_pass']
               and value['root'] == expected['root'] for value in old_children.values()):
        raise RuntimeError('The original two complete native endpoints are required')
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
    root_checks = {}
    prefix_reproduction = None
    started = time.monotonic()

    def counts():
        return dict(actor_updates=0,critic_updates=0,value_updates=0,alpha_updates=0,
            D_refits=0,world_updates=0,trust_updates=0,admitted_labels=0)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,current=stream.now if stream else 0.,
            total=24000 if branch else declaration['time'],unit='sim_seconds',case=case,
            future_id=future_id,pairs_complete=len(pairs),elapsed_seconds=time.monotonic()-started)),flush=True)

    def policy(state):
        nonlocal clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0] == np.float32(np.float32(event_clock)/24000))
        return controller(state)

    def selector(event):
        nonlocal branch,out,trace,probe,root,choice,future_id,future_seed,rng_receipt
        nonlocal root_actions,native_started_wall,root_checks,prefix_reproduction,api
        if branch is not None:
            return None
        if event['sequence'] < declaration['sequence']:
            return None
        if {key:event[key] for key in IDENTITY} != expected_identity:
            raise RuntimeError('Exact frozen root missing or changed; no later replacement')
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
        reason = context_reason(service,condition,'7X',model,event['started_at'])
        predictions = None if reason else frozen_predictions(model,'7X',service,condition['value'],
            choice['reference_hold_seconds'],choice['candidate_hold_seconds'])
        root = selected_state(event,stream._snapshot())
        root.update(reference_proposal=controller.last_proposal.tolist(),service_binding=binding,
            service=service,history_condition=condition,predictions=predictions,
            training_support=model_support(model,'7X',condition['value']))
        prefix_reproduction = dict(all_pass=trace.comparison_exact and trace.matched_issues ==
            trace.observed_issues == declaration['sequence'],reference_path=str(old_run/'prefix_trace.jsonl'),
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
        write(root_out/'root.json',dict(root=root,context=ctx,choice=choice,reference=reference,
            checks=root_checks,prefix_reproduction=prefix_reproduction,source_root=str(old_pair/'root.json')))
        if not all(root_checks.values()):
            raise RuntimeError('Frozen root/prefix/command/context reproduction failed')
        parent_native = native_probe.save_snapshot(api,root_out/'root_native.xml')
        for future_id,future_seed in FUTURES:
            pair_path = root_out/'futures'/future_id

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
            reproduction = original_reproduction(arms,frozen_original) if future_id == 'F0' else None
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
                parent_at_fixed_root=stream.now == declaration['time'],
                trace_persisted=all(value['trace_audit']['all_pass'] for value in arms.values()),
                rng_only_change=all(value['rng']['all_pass'] for value in arms.values()),
                rng_used=all(value['rng_usage']['all_pass'] for value in arms.values()),
                original_endpoint_reproduction=reproduction is None or reproduction['all_pass'])
            checks.update(native_pair_checks)
            record = dict(future_id=future_id,future_seed=future_seed,all_pass=pair_integrity(checks),checks=checks,
                physical_noop=checks['actual_departures_observed'] and not checks['physical_departure_separation'],
                remaining={role:value['remaining'] for role,value in arms.items()},
                delta={key:cand['remaining'][key]-ref['remaining'][key] for key in ('wait','ride','cost')},
                physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                    next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds']) for role,value in arms.items()},
                rng={role:value['rng'] for role,value in arms.items()},
                rng_usage={role:value['rng_usage'] for role,value in arms.items()},
                trace_audit={role:value['trace_audit'] for role,value in arms.items()},
                F0_reproduction=reproduction,overlap_seconds=receipt['overlap_seconds'])
            write(pair_path/'pair_summary.json',record)
            pairs.append(record)
            if not record['all_pass']:
                raise RuntimeError('Formal future pair failed; no replacement or additional future')
            progress('formal_future_complete_parent_still_paused')
        raise ParentFinished()

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = controller.last_proposal.copy()
        if branch is not None and event['sequence'] != root['sequence']:
            continuation_exact &= bool(np.array_equal(event['a'],controller.action_transform.numpy(event['s'],controller.last_proposal)))
        trace.observe(event,stream._snapshot())
        history.observe(event)

    def spent_budget():
        rows = []
        for fid,_seed in FUTURES:
            for key,role in ROLES.items():
                directory = root_out/'futures'/fid/key
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
                rows.append(dict(future_id=fid,role=role,status=status,seconds=seconds))
        prefix = stream.now if stream is not None else 0.
        suffix = sum(row['seconds'] for row in rows if row['seconds'] is not None)
        unknown = sum(row['seconds'] is None for row in rows)
        return dict(prefixes=int(stream is not None),suffixes=len(rows),native_prefix_seconds=prefix,
            suffix_native_seconds=suffix,total_native_seconds=prefix+suffix,
            total_native_seconds_is_lower_bound=unknown>0,unknown_suffix_costs=unknown,
            labels_observed=sum(row['status']=='complete' for row in rows),details=rows,
            new_initialization_updates=0,prefix_training_updates=0,suffix_updates=0,
            labels_fed_back=0,new_fits=0,new_neural_updates=0,native_cap=declaration['cap'])

    def finish_parent(error=None):
        trace_audit = None
        if trace is not None:
            trace.close()
            trace_audit = audit_trace(trace.path,0,trace.observed_issues)
        spent = spent_budget()
        checks = dict(three_fixed_futures=[row['future_id'] for row in pairs] == [row[0] for row in FUTURES],
            pairs_complete=all(row['all_pass'] for row in pairs),exact_root=bool(root_checks) and all(root_checks.values()),
            exact_prefix=prefix_reproduction is not None and prefix_reproduction['all_pass'],
            prefix_persisted=trace_audit is not None and trace_audit['all_pass'],
            parent_stopped_at_root=stream is not None and stream.now == declaration['time'],
            native_budget=spent['total_native_seconds'] <= declaration['cap'],
            known_spend=spent['unknown_suffix_costs'] == 0,error_free=error is None)
        value = dict(status='complete' if all(checks.values()) else 'failed',case=case,
            external_seed=external_seed,scenario='stable',root=expected_identity,
            integrity=dict(all_pass=all(checks.values()),checks=checks),root_checks=root_checks,
            prefix_reproduction=prefix_reproduction,trace_audit=trace_audit,pairs=pairs,
            root_ledger=root['issue_ledger'] if root is not None else None,
            frozen_original_root_ledger=expected['root']['issue_ledger'],
            frozen_original_remaining=frozen_original,budget=spent,error=error,
            interpretation='Fixed-root lane-driving RNG sensitivity only; no learning, fit, confidence recalibration or V release.')
        write(root_out/'summary.json',value)
        return value

    try:
        torch.set_num_threads(1)
        controller,mdp,stations,reference = load_reference('stable')
        reference['heldout_external_seed'] = external_seed
        actor_before = controller.snapshot_state()
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl',old_run/'prefix_trace.jsonl')
        history = CorridorHistory(EI)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000,seed=external_seed,sumo_cfg=isolated_cfg)
        stream = CollectionStream(env,stations,mdp,domain='target',selector=selector,
            delay_seconds=30,execute_transform=False,on_issue=issue,prefix_limit_seconds=12600)
        stream.reset(); stream.enable_prefix_limit()
        context_reader = NativeContextReader(env)
        hook = install_scenario(env,'stable',shock_at=9000,factor=.65,metric_every=300)
        last_progress = time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy); stream.release_intervals()
            if time.monotonic()-last_progress >= 45:
                progress('frozen_suffix' if branch else 'frozen_prefix_reproduction')
                last_progress = time.monotonic()
        if branch is None:
            raise RuntimeError('Fixed root not reached; no replacement')
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
            case=case,external_seed=external_seed,scenario='stable',role=role,future_id=future_id,future_seed=future_seed,
            executed_action=choice[role+'_action'],executed_hold_seconds=hold,final_clock=stream.now,
            suffix_updates=0,suffix_admitted_labels=0,remaining=remaining,physical=observation,
            rng=rng_receipt,rng_usage=rng_usage,trace_audit=trace_audit,
            tail_boundary=next((row for row in hook.metrics if row['time_seconds']==18300.),None),
            execution_wall=dict(pid=os.getpid(),native_started_wall=native_started_wall,native_finished_wall=time.time()))
        write(out/'summary.json',value)
        if not all(checks.values()):
            raise RuntimeError('Native future child integrity failed')
        progress('complete'); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except ParentFinished:
        return finish_parent()
    except Exception as error:
        value = dict(type=type(error).__name__,message=str(error),role=ROLES.get(branch),
            future_id=future_id,native_time=stream.now if stream else None,labels_fed_back=0)
        if trace is not None:
            trace.close()
        if branch is not None:
            value['suffix_native_seconds'] = max(0.,stream.now-root['started_at'])
        else:
            value['consumed_budget'] = spent_budget()
        write(out/'failure.json',value)
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        if not (root_out/'summary.json').exists():
            finish_parent(error=value)
        raise
    finally:
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=tuple(CASES),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--libsumo-library',type=Path,required=True)
    parser.add_argument('--sumo-cfg',type=Path)
    args = parser.parse_args()
    run(args.case,args.out,args.adapter,args.libsumo_library,args.sumo_cfg)
