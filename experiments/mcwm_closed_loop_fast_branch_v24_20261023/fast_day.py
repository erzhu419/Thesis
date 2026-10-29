"""v23 science with live-state paired suffixes; day-only execution benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from bootstrap import HERE, formal, load_learner, write
import numpy as np
import torch
from stream import create_target_env
from experiments.mcwm_closed_loop_formal_20261020.scenarios import UPSTREAM_STOP, install_scenario
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedTrainer
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld
from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import counters, train_packet, generalized
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import ActionResponseMemory
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_state_response_v18_20261022.budget import PairPacketBudget, paired_label_ids, transition_to_dict, transition_from_dict
from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import DirectDifferenceReadout, issue_features
from candidate import HistoryResponseReadout
from history import NativeRunHistory, future_services
from shift_model import ServiceShiftReadout, shift_targets
from sampling import StratifiedPairSampler
from fork_state import fork_children

FORK_CHILD = False


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def selected_state(event, snapshot):
    result = {key: event[key] for key in ('sequence','started_at','line_id','vehicle_id',
        'trip_id','station_id','s','raw_s','event')}
    result['issue_ledger'] = {key: float(getattr(snapshot, key)) for key in (
        'time_seconds','waiting_passengers','in_vehicle_passengers',
        'accrued_wait_seconds','accrued_in_vehicle_seconds')}
    return plain(result)


def action_choices(transform, state):
    calibration = ControlledCalibration.__new__(ControlledCalibration)
    calibration.transform = transform
    choices = {}
    for arm, seconds in (('low',0.),('high',7.5)):
        choice = calibration._action_for_hold(state, seconds)
        if choice is None:
            return None
        choices[arm] = choice[1]
    return choices


def planned_cells(p, seed):
    """Window/line choices depend on clocks and action support, never outcomes."""
    rng = np.random.default_rng(seed+p['calibration_seed_offset'])
    train, heldout = [], []
    index = 0
    for line in p['target_lines']:
        windows = p.get('training_windows_by_line',{}).get(line,p['training_windows'])
        order = []
        while len(order) < len(windows):
            order.extend(('low','high') if rng.integers(2) == 0 else ('high','low'))
        for i, window in enumerate(windows):
            train.append(dict(id=f"train/{line}/{window['name']}",role='train',line_id=line,
                window_name=window['name'],window=window,index=index,nominal_arm=order[i]))
            index += 1
        for window in p['heldout_windows']:
            heldout.append(dict(id=f"heldout/{line}/{window['name']}",role='heldout',line_id=line,
                window_name=window['name'],window=window,index=None))
    return train+heldout


def encode_pair(record):
    result = {key: plain(value) for key,value in record.items() if key != 'outcomes'}
    result['outcomes'] = {arm:dict(truth=plain(value['truth']),
        transition=transition_to_dict(value['transition'])) for arm,value in record['outcomes'].items()}
    return result


def live_pair(record, values):
    """Keep v23 contrast and two-slot identities after the live siblings finish."""
    low, high = values['low'], values['high']
    for value in (low, high):
        if value['status'] != 'complete' or not value['integrity']['all_pass']:
            raise RuntimeError('Live training branch failed native integrity')
        for key in ('selected','prefix','prediction','service_history','service_prediction'):
            if value[key] != record[key]:
                raise RuntimeError('Live training branch changed common '+key)
    if low['native']['other_batch_actions'] != high['native']['other_batch_actions']:
        raise RuntimeError('Live sibling batch actions differ')
    return dict(id=record['id'],index=record['index'],role='train',
        source_sequence=record['selected']['sequence'],issued_at=record['selected']['started_at'],
        line_id=record['line_id'],features=record['features'],
        outcomes={arm:dict(truth=value['truth']['fixed300'],
            transition=transition_from_dict(value['global_transition'])) for arm,value in values.items()},
        service_history=record['service_history'],
        service_targets=shift_targets(low['service_shift'],high['service_shift']))


def branch_budget(values):
    keys = ('prefix_sim_seconds','suffix_sim_seconds','replay_target_labels','replay_policy_updates')
    return {key:sum(value.get('budget',{}).get(key,0) for value in values) for key in keys}


def load_protocol():
    """Preflight uses the same initialization, data budget and clock as the pilot."""
    return json.loads((HERE/'protocol.json').read_text())


def make_readouts(p, *, service_only=False):
    """Each passive readout receives the same admitted two-slot pairs."""
    result = dict(shared=DirectDifferenceReadout(line_only=False,ridge=p['difference_ridge']),
        route=DirectDifferenceReadout(line_only=True,ridge=p['difference_ridge']))
    if not service_only:
        result['candidate'] = HistoryResponseReadout(ridge=p['difference_ridge'])
    return result


def design_report(p, plans, records, admitted):
    """Coverage and label age are separate from integrity of observed evidence."""
    expected_train = [row for row in plans if row['role'] == 'train']
    expected_heldout = [row for row in plans if row['role'] == 'heldout']
    selected_train = [row for row in records if row['role'] == 'train']
    selected_heldout = [row for row in records if row['role'] == 'heldout']
    selected_ids = {row['id'] for row in records}
    admitted_ids = {row['id'] for row in admitted}
    coverage = dict(training_expected=len(expected_train),training_selected=len(selected_train),
        training_admitted=len(admitted),heldout_expected=len(expected_heldout),
        heldout_selected=len(selected_heldout),by_line={})
    def clocks(pairs):
        return dict(last_pair_issued_at=max((row['issued_at'] for row in pairs),default=None),
            last_pair_admitted_at=max((row['admitted_at'] for row in pairs),default=None))
    freshness = dict(**clocks(admitted),by_line={},heldout={})
    for line in p['target_lines']:
        train = [row for row in expected_train if row['line_id'] == line]
        heldout = [row for row in expected_heldout if row['line_id'] == line]
        pairs = [row for row in admitted if row['line_id'] == line]
        coverage['by_line'][line] = dict(training_expected=len(train),
            training_selected=sum(row['line_id'] == line for row in selected_train),
            training_admitted=len(pairs),heldout_expected=len(heldout),
            heldout_selected=sum(row['line_id'] == line for row in selected_heldout),
            missing_training_windows=[row['window_name'] for row in train if row['id'] not in selected_ids],
            unadmitted_training_windows=[row['window_name'] for row in train if row['id'] not in admitted_ids],
            missing_heldout_windows=[row['window_name'] for row in heldout if row['id'] not in selected_ids])
        freshness['by_line'][line] = clocks(pairs)
    by_id = {row['id']:row for row in selected_heldout}
    for plan in expected_heldout:
        row = by_id.get(plan['id'])
        last = freshness['by_line'][plan['line_id']]['last_pair_admitted_at']
        at = row['selected']['started_at'] if row else None
        freshness['heldout'][plan['id']] = dict(line_id=plan['line_id'],window_name=plan['window_name'],
            status='selected' if row else 'selection_missing',at=at,
            age_since_last_line_pair_admitted=at-last if at is not None and last is not None else None)
    coverage['all_pass'] = (len(selected_train) == len(expected_train) == len(admitted)
        and {row['id'] for row in expected_train} == {row['id'] for row in selected_train} == admitted_ids
        and len(selected_heldout) == len(expected_heldout)
        and {row['id'] for row in expected_heldout} == {row['id'] for row in selected_heldout})
    return coverage,freshness


def run(args):
    global FORK_CHILD
    p = load_protocol()
    if args.preflight:
        p['target_seed_offset'] = p['preflight']['target_seed_offset']
    out = args.out.resolve()
    day_root = out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    env = hook = trace = stream = history = None
    started = time.monotonic()
    is_branch, source, journal, expected = False, out, [], None
    cell = dict(seed=args.seed,scenario=args.scenario,mode=args.mode)

    def progress(phase, **values):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,current=stream.now if stream else 0.,
            total=p['horizon_seconds'],unit='sim_seconds',
            elapsed_seconds=time.monotonic()-started,**values)),flush=True)
    try:
        metadata = json.loads((HERE/p['reference_inputs']/'history_metadata.json').read_text())
        contract = formal.TransitMDPConfigV7(**metadata['contract'])
        transform = ExecutionAction(contract)
        learner,replay,_,contract,stations,initializer = load_learner(args.seed,
            trainer_class=ProbedTrainer,arm='model',action_transform=transform)
        initial_updates = learner.total_updates
        physical = LocalIntervalCalibration(json.loads((HERE/p['reference_source_intervals']).read_text()),
            contract,shock_at=p['shock_at_seconds'],delay_seconds=p['feedback_delay_seconds'],
            learning_end_seconds=0.,min_source_per_line=p['source_intervals_per_line_minimum'],
            max_source_per_line=p['source_intervals_per_line_limit'])
        progress('initial_world_fit')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+20000)
            world = PairedWorld(replay,seed=args.seed,updates=p['model_init_updates'],
                action_transform=transform,physical=physical,enabled=False)
        world.name,world.context_enabled,world.paired_probe_deadlines = 'action_model',False,[]
        learner.world = world
        models = make_readouts(p,service_only=args.preflight)
        service_models = {name:ServiceShiftReadout(use_history=name=='history',ridge=p['difference_ridge'])
            for name in ('shared','history')}
        clock_audit = dict(actor_real_clock=True,actor_clock_max_abs_error=0.,native_policy_issues=0)
        budget = PairPacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        endpoints = ResponseEndpointCollector(delay_seconds=p['feedback_delay_seconds'],duration_seconds=300.)
        plans = planned_cells(p,args.seed)
        sampler = StratifiedPairSampler(plans,p['target_lines'])
        pre_recorded, completed_pairs = {}, {}
        chosen, decisions, records = set(),{},[]
        cache_records, child_records = [],[]
        last_proposal = frozen_counts = fork_evidence = None
        readout_refits = comparisons = 0
        freeze_clock = None

        def model_rows(model):
            return len(model.records)

        def all_counts():
            return dict(**counters(learner,world,physical),target_labels=len(budget.selected_sequences),
                **{name+'_pairs':model_rows(model) for name,model in models.items()},
                **{'service_'+name+'_pairs':len(model.records) for name,model in service_models.items()},readout_refits=readout_refits)

        def prefix():
            return plain(dict(initializer=str(initializer),initial_updates=initial_updates,
                external_seed=args.seed+p['target_seed_offset'],counters=all_counts(),
                receipts=budget.receipts,native_steps=int(env._bridge.steps),issued_count=trace.observed_issues))

        def policy(state):
            nonlocal last_proposal
            event_time = float(env._current_events[0].sim_time)
            expected_clock = np.float32(np.float32(event_time)/p['horizon_seconds'])
            clock_audit['actor_real_clock'] &= bool(state[0] == expected_clock)
            clock_audit['actor_clock_max_abs_error'] = max(clock_audit['actor_clock_max_abs_error'],
                abs(float(state[0])-float(expected_clock))*p['horizon_seconds'])
            clock_audit['native_policy_issues'] += 1
            last_proposal = learner.act(state[None],deterministic=True)[0]
            return transform.numpy(state,last_proposal)

        def selector(event):
            if stream.fork is not None:
                return None
            if str(event['station_id']) != UPSTREAM_STOP:
                return None
            for plan in plans:
                if plan['id'] in chosen or plan['line_id'] != event['line_id']:
                    continue
                window = plan['window']
                if not window['start'] <= event['started_at'] < window['end']:
                    continue
                if plan['role'] == 'heldout' and frozen_counts is None:
                    continue
                choices = action_choices(transform,event['s'])
                if choices is None:
                    continue
                sampling = None
                if plan['role'] == 'train':
                    if plan['window_name'] != 'bootstrap' and not any(
                            r['line_id'] == event['line_id'] for r in budget.admitted):
                        continue
                    sampling = sampler.consider(plan, issued_at=event['started_at'],
                        line_id=event['line_id'], features=plain(issue_features(event)),
                        history=history.snapshot(event))
                    if sampling is None:
                        continue
                chosen.add(plan['id'])
                row = dict(plan,actions=plain(choices),sampling=sampling)
                decisions[event['sequence']] = row
                return choices[plan['nominal_arm']] if plan['role'] == 'train' else None
            return None

        def acquisition(nominal, row):
            choices = action_choices(transform,nominal['s'])
            features = plain(issue_features(nominal))
            history_snapshot = history.snapshot(nominal)
            prediction = {name:plain(model.predict(features,history_snapshot) if name=='candidate'
                else model.predict(features)) for name,model in models.items()}
            service_prediction = {name:plain(model.predict(features,history_snapshot))
                for name,model in service_models.items()}
            return dict(id=row['id'],role=row['role'],index=row['index'],status='selected',
                line_id=row['line_id'],window_name=row['window_name'],
                selected=selected_state(nominal,stream._snapshot()),nominal_a=plain(nominal['a']),
                actions=plain(choices),features=features,prediction=prediction,prefix=prefix(),
                service_history=plain(history_snapshot),service_prediction=service_prediction,
                sampling=row.get('sampling'))

        def fork_selector(event):
            row = decisions.get(event['sequence'])
            if row is None or row['role'] != 'train':
                return None
            # Record the common nominal issue exactly once before copying its
            # native/Python state. on_issue resumes this same half-issued event.
            event['proposed_a'] = last_proposal.copy()
            trace.observe(event,stream._snapshot())
            current = acquisition(event,row)
            pre_recorded[event['sequence']] = current
            pair_root = day_root/'training_branches'/str(row['index'])

            def prepare_child(arm):
                global FORK_CHILD
                nonlocal out,cell,started,native_started_wall,is_branch,source,expected,journal
                FORK_CHILD = True
                is_branch, source, expected = True, day_root, current
                args.mode,args.branch,args.target_id = 'branch',arm,current['id']
                out = pair_root/arm
                cell = dict(seed=args.seed,scenario=args.scenario,mode='branch',
                    target_id=current['id'],branch=arm)
                decisions[event['sequence']] = current
                journal = records+[current]
                # Earlier acquisitions and the current trace row are inherited,
                # rather than re-read or independently replayed in this child.
                trace.writer.close()
                trace.path = out/'suffix_trace.jsonl'
                trace.writer = trace.path.open('x',encoding='utf-8')
                if trace.reference is not None:
                    trace.reference.close()
                trace.reference = None
                logfd = os.open(pair_root/(arm+'.log'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                os.dup2(logfd,1);os.dup2(logfd,2);os.close(logfd)
                started,native_started_wall = time.monotonic(),time.time()

            result = fork_children(pair_root,trace,lambda:stream.now,all_counts,prepare_child)
            if isinstance(result,str):
                return np.asarray(current['actions'][result],np.float32)
            values = {arm:json.loads((pair_root/arm/'summary.json').read_text())
                for arm in ('low','high')}
            completed_pairs[event['sequence']] = values
            receipt = dict(pair_id=current['id'],launches={arm:dict(
                pid=result['child_pids'][arm],started_wall=result['children'][arm]['native_started_wall'],
                ended_wall=result['children'][arm]['native_finished_wall']) for arm in ('low','high')},
                native_wall=result['children'],overlap_seconds=result['overlap_seconds'],
                parent_native_before=result['native_before'],parent_native_after=result['native_after'],
                all_pass=result['all_pass'],prefix_execution='inherited_live_parent_state')
            write(pair_root/'parallel_execution.json',receipt)
            return None

        def issue(event):
            nonlocal comparisons,fork_evidence
            event['proposed_a'] = last_proposal.copy()
            at_fork = stream.fork is not None and event['sequence'] == expected['selected']['sequence'] if is_branch else False
            nominal = dict(event,a=np.asarray(stream.fork['nominal_a'],np.float32)) if at_fork else event
            if stream.fork is not None and not at_fork:
                return
            if event['sequence'] not in pre_recorded:
                trace.observe(event,stream._snapshot(),nominal_action=nominal['a'] if at_fork else None)
            row = decisions.get(event['sequence'])
            if row is None:
                if frozen_counts is None:
                    world.register(nominal)
                return
            current = acquisition(nominal,row)
            choices = {arm:np.asarray(action,np.float32) for arm,action in current['actions'].items()}
            if event['sequence'] in pre_recorded and current != pre_recorded[event['sequence']]:
                raise RuntimeError('Live acquisition changed after child suffix execution')
            if is_branch:
                if current != row:
                    differing = [key for key in current if current[key] != row.get(key)]
                    raise RuntimeError('New source acquisition differs: '+','.join(differing))
                comparisons += 1
            records.append(current)
            if row['role'] == 'train':
                budget.reserve(event['sequence'],row['id'])
                for arm,label_id in paired_label_ids(row['index']).items():
                    world.register(dict(nominal,sequence=label_id,a=choices[arm]))
            elif frozen_counts is None:
                world.register(nominal)
            if at_fork:
                endpoints.register(event,stream._snapshot(),role='heldout',
                    hold_seconds=0. if args.branch == 'low' else 7.5)
                fork_evidence = current
                return
            if not is_branch:
                write(out/'acquisitions.json',dict(records=records))
                trace.writer.flush()
            if row['role'] == 'train':
                values = completed_pairs.pop(event['sequence'])
                pair = live_pair(current,values)
                pair = budget.add_pair(pair)
                cache_records.append(pair)
                if not is_branch:
                    write(out/'pair_cache'/f"{row['index']}.json",encode_pair(pair))
                    for arm in ('low','high'):
                        directory = out/'training_branches'/str(row['index'])/arm
                        child_records.append(json.loads((directory/'summary.json').read_text()))

        write(out/'protocol.json',dict(arguments={**vars(args),'out':str(out),
            'source':str(args.source) if args.source else None},protocol=p))
        if not is_branch:
            write(out/'acquisitions.json',dict(records=[]))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'],seed=args.seed+p['target_seed_offset'])
        stream = PairedControlledStream(env,stations,contract,domain='target',selector=selector,
            fork_selector=fork_selector,fork_duration_seconds=300.,
            endpoints=endpoints,response_memory=ActionResponseMemory(),
            delay_seconds=p['feedback_delay_seconds'],execute_transform=False,on_issue=issue)
        stream.reset()
        stream.attach_native_clock()
        history = NativeRunHistory(env._bridge)
        hook = install_scenario(env,args.scenario,shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'],metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        native_started_wall = time.time()
        stop = p['horizon_seconds']
        while not stream.done and not stream.cutoff_reached and (is_branch or stream.now < stop):
            released = stream.step(policy)
            stream.release_intervals()
            endpoints.release(stream.now)
            if stream.fork is None:
                budget.ingest(released)
                while budget.due(stream.now):
                    packet,discarded = budget.take(stream.now)
                    valid = [row for row in packet if row.dt > 0 and not row.terminal]
                    excluded = [row.sequence for row in packet if row.dt <= 0 or row.terminal]
                    if not valid:
                        raise RuntimeError('No positive transition in the original update packet')
                    train_packet(learner,replay,world,valid,discarded+excluded,budget.receipts[-1],
                        stream.now,p['model_refit_updates'],contract)
                    for model in models.values():
                        model.accept(budget.last_pairs,stream.now)
                        model.refit()
                    for model in service_models.values():
                        model.accept(budget.last_pairs,stream.now)
                        model.refit()
                    readout_refits += 1
                    progress('packet_training',packets=len(budget.receipts),target_labels=len(budget.selected_sequences))
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    freeze_clock = stream.now
                    frozen_counts = all_counts()
            if time.monotonic()-last_progress >= 45:
                progress('frozen_suffix' if stream.fork is not None else 'native_day')
                last_progress = time.monotonic()
        native_finished_wall = time.time()
        endpoints.finish(stream.now,reason='exact_native300_cutoff' if is_branch else 'day_end')
        b,end_counts = budget.summary(),all_counts()
        common_checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_budget=len(replay.history['s']) == p['history_rows'] and len(replay.source['s']) == p['source_rows'],
            target_horizon_unchanged=contract.horizon_seconds == p['horizon_seconds'],
            physical_source_fixed=sum(physical.learned_by_line.values()) == 0,
            original_trust_rule=world.trust_updates == len(budget.receipts)*p['policy_updates_per_packet'],
            readouts_admitted_pairs_only=all(model_rows(model) == len(budget.admitted) for model in models.values()),
            no_heldout_fit=all(pair['role'] == 'train' for pair in budget.admitted),
            service_models_admitted_pairs_only=all(len(model.records)==len(budget.admitted) for model in service_models.values()),
            service_gate_candidate_not_fitted=(not args.preflight or 'candidate' not in models),
            actor_real_clock=clock_audit['actor_real_clock'])
        if is_branch:
            fork = stream.fork
            record = endpoints.records.get(expected['selected']['sequence'])
            delta = {key:end_counts[key]-fork_evidence['prefix']['counters'][key] for key in end_counts} if fork_evidence else {}
            checks = dict(**common_checks,
                live_prefix_inherited=FORK_CHILD and fork is not None and trace.observed_issues == expected['selected']['sequence']+1,
                current_issue_compared=comparisons == 1,
                selected_evidence_exact=fork_evidence == expected,
                source_receipts_exact=budget.receipts == expected['prefix']['receipts'],
                source_counters_exact=end_counts == expected['prefix']['counters'],
                suffix_zero_training=bool(delta) and all(value == 0 for value in delta.values()),
                native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == 300,
                actual_action_exact=record is not None and record.hold_seconds == (0. if args.branch == 'low' else 7.5)
                    and float(transform.hold_seconds(torch.as_tensor(record.a))[0]) == record.hold_seconds
                    and any(control['sequence'] == record.sequence and control['hold_seconds'] == record.hold_seconds
                        for control in stream.executed_controls),
                fixed300_truth=record is not None and record.endpoints['fixed300']['status'] == 'complete' and record.endpoints['fixed300']['dt'] == 300,
                no_future_calibration=all(item['sequence'] <= expected['selected']['sequence'] for item in stream.executed_controls),
                global_transition_observed=stream.fork_transition is not None)
            summary = dict(**(fork_evidence or {}),cell=cell,source_evidence_id=args.target_id,
                execution=dict(requested_hold_seconds=0. if args.branch == 'low' else 7.5,
                    executed_hold_seconds=record.hold_seconds if record else None),
                truth=record.endpoints if record else {},
                service_shift=future_services(env._bridge,expected['selected'],stream.now),
                global_transition=transition_to_dict(stream.fork_transition) if stream.fork_transition else None,
                suffix=dict(seconds=stream.native_suffix_steps,training_delta=delta),
                native=dict(other_batch_actions=stream.fork_batch_other_controls,
                    prefix_matched_issues=trace.observed_issues),
                budget=dict(prefix_sim_seconds=fork['started_at'] if fork else stream.now,
                    suffix_sim_seconds=stream.native_suffix_steps,replay_target_labels=len(budget.selected_sequences),
                    replay_policy_updates=learner.total_updates-initial_updates,truth_labels_enter_training=0))
            if FORK_CHILD:
                summary['fork_reuse'] = dict(prefix_execution='inherited_not_independently_replayed',
                    parent_prefix_trace=str(day_root/'prefix_trace.jsonl'),
                    parent_recorded_issues=trace.observed_issues,
                    parent_verified_prefix_seconds=fork['started_at'],
                    inherited_target_labels=len(budget.selected_sequences),
                    inherited_policy_updates=learner.total_updates-initial_updates)
                summary['budget'].update(prefix_sim_seconds=0,replay_target_labels=0,replay_policy_updates=0)
        else:
            checks = dict(**common_checks,target_budget_exact=len(budget.selected_sequences) == p['target_record_budget'],
                policy_update_budget=learner.total_updates-initial_updates == p['policy_update_budget'],
                native_horizon=stream.now == p['horizon_seconds'],
                post_budget_zero_training=frozen_counts is not None and end_counts == frozen_counts,
                main_trace_complete=trace.observed_issues == stream.issued_count,
                training_children_integrity=all(value['integrity']['all_pass'] for value in child_records),
                factual_pair_rows_not_selected=all(row['selected']['sequence'] not in budget.selected_sequences for row in records if row['role']=='train'))
            missing = [dict(id=row['id'],role=row['role'],line_id=row['line_id'],window_name=row['window_name'],status='selection_missing')
                for row in plans if row['id'] not in chosen]
            training_pairs = [dict(id=pair['id'],index=pair['index'],role=pair['role'],line_id=pair['line_id'],
                issued_at=pair['issued_at'],source_sequence=pair['source_sequence'],available_at=pair['available_at'],
                admitted_at=pair.get('admitted_at'),admitted_deadline=pair.get('admitted_deadline'),
                outcome_ids=paired_label_ids(pair['index']),admitted=pair['id'] in {value['id'] for value in budget.admitted}) for pair in cache_records]
            design_coverage,freshness = design_report(p,plans,records,budget.admitted)
            parallel_pairs = [json.loads((out/'training_branches'/str(pair['index'])/
                'parallel_execution.json').read_text()) for pair in cache_records]
            checks['parallel_pair_execution'] = all(r['all_pass'] for r in parallel_pairs)
            summary = dict(cell=cell,initializer=dict(path=str(initializer),initial_updates=initial_updates,
                    history_rows=len(replay.history['s']),source_rows=len(replay.source['s'])),
                budget=dict(**b,native_seconds=stream.now,policy_updates=learner.total_updates-initial_updates,
                    trust_updates=world.trust_updates,world_updates=world.model_updates,training_stopped_at=freeze_clock),
                receipts=budget.receipts,readouts={name:plain(model.summary()) for name,model in models.items()},
                acquisitions=records,anchors=[row for row in records if row['role']=='heldout'],missing_cells=missing,
                design_coverage=design_coverage,freshness=freshness,
                training_pairs=training_pairs,training_branch_budget=branch_budget(child_records),
                training_branches=[dict(path=str(out/'training_branches'/str(pair['index'])/arm),
                    status=value['status'],budget=value['budget'],integrity=value['integrity'])
                    for index,pair in enumerate(cache_records)
                    for arm,value in zip(('low','high'),child_records[2*index:2*index+2])],
                service_readouts={name:plain(model.summary()) for name,model in service_models.items()},
                history_coverage=dict(counts=dict(history.counts),observed_segments=len(history.rows)),
                cost=dict(full_day=generalized(stream._snapshot()),trajectory='New v21 common action_model flow; all readouts passive'),
                scenario=hook.metadata())
            summary.update(sampling_summary=sampler.report(),parallel_pairs=parallel_pairs)
        summary['execution_wall'] = dict(pid=os.getpid(), native_started_wall=native_started_wall,
            native_finished_wall=native_finished_wall)
        summary['clock_audit'] = clock_audit
        summary.update(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),wall_seconds=time.monotonic()-started)
        write(out/'summary.json',plain(summary))
        if not is_branch:
            write(out/'gate.json',plain({key:summary[key] for key in (
                'status','cell','integrity','budget','design_coverage')}))
        progress(summary['status'])
        return summary
    except Exception as error:
        write(out/'failure.json',dict(status='failed',cell=cell,type=type(error).__name__,
            message=str(error),time=stream.now if stream else None))
        raise
    finally:
        # A fork child must not unlink the parent's bridge runtime files or run
        # inherited native cleanup/atexit handlers. Its address space ends below.
        if FORK_CHILD:
            if trace is not None:
                trace.close()
            sys.stdout.flush();sys.stderr.flush()
        if not FORK_CHILD and history is not None:
            history.close()
        if not FORK_CHILD and trace is not None:
            trace.close()
        if not FORK_CHILD and hook is not None:
            hook.close()
        if not FORK_CHILD and env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('day',),required=True)
    parser.add_argument('--seed',type=int,choices=(6201,6202),required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.set_defaults(source=None,target_id=None,branch=None)
    parser.add_argument('--preflight',action='store_true')
    try:
        run(parser.parse_args())
    except BaseException:
        if FORK_CHILD:
            import traceback
            traceback.print_exc()
            sys.stdout.flush();sys.stderr.flush()
            os._exit(1)
        raise
    if FORK_CHILD:
        sys.stdout.flush();sys.stderr.flush()
        os._exit(0)
