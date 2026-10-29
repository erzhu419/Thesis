"""Replay v18 exactly, then intervene on the post-fork execution mapping only."""
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
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedTrainer
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld
from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import counters, train_packet
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import ActionResponseMemory
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector
from phase_stream import PhaseControlledStream, PhasePolicy, tail_action
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_state_response_v18_20261022.budget import (
    PairPacketBudget, paired_label_ids, transition_to_dict, transition_from_dict)
from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import DirectDifferenceReadout
from experiments.mcwm_closed_loop_state_response_v18_20261022.candidate import StateResponseReadout, issue_state_features


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


def saved_command(record):
    """Replay the command saved in an acquisition, not an unsaved plan field."""
    return np.asarray(record['nominal_a'],np.float32)


def decode_pair(record):
    result = {key:value for key,value in record.items() if key != 'outcomes'}
    result['outcomes'] = {arm:dict(truth=value['truth'],transition=transition_from_dict(value['transition']))
        for arm,value in record['outcomes'].items()}
    return result


def cached_pair(source, index):
    return decode_pair(json.loads((source/'pair_cache'/f'{index}.json').read_text()))


def make_readouts(p):
    """Each passive readout receives the same admitted two-slot pairs."""
    return dict(shared=DirectDifferenceReadout(line_only=False,ridge=p['difference_ridge']),
        route=DirectDifferenceReadout(line_only=True,ridge=p['difference_ridge']),
        candidate=StateResponseReadout(ridge=p['difference_ridge']))


def run(args):
    source = args.source.resolve()
    p = json.loads((source/'protocol.json').read_text())['protocol']
    if (args.seed,args.scenario) != (6201,'stable') or args.target_id not in (
            'train/7X/late','train/102X/late','train/705X/late'):
        raise ValueError('Only the three predeclared v18 late training states are allowed')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    env = hook = trace = stream = None
    started = time.monotonic()
    journal = json.loads((source/'acquisitions.json').read_text())['records']
    expected = next((row for row in journal if row['id'] == args.target_id),None)
    cell = dict(seed=args.seed,scenario=args.scenario,mode='branch',target_id=args.target_id,branch=args.branch)

    def progress(phase, **values):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,current=stream.now if stream else 0.,
            total=p['horizon_seconds'],unit='sim_seconds',
            elapsed_seconds=time.monotonic()-started,**values)),flush=True)
    try:
        if expected is None:
            raise RuntimeError('The predeclared source anchor is missing')
        native_path = source/'training_branches'/str(expected['index'])/args.branch/'summary.json'
        native_reference = json.loads(native_path.read_text())
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
        models = make_readouts(p)
        budget = PairPacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl',reference=source/'prefix_trace.jsonl')
        endpoints = ResponseEndpointCollector(delay_seconds=p['feedback_delay_seconds'],duration_seconds=300.)
        by_sequence = {row['selected']['sequence']:row for row in journal}
        decisions = {}
        frozen_counts = fork_evidence = None
        readout_refits = comparisons = 0
        phase = dict(mapping='tail_after_fork_batch',source_native_summary=str(native_path),
            first_tail_sequence=None,first_tail_time=None,native_policy_issues=0,
            tail_policy_issues=0,actor_real_clock=True,actor_clock_max_abs_error=0.,
            actor_clock_error_unit='native_seconds',fork_batch_native=False,tail_formula_exact=True)

        def model_rows(model):
            return len(model.records)

        def all_counts():
            return dict(**counters(learner,world,physical),target_labels=len(budget.selected_sequences),
                **{name+'_pairs':model_rows(model) for name,model in models.items()},readout_refits=readout_refits)

        def prefix():
            return plain(dict(initializer=str(initializer),initial_updates=initial_updates,
                external_seed=args.seed+p['target_seed_offset'],counters=all_counts(),
                receipts=budget.receipts,native_steps=int(env._bridge.steps),issued_count=trace.observed_issues))

        def selector(event):
            if stream.fork is not None:
                return None
            row = by_sequence.get(event['sequence'])
            if row is None:
                return None
            decisions[event['sequence']] = row
            return saved_command(row)

        def fork_selector(event):
            if event['sequence'] != expected['selected']['sequence']:
                return None
            if selected_state(event,stream._snapshot()) != expected['selected']:
                raise RuntimeError('New source state differs before native fork')
            if event['a'].tolist() != expected['nominal_a']:
                raise RuntimeError('New source nominal action differs before native fork')
            choices = action_choices(transform,event['s'])
            if choices is None or plain(choices) != expected['actions']:
                raise RuntimeError('New source action support differs before native fork')
            return choices[args.branch]

        def issue(event):
            nonlocal comparisons,fork_evidence
            event['proposed_a'] = policy.last_proposal.copy()
            at_fork = stream.fork is not None and event['sequence'] == expected['selected']['sequence']
            nominal = dict(event,a=np.asarray(stream.fork['nominal_a'],np.float32)) if at_fork else event
            if stream.fork is not None and not at_fork:
                if stream.tail_policy_active:
                    if phase['first_tail_sequence'] is None:
                        phase['first_tail_sequence'] = event['sequence']
                        phase['first_tail_time'] = event['started_at']
                    phase['tail_formula_exact'] &= bool(np.array_equal(event['a'],tail_action(transform,policy.last_proposal)))
                return
            trace.observe(event,stream._snapshot(),nominal_action=nominal['a'] if at_fork else None)
            row = decisions.get(event['sequence'])
            if row is None:
                if frozen_counts is None:
                    world.register(nominal)
                return
            choices = action_choices(transform,nominal['s'])
            features = plain(issue_state_features(nominal))
            prediction = {name:plain(model.predict(features)) for name,model in models.items()}
            current = dict(id=row['id'],role=row['role'],index=row['index'],status='selected',
                line_id=row['line_id'],window_name=row['window_name'],
                selected=selected_state(nominal,stream._snapshot()),nominal_a=plain(nominal['a']),
                actions=plain(choices),features=features,prediction=prediction,prefix=prefix())
            if current != row:
                differing = [key for key in current if current[key] != row.get(key)]
                raise RuntimeError('New source acquisition differs: '+','.join(differing))
            comparisons += 1
            if row['role'] == 'train':
                budget.reserve(event['sequence'],row['id'])
                for arm,label_id in paired_label_ids(row['index']).items():
                    world.register(dict(nominal,sequence=label_id,a=choices[arm]))
            elif frozen_counts is None:
                world.register(nominal)
            if at_fork:
                phase['fork_batch_native'] = not stream.tail_policy_active
                endpoints.register(event,stream._snapshot(),role='heldout',
                    hold_seconds=0. if args.branch == 'low' else 7.5)
                fork_evidence = current
                return
            if row['role'] == 'train':
                budget.add_pair(cached_pair(source,row['index']))

        write(out/'protocol.json',dict(arguments={**vars(args),'out':str(out),
            'source':str(args.source) if args.source else None},protocol=p))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'],seed=args.seed+p['target_seed_offset'])
        stream = PhaseControlledStream(env,stations,contract,domain='target',selector=selector,
            fork_selector=fork_selector,fork_duration_seconds=300.,
            endpoints=endpoints,response_memory=ActionResponseMemory(),
            delay_seconds=p['feedback_delay_seconds'],execute_transform=False,on_issue=issue)
        stream.reset()
        stream.attach_native_clock()
        policy = PhasePolicy(learner,transform,stream,p['horizon_seconds'],phase)
        hook = install_scenario(env,args.scenario,shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'],metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        stop = p['horizon_seconds']
        while not stream.done and not stream.cutoff_reached and stream.now < stop:
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
                    readout_refits += 1
                    progress('packet_training',packets=len(budget.receipts),target_labels=len(budget.selected_sequences))
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    frozen_counts = all_counts()
            if time.monotonic()-last_progress >= 45:
                progress('frozen_suffix' if stream.fork is not None else 'native_day')
                last_progress = time.monotonic()
        endpoints.finish(stream.now,reason='exact_native300_cutoff')
        end_counts = all_counts()
        common_checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_budget=len(replay.history['s']) == p['history_rows'] and len(replay.source['s']) == p['source_rows'],
            target_horizon_unchanged=contract.horizon_seconds == p['horizon_seconds'],
            physical_source_fixed=sum(physical.learned_by_line.values()) == 0,
            original_trust_rule=world.trust_updates == len(budget.receipts)*p['policy_updates_per_packet'],
            readouts_admitted_pairs_only=all(model_rows(model) == len(budget.admitted) for model in models.values()),
            no_heldout_fit=all(pair['role'] == 'train' for pair in budget.admitted))
        fork = stream.fork
        record = endpoints.records.get(expected['selected']['sequence'])
        delta = {key:end_counts[key]-fork_evidence['prefix']['counters'][key] for key in end_counts} if fork_evidence else {}
        checks = dict(**common_checks,
            full_native_prefix_exact=trace.comparison_exact and fork is not None and trace.matched_issues == expected['selected']['sequence']+1,
            acquisition_predictions_exact=comparisons == sum(row['selected']['sequence'] <= expected['selected']['sequence'] for row in journal),
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
            global_transition_observed=stream.fork_transition is not None,
            native_reference_complete=native_reference.get('status') == 'complete'
                and native_reference.get('integrity',{}).get('all_pass') is True,
            native_reference_fork_exact=all(native_reference.get(key) == (fork_evidence or {}).get(key)
                for key in ('selected','prefix','prediction','nominal_a','actions','features')),
            native_reference_other_batch_exact=native_reference.get('native',{}).get('other_batch_actions')
                == stream.fork_batch_other_controls,
            tail_only_after_fork_batch=phase['fork_batch_native']
                and phase['native_policy_issues'] >= trace.matched_issues
                and (phase['first_tail_sequence'] is None or phase['first_tail_sequence'] > expected['selected']['sequence']),
            actor_real_clock=phase['actor_real_clock'],tail_formula_exact=phase['tail_formula_exact'],
            no_new_training_labels=len(budget.selected_sequences) == expected['prefix']['counters']['target_labels'])
        phase['tail_batches'] = stream.tail_batches
        summary = dict(**(fork_evidence or {}),cell=cell,source_evidence_id=args.target_id,
            fixed_zero_prediction=dict(dt=0.,wait=0.,ride=0.,cost=0.),
            execution=dict(requested_hold_seconds=0. if args.branch == 'low' else 7.5,
                executed_hold_seconds=record.hold_seconds if record else None),
            truth=record.endpoints if record else {},
            global_transition=transition_to_dict(stream.fork_transition) if stream.fork_transition else None,
            suffix=dict(seconds=stream.native_suffix_steps,training_delta=delta),
            phase_intervention=phase,
            native=dict(other_batch_actions=stream.fork_batch_other_controls,prefix_matched_issues=trace.matched_issues),
            budget=dict(prefix_sim_seconds=fork['started_at'] if fork else stream.now,
                suffix_sim_seconds=stream.native_suffix_steps,replay_target_labels=len(budget.selected_sequences),
                replay_policy_updates=learner.total_updates-initial_updates,truth_labels_enter_training=0,
                new_target_training_labels=0,new_diagnostic_action_outcomes=1,reused_native_action_outcomes=1))
        summary.update(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),wall_seconds=time.monotonic()-started)
        write(out/'summary.json',plain(summary))
        progress(summary['status'])
        return summary
    except Exception as error:
        write(out/'failure.json',dict(status='failed',cell=cell,type=type(error).__name__,
            message=str(error),time=stream.now if stream else None))
        raise
    finally:
        if trace is not None:
            trace.close()
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('branch',),required=True)
    parser.add_argument('--seed',type=int,choices=(6201,),required=True)
    parser.add_argument('--scenario',choices=('stable',),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target-id',required=True)
    parser.add_argument('--branch',choices=('low','high'),required=True)
    parser.add_argument('--preflight',action='store_true')
    run(parser.parse_args())
