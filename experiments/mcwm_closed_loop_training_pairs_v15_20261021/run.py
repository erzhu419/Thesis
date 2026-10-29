"""Replay a v14 training prefix, intervene once, and freeze every learner for300s."""
import argparse
import json
import os
from pathlib import Path
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
from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import counters, train_packet
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration, CalibrationPacketBudget
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector, ControlledResponseReadout
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.global_comparator import Global58
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import ResidualResponseReadout
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import StateInteractionReadout
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.separated import SeparatedContrastReadout
from experiments.mcwm_closed_loop_assignment_innovation_v14_20261021.innovation import AssignmentInnovationReadout


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def selected_state(event, snapshot):
    result = {key: event[key] for key in ('sequence', 'started_at', 'line_id',
        'vehicle_id', 'trip_id', 'station_id', 's', 'raw_s', 'event')}
    result['issue_ledger'] = {key: float(getattr(snapshot, key)) for key in (
        'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
        'accrued_wait_seconds', 'accrued_in_vehicle_seconds')}
    return plain(result)


def find_selection(manifest, args):
    rows = [row for row in manifest['states'] if row['cell']['seed'] == args.seed
        and row['cell']['scenario'] == args.scenario
        and row['cell']['target_line'] == args.target_line
        and row['cell']['window_name'] == args.window]
    if len(rows) != 1:
        raise ValueError('A branch must identify exactly one planned training-state cell')
    return rows[0]


def nominal_issue(event, action):
    """Predict at the source action while preserving the actual intervened event."""
    return dict(event, a=np.asarray(action, dtype=np.float32).copy())


def choose_prefix_calibration(event, calibration, budget, decisions, *, selected_sequence,
                              already_forked, last_deadline):
    """Replay the original assignment, reserving only non-diagnostic prefix labels."""
    if already_forked or event['started_at'] >= last_deadline:
        return None
    decision = calibration.choose(event, role='train')
    if decision is None:
        return None
    decisions[event['sequence']] = decision
    if event['sequence'] != selected_sequence:
        budget.reserve(event['sequence'])
    return decision['action']


def background_scope_valid(context, event):
    snapshot = context['innovation_background']
    return snapshot['sequence'] == event['sequence'] and snapshot['issued_at'] == event['started_at'] \
        and all(value['max_available_at'] is None or value['max_available_at'] <= event['started_at']
                for value in snapshot['endpoints'].values()) \
        and all(value['max_sequence'] is None or value['max_sequence'] < event['sequence']
                for value in snapshot['endpoints'].values())


def expected_prefix_counts(p, receipts, initial_updates):
    packets = len(receipts)
    labels = sum(row['calibration_records'] for row in receipts)
    return dict(policy=initial_updates+packets*p['policy_updates_per_packet'],
        world=2*packets*p['model_refit_updates'], trust=packets*p['policy_updates_per_packet'],
        physical=0, global58_refits=packets,
        global58_labels=sum(row['learned_global_records'] for row in receipts),
        response_refits=packets, legacy_labels=labels, residual_labels=labels,
        interaction_labels=labels, separated_labels=labels, innovation_labels=labels,
        innovation_line_labels=labels)


def compact_innovation(model):
    return dict(endpoint_coverage=model.endpoint_coverage(), diagnostics={endpoint: {
        key: value.get(key) for key in ('status', 'background_rows', 'first_rows', 'second_rows',
            'first_coverage', 'cold_start_first_rows', 'snapshot_scope_valid', 'action_design',
            'innovation_error_rmse', 'score_error_rmse')}
        for endpoint, value in model.innovation_diagnostics.items()})


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.preflight:
        p.update(p['preflight'])
    out, started = args.out.resolve(), time.monotonic()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    env = hook = stream = trace = None
    cell = dict(seed=args.seed, scenario=args.scenario, target_line=args.target_line,
        window=args.window, branch=args.branch)

    def progress(stage, now=0., **fields):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=stage, current=now,
            total=p['horizon_seconds'], unit='sim_seconds', elapsed_seconds=time.monotonic()-started,
            **fields)), flush=True)

    try:
        manifest = json.loads(args.manifest.read_text())
        expected = find_selection(manifest, args)
        if expected['status'] != 'selected':
            summary = dict(status='selection_missing', state_id=expected['state_id'], cell=cell,
                integrity=dict(all_pass=False, checks=dict(selection_available=False)),
                budget=dict(prefix_sim_seconds=0, suffix_sim_seconds=0,
                    replay_target_labels=0, replay_policy_updates=0, truth_labels_enter_training=0),
                reason=expected.get('reason', 'No eligible training issue in the planned window'))
            write(out/'summary.json', summary)
            return summary
        old_day = (HERE/p['reference_v14_root']/p['reference_v14_preflight_directory']/'day'
            if args.preflight else HERE/p['reference_v14_root']/'runs'/args.scenario/f'seed{args.seed}')
        old_reference = json.loads((old_day/'summary.json').read_text())
        old_predictions = {row['sequence']: row['prediction'] for row in old_reference['evaluations']}
        trace = NativePrefixTrace(out/'prefix_trace.jsonl', reference=old_day/'prefix_trace.jsonl')
        progress('load_initializer')
        metadata = json.loads((HERE/p['reference_inputs']/'history_metadata.json').read_text())
        contract = formal.TransitMDPConfigV7(**metadata['contract'])
        transform = ExecutionAction(contract)
        learner, replay, _, contract, stations, initializer = load_learner(args.seed,
            trainer_class=ProbedTrainer, arm='model', action_transform=transform)
        initial_updates = learner.total_updates
        source = json.loads((HERE/p['reference_source_intervals']).resolve().read_text())
        physical = LocalIntervalCalibration(source, contract, shock_at=p['shock_at_seconds'],
            delay_seconds=p['feedback_delay_seconds'], learning_end_seconds=0.,
            min_source_per_line=p['source_intervals_per_line_minimum'],
            max_source_per_line=p['source_intervals_per_line_limit'])
        progress('initial_world_fit')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+20000)
            world = PairedWorld(replay, seed=args.seed, updates=p['model_init_updates'],
                action_transform=transform, physical=physical, enabled=False)
        world.name, world.context_enabled, world.paired_probe_deadlines = 'action_model', False, []
        learner.world = world
        response = Global58(world, physical, ridge=p['response_ridge'])
        legacy = ControlledResponseReadout(ridge=p['response_ridge'])
        residual = ResidualResponseReadout(ridge=p['response_ridge'])
        interaction = StateInteractionReadout(ridge=p['response_ridge'])
        separated = SeparatedContrastReadout(ridge=p['response_ridge'])
        innovation = AssignmentInnovationReadout(ridge=p['response_ridge'])
        innovation_line = AssignmentInnovationReadout(ridge=p['response_ridge'], line_only=True)
        models = (legacy, residual, interaction, separated, innovation, innovation_line)
        endpoints = ResponseEndpointCollector(delay_seconds=p['feedback_delay_seconds'],
            duration_seconds=p['response_endpoint_seconds'])
        calibration = ControlledCalibration(p, args.seed, transform, station_id=UPSTREAM_STOP)
        budget = CalibrationPacketBudget(p)
        decisions, predictions = {}, {}
        last_proposal = fork_evidence = None
        response_refits = old_prediction_matches = 0
        selected_sequence = expected['selected']['sequence']

        def all_counts():
            return dict(**counters(learner, world, physical),
                global58_refits=response.refits, global58_labels=response.labels,
                response_refits=response_refits, legacy_labels=len(legacy.records),
                residual_labels=len(residual.records), interaction_labels=len(interaction.records),
                separated_labels=len(separated.records), innovation_labels=len(innovation.records),
                innovation_line_labels=len(innovation_line.records))

        def prefix():
            return plain(dict(initializer=str(initializer), initial_updates=initial_updates,
                external_seed=args.seed+p['target_seed_offset'], counters=all_counts(),
                receipts=budget.receipts, freeze_clock=stream.fork['started_at'],
                native_steps=int(stream.env._bridge.steps), issued_count=trace.observed_issues))

        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def selector(event):
            return choose_prefix_calibration(event, calibration, budget, decisions,
                selected_sequence=selected_sequence, already_forked=stream.fork is not None,
                last_deadline=p['feedback_last_deadline'])

        def fork_selector(event):
            if event['sequence'] != selected_sequence:
                return None
            if event['sequence'] not in decisions:
                raise RuntimeError('Chosen source training issue lost its calibration assignment')
            if selected_state(event, stream._snapshot()) != expected['selected']:
                raise RuntimeError('Source training-state mismatch before intervention')
            if not np.array_equal(event['a'], np.asarray(expected['nominal_a'], np.float32)):
                raise RuntimeError('Source nominal command mismatch')
            for arm, seconds in (('low', 0.), ('high', 7.5)):
                choice = calibration._action_for_hold(event['s'], seconds)
                if choice is None or not np.array_equal(choice[1],
                        np.asarray(expected['executable'][arm]['action'], np.float32)):
                    raise RuntimeError('Source jointly executable command lost support')
            seconds = 0. if args.branch == 'low' else 7.5
            return calibration._action_for_hold(event['s'], seconds)[1]

        def issue(event):
            nonlocal fork_evidence, old_prediction_matches
            event['proposed_a'] = last_proposal.copy()
            is_fork = stream.fork is not None and event['sequence'] == selected_sequence
            reference_event = nominal_issue(event, stream.fork['nominal_a']) if is_fork else event
            context = response.context(reference_event)
            context['issue_ledger'] = selected_state(reference_event, stream._snapshot())['issue_ledger']
            if stream.fork is None or is_fork:
                world.register(reference_event)
                trace.observe(event, stream._snapshot(),
                    nominal_action=stream.fork['nominal_a'] if is_fork else None)
            decision = decisions.get(event['sequence'])
            if decision is not None:
                success_index = calibration.line_successes[('train', event['line_id'])]-1
                context.update(assignment_success_index=success_index,
                    assignment_position=success_index % 2, assignment_block_index=success_index // 2)
                before = all_counts()
                global_prediction = response.predict_issue(reference_event, context,
                    seed=args.seed+300000+event['sequence'])
                context['global_baselines'] = {name: global_prediction['arms']['base_analytic'][name]
                    for name in ('low', 'high')}
                innovation.capture_context(reference_event, context)
                saved = plain(dict(global_comparator=global_prediction,
                    legacy=legacy.predict_issue(reference_event, context),
                    baseline=residual.baseline_predict_issue(reference_event, context),
                    residual=residual.predict_issue(reference_event, context),
                    interaction=interaction.predict_issue(reference_event, context),
                    interaction_diag=interaction.contrast_diagnostic(reference_event, context),
                    separated=separated.predict_issue(reference_event, context),
                    separated_no_cost=separated.predict_issue(reference_event, context, omit_cost_rate=True),
                    separated_diag=separated.contrast_diagnostic(reference_event, context),
                    separated_no_cost_diag=separated.contrast_diagnostic(reference_event, context, omit_cost_rate=True),
                    innovation=innovation.predict_issue(reference_event, context),
                    innovation_line=innovation_line.predict_issue(reference_event, context),
                    innovation_diag=innovation.contrast_diagnostic(reference_event, context),
                    innovation_line_diag=innovation_line.contrast_diagnostic(reference_event, context)))
                previous = old_predictions.get(event['sequence'])
                if previous is None or saved != previous:
                    raise RuntimeError('Read-only v14 prediction mismatch at issue '+str(event['sequence']))
                old_prediction_matches += 1
                hold = (0. if args.branch == 'low' else 7.5) if is_fork else decision['executed_hold_seconds']
                endpoints.register(event, stream._snapshot(), context,
                    role='heldout' if is_fork else 'train', hold_seconds=hold,
                    assignment=decision if not is_fork else dict(assignment=args.branch,
                        target_hold_seconds=hold, executed_hold_seconds=hold))
                predictions[event['sequence']] = saved
                if all_counts() != before:
                    raise RuntimeError('Passive prediction changed parameter counters')
                if is_fork:
                    fork_evidence = dict(selected=selected_state(reference_event, stream._snapshot()),
                        prefix=prefix(), prediction=saved, context=plain(context),
                        coverage=dict(calibration=budget.summary(),
                            innovation=compact_innovation(innovation),
                            innovation_line=compact_innovation(innovation_line)))
                    if saved != expected['prediction']:
                        raise RuntimeError('Selected source prediction differs from manifest')
            # The history follows the actual intervention in the suffix; no fit
            # or future controlled assignment is allowed after this point.
            response.register(event, context)

        write(out/'protocol.json', dict(arguments={**vars(args), 'out': str(out),
            'manifest': str(args.manifest)}, protocol=p))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.seed+p['target_seed_offset'])
        stream = PairedControlledStream(env, stations, contract, domain='target',
            selector=selector, fork_selector=fork_selector,
            fork_duration_seconds=p['paired_suffix_seconds'], endpoints=endpoints,
            response_memory=response.memory, delay_seconds=p['feedback_delay_seconds'],
            execute_transform=False, on_issue=issue)
        stream.reset()
        stream.attach_native_clock()
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        while not stream.done and not stream.cutoff_reached:
            released = stream.step(policy)
            stream.release_intervals()
            ready = endpoints.release(stream.now)
            if stream.fork is None:
                budget.ingest(released)
                budget.ingest_calibration([r for r in ready if r.role == 'train'])
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    valid = [r for r in packet if r.dt > 0 and not r.terminal]
                    excluded = [r.sequence for r in packet if r.dt <= 0 or r.terminal]
                    if not valid:
                        raise RuntimeError('Prefix packet has no valid transition')
                    response.accept(valid, discarded+excluded, stream.now)
                    train_packet(learner, replay, world, valid, discarded+excluded,
                        budget.receipts[-1], stream.now, p['model_refit_updates'], contract)
                    response.refit()
                    for model in models:
                        model.accept(budget.last_calibration_records, stream.now)
                        model.refit()
                    response_refits += 1
                    progress('prefix_training', stream.now, packets=len(budget.receipts),
                        labels=budget.summary()['target_labels'])
            if time.monotonic()-last_progress >= 45:
                progress('frozen_native_suffix' if stream.fork is not None else 'prefix_replay', stream.now)
                last_progress = time.monotonic()
        endpoints.finish(stream.now, reason='training_period_frozen_native_300_second_cutoff')
        b, end_counts = budget.summary(), all_counts()
        fork = stream.fork
        record = endpoints.records.get(selected_sequence)
        delta = {key: end_counts[key]-fork_evidence['prefix']['counters'][key]
            for key in end_counts} if fork_evidence is not None else {}
        source_receipts = expected['source']['prefix_receipts']
        expected_counts = expected_prefix_counts(p, source_receipts, initial_updates)
        source_selected = set(expected['source']['selected_calibration_sequence_ids_before_issue'])
        selected_ids = {row.sequence for row in budget.admitted}
        first_scope = all(background_scope_valid(row.context, dict(sequence=row.sequence,
            started_at=row.started_at)) for row in budget.admitted)
        checks = dict(source_selection_present=True,
            initializer_50000=initial_updates == p['initializer_updates'],
            history_source_budget=len(replay.history['s']) == p['history_rows']
                and len(replay.source['s']) == p['source_rows'],
            target_horizon_unchanged=contract.horizon_seconds == p['horizon_seconds'],
            full_native_prefix_exact=trace.comparison_exact and fork is not None
                and trace.matched_issues == selected_sequence+1,
            v14_old_predictions_exact=old_prediction_matches == len(predictions)
                and old_prediction_matches == sum(seq <= selected_sequence for seq in old_predictions),
            selected_prediction_exact=fork_evidence is not None
                and fork_evidence['prediction'] == expected['prediction'],
            selected_state_exact=fork_evidence is not None
                and fork_evidence['selected'] == expected['selected'],
            prefix_receipts_exact=budget.receipts == source_receipts,
            prefix_counters_exact=end_counts == expected_counts,
            prefix_label_budget_exact=b['target_labels'] == expected['source']['prefix_target_labels'],
            prefix_policy_updates_exact=learner.total_updates-initial_updates
                == expected['source']['prefix_policy_updates'],
            selected_prefix_calibration_exact=selected_ids == source_selected,
            no_early_endpoint_fitting=all(row.available_at <= receipt['deadline']
                for receipt in budget.receipts for row in budget.admitted
                if row.sequence in receipt['calibration_sequence_ids']),
            background_snapshot_scope=first_scope and fork_evidence is not None
                and background_scope_valid(fork_evidence['context'], fork_evidence['selected']),
            diagnostic_not_reserved=selected_sequence not in budget.reserved,
            diagnostic_never_fit=selected_sequence not in budget.selected_sequences
                and all(selected_sequence not in {row.sequence for row in model.records} for model in models),
            selected_endpoints_only=all(len(model.records) == len(budget.admitted) for model in models),
            original_trust_update_rule=world.trust_updates == len(budget.receipts)*p['policy_updates_per_packet'],
            source_physical_fixed=sum(physical.learned_by_line.values()) == 0,
            native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == p['paired_suffix_seconds'],
            suffix_zero_training=bool(delta) and all(value == 0 for value in delta.values()),
            actual_low_high_command=record is not None and fork is not None
                and record.hold_seconds == (0. if args.branch == 'low' else 7.5)
                and abs(30.*(1.+float(fork['a'][0]))-record.hold_seconds) <= 1e-5,
            fixed300_truth=record is not None and record.endpoints['fixed300']['status'] == 'complete'
                and record.endpoints['fixed300']['dt'] == 300.,
            prefix_training_budget_not_increased=b['target_labels'] <= p['target_record_budget'],
            no_future_calibration=all(item['sequence'] <= selected_sequence for item in stream.executed_controls))
        summary = dict(cell=cell, state_id=expected['state_id'], mode='training_pair', preflight=args.preflight,
            **(fork_evidence or {}), execution=dict(requested_hold_seconds=0. if args.branch == 'low' else 7.5,
                executed_hold_seconds=record.hold_seconds if record else None,
                action=plain(fork['a']) if fork else None, nominal_a=plain(fork['nominal_a']) if fork else None),
            source=expected['source'], source_truth=expected.get('source_truth'),
            source_original_hold_seconds=float(30.*(1.+expected['nominal_a'][0])),
            source_truth_scope='Original continuing-learning/calibration outcome, descriptive only; not the frozen-suffix reference.',
            truth=record.endpoints if record else {}, suffix=dict(seconds=stream.native_suffix_steps,
                training_delta=delta), budget=dict(prefix_sim_seconds=fork['started_at'] if fork else stream.now,
                suffix_sim_seconds=stream.native_suffix_steps, replay_target_labels=b['target_labels'],
                replay_policy_updates=learner.total_updates-initial_updates, truth_labels_enter_training=0),
            native=dict(other_batch_actions=stream.fork_batch_other_controls,
                prefix_matched_issues=trace.matched_issues, v14_matched_predictions=old_prediction_matches,
                fork_transition_observed=stream.fork_transition is not None),
            wall_seconds=time.monotonic()-started)
        summary.update(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks))
        write(out/'summary.json', plain(summary))
        progress(summary['status'], stream.now)
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), time=stream.now if stream is not None else None))
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
    parser.add_argument('--seed', type=int, choices=(6201,), required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--target-line', choices=('7X', '102X', '705X'), required=True)
    parser.add_argument('--window', choices=('pre_shock', 'post_shock', 'preflight'), required=True)
    parser.add_argument('--branch', choices=('low', 'high'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
