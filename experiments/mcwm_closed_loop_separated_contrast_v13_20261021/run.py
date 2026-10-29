"""Separate background fitting and action contrasts on unchanged native flows."""
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
from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import counters, generalized, train_packet
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration, CalibrationPacketBudget
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector, ControlledResponseReadout
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.global_comparator import Global58
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import ResidualResponseReadout
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream
from experiments.mcwm_closed_loop_response_residual_v11_20261021.coverage import HeldoutCoverage
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import StateInteractionReadout
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.diagnostics import summarize_action_differences
from separated import SeparatedContrastReadout
from diagnostics import summarize_separation


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def selected_state(event, snapshot):
    result = {key: event[key] for key in ('sequence', 'started_at', 'line_id',
        'vehicle_id', 'trip_id', 'station_id', 's', 'raw_s', 'event')}
    result['issue_ledger'] = {key: float(getattr(snapshot, key)) for key in (
        'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
        'accrued_wait_seconds', 'accrued_in_vehicle_seconds')}
    return plain(result)


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.preflight:
        p.update(p['preflight'])
    out, started = args.out.resolve(), time.monotonic()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    env = hook = stream = trace = None
    cell = dict(seed=args.seed, scenario=args.scenario)
    if args.mode == 'pair':
        cell.update(target_line=args.target_line, branch=args.branch, anchor_index=args.anchor_index)
    stop = p.get('preflight_stop_seconds', p['horizon_seconds'])

    def progress(stage, now=0., **fields):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=stage, current=now,
            total=stop, unit='sim_seconds', elapsed_seconds=time.monotonic()-started,
            **fields)), flush=True)

    try:
        expected = None
        old_day = (HERE/p['reference_v12_root']/p['reference_v12_preflight_directory']/'day'
            if args.preflight else HERE/p['reference_v12_root']/'runs'/args.scenario/f'seed{args.seed}')
        old_reference = json.loads((old_day/'summary.json').read_text())
        old_predictions = {row['sequence']: row['prediction'] for row in old_reference['evaluations']}
        if args.mode == 'pair':
            day = HERE/p['preflight_directory']/'day' if args.preflight else HERE/'runs'/args.scenario/f'seed{args.seed}'
            reference = json.loads((day/'summary.json').read_text())
            candidates = reference['anchors'].get(args.target_line, [])
            expected = candidates[args.anchor_index] if args.anchor_index < len(candidates) else None
            if expected is None:
                summary = dict(status='selection_missing', mode=args.mode, cell=cell,
                    integrity=dict(all_pass=False, checks=dict(anchor_present=False)),
                    budget=dict(prefix_sim_seconds=0, suffix_sim_seconds=0,
                        replay_target_labels=0, truth_labels_enter_training=0),
                    reason='No eligible post-freeze main-day anchor on this line')
                write(out/'summary.json', summary)
                return summary
        trace = NativePrefixTrace(out/'prefix_trace.jsonl',
            reference=day/'prefix_trace.jsonl' if expected is not None else old_day/'prefix_trace.jsonl')
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
        endpoints = ResponseEndpointCollector(delay_seconds=p['feedback_delay_seconds'],
            duration_seconds=p['response_endpoint_seconds'])
        calibration = ControlledCalibration(p, args.seed, transform, station_id=UPSTREAM_STOP)
        heldout = None
        budget = CalibrationPacketBudget(p)
        decisions, predictions, scores, scored, anchors = {}, {}, [], set(), {}
        last_proposal = frozen_counts = freeze_clock = fork_evidence = None
        response_refits = 0
        old_prediction_matches = 0

        def all_counts():
            return dict(**counters(learner, world, physical),
                global58_refits=response.refits, global58_labels=response.labels,
                response_refits=response_refits, legacy_labels=len(legacy.records),
                residual_labels=len(residual.records), interaction_labels=len(interaction.records),
                separated_labels=len(separated.records))

        def prefix():
            return plain(dict(initializer=str(initializer), initial_updates=initial_updates,
                external_seed=args.seed+p['target_seed_offset'], counters=all_counts(),
                receipts=budget.receipts, freeze_clock=freeze_clock,
                native_steps=int(stream.env._bridge.steps), issued_count=trace.observed_issues))

        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def selector(event):
            if stream.fork is not None:
                return None
            if event['started_at'] < p['feedback_last_deadline']:
                decision = calibration.choose(event, role='train')
            elif heldout is not None:
                decision = heldout.propose(event)
            else:
                decision = None
            if decision is None:
                return None
            decisions[event['sequence']] = decision
            if decision['role'] == 'train':
                budget.reserve(event['sequence'])
            return decision['action']

        def fork_selector(event):
            if expected is None or event['sequence'] != expected['selected']['sequence']:
                return None
            if frozen_counts is None:
                raise RuntimeError('Paired anchor preceded the final parameter update')
            target = 0. if args.branch == 'low' else 7.5
            choice = calibration._action_for_hold(event['s'], target)
            if choice is None:
                raise RuntimeError('Main-day jointly executable anchor lost action support')
            return choice[1]

        def issue(event):
            nonlocal fork_evidence, old_prediction_matches
            event['proposed_a'] = last_proposal.copy()
            context = response.context(event)
            context['issue_ledger'] = selected_state(event, stream._snapshot())['issue_ledger']
            if event['started_at'] < p['feedback_last_deadline']:
                world.register(event)
            is_fork = stream.fork is not None and event['sequence'] == stream.fork['sequence']
            if stream.fork is None or is_fork:
                trace.observe(event, stream._snapshot(),
                    nominal_action=stream.fork['nominal_a'] if is_fork else None)
            decision = decisions.get(event['sequence'])
            if is_fork:
                hold = 0. if args.branch == 'low' else 7.5
                decision = dict(role='heldout', assignment=args.branch,
                    target_hold_seconds=hold, executed_hold_seconds=hold,
                    action=event['a'].copy(), jointly_executable=True)
                decisions[event['sequence']] = decision
            if decision is not None:
                if decision['role'] == 'train':
                    context['assignment_block_index'] = (
                        calibration.line_successes[('train', event['line_id'])]-1)//2
                before = all_counts()
                global_prediction = response.predict_issue(event, context,
                    seed=args.seed+300000+event['sequence'])
                context['global_baselines'] = {name: global_prediction['arms']['base_analytic'][name]
                    for name in ('low', 'high')}
                saved = plain(dict(global_comparator=global_prediction,
                    legacy=legacy.predict_issue(event, context),
                    baseline=residual.baseline_predict_issue(event, context),
                    residual=residual.predict_issue(event, context),
                    interaction=interaction.predict_issue(event, context),
                    interaction_diag=interaction.contrast_diagnostic(event, context),
                    separated=separated.predict_issue(event, context),
                    separated_no_cost=separated.predict_issue(event, context, omit_cost_rate=True),
                    separated_diag=separated.contrast_diagnostic(event, context),
                    separated_no_cost_diag=separated.contrast_diagnostic(event, context, omit_cost_rate=True)))
                previous = old_predictions.get(event['sequence'])
                if previous is None or {key: saved[key] for key in previous} != previous:
                    raise RuntimeError('Read-only v12 prediction mismatch at issue '+str(event['sequence']))
                old_prediction_matches += 1
                record = endpoints.register(event, stream._snapshot(), context,
                    role=decision['role'], hold_seconds=decision['executed_hold_seconds'],
                    assignment=decision)
                predictions[event['sequence']] = saved
                if all_counts() != before:
                    raise RuntimeError('Passive prediction changed parameter training counters')
                if decision['role'] == 'heldout':
                    evidence = dict(selected=selected_state(event, stream._snapshot()),
                        prefix=prefix(), prediction=saved)
                    if is_fork:
                        fork_evidence = evidence
                        if evidence != expected:
                            different = [key for key in evidence if evidence[key] != expected[key]]
                            raise RuntimeError('Main-day native prefix mismatch: '+','.join(different))
                    else:
                        heldout.commit(event, decision)
                        anchors.setdefault(event['line_id'], []).append(evidence)
            response.register(event, context)

        def score_available():
            for seq, record in endpoints.records.items():
                if seq in scored or record.status == 'pending':
                    continue
                saved = predictions[seq]
                scores.append(plain(dict(sequence=seq, issued_at=record.started_at,
                    line_id=record.line_id, role=record.role, hold_seconds=record.hold_seconds,
                    status=record.status, truth=record.endpoints, prediction=saved,
                    errors={name: model.chosen_outcome_errors(record, saved[name])
                        for name, model in (('legacy', legacy), ('baseline', residual),
                            ('residual', residual), ('interaction', interaction),
                            ('separated', separated), ('separated_no_cost', separated))})))
                scored.add(seq)

        write(out/'protocol.json', dict(arguments={**vars(args), 'out': str(out)}, protocol=p))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.seed+p['target_seed_offset'])
        stream = PairedControlledStream(env, stations, contract, domain='target',
            selector=selector, fork_selector=fork_selector if expected is not None else None,
            fork_duration_seconds=p['paired_suffix_seconds'], endpoints=endpoints,
            response_memory=response.memory, delay_seconds=p['feedback_delay_seconds'],
            execute_transform=False, on_issue=issue)
        stream.reset()
        stream.attach_native_clock()
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        while not stream.done and not stream.cutoff_reached and (expected is not None or stream.now < stop):
            released = stream.step(policy)
            stream.release_intervals()
            ready = endpoints.release(stream.now)
            score_available()
            if stream.fork is None:
                budget.ingest(released)
                budget.ingest_calibration([r for r in ready if r.role == 'train'])
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    valid = [r for r in packet if r.dt > 0 and not r.terminal]
                    excluded = [r.sequence for r in packet if r.dt <= 0 or r.terminal]
                    if not valid:
                        raise RuntimeError('Packet has no positive transition for original update rule')
                    response.accept(valid, discarded+excluded, stream.now)
                    train_packet(learner, replay, world, valid, discarded+excluded,
                        budget.receipts[-1], stream.now, p['model_refit_updates'], contract)
                    response.refit()
                    for model in (legacy, residual, interaction, separated):
                        model.accept(budget.last_calibration_records, stream.now)
                        model.refit()
                    response_refits += 1
                    progress('packet_training', stream.now, packets=len(budget.receipts),
                        labels=budget.summary()['target_labels'])
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    freeze_clock = stream.now
                    frozen_counts = all_counts()
                    heldout = HeldoutCoverage(p, args.seed, transform, station_id=UPSTREAM_STOP,
                        start_after=freeze_clock)
            if time.monotonic()-last_progress >= 45:
                progress('paired_suffix' if stream.fork is not None else
                    ('heldout_frozen' if frozen_counts else 'calibration'), stream.now)
                last_progress = time.monotonic()
        endpoints.finish(stream.now, reason='paired_native_300_second_cutoff' if expected is not None
            else ('preflight_stop' if args.preflight else 'target_horizon'))
        score_available()
        b, end_counts = budget.summary(), all_counts()
        selected = budget.admitted
        action_difference = dict(interaction=summarize_action_differences(predictions.values()),
            **{name: summarize_action_differences(dict(interaction=pred[name], interaction_diag=pred[diag])
                for pred in predictions.values()) for name, diag in (
                    ('separated', 'separated_diag'), ('separated_no_cost', 'separated_no_cost_diag'))})
        separation_diagnostic = summarize_separation(predictions.values(), separated)
        actions_exact = len(stream.executed_controls) == len(endpoints.records) and all(
            abs(item['hold_seconds']-endpoints.records[item['sequence']].hold_seconds) <= 1e-5
            for item in stream.executed_controls)
        checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_budget=len(replay.history['s']) == p['history_rows'] and len(replay.source['s']) == p['source_rows'],
            target_issue_budget=b['target_labels'] == p['target_record_budget'],
            policy_update_budget=learner.total_updates-initial_updates == p['policy_update_budget'],
            no_label_duplication=sum(r['records'] for r in budget.receipts) == len(budget.selected_sequences),
            no_early_endpoint_fitting=all(r.available_at <= receipt['deadline'] for receipt in budget.receipts
                for r in selected if r.sequence in receipt['calibration_sequence_ids']),
            selected_endpoints_only=len(legacy.records) == len(residual.records) == len(interaction.records) == len(separated.records) == len(selected),
            heldout_never_fit=all(r.role == 'train' for model in (residual, interaction, separated) for r in model.records),
            v12_old_predictions_exact=old_prediction_matches == len(predictions) and bool(predictions),
            separated_crossfit_endpoints_fitted=all(
                name in separated.readouts and name in separated.no_cost_readouts
                and separated.nuisance_diagnostics.get(name, {}).get('status') == 'fitted'
                for name in ('global_next', 'same_vehicle_next', 'fixed300')),
            learned_global_count=response.labels == sum(r['learned_global_records'] for r in budget.receipts),
            zero_global_never_learned=all(r['row'].dt > 0 for r in response.records),
            original_trust_update_rule=world.trust_updates == p['policy_update_budget'],
            no_post_budget_training=frozen_counts is not None and end_counts == frozen_counts,
            calibration_actually_executed=actions_exact,
            source_physical_fixed=sum(physical.learned_by_line.values()) == 0,
            exact_300_clocks=not any(e.get('reason') == 'exact_native_endpoint_clock_not_observed'
                for r in endpoints.records.values() for e in r.endpoints.values()),
            analytic_fixed300_duration=all(arm['endpoints']['fixed300'] is None or
                arm['endpoints']['fixed300']['dt'] == 300. for pred in predictions.values()
                for name in ('baseline', 'residual', 'interaction', 'separated', 'separated_no_cost')
                for arm in (pred[name]['low'], pred[name]['high'])))
        common = dict(cell=cell, mode=args.mode, preflight=args.preflight,
            action_difference=action_difference,
            separation_diagnostic=separation_diagnostic,
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            wall_seconds=time.monotonic()-started)
        for name, diagnosis in action_difference.items():
            checks.update({name+'_'+key: value for key, value in diagnosis['checks'].items()})
        checks.update(separation_diagnostic['checks'])
        if args.mode == 'pair':
            fork = stream.fork
            checks.update(anchor_reproduced=fork_evidence == expected,
                v12_common_flow_exact=reference['integrity']['checks']['v12_common_flow_exact'],
                full_native_prefix_exact=trace.comparison_exact and fork is not None and
                    trace.matched_issues == int(fork['sequence'])+1,
                native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == p['paired_suffix_seconds'],
                suffix_zero_training=fork_evidence is not None and end_counts == fork_evidence['prefix']['counters'],
                requested_hold_executed_exactly=fork is not None and
                    abs(30.*(1.+float(fork['a'][0]))-(0. if args.branch == 'low' else 7.5)) <= 1e-5)
            record = endpoints.records.get(fork['sequence']) if fork else None
            old_pair = (HERE/p['reference_v12_root']/p['reference_v12_preflight_directory']/'pair'/args.branch
                if args.preflight else HERE/p['reference_v12_root']/'pairs'/args.scenario/args.target_line/f'seed{args.seed}'/f'state{args.anchor_index}'/args.branch)
            old_truth = json.loads((old_pair/'summary.json').read_text())['truth']
            checks['v12_reference_truth_exact'] = record is not None and record.endpoints == old_truth
            suffix_delta = {k: end_counts[k]-fork_evidence['prefix']['counters'][k] for k in end_counts} if fork_evidence else {}
            summary = dict(**common, **(fork_evidence or {}),
                execution=dict(requested_hold_seconds=0. if args.branch == 'low' else 7.5,
                    executed_hold_seconds=record.hold_seconds if record else None,
                    action=plain(fork['a']) if fork else None, nominal_a=plain(fork['nominal_a']) if fork else None),
                truth=record.endpoints if record else {},
                suffix=dict(seconds=stream.native_suffix_steps, training_delta=suffix_delta),
                budget=dict(prefix_sim_seconds=fork['started_at'] if fork else stream.now,
                    suffix_sim_seconds=stream.native_suffix_steps, replay_target_labels=b['target_labels'],
                    replay_policy_updates=learner.total_updates-initial_updates, truth_labels_enter_training=0),
                native=dict(other_batch_actions=stream.fork_batch_other_controls,
                    prefix_matched_issues=trace.matched_issues,
                    fork_transition_observed=stream.fork_transition is not None))
        else:
            checks['native_horizon'] = stream.now >= stop if args.preflight else stream.now == p['horizon_seconds']
            checks['at_least_one_post_freeze_anchor'] = bool(anchors)
            checks['prefix_trace_recorded'] = trace.observed_issues == stream.issued_count
            checks['v12_common_flow_exact'] = trace.comparison_exact and trace.matched_issues == stream.issued_count and trace.reference.readline() == ''
            summary = dict(**common, budget=dict(**b, native_seconds=stream.now,
                policy_updates=learner.total_updates-initial_updates,
                world_updates=world.model_updates, trust_updates=world.trust_updates,
                global58_labels=response.labels, endpoint_labels=len(residual.records),
                interaction_labels=len(interaction.records), separated_labels=len(separated.records),
                v12_matched_predictions=old_prediction_matches,
                training_stopped_at=freeze_clock, new_native_suffix_seconds=0),
                coverage=dict(train=calibration.summary(stream.now)['coverage']['train'],
                    heldout=heldout.summary(stream.now) if heldout else None), anchors=anchors,
                residual_model=residual.summary(), legacy_model=legacy.summary(),
                interaction_model=interaction.summary(),
                separated_model=separated.summary(),
                global58=response.summary(stream.now), receipts=budget.receipts,
                endpoints=endpoints.summary(), evaluations=scores,
                cost=dict(full_day=generalized(stream._snapshot()),
                    trajectory='one common flow; passive readouts, not deployed candidate policies'),
                native=dict(observation_ticks=stream.native_ticks, v12_matched_issues=trace.matched_issues), scenario=hook.metadata())
        summary.update(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(checks=checks, all_pass=all(checks.values())))
        write(out/'summary.json', plain(summary))
        if args.preflight and args.mode == 'day':
            write(out/'gate.json', {key: plain(summary[key]) for key in ('status', 'cell', 'integrity', 'budget')})
        progress(summary['status'], stream.now)
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', mode=args.mode, cell=cell,
            type=type(error).__name__, message=str(error), time=stream.now if stream is not None else None))
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
    parser.add_argument('--mode', choices=('day', 'pair'), required=True)
    parser.add_argument('--seed', type=int, choices=(6201, 6202), required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--target-line', choices=('7X', '102X', '705X'))
    parser.add_argument('--branch', choices=('low', 'high'))
    parser.add_argument('--anchor-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
