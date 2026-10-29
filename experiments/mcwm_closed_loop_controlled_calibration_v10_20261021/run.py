"""One budget-matched controlled day, shared by passive response comparators."""
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
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.controlled_stream import ControlledStream
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.global_comparator import Global58


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.preflight:
        p.update(p['preflight'])
    out, started = args.out.resolve(), time.monotonic()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    env = hook = stream = None
    cell = dict(seed=args.seed, scenario=args.scenario)
    stop = p.get('preflight_stop_seconds', p['horizon_seconds'])

    def progress(stage, now=0., **fields):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=stage, current=now,
            total=stop, unit='sim_seconds', elapsed_seconds=time.monotonic()-started, **fields)), flush=True)

    try:
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
        endpoint_model = ControlledResponseReadout(ridge=p['response_ridge'])
        endpoints = ResponseEndpointCollector(delay_seconds=p['feedback_delay_seconds'],
            duration_seconds=p['response_endpoint_seconds'])
        calibration = ControlledCalibration(p, args.seed, transform, station_id=UPSTREAM_STOP)
        budget = CalibrationPacketBudget(p)
        decisions, predictions, scores, scored = {}, {}, [], set()
        last_proposal = None
        frozen_counts = None
        freeze_clock = None

        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def selector(event):
            role = 'train' if event['started_at'] < p['feedback_last_deadline'] else 'heldout'
            decision = calibration.choose(event, role=role)
            if decision is None:
                return None
            decisions[event['sequence']] = decision
            if role == 'train':
                budget.reserve(event['sequence'])
            return decision['action']

        def issue(event):
            event['proposed_a'] = last_proposal.copy()
            context = response.context(event)
            # No additional physical target-label collection or fitting.
            if event['started_at'] < p['feedback_last_deadline']:
                world.register(event)
            decision = decisions.get(event['sequence'])
            if decision is not None:
                role = decision['role']
                before = counters(learner, world, physical)
                record = endpoints.register(event, stream._snapshot(), context, role=role,
                    hold_seconds=decision['executed_hold_seconds'], assignment=decision)
                predictions[event['sequence']] = dict(
                    global_comparator=response.predict_issue(event, context, seed=args.seed+300000+event['sequence']),
                    endpoint_candidate=endpoint_model.predict_issue(event, context))
                if counters(learner, world, physical) != before:
                    raise RuntimeError('Passive candidate prediction changed active training counters')
            response.register(event, context)

        def score_available():
            for seq, record in endpoints.records.items():
                if seq in scored or record.status == 'pending':
                    continue
                saved = predictions[seq]
                truth = record.endpoints
                chosen = 'low' if np.isclose(record.hold_seconds, 0.) else 'high'
                comparison = {}
                if truth['global_next']['status'] == 'complete':
                    for arm, values in saved['global_comparator']['arms'].items():
                        comparison[arm] = dict(
                            cost_signed_error=values[chosen]['cost']-truth['global_next']['cost'],
                            dt_signed_error=values[chosen]['dt']-truth['global_next']['dt'],
                            cost_gain_low_minus_high=values['cost_gain_low_minus_high'],
                            raw_cost_gain_low_minus_high=values['raw_cost_gain_low_minus_high'])
                scores.append(dict(sequence=seq, issued_at=record.started_at, line_id=record.line_id,
                    role=record.role, hold_seconds=record.hold_seconds, status=record.status,
                    issue_phase='before' if record.started_at < p['shock_at_seconds'] else 'after',
                    truth=truth, global_comparator=comparison,
                    endpoint_prediction=saved['endpoint_candidate'],
                    endpoint_error=endpoint_model.chosen_outcome_errors(record, saved['endpoint_candidate'])))
                scored.add(seq)

        write(out/'protocol.json', dict(arguments={**vars(args), 'out': str(out)}, protocol=p,
            initializer=dict(path=str(initializer), updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s']))))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.seed+p['target_seed_offset'])
        stream = ControlledStream(env, stations, contract, domain='target',
            selector=selector, endpoints=endpoints, response_memory=response.memory,
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=issue)
        stream.reset()
        stream.attach_native_clock()
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        while not stream.done and stream.now < stop:
            released = stream.step(policy)
            budget.ingest(released)
            # Physical intervals are observable stream records, but unused targets.
            stream.release_intervals()
            ready = endpoints.release(stream.now)
            budget.ingest_calibration([r for r in ready if r.role == 'train'])
            score_available()
            while budget.due(stream.now):
                packet, discarded = budget.take(stream.now)
                valid = [r for r in packet if r.dt > 0 and not r.terminal]
                excluded = [r.sequence for r in packet if r.dt <= 0 or r.terminal]
                if not valid:
                    raise RuntimeError('Packet has no positive global transition for original update rule')
                response.accept(valid, discarded+excluded, stream.now)
                train_packet(learner, replay, world, valid, discarded+excluded,
                    budget.receipts[-1], stream.now, p['model_refit_updates'], contract)
                response.refit()
                endpoint_model.accept(budget.last_calibration_records, stream.now)
                endpoint_model.refit()
                progress('packet_training', stream.now, packets=len(budget.receipts),
                    labels=budget.summary()['target_labels'], calibration=len(endpoint_model.records))
            if budget.index == len(budget.deadlines) and frozen_counts is None:
                frozen_counts = counters(learner, world, physical)
                freeze_clock = stream.now
            if time.monotonic()-last_progress >= 45:
                progress('heldout_frozen' if frozen_counts else 'calibration', stream.now)
                last_progress = time.monotonic()
        endpoints.finish(stream.now, reason='preflight_stop' if args.preflight else 'target_horizon')
        score_available()
        selected = budget.admitted
        b = budget.summary()
        end_counts = counters(learner, world, physical)
        actions_exact = len(stream.executed_controls) == len(endpoints.records) and all(
            np.isclose(item['hold_seconds'], endpoints.records[item['sequence']].hold_seconds, atol=1e-5)
            for item in stream.executed_controls)
        checks = dict(
            initializer_50000=initial_updates == p['initializer_updates'],
            history_source_budget=len(replay.history['s']) == p['history_rows'] and len(replay.source['s']) == p['source_rows'],
            target_issue_budget=b['target_labels'] == p['target_record_budget'],
            policy_update_budget=learner.total_updates-initial_updates == p['policy_update_budget'],
            no_label_duplication=sum(r['records'] for r in budget.receipts) == len(budget.selected_sequences),
            no_early_endpoint_fitting=all(r.available_at <= receipt['deadline']
                for receipt in budget.receipts for r in selected if r.sequence in receipt['calibration_sequence_ids']),
            selected_endpoints_only=len(endpoint_model.records) == len(selected),
            heldout_never_fit=all(r.role == 'train' for r in endpoint_model.records),
            learned_global_count=response.labels == sum(r['learned_global_records'] for r in budget.receipts),
            zero_global_never_learned=all(r['row'].dt > 0 for r in response.records),
            original_trust_update_rule=world.trust_updates == p['policy_update_budget'],
            no_post_budget_training=frozen_counts is not None and end_counts == frozen_counts,
            calibration_actually_executed=actions_exact,
            exact_300_clocks=not any(e.get('reason') == 'exact_native_endpoint_clock_not_observed'
                for r in endpoints.records.values() for e in r.endpoints.values()),
            source_physical_fixed=sum(physical.learned_by_line.values()) == 0,
            native_horizon=stream.now >= stop if args.preflight else stream.now == p['horizon_seconds'],
            at_least_one_calibration_selected=len(selected) > 0)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed', cell=cell,
            preflight=args.preflight, integrity=dict(checks=checks, all_pass=all(checks.values())),
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            budget=dict(**b, native_seconds=stream.now, policy_updates=learner.total_updates-initial_updates,
                world_updates=world.model_updates, world_refits=world.refits, trust_updates=world.trust_updates,
                global58_labels=response.labels, endpoint_labels=len(endpoint_model.records),
                heldout_evaluation_issues=sum(r.role == 'heldout' for r in endpoints.records.values()),
                training_stopped_at=freeze_clock, new_native_suffix_seconds=0),
            calibration=calibration.summary(stream.now), endpoints=endpoints.summary(),
            endpoint_model=endpoint_model.summary(), global58=response.summary(stream.now),
            receipts=budget.receipts, controlled_records=[r.summary() for r in endpoints.records.values()],
            evaluations=scores, cost=dict(full_day=generalized(stream._snapshot()),
                trajectory='one controlled common flow, not a candidate-policy comparison'),
            scenario=hook.metadata(), native=dict(sumo_version=hook.api.getVersion()[1],
                observation_ticks=stream.native_ticks), wall_seconds=time.monotonic()-started)
        write(out/'summary.json', summary)
        if args.preflight:
            write(out/'gate.json', {k: summary[k] for k in ('status', 'cell', 'integrity', 'budget', 'endpoints')})
        progress(summary['status'], stream.now)
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), time=stream.now if stream is not None else None))
        raise
    finally:
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(6201, 6202), required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
