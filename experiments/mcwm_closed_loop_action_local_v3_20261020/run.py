#!/usr/bin/env python3
"""Finite action-consistent, local-calibrated MC-WM development comparison."""
import argparse
import json
import os
from pathlib import Path
import time
from bootstrap import HERE, formal, load_learner, write
import numpy as np
import torch
from stream import EventStream, create_target_env, policy_action_to_hold_seconds_v7
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, AFFECTED_LINES, UPSTREAM_STOP
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.run import Moments
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import DiagnosticTrainer
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.world_diagnostics import DiagnosticWorld
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration, phase
from experiments.mcwm_closed_loop_action_local_v3_20261020.probes import ProbedWorld, ProbedTrainer


def action_probe(learner, packet, contract, transform):
    states = np.stack([row.s for row in packet])
    proposed = learner.act(states, deterministic=True)
    executed = (transform.numpy(states, proposed) if transform is not None else
        np.stack([completion_safety_action(row.raw_s, action, 'quarter-load-sixty', contract)
                  for row, action in zip(packet, proposed)]))
    hold = lambda actions: np.array([float(policy_action_to_hold_seconds_v7(a[0], contract)) for a in actions])
    return hold(proposed), hold(executed)


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.preflight:
        p.update(model_init_updates=2, model_refit_updates=1, policy_updates_per_packet=1,
                 feedback_last_deadline=3900, target_record_budget=50,
                 policy_update_budget=10, shock_at_seconds=1350)
    if args.seed not in p['training_seeds'] or args.arm not in p['arms']:
        raise ValueError('Unknown development seed or arm')
    period = lambda at: phase(at, p['shock_at_seconds'])
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(stage, now=0., **extra):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=now, total=24000,
            unit='sim_seconds', phase=stage, elapsed_seconds=time.monotonic()-started,
            **extra)), flush=True)
    progress('load_initializer')
    old = args.arm == 'old_model'
    metadata = json.loads((HERE.parent/'mcwm_closed_loop_formal_20261020/inputs/history_metadata.json').read_text())
    contract = formal.TransitMDPConfigV7(**metadata['contract'])
    transform = None if old else ExecutionAction(contract)
    kwargs = dict(arm='real075' if args.arm == 'action_real' else 'model')
    if not old:
        kwargs['action_transform'] = transform
    learner, replay, _, contract, stations, initializer = load_learner(args.seed,
        trainer_class=DiagnosticTrainer if old else ProbedTrainer, **kwargs)
    initial_updates = learner.total_updates
    source = json.loads((HERE/p['reference_source_intervals']).resolve().read_text())
    physical = LocalIntervalCalibration(source, contract, shock_at=p['shock_at_seconds'],
        delay_seconds=p['feedback_delay_seconds'], learning_end_seconds=p['physical_feedback_end_seconds'],
        min_source_per_line=p['source_intervals_per_line_minimum'],
        max_source_per_line=p['source_intervals_per_line_limit'])
    world = None
    if args.arm != 'action_real':
        progress('initial_world_fit')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+20000)
            world = (DiagnosticWorld(replay, seed=args.seed, updates=p['model_init_updates']) if old else
                ProbedWorld(replay, seed=args.seed, updates=p['model_init_updates'],
                    action_transform=transform, physical=physical, enabled=args.arm == 'action_local'))
        world.name, world.context_enabled = args.arm, False
        learner.world = world
    write(out/'protocol.json', dict(arguments={**vars(args), 'out': str(out)}, protocol=p,
        initializer=str(initializer), initial_updates=initial_updates,
        history_rows=len(replay.history['s']), source_rows=len(replay.source['s']),
        policy_config=formal.asdict(learner.config), physical_labels_enter_policy=args.arm == 'action_local',
        proposal_entropy='latent proposal entropy; new Q objectives use shared executed actions',
        policy_objectives={'old_model':[.75,.25], 'action_real':[.75,0.],
                           'action_model':[.75,.25], 'action_local':[.75,.25]}))
    budget = formal.PacketBudget(p)
    action_stats, update_stats, contract_stats = Moments(), Moments(), Moments()
    feedback_coverage = {}
    last_proposal = None
    def policy(state):
        nonlocal last_proposal
        last_proposal = learner.act(state[None], deterministic=True)[0]
        return last_proposal if old else transform.numpy(state, last_proposal)
    def issue(event):
        if not old:
            event['proposed_a'] = last_proposal.copy()
        physical.register(event)
        if world is not None:
            world.register(event)
        hold = lambda a: float(policy_action_to_hold_seconds_v7(a[0], contract))
        proposed, executed = hold(event['proposed_a']), hold(event['a'])
        native = hold(completion_safety_action(event['raw_s'], event['proposed_a'],
                                               'quarter-load-sixty', contract))
        group = period(event['started_at'])
        action_stats.add(dict(proposed_hold_seconds=proposed, executed_hold_seconds=executed,
            execution_changed=float(abs(executed-proposed)>1e-5)), group)
        contract_stats.add(dict(native_v19_disagreement=float(abs(native-executed)>1e-4),
            native_v19_absolute_hold_gap_seconds=abs(native-executed)), group)
    env = hook = stream = None
    rounds = total_rows = after_update_decisions = 0
    last_progress = time.monotonic()
    try:
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.seed+p['target_seed_offset'])
        stream = EventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=old, on_issue=issue)
        stream.reset()
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        native_version = hook.api.getVersion()[1]
        with (out/'prequential.jsonl').open('w') as predictions, \
             (out/'interval_predictions.jsonl').open('w') as intervals, \
             (out/'td_diagnostics.jsonl').open('w') as td_file:
            while not stream.done:
                previous = stream.issued_count
                budget.ingest(stream.step(policy))
                if rounds:
                    after_update_decisions += stream.issued_count-previous
                for log in physical.observe(stream.release_intervals(), stream.now):
                    intervals.write(json.dumps(log)+'\n')
                intervals.flush()
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    total_rows += len(packet)
                    for row in packet:
                        group = feedback_coverage.setdefault(period(row.started_at), dict(records=0, by_line={},
                            directly_affected_upstream_records=0,
                            directly_affected_upstream_by_line={line:0 for line in AFFECTED_LINES}))
                        group['records'] += 1
                        group['by_line'][row.line_id] = group['by_line'].get(row.line_id,0)+1
                        if row.line_id in AFFECTED_LINES and row.station_id == UPSTREAM_STOP:
                            group['directly_affected_upstream_records'] += 1
                            group['directly_affected_upstream_by_line'][row.line_id] += 1
                    if world is not None:
                        for seq in discarded:
                            world.pending.pop(seq, None)
                        logs = world.observe(packet, stream.now)
                        if len(logs) != len(packet):
                            raise RuntimeError('Selected global label lacks issue-time prediction')
                        for log in logs:
                            predictions.write(json.dumps(log)+'\n')
                        predictions.flush()
                    before = action_probe(learner, packet, contract, transform)
                    replay.add_target(packet)
                    if world is not None:
                        world.refit(p['model_refit_updates'])
                        world.recenter_context(packet)
                    diagnostic = local_counterfactual = None
                    for step in range(p['policy_updates_per_packet']):
                        if world is not None and not old:
                            world.capture_counterfactual = step == 0
                        learner.local_counterfactual = None
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        update_stats.add(metrics, period(stream.now))
                        if step == 0:
                            diagnostic = metrics.get('td_diagnostic')
                            local_counterfactual = learner.local_counterfactual
                        if world is not None and not old:
                            world.capture_counterfactual = False
                            world.counterfactual_batch = None
                    after = action_probe(learner, packet, contract, transform)
                    td_file.write(json.dumps(dict(packet=rounds+1, deadline=budget.receipts[-1]['deadline'],
                        processed_at=stream.now, phase=period(stream.now), diagnostic=diagnostic,
                        local_counterfactual=local_counterfactual,
                        proposed_action_change_seconds=float(np.abs(after[0]-before[0]).mean()),
                        executed_action_change_seconds=float(np.abs(after[1]-before[1]).mean())))+'\n')
                    td_file.flush()
                    rounds += 1
                    progress('target_loop', stream.now, packets=rounds, target_records=total_rows,
                             policy_updates=learner.total_updates-initial_updates)
                    write(out/'progress.json', dict(target_time_seconds=stream.now, packets=rounds,
                        target_records=total_rows, policy_updates=learner.total_updates-initial_updates))
                    last_progress = time.monotonic()
                if time.monotonic()-last_progress >= 45:
                    progress('target_loop', stream.now, packets=rounds, target_records=total_rows)
                    last_progress = time.monotonic()
                if args.preflight and stream.now >= 4200:
                    break
        hook.capture_final()
        snapshot = env._bridge.passenger_time_ledger.snapshot()
        if total_rows != p['target_record_budget'] or learner.total_updates-initial_updates != p['policy_update_budget']:
            raise RuntimeError('Development data/update budget differs')
        if not args.preflight and (not stream.done or snapshot.time_seconds != 24000):
            raise RuntimeError('Target did not finish full horizon')
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        from SimpleSAC.transit_v3.evaluation import summarize_service
        passengers = summarize_passengers_v5(env._bridge, snapshot)
        if passengers['active_state_count_balance_error'] != 0:
            raise RuntimeError('Passenger ledger imbalance')
        cost = 2*snapshot.accrued_wait_seconds+snapshot.accrued_in_vehicle_seconds
        with (out/'metrics.jsonl').open('w') as f:
            for row in hook.metrics:
                f.write(json.dumps(row)+'\n')
        world_summary = world.summary() if world is not None else None
        summary = dict(status='complete', seed=args.seed, scenario=args.scenario, arm=args.arm,
            mode='control', preflight=args.preflight, full_horizon_completed=stream.done,
            target_time_seconds=stream.now, target_online_training_events=total_rows,
            policy_updates=learner.total_updates-initial_updates, packets=rounds,
            decisions_after_policy_update=after_update_decisions, sumo_version=native_version,
            generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers['departed_passengers'],
            passengers=passengers, service=summarize_service(env._bridge),
            scenario_metadata=hook.metadata(), feedback_receipts=budget.receipts,
            worlds={args.arm:world_summary} if world is not None else {},
            local_world=world_summary.get('local_adapter') if not old and world is not None else None,
            physical_intervals=physical.summary(), physical_learning_enters_policy=args.arm == 'action_local',
            action_statistics=action_stats.summary(), execution_contract=contract_stats.summary(),
            global_feedback_coverage=feedback_coverage,
            update_statistics=update_stats.summary(), wall_seconds=time.monotonic()-started)
        write(out/'summary.json', summary)
        progress('complete', stream.now, target_records=total_rows)
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            seed=args.seed, arm=args.arm, scenario=args.scenario,
            target_time_seconds=stream.now if stream is not None and stream.episode_id >= 0 else None))
        raise
    finally:
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('old_model','action_real','action_model','action_local'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable','local_speed_drop'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
