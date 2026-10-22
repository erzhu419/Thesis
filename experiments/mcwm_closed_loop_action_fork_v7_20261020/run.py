"""Replay action_model to one fixed issue, intervene once and freeze learning."""

import argparse
import json
import os
from pathlib import Path
import time

from bootstrap import HERE, formal, load_learner, write
import numpy as np
import torch
from stream import create_target_env
from experiments.mcwm_closed_loop_formal_20261020.scenarios import (
    AFFECTED_LINES, UPSTREAM_STOP, install_scenario,
)
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedTrainer
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld
from experiments.mcwm_closed_loop_paired_world_v6_20261020.run import action_probe
from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.predictions import ActionForkPrediction
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5


def generalized(snapshot):
    return 2 * snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds


def counters(learner, world, physical):
    return dict(policy=learner.total_updates, world=world.model_updates,
                trust=world.trust_updates, physical=physical.parameter_version)


def train_packet(learner, replay, world, packet, discarded, receipt, now, updates, contract):
    for sequence in discarded:
        world.pending.pop(sequence, None)
    if len(world.observe(packet, now)) != len(packet):
        raise RuntimeError('Prefix selected label lacks its issue-time prediction')
    action_probe(learner, packet, contract, learner.action_transform)
    replay.add_target(packet)
    world.refit(updates)
    world.recenter_context(packet)
    world.diagnose_packet(learner, packet, receipt['deadline'], now)
    for step in range(8):
        world.capture_counterfactual = step == 0
        learner.local_counterfactual = learner.head_counterfactual = None
        learner.continue_step(collect_diagnostics=step == 0)
        world.capture_counterfactual = False
        for name in ('counterfactual_batch', 'head_counterfactual_batches',
                     'head_counterfactual_weights', 'head_counterfactual_outputs'):
            setattr(world, name, None)
    action_probe(learner, packet, contract, learner.action_transform)


def run(args):
    p = json.loads((HERE / 'protocol.json').read_text())
    if args.preflight:
        p.update(p['preflight'])
    cell = dict(seed=args.seed, scenario=args.scenario, target_line=args.target_line, branch=args.branch)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = hook = stream = None
    def progress(stage, now=0., **fields):
        print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=now,
            total=p['selection_window_seconds'][1] + p['suffix_seconds'], unit='sim_seconds',
            phase=stage, elapsed_seconds=time.monotonic() - started, **fields)), flush=True)
    try:
        progress('load_initializer')
        metadata = json.loads((HERE.parent / 'mcwm_closed_loop_formal_20261020/inputs/history_metadata.json').read_text())
        contract = formal.TransitMDPConfigV7(**metadata['contract'])
        transform = ExecutionAction(contract)
        learner, replay, _, contract, stations, initializer = load_learner(args.seed,
            trainer_class=ProbedTrainer, arm='model', action_transform=transform)
        initial_updates = learner.total_updates
        source = json.loads((HERE / p['reference_source_intervals']).resolve().read_text())
        physical = LocalIntervalCalibration(source, contract, shock_at=p['shock_at_seconds'],
            delay_seconds=p['feedback_delay_seconds'], learning_end_seconds=p['physical_feedback_end_seconds'],
            min_source_per_line=p['source_intervals_per_line_minimum'],
            max_source_per_line=p['source_intervals_per_line_limit'])
        progress('initial_world_fit')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 20000)
            world = PairedWorld(replay, seed=args.seed, updates=p['model_init_updates'],
                                action_transform=transform, physical=physical, enabled=False)
        world.name, world.context_enabled, world.paired_probe_deadlines = 'action_model', False, []
        learner.world = world
        budget = formal.PacketBudget(p)
        probe = None
        selected = prefix = frozen_counts = None
        collapsed_candidates = 0
        last_proposal = None
        external_seed = args.seed + p['target_seed_offset']
        initial_receipt = dict(path=str(initializer), initial_updates=initial_updates,
            history_rows=len(replay.history['s']), source_rows=len(replay.source['s']),
            model_init_updates=p['model_init_updates'], model_refit_updates=p['model_refit_updates'])
        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)
        def issue(event):
            event['proposed_a'] = last_proposal.copy()
            if stream.fork is None:
                physical.register(event)
                world.register(event)
        def selector(event):
            nonlocal probe, selected, prefix, frozen_counts, collapsed_candidates
            now = event['started_at']
            lo, hi = p['selection_window_seconds']
            if not (lo <= now <= hi and event['line_id'] == args.target_line
                    and event['station_id'] == UPSTREAM_STOP):
                return None
            state = event['s']
            low = transform.numpy(state, np.array([-1.], np.float32))
            high = transform.numpy(state, np.array([1.], np.float32))
            if float(high[0] - low[0]) <= 1e-7:
                collapsed_candidates += 1
                return None
            probe = ActionForkPrediction(learner, state,
                seed=args.seed + 300000 + AFFECTED_LINES.index(args.target_line))
            if not probe.valid_pair:
                raise RuntimeError('The selected executed actions collapsed inside the prediction probe')
            snapshot = stream._snapshot()
            passengers = summarize_passengers_v5(env._bridge, snapshot)
            selected = dict(time=now, sequence=event['sequence'], line=event['line_id'],
                vehicle_id=event['vehicle_id'], trip_id=event['trip_id'], station=event['station_id'],
                state=state.tolist(), raw_state=event['raw_s'].tolist(),
                low_action=float(low[0]), high_action=float(high[0]), remaining_actions=[],
                prefix_cost=generalized(snapshot), prefix_departed=passengers['departed_passengers'])
            reference_dir = HERE.parent / 'mcwm_closed_loop_paired_world_v6_20261020'
            reference_dir /= ('preflight' if args.preflight else 'runs')
            reference = json.loads((reference_dir / 'control' / args.scenario / 'action_model'
                                    / f'seed{args.seed}' / 'summary.json').read_text())
            receipts = list(budget.receipts)
            prefix = dict(initializer_receipt=initial_receipt, feedback_receipts=receipts,
                policy_updates=learner.total_updates - initial_updates,
                world_updates=world.model_updates, world_refits=world.refits,
                trust_updates=world.trust_updates,
                physical_labels=sum(physical.learned_by_line.values()), external_seed=external_seed,
                target_labels=sum(row['records'] for row in receipts),
                reference_feedback_prefix_exact=receipts == [row for row in reference['feedback_receipts']
                                                              if row['processed_at'] <= now])
            if not prefix['reference_feedback_prefix_exact']:
                raise RuntimeError('Prefix feedback differs from the resident v6 action_model reference')
            frozen_counts = counters(learner, world, physical)
            progress('fork_selected', now, sequence=event['sequence'], line=args.target_line)
            return np.array([probe.candidates[args.branch]['executed_action']], np.float32)
        write(out / 'protocol.json', dict(arguments={**vars(args), 'out': str(out)}, protocol=p,
                                         initializer_receipt=initial_receipt))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT']) / 'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=external_seed)
        stream = ForkEventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False,
            on_issue=issue, selector=selector, fork_duration_seconds=p['suffix_seconds'])
        stream.reset()
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
                                factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        last_progress = time.monotonic()
        while not stream.done and not stream.cutoff_reached:
            released = stream.step(policy)
            if stream.fork is None:
                if stream.now > p['selection_window_seconds'][1]:
                    break
                budget.ingest(released)
                physical.observe(stream.release_intervals(), stream.now)
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    train_packet(learner, replay, world, packet, discarded,
                                 budget.receipts[-1], stream.now, p['model_refit_updates'], contract)
                    progress('prefix_training', stream.now, packets=len(budget.receipts))
            if time.monotonic() - last_progress >= 45:
                progress('suffix' if stream.fork is not None else 'prefix', stream.now)
                write(out / 'progress.json', dict(time=stream.now, selected=stream.fork is not None))
                last_progress = time.monotonic()
        if stream.fork is None:
            result = dict(status='selection_missing', cell=cell, preflight=args.preflight,
                collapsed_candidates=collapsed_candidates, selection_window_seconds=p['selection_window_seconds'],
                budget=dict(prefix_sim_seconds=stream.now, suffix_sim_seconds=0,
                            truth_next_global_records=0, truth_labels_enter_training=0))
            write(out / 'summary.json', result)
            return result
        if not stream.cutoff_reached or stream.fork_transition is None:
            raise RuntimeError('The fork did not produce a true next global event and exact300-second cutoff')
        after = counters(learner, world, physical)
        if after != frozen_counts:
            raise RuntimeError('Learner, world, trust or physical parameters updated during the suffix')
        start, end = stream.fork_snapshot, stream.cutoff_snapshot
        passengers = summarize_passengers_v5(env._bridge, end)
        selected['remaining_actions'] = stream.fork_batch_other_controls
        actual = probe.score_transition(stream.fork_transition, now=stream.now, candidate=args.branch)
        suffix = dict(start_time=start.time_seconds, end_time=end.time_seconds,
            seconds=end.time_seconds - start.time_seconds, cost=generalized(end) - generalized(start),
            wait=end.accrued_wait_seconds - start.accrued_wait_seconds,
            ride=end.accrued_in_vehicle_seconds - start.accrued_in_vehicle_seconds,
            departed=passengers['departed_passengers'], ledger_balance=passengers['active_state_count_balance_error'],
            training_updates=after['policy'] - frozen_counts['policy'],
            world_updates=after['world'] - frozen_counts['world'],
            trust_updates=after['trust'] - frozen_counts['trust'],
            physical_updates=after['physical'] - frozen_counts['physical'],
            native_steps=stream.native_suffix_steps)
        result = dict(status='complete', cell=cell, preflight=args.preflight,
            selected=selected, prefix=prefix, prediction=probe.summary(), next_event=actual,
            suffix=suffix, collapsed_candidates=collapsed_candidates,
            native=dict(sumo_version=hook.api.getVersion()[1],
                passenger_ledger_balanced=suffix['ledger_balance'] == 0,
                first_action_applied=float(stream.fork['a'][0]),
                other_batch_actions_unchanged=True, cutoff_not_terminal=not stream.done),
            budget=dict(prefix_sim_seconds=selected['time'], suffix_sim_seconds=suffix['seconds'],
                truth_next_global_records=1, truth_labels_enter_training=0,
                prefix_target_labels=prefix['target_labels'], prefix_policy_updates=prefix['policy_updates']),
            scenario_metadata=hook.metadata(), wall_seconds=time.monotonic() - started)
        write(out / 'summary.json', result)
        progress('complete', stream.now, suffix_seconds=suffix['seconds'])
        return result
    except Exception as error:
        write(out / 'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
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
    parser.add_argument('--target-line', choices=AFFECTED_LINES, required=True)
    parser.add_argument('--branch', choices=('low', 'high'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
