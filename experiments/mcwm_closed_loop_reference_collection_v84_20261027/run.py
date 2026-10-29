"""Collect three reference/candidate remaining-day pairs from one frozen prefix."""
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
    formal, source_protocol, load_learner, DecoupledPacketBudget, snapshot,
    strip_profiler, fork_children, NativePrefixTrace, install_scenario,
    _equal, _state, create_target_env, selected_state, contract, physical)
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
import numpy as np
import torch

LINES = ('7X', '102X', '705X')
ROLES = {'low': 'reference', 'high': 'candidate'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


class CollectionFinished(Exception):
    pass


def run(scenario, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root_out = out
    seed, external_seed = 6207, 106207
    p = source_protocol()
    source = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'/(
        f'runs/executed_cql/validation/seed{seed}/{scenario}/fresh')
    original = json.loads((source/'summary.json').read_text())
    source_updates = (source/'updates.jsonl').open()
    env = hook = stream = trace = probe = learner = updates = None
    branch = root = freeze = frozen_state = frozen_counts = None
    pair_path = choice = native_info = None
    pairs = []
    selection = {line:dict(selected=None, attempted=0, reasons={}, status='missing') for line in LINES}
    source_prefix_rows = source_matched_rows = source_update_rows = 0
    last_proposal = None
    root_actions = 0
    clock_exact = continuation_exact = True
    native_started_wall = None
    started = time.monotonic()

    def counts():
        if learner is None:
            return {}
        audit = learner.actor_schedule_audit()
        return {key:audit[key+'_updates'] for key in contract.COUNTS}

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase, current=stream.now if stream else 0.,
            total=24000 if branch else 12600, unit='sim_seconds', scenario=scenario,
            pairs_complete=len(pairs), elapsed_seconds=time.monotonic()-started)), flush=True)

    def policy(state):
        nonlocal last_proposal, clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0] == np.float32(np.float32(event_clock)/24000))
        last_proposal = learner.act(state[None], deterministic=True)[0]
        return learner.action_transform.numpy(state, last_proposal)

    def selector(event):
        nonlocal branch, out, trace, probe, root, choice, pair_path, native_info
        nonlocal native_started_wall, root_actions
        line = event['line_id']
        if branch is not None or freeze is None or line not in LINES:
            return None
        slot = selection[line]
        if slot['selected'] is not None or not 12000 <= event['started_at'] < 12600:
            return None
        slot['attempted'] += 1
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        try:
            step = float(api.simulation.getDeltaT())
            action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            at_stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
        except Exception:
            step = action_step = None
            at_stop = False
        if not at_stop or step is None or action_step is None:
            reason = 'native_stop_or_clock_unavailable'
        else:
            choice = choose_pair(learner.action_transform, event['s'], last_proposal,
                                 simulation_step=step, action_step=action_step)
            reason = None if choice['supported'] else 'singleton_or_insufficient_gap'
        if reason:
            slot['reasons'][reason] = slot['reasons'].get(reason, 0)+1
            return None
        if not np.array_equal(event['a'], np.asarray(choice['reference_action'], np.float32)):
            raise RuntimeError('Reference executed action differs from the visible original actor/F')
        root = selected_state(event, stream._snapshot())
        root['reference_proposal'] = last_proposal.tolist()
        native_info = dict(sim_step_seconds=step, action_step_seconds=action_step, at_stop=at_stop)
        slot['selected'] = {key:root[key] for key in ('started_at','sequence','trip_id','station_id')}
        slot['status'] = 'selected'
        pair_path = root_out/'pairs'/line
        write(pair_path/'root.json', dict(root=root, reference=freeze, choice=choice, **native_info))
        updates.flush()

        def prepare_child(key):
            nonlocal branch, out, trace, probe, native_started_wall
            branch, out = key, pair_path/key
            old_rows = trace.observed_issues
            trace.close()
            trace = NativePrefixTrace(out/'trace.jsonl')
            trace.observed_issues = old_rows
            updates.close()
            source_updates.close()
            stream.enter_suffix()
            probe = physical.install(env._bridge, root)
            native_started_wall = time.time()

        receipt = fork_children(pair_path, trace, lambda:stream.now, counts, prepare_child,
                                timeout_seconds=3600)
        if isinstance(receipt, str):
            root_actions += 1
            return np.asarray(choice[ROLES[branch]+'_action'], np.float32)
        arms = {ROLES[key]:json.loads((pair_path/key/'summary.json').read_text()) for key in ROLES}
        ref, cand = arms['reference'], arms['candidate']
        departures = [arms[role]['physical']['departure']['clock_seconds'] for role in arms]
        actual_departures = all(value is not None for value in departures)
        checks = dict(same_complete_root=ref['root'] == cand['root'],
            same_issue_reference=ref['reference'] == cand['reference'] == freeze,
            visible_reference=freeze['max_available_at'] <= 12000 <= freeze['published_clock'] <= root['started_at'],
            same_external_seed=ref['external_seed'] == cand['external_seed'] == external_seed,
            original_actor_root_exact=ref['executed_action'] == choice['reference_action'],
            predeclared_candidate_exact=cand['executed_action'] == choice['candidate_action'],
            command_support=choice['supported'], actual_departures_observed=actual_departures,
            physical_departure_separation=bool(actual_departures and abs(departures[1]-departures[0]) >= step),
            actual_next_arrivals=all(value['physical']['coverage']['next_arrival_observed'] for value in arms.values()),
            both_real_terminal=all(value['final_clock'] == 24000 for value in arms.values()),
            same_observed_duration=ref['remaining']['dt'] == cand['remaining']['dt'] > 0,
            both_labels_at24030=all(value['remaining']['available_at'] == 24030 for value in arms.values()),
            suffix_zero_training=all(value['suffix_updates'] == 0 for value in arms.values()),
            suffix_zero_admission=all(value['suffix_admitted_labels'] == 0 for value in arms.values()),
            native_fork=receipt['all_pass'], actual_parallel=receipt['overlap_seconds'] > 0,
            parent_parameters_adam_frozen=_equal(frozen_state, _state(learner)),
            parent_counts_frozen=counts() == frozen_counts, no_replacement_root=True)
        record = dict(status='complete', all_pass=all(checks.values()), checks=checks,
            line_id=line, root={key:root[key] for key in ('started_at','sequence','trip_id','station_id')},
            reference_hold_seconds=choice['reference_hold_seconds'],
            candidate_hold_seconds=choice['candidate_hold_seconds'],
            overlap_seconds=receipt['overlap_seconds'],
            remaining={role:value['remaining'] for role,value in arms.items()},
            delta={key:cand['remaining'][key]-ref['remaining'][key] for key in ('wait','ride','cost')},
            physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds'],
                missing=value['physical']['missing']) for role,value in arms.items()})
        write(pair_path/'pair_summary.json', record)
        pairs.append(record)
        progress('pair_complete_parent_reference_resumes')
        if len(pairs) == 3:
            raise CollectionFinished()
        return None

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = last_proposal.copy()
        if branch is not None and event['sequence'] != root['sequence']:
            expected = learner.action_transform.numpy(event['s'], last_proposal)
            continuation_exact &= bool(np.array_equal(event['a'], expected))
        trace.observe(event, stream._snapshot())

    def spent_budget():
        suffixes = []
        for line in LINES:
            for key in ROLES:
                directory = root_out/'pairs'/line/key
                if not directory.exists():
                    continue
                final = directory/'summary.json'
                failure = directory/'failure.json'
                if final.exists():
                    value = json.loads(final.read_text())
                    suffixes.append(dict(line=line, arm=ROLES[key], status=value['status'],
                        native_seconds=value['remaining']['dt']))
                elif failure.exists():
                    value = json.loads(failure.read_text())
                    suffixes.append(dict(line=line, arm=ROLES[key], status='failed',
                        native_seconds=value.get('suffix_native_seconds')))
                else:
                    suffixes.append(dict(line=line, arm=ROLES[key], status='started_unresolved', native_seconds=None))
        known = [row['native_seconds'] for row in suffixes if row['native_seconds'] is not None]
        prefix = stream.now if stream is not None else 0.
        return dict(prefixes=1, suffixes=len(suffixes), native_prefix_seconds=prefix,
            suffix_native_seconds=sum(known), total_native_seconds=prefix+sum(known),
            unknown_suffix_costs=sum(row['native_seconds'] is None for row in suffixes),
            restored_covariates=0 if learner is None else replay.target_received,
            restored_complete_outcomes=0 if learner is None else len(replay.outcome_ids),
            restored_updates_per_optimizer=counts().get('critic', 0),
            suffix_updates=0, labels_fed_back=0, new_initialization_updates=0,
            labels_observed=sum(row['status']=='complete' for row in suffixes), details=suffixes)

    def finish_parent():
        budget_value = spent_budget()
        checks = dict(exact_original_prefix=source_prefix_rows == source_matched_rows and source_prefix_rows > 0,
            exact_37_receipts=budget.receipts == original['receipts'][:37], exact_logged_updates=source_update_rows == 74,
            exact_restoration_counts=counts() == contract.COUNTS,
            exact_restoration_pools=replay.target_received == len(replay.outcome_ids) == 185,
            parent_parameters_adam_frozen=_equal(frozen_state, _state(learner)),
            all_three_lines=len(pairs) == 3, all_pair_integrity=all(row['all_pass'] for row in pairs),
            prefix_within_budget=stream.now <= 12600,
            scenario_native_budget=budget_value['total_native_seconds'] <= 84600,
            no_unknown_native_cost=budget_value['unknown_suffix_costs'] == 0,
            no_new_response_training=True, no_label_feedback=True)
        value = dict(status='complete' if len(pairs)==3 else 'incomplete',
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            scenario=scenario, seed=seed, external_seed=external_seed, departed_passengers=15079,
            reference_snapshot=freeze, source_prefix_rows=source_prefix_rows,
            source_update_rows=source_update_rows, budget=budget_value, pairs=pairs, selection=selection,
            wall_seconds=time.monotonic()-started, training_started=False,
            original_endpoint_preflight_used_as_training=False)
        write(root_out/'collector_summary.json', value)
        progress('collection_complete_no_training')
        return value

    try:
        progress('load_original_initializer')
        learner, replay, mdp, stations, initializer = load_learner(seed, arm='fresh')
        budget = DecoupledPacketBudget(p)
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl', source/'prefix_trace.jsonl')
        updates = (root_out/'updates.jsonl').open('x')
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=external_seed)
        stream = CollectionStream(env, stations, mdp, domain='target', selector=selector,
            delay_seconds=30, execute_transform=False, on_issue=issue)
        stream.reset()
        hook = install_scenario(env, scenario, shock_at=9000, factor=.65, metric_every=300)
        last_progress = time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            rows = stream.step(policy)
            if freeze is None:
                budget.ingest(rows)
            stream.release_intervals()
            while freeze is None and budget.index < 37 and budget.due(stream.now):
                packet, outcome_packet = budget.take(stream.now)
                replay.add_target(packet, outcome_packet)
                learner.begin_actor_packet()
                for index in range(8):
                    metrics = learner.continue_step(collect_diagnostics=index == 0)
                    if index in (0,7):
                        row = dict(packet=budget.index, step_in_packet=index+1, processed_at=stream.now, **metrics)
                        expected = source_updates.readline()
                        if not expected or strip_profiler(row) != strip_profiler(json.loads(expected)):
                            raise RuntimeError('Original logged update sequence differs')
                        source_update_rows += 1
                        updates.write(json.dumps(row, allow_nan=False)+'\n')
                updates.flush()
                progress('restore_original_prefix')
            if freeze is None and budget.index == 37:
                if counts() != contract.COUNTS or budget.receipts != original['receipts'][:37]:
                    raise RuntimeError('Packet37 restoration budget or receipts differ')
                source_prefix_rows, source_matched_rows = trace.observed_issues, trace.matched_issues
                if not trace.comparison_exact or source_prefix_rows != source_matched_rows:
                    raise RuntimeError('Original v75 prefix did not reproduce')
                freeze = contract.source_snapshot_metadata(seed=seed, scenario=scenario,
                    published_clock=stream.now,
                    max_available_at=max(row['max_available_at'] for row in budget.receipts))
                frozen_counts, frozen_state = counts(), deepcopy(_state(learner))
                torch.save(dict(metadata=freeze, actor=learner.actor.state_dict()), root_out/'reference_actor.pt')
                write(root_out/'reference.json', freeze)
                trace.reference.close()
                trace.reference = None
                stream.enable_prefix_limit()
            if time.monotonic()-last_progress >= 45:
                progress('remaining_day_suffix' if branch else 'parent_frozen_prefix')
                last_progress = time.monotonic()
        if branch is None:
            return finish_parent()
        final = snapshot(stream.final_snapshot)
        remaining = contract.remaining_cost(root['issue_ledger'], final)
        observation = probe.summary(endpoint_reached=True)
        applied = observation['native_action']
        role = ROLES[branch]
        assigned = choice[role+'_action']
        hold = choice[role+'_hold_seconds']
        checks = dict(real_terminal=stream.now == 24000 and stream.done,
            root_action_applied_once=applied is not None and root_actions == 1,
            actual_command_exact=applied is not None and abs(applied['applied_hold_seconds']-hold) <= 1e-5,
            own_reference_continuation=continuation_exact, real_actor_clock=clock_exact,
            parameters_adam_frozen=_equal(frozen_state, _state(learner)),
            suffix_zero_training=counts() == frozen_counts,
            suffix_zero_admission=replay.target_received == len(replay.outcome_ids) == 185,
            real_nonzero_duration=remaining['dt'] > 0,
            label_not_released=not contract.label_available(remaining['available_at'], stream.now),
            all_issues_logged=trace.observed_issues == stream.issued_count)
        value = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), root=root, reference=freeze,
            role=role, external_seed=external_seed, executed_action=assigned, executed_hold_seconds=hold,
            **native_info, final_clock=stream.now, suffix_updates=0, suffix_admitted_labels=0,
            remaining=remaining, physical=observation,
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started_wall,
                                native_finished_wall=time.time()))
        write(out/'summary.json', value)
        if not all(checks.values()):
            raise RuntimeError('Reference/candidate suffix integrity failed')
        progress('complete')
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except CollectionFinished:
        return finish_parent()
    except Exception as error:
        value = dict(type=type(error).__name__, message=str(error), scenario=scenario,
            native_time=stream.now if stream is not None else None, role=ROLES.get(branch),
            counts=counts(), labels_fed_back=0, wall_seconds=time.monotonic()-started)
        if branch is None:
            value['consumed_budget'] = spent_budget()
        else:
            value['suffix_native_seconds'] = max(0., stream.now-root['started_at'])
        write(out/'failure.json', value)
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        raise
    finally:
        source_updates.close()
        if updates is not None: updates.close()
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable','local_speed_drop'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.scenario, args.out)
