"""Restore v75 packet51 prefixes, then compare frozen actors from native complete entries."""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal, write
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.learning import (
    FreshOutcomeReplay, load_learner, cql_checks, recency_checks, target_checks)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.preflight import native_preflight
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
    DecoupledPacketBudget, summarize_selection, selection_checks)
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.service_trace import TailServiceTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import EventStream, create_target_env
import numpy as np



from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027.support import (
    PrefixComplete, ARMS, COUNTS, PACKETS, RECORDS, UPDATES, publish_actor,
    opposite_actor, trace_child, cost_windows, snapshot, strip_profiler, acquire_tail_slot)
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children


def protocol():
    from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.control import protocol as original
    return original()


def run(args):
    p = protocol()
    if args.seed not in (6208, 6207) or args.scenario != 'stable' or args.arm not in ARMS.values():
        raise ValueError('Only the two fixed stable diagnostic seeds and original arms are supported')
    external_seed = args.seed+p['target_seed_offset'] if args.external_seed is None else args.external_seed
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    root_out = out
    reference_dir = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'/f'runs/executed_cql/validation/seed{args.seed}/stable/{args.arm}'
    reference = json.loads((reference_dir/'summary.json').read_text())
    reference_updates = (reference_dir/'updates.jsonl').open()
    cell = dict(seed=args.seed, scenario='stable', entry_arm=args.arm, external_seed=external_seed)
    started = time.monotonic()
    env = hook = history = trace = stream = learner = replay = budget = service = None
    initial_updates = None
    native_started_wall = native_finished_wall = None
    branch = None
    frozen_counts = freeze_clock = own_snapshot = selected_snapshot = None
    switch_time = cut_snapshot = original_next_issue = None
    prefix_trace_rows = prefix_matched_rows = None
    cut_input_exact = False
    first_issue = first_own_issue_exact = None
    update_rows_matched = 0
    suffix_feedback_discarded = 0

    def counts():
        audit = learner.actor_schedule_audit() if learner is not None else {}
        values = {key: audit.get(key+'_updates', 0) for key in ('critic', 'value', 'actor', 'alpha')}
        return dict(**values, policy=values['actor'], world=0, trust=0)

    def progress(phase, **values):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None else 0., total=p['horizon_seconds'],
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started, **values)), flush=True)

    try:
        progress('load_initializer')
        learner, replay, contract, stations, initializer = load_learner(args.seed, arm=args.arm)
        initial_updates = learner.total_updates
        policy_probe = PacketPolicyProbe()
        transform = learner.action_transform
        budget = DecoupledPacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl', reference_dir/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        pre_update_prefix = None
        last_proposal = None

        def policy(state):
            nonlocal last_proposal, branch, out, trace, native_started_wall, started
            nonlocal switch_time, cut_snapshot, original_next_issue, cut_input_exact
            nonlocal prefix_trace_rows, prefix_matched_rows, selected_snapshot
            event_clock = float(env._current_events[0].sim_time)
            expected = np.float32(np.float32(event_clock)/p['horizon_seconds'])
            clock_audit['actor_real_clock'] &= bool(state[0] == expected)
            clock_audit['max_actor_clock_error_seconds'] = max(
                clock_audit['max_actor_clock_error_seconds'],
                abs(float(state[0])-float(expected))*p['horizon_seconds'])
            clock_audit['native_policy_issues'] += 1
            if branch is None and budget.index == PACKETS:
                if counts() != COUNTS or replay.target_received != RECORDS or own_snapshot is None:
                    raise RuntimeError('First post-packet51 control lacks its exact visible actor budget')
                switch_time = float(stream.now)
                cut_snapshot = snapshot(stream._snapshot())
                if not (16200 <= switch_time < 24000 and cut_snapshot['time_seconds'] == switch_time):
                    raise RuntimeError('The actual controllable cut is not a real post16200 ledger state')
                prefix_trace_rows, prefix_matched_rows = trace.observed_issues, trace.matched_issues
                if prefix_trace_rows != prefix_matched_rows or not trace.comparison_exact:
                    raise RuntimeError('Restored complete prefix differs from v75')
                if budget.receipts != reference['receipts'][:PACKETS] or update_rows_matched != 2*PACKETS:
                    raise RuntimeError('Restored packet51 receipt or update sequence differs from v75')
                cookie = trace.reference.tell()
                original_next_issue = json.loads(trace.reference.readline())
                trace.reference.seek(cookie)
                cut_input_exact = bool(original_next_issue['started_at'] == switch_time
                    and np.array_equal(np.asarray(original_next_issue['s'], dtype=np.float32), state))
                if not cut_input_exact:
                    raise RuntimeError('The first frozen-policy state differs from the original v75 issue')
                peer = 'no_fresh' if args.arm == 'fresh' else 'fresh'
                peer_weights, peer_snapshot = opposite_actor(root_out.parent/peer/'actor_at_packet51.pt',
                    seed=args.seed, arm=peer)
                tail_lock = acquire_tail_slot(root_out.parent, args.arm, switch_time)
                updates.flush(); policy_records.flush(); service.flush()

                def prepare_child(key):
                    nonlocal branch, out, trace, native_started_wall, started, selected_snapshot
                    branch = key
                    tail_lock.close()
                    out = root_out/'suffixes'/key
                    updates.close(); policy_records.close(); reference_updates.close()
                    trace = trace_child(trace, out/'prefix_trace.jsonl')
                    service.child_detach(out/'service_trace.jsonl')
                    if ARMS[key] != args.arm:
                        learner.actor.load_state_dict(peer_weights)
                        selected_snapshot = peer_snapshot
                    else:
                        selected_snapshot = own_snapshot
                    native_started_wall = time.time()
                    started = time.monotonic()

                try:
                    selected = fork_children(root_out/'suffixes', trace, lambda: stream.now,
                        counts, prepare_child, timeout_seconds=3600)
                finally:
                    if branch is None:
                        fcntl.flock(tail_lock.fileno(), fcntl.LOCK_UN)
                        tail_lock.close()
                if isinstance(selected, dict):
                    raise PrefixComplete(selected)
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def issue(event):
            nonlocal first_issue, first_own_issue_exact
            event['proposed_a'] = last_proposal.copy()
            trace.observe(event, stream._snapshot())
            if branch is not None and first_issue is None:
                first_issue = dict(clock_seconds=float(event['started_at']), trip_id=str(event['trip_id']),
                    station_id=str(event['station_id']), proposed_a=last_proposal.tolist(),
                    executed_a=event['a'].tolist(),
                    holding_seconds=(float(event['a'][0])+1)*.5*contract.max_hold_seconds)
                if ARMS[branch] == args.arm:
                    trace.writer.flush()
                    with trace.path.open() as own_trace:
                        first_own_issue_exact = json.loads(own_trace.readline()) == original_next_issue
                    if not first_own_issue_exact:
                        raise RuntimeError('The visible own-actor snapshot failed its original first issue')
            native = completion_safety_action(event['raw_s'], last_proposal,
                'quarter-load-sixty', contract)
            expected = transform.numpy(event['s'], last_proposal)
            shared_gap = abs(float(expected[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
            raw_gap = abs(float(native[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
            action_audit['compared_issues'] += 1
            action_audit['max_shared_hold_gap_seconds'] = max(action_audit['max_shared_hold_gap_seconds'], shared_gap)
            action_audit['max_raw_legacy_hold_gap_seconds'] = max(action_audit['max_raw_legacy_hold_gap_seconds'], raw_gap)
            action_audit['raw_legacy_disagreements'] += raw_gap > 1e-4
            action_audit['shared_rule_exact'] &= shared_gap <= 1e-6
            recorded = history.snapshot(event) if str(event['station_id']) == UPSTREAM_STOP else None
            audit.observe(event, last_proposal, recorded)

        runtime = dict(**p['runtime_contract'], td_slots=dict(source=128, historical_target=64,
            fresh_target=64 if args.arm == 'fresh' else 0,
            historical_replacement=64 if args.arm == 'no_fresh' else 0))
        write(out/'protocol.json', dict(cell=cell, source_method_protocol=p,
            diagnostic_protocol=json.loads((HERE/'protocol.json').read_text()),
            runtime_contract=runtime, policy_config=asdict(learner.config), initializer=str(initializer)))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=external_seed)
        stream = EventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=issue)
        stream.reset()
        history = NativeRunHistory(env._bridge)
        service = TailServiceTrace(env._bridge, out/'service_trace.jsonl', detail_from=0.)
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        native_started_wall = time.time()
        last_progress = time.monotonic()
        with ((out/'updates.jsonl').open('x') as updates,
              (out/'packet_policy_audit.jsonl').open('x') as policy_records):
            while not stream.done:
                rows = stream.step(policy)
                if branch is None:
                    budget.ingest(rows)
                else:
                    suffix_feedback_discarded += len(rows)
                stream.release_intervals()
                while branch is None and budget.index < PACKETS and budget.due(stream.now):
                    packet, outcome_packet = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    if budget.index == 1:
                        trace.writer.flush()
                        pre_update_prefix = dict(trace_path=str(trace.path), rows=trace.observed_issues,
                            clock=stream.now, initial_updates=learner.total_updates)
                    replay.add_target(packet, outcome_packet)
                    learner.begin_actor_packet()
                    for step in range(p['critic_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if step == 0 and (metrics['td_diagnostic']['real_scale'] != 1
                                or metrics['td_diagnostic']['model_scale'] != 0):
                            raise RuntimeError('The original real-only objective changed')
                        if step in (0, 7):
                            row = dict(packet=budget.index, step_in_packet=step+1,
                                processed_at=stream.now, **metrics)
                            expected_line = reference_updates.readline()
                            if not expected_line or strip_profiler(row) != strip_profiler(json.loads(expected_line)):
                                raise RuntimeError('Restored non-profiler update diagnostics differ from v75')
                            update_rows_matched += 1
                            updates.write(json.dumps(row, allow_nan=False)+'\n')
                    updates.flush()
                    policy_records.write(json.dumps(policy_probe.record(learner, packet,
                        processed_at=stream.now), allow_nan=False)+'\n')
                    policy_records.flush()
                    if budget.index == 1:
                        write(out/'preflight.json', native_preflight(learner, replay, contract,
                            initializer, packet, budget.receipts[-1], clock_audit, action_audit,
                            seed=args.seed, scenario=args.scenario, arm=args.arm,
                            runtime_contract=p['runtime_contract'],
                            prefix_before_first_update=pre_update_prefix, outcome_packet=outcome_packet))
                    progress('prefix_packet_training', packets=budget.index,
                        covariate_slots=replay.target_received, updates=counts()['critic'])
                if budget.index == PACKETS and frozen_counts is None:
                    frozen_counts, freeze_clock = counts(), stream.now
                    own_snapshot = publish_actor(root_out/'actor_at_packet51.pt', learner,
                        seed=args.seed, arm=args.arm, counts=counts(), published_clock=stream.now,
                        max_available_at=max(row['max_available_at'] for row in budget.receipts))
                if time.monotonic()-last_progress >= 45:
                    progress('frozen_suffix' if branch else 'prefix_restore', packets=budget.index,
                        critic_updates=counts()['critic'], actor_updates=counts()['actor'])
                    last_progress = time.monotonic()
        native_finished_wall = time.time()
        hook.capture_final()
        if branch is None:
            raise RuntimeError('No native post-packet51 controllable fork was reached')
        service_result = service.finish()
        service_result['service_sample_scope'] = ('Native service clocks and physical inventories '
            'recorded from0 seconds; inventories retain their observed_at clock; no extra simulation.')
        write(out/'service_summary.json', service_result)
        ledger = env._bridge.passenger_time_ledger.snapshot()
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        passengers = summarize_passengers_v5(env._bridge, ledger)
        departed = passengers['departed_passengers']
        actual_metrics = [dict(row, departed_passengers=departed) for row in hook.metrics]
        with (out/'metrics.jsonl').open('x') as metrics_file:
            for row in actual_metrics:
                metrics_file.write(json.dumps(row)+'\n')
        points, windows = cost_windows(actual_metrics, departed, cut_snapshot)
        schedule = learner.actor_schedule_audit()
        checks = dict(initializer_50000=initial_updates == 50000,
            exact_restoration_budget=counts() == COUNTS and replay.target_received == RECORDS,
            exact_prefix_receipts=budget.receipts == reference['receipts'][:PACKETS],
            prefix_updates_exact=update_rows_matched == 2*PACKETS,
            source_prefix_exact=prefix_trace_rows == prefix_matched_rows and cut_input_exact,
            prefix_selection_exact=all(selection_checks(budget.receipts,
                expected_packets=PACKETS, expected_records=RECORDS).values()),
            no_fresh_outcomes_retained=(args.arm == 'fresh' or len(replay.outcome_ids) == 0),
            prefix_outcomes= len(replay.outcome_ids) == (RECORDS if args.arm == 'fresh' else 0),
            fixed_recent30=all(recency_checks(replay.summary(), args.arm, UPDATES, args.seed).values()),
            original_targets=all(target_checks(learner.target_audit(), args.arm, calls=UPDATES).values()),
            original_actor_schedule=schedule['packets'] == PACKETS and schedule['optimizer_order_exact']
                and all(schedule['packet_'+family+'_steps'] == [8]*PACKETS
                    for family in ('critic', 'value', 'actor', 'alpha')),
            original_CQL=all(cql_checks(learner.cql_audit(), expected_calls=UPDATES, rows_per_call=128).values()),
            suffix_zero_training=counts() == frozen_counts,
            suffix_zero_admission=budget.index == PACKETS and replay.target_received == RECORDS,
            source_reference_detached=trace.reference is None,
            first_own_issue_exact=(first_own_issue_exact is True if ARMS[branch] == args.arm else True),
            actor_real_clock=clock_audit['actor_real_clock'], shared_F=action_audit['shared_rule_exact'],
            complete_actual_endpoints=set(points) == {'18300', '20100', '24000'},
            full_native_horizon=stream.done and stream.now == 24000,
            trace_complete=trace.observed_issues == stream.issued_count,
            balanced_passengers=passengers['active_state_count_balance_error'] == 0)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            cell=dict(seed=args.seed, scenario='stable', entry_arm=args.arm, policy_arm=ARMS[branch]),
            external_seed=external_seed, initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            actor_snapshot=selected_snapshot, own_actor_snapshot=own_snapshot,
            switch_time=switch_time, cut_snapshot=cut_snapshot, endpoints=points, cost_windows=windows,
            first_issue=first_issue, first_own_issue_exact=first_own_issue_exact,
            final_waiting=points['24000']['waiting_passengers'], zero_suffix_training=counts() == frozen_counts,
            cost=dict(departed_passengers=departed, accrued_wait_seconds=ledger.accrued_wait_seconds,
                accrued_in_vehicle_seconds=ledger.accrued_in_vehicle_seconds,
                generalized_passenger_seconds=2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds,
                generalized_seconds_per_departed_passenger=(2*ledger.accrued_wait_seconds+
                    ledger.accrued_in_vehicle_seconds)/departed),
            budget=dict(restoration_feedback_slots=RECORDS,
                restoration_complete_outcomes=len(replay.outcome_ids), restoration_packets=budget.index,
                **{'restoration_'+key+'_updates':counts()[key] for key in ('critic','value','actor','alpha')},
                suffix_native_seconds=stream.now-switch_time, suffix_training_updates=0,
                suffix_admitted_labels=0, diagnostic_labels_fed_back=0),
            post_budget_counts=dict(before=frozen_counts, after=counts()),
            replay=replay.summary(), runtime_contract=runtime,
            factual_trace_rows=trace.observed_issues, suffix_trace_rows=trace.observed_issues-prefix_trace_rows,
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started_wall,
                native_finished_wall=native_finished_wall), wall_seconds=time.monotonic()-started,
            diagonal_reference=dict(requires_original_suffix_exact=False,
                reason='Frozen after408 updates; source v75 continued to456, so its suffix is not the target replay.'),
            service=dict(resident_service_path=service_result['resident_service_path'],
                resident_summary_path=service_result['resident_summary_path'],
                final_waiting=service_result['final']['waiting_unique'],
                missing_observations=service_result['missing_observations']))
        write(out/'summary.json', summary)
        write(out/'gate.json', {key:summary[key] for key in ('status', 'integrity', 'cell', 'budget')})
        if not all(checks.values()):
            raise RuntimeError('Frozen suffix integrity failed')
        progress('complete')
        service.close(); trace.close(); history.close(); hook.close(); env.close()
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except PrefixComplete as completed:
        checks = dict(full_prefix_exact=prefix_trace_rows == prefix_matched_rows and trace.comparison_exact,
            cut_input_exact=cut_input_exact, prefix_receipts_exact=budget.receipts == reference['receipts'][:PACKETS],
            prefix_update_values_exact=update_rows_matched == 2*PACKETS,
            exact_restoration_budget=counts() == COUNTS and replay.target_received == RECORDS,
            frozen_parent_preserved=completed.receipt['all_pass'],
            first_legal_post_packet51_event=16200 <= switch_time < 24000)
        result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            cell=cell, external_seed=external_seed, prefix_clock=switch_time,
            trace_rows=prefix_trace_rows, matched_prefix_rows=prefix_matched_rows,
            prefix_update_rows_compared=update_rows_matched, cut_snapshot=cut_snapshot,
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            actor_snapshot=own_snapshot,
            budget=dict(restoration_feedback_slots=replay.target_received,
                restoration_complete_outcomes=len(replay.outcome_ids), restoration_packets=budget.index,
                native_prefix_seconds=switch_time,
                **{'restoration_'+key+'_updates':counts()[key] for key in ('critic','value','actor','alpha')}),
            actor_snapshot_path=str(root_out/'actor_at_packet51.pt'),
            fork_path=str(root_out/'suffixes/live_fork.json'))
        write(root_out/'prefix_summary.json', result)
        if not all(checks.values()):
            raise RuntimeError('Restored prefix integrity failed')
        return result
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None,
            consumed_budget=dict(feedback_slots=replay.target_received if replay is not None else 0,
                packets=budget.index if budget is not None else 0, counts=counts(),
                diagnostic_labels_fed_back=0), wall_seconds=time.monotonic()-started))
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        raise
    finally:
        reference_updates.close()
        if service is not None: service.close()
        if history is not None: history.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()
        if branch is not None: os._exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(6208,6207))
    parser.add_argument('--scenario', choices=('stable',), required=True)
    parser.add_argument('--arm', choices=('fresh','no_fresh'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external-seed', type=int)
    run(parser.parse_args())
