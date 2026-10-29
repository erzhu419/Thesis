"""Exact v70 prefix restoration and full-native-state final-policy exchange."""
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
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.learning import (
    FreshOutcomeReplay, load_learner, cql_checks, recency_checks, target_checks)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.support import (
    PrefixComplete, ARMS, COUNTS, publish_actor, opposite_actor, trace_child, cost_windows, diagonal_checks, acquire_tail_slot)
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.service_trace import TailServiceTrace
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import EventStream, create_target_env
import numpy as np




def protocol():
    from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.control import protocol as original
    return original()


def target_audit_exact(learner, arm):
    return all(target_checks(learner.target_audit(), arm, calls=456).values())


def run(args):
    p = protocol()
    if args.arm not in p['arms'] or args.seed not in p['training_seeds']:
        raise ValueError('Unsupported actual-control candidate or arm')
    # This batch uses only the two prespecified development seeds.
    # load_learner validates every new seed against its real50000-step archive.
    external_seed = getattr(args, 'external_seed', None)
    external_seed = args.seed+p['target_seed_offset'] if external_seed is None else external_seed
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cell = dict(seed=args.seed, scenario=args.scenario, arm=args.arm,
        candidate='executed_cql', recipe='recent30_mixed_cql', external_seed=external_seed)
    env = hook = history = trace = stream = learner = replay = budget = None
    service = None
    branch = None
    root_out = out
    switch_time = prefix_trace_rows = prefix_matched_rows = None
    reference_dir = HERE.parent/'mcwm_closed_loop_recent30_mixed_cql_v70_20261026'/f'runs/executed_cql/development/seed{args.seed}/stable/{args.arm}'
    reference = json.loads((reference_dir/'summary.json').read_text())
    reference_metrics = [json.loads(line) for line in (reference_dir/'metrics.jsonl').read_text().splitlines()]
    initial_updates = None
    native_started_wall = native_finished_wall = None

    def counts():
        audit = learner.actor_schedule_audit() if learner is not None else {}
        values = {key:audit.get(key+'_updates', 0) for key in ('critic', 'value', 'actor', 'alpha')}
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
        budget = formal.PacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl',reference_dir/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        frozen_counts, freeze_clock = None, None
        last_proposal = None

        def policy(state):
            nonlocal last_proposal,branch,out,trace,native_started_wall,started
            nonlocal switch_time,prefix_trace_rows,prefix_matched_rows
            event_clock = float(env._current_events[0].sim_time)
            expected = np.float32(np.float32(event_clock)/p['horizon_seconds'])
            clock_audit['actor_real_clock'] &= bool(state[0] == expected)
            clock_audit['max_actor_clock_error_seconds'] = max(
                clock_audit['max_actor_clock_error_seconds'],
                abs(float(state[0])-float(expected))*p['horizon_seconds'])
            clock_audit['native_policy_issues'] += 1
            if branch is None and stream.now>=18300:
                if counts()!=COUNTS or replay.target_received!=285 or frozen_counts is None:
                    raise RuntimeError('Reached tail entry without exact restored285/456 budget')
                switch_time=stream.now
                prefix_trace_rows,prefix_matched_rows=trace.observed_issues,trace.matched_issues
                if prefix_trace_rows!=prefix_matched_rows or not trace.comparison_exact:
                    raise RuntimeError('Restored prefix does not exactly match v70')
                peer='no_fresh' if args.arm=='fresh' else 'fresh'
                peer_weights=opposite_actor(root_out.parent/peer/'actor_final.pt',seed=args.seed,arm=peer)
                tail_lock=acquire_tail_slot(root_out.parent,args.arm,switch_time)
                reference_cookie=trace.reference.tell()
                updates.flush();policy_records.flush();service.flush()

                def prepare_child(key):
                    nonlocal branch,out,trace,native_started_wall,started
                    branch=key
                    tail_lock.close()
                    policy_arm=ARMS[key]
                    out=root_out/'suffixes'/key
                    updates.close();policy_records.close()
                    trace=trace_child(trace,out/'prefix_trace.jsonl',
                        reference_path=reference_dir/'prefix_trace.jsonl',reference_cookie=reference_cookie,
                        diagonal=policy_arm==args.arm)
                    service.child_detach(out/'service_trace.jsonl')
                    if policy_arm!=args.arm:
                        learner.actor.load_state_dict(peer_weights)
                    native_started_wall=time.time();started=time.monotonic()

                try:
                    selected=fork_children(root_out/'suffixes',trace,lambda:stream.now,counts,
                        prepare_child,timeout_seconds=3600)
                finally:
                    if branch is None:
                        fcntl.flock(tail_lock.fileno(),fcntl.LOCK_UN)
                        tail_lock.close()
                if isinstance(selected,dict):
                    raise PrefixComplete(selected)
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def issue(event):
            event['proposed_a'] = last_proposal.copy()
            trace.observe(event, stream._snapshot())
            native = completion_safety_action(event['raw_s'], last_proposal,
                'quarter-load-sixty', contract)
            expected = transform.numpy(event['s'], last_proposal)
            shared_gap = abs(float(expected[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
            raw_gap = abs(float(native[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
            action_audit['compared_issues'] += 1
            action_audit['max_shared_hold_gap_seconds'] = max(
                action_audit['max_shared_hold_gap_seconds'], shared_gap)
            action_audit['max_raw_legacy_hold_gap_seconds'] = max(
                action_audit['max_raw_legacy_hold_gap_seconds'], raw_gap)
            action_audit['raw_legacy_disagreements'] += raw_gap > 1e-4
            action_audit['shared_rule_exact'] &= shared_gap <= 1e-6
            snapshot = history.snapshot(event) if str(event['station_id']) == UPSTREAM_STOP else None
            audit.observe(event, last_proposal, snapshot)

        runtime = dict(**p['runtime_contract'], td_slots=dict(source=128, historical_target=64,
            fresh_target=64 if args.arm == 'fresh' else 0,
            historical_replacement=64 if args.arm == 'no_fresh' else 0))
        write(out/'protocol.json', dict(cell=cell, protocol=p, runtime_contract=runtime,
            policy_config=asdict(learner.config), initializer=str(initializer)))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=external_seed)
        stream = EventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=issue)
        stream.reset()
        history = NativeRunHistory(env._bridge)
        service = TailServiceTrace(env._bridge,out/'service_trace.jsonl')
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        native_started_wall = time.time()
        last_progress = time.monotonic()
        with ((out/'updates.jsonl').open('x') as updates,
              (out/'packet_policy_audit.jsonl').open('x') as policy_records):
            while not stream.done:
                budget.ingest(stream.step(policy))
                stream.release_intervals()
                while budget.due(stream.now):
                    packet, _ = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    if branch is not None:
                        raise RuntimeError('A suffix attempted to train')
                    replay.add_target(packet)
                    learner.begin_actor_packet()
                    for step in range(p['critic_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if step == 0:
                            diagnostic = metrics['td_diagnostic']
                            if diagnostic['real_scale'] != 1 or diagnostic['model_scale'] != 0:
                                raise RuntimeError('The original real-only objective changed')
                        if step in (0, 7):
                            updates.write(json.dumps(dict(packet=budget.index, step_in_packet=step+1,
                                processed_at=stream.now, **metrics), allow_nan=False)+'\n')
                    updates.flush()
                    policy_records.write(json.dumps(policy_probe.record(learner, packet,
                        processed_at=stream.now), allow_nan=False)+'\n')
                    policy_records.flush()
                    if budget.index == 1:
                        write(out/'preflight.json', native_preflight(learner, replay, contract,
                            initializer, packet, budget.receipts[-1], clock_audit, action_audit,
                            seed=args.seed, scenario=args.scenario, arm=args.arm,
                            runtime_contract=p['runtime_contract']))
                    progress('packet_training', packets=budget.index,
                        feedback_slots=replay.target_received, critic_updates=counts()['critic'],
                        actor_updates=counts()['actor'], alpha_updates=counts()['alpha'])
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    frozen_counts, freeze_clock = counts(), stream.now
                    publish_actor(root_out/'actor_final.pt',learner,seed=args.seed,arm=args.arm,counts=counts())
                if time.monotonic()-last_progress >= 45:
                    progress('native_day', packets=budget.index, critic_updates=counts()['critic'],
                        actor_updates=counts()['actor'])
                    last_progress = time.monotonic()
        native_finished_wall = time.time()
        hook.capture_final()
        ledger = env._bridge.passenger_time_ledger.snapshot()
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        passengers = summarize_passengers_v5(env._bridge, ledger)
        departed = passengers['departed_passengers']
        cost = 2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds
        with (out/'metrics.jsonl').open('x') as metrics:
            for row in hook.metrics:
                metrics.write(json.dumps(dict(row, departed_passengers=departed))+'\n')
        feedback_ids = [seq for receipt in budget.receipts for seq in receipt['sequence_ids']]
        checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_unchanged=len(replay.history['s']) == p['history_rows']
                and len(replay.source['s']) == p['source_rows'],
            factual_feedback_285=replay.target_received == p['target_record_budget'],
            unique_factual_labels=len(feedback_ids) == len(set(feedback_ids)) == p['target_record_budget'],
            full_packets=budget.index == 57 and all(r['records'] == 5 for r in budget.receipts),
            no_early_feedback=all(r['max_available_at'] <= r['deadline'] <= r['processed_at']
                for r in budget.receipts),
            critic_value_updates_456=counts()['critic'] == counts()['value'] == 456,
            actor_alpha_updates_456=counts()['policy'] == counts()['actor'] == counts()['alpha'] == 456,
            no_world=learner.world is None,
            no_fresh_outcomes_retained=(args.arm == 'fresh' or all(
                set(row) == {'s', 'a'} for row in replay.recent)),
            current_covariates_285=len(replay.recent) == 285 and replay.recent.maxlen == 285,
            fixed_recent30_replay=type(replay) is FreshOutcomeReplay,
            recent30_scope=all(recency_checks(replay.summary(), args.arm, 456, args.seed).values()),
            fresh_mixed_target_scope=target_audit_exact(learner, args.arm),
            original_actor_schedule=learner.actor_schedule_audit()['packets'] == 57
                and learner.actor_schedule_audit()['optimizer_order_exact']
                and all(learner.actor_schedule_audit()['packet_'+family+'_steps'] == [8]*57
                        for family in ('critic','value','actor','alpha')),
            executed_cql_contract=all(cql_checks(learner.cql_audit(), expected_calls=456, rows_per_call=128).values()),
            paid_policy_probes=policy_probe.packets == 57 and policy_probe.states == 285,
            actor_real_clock=clock_audit['actor_real_clock'],
            shared_execution_rule=action_audit['shared_rule_exact'],
            full_native_day=stream.done and stream.now == p['horizon_seconds'],
            trace_complete=trace.observed_issues == stream.issued_count,
            post_budget_zero_training=frozen_counts is not None and counts() == frozen_counts,
            passenger_ledger_balanced=passengers['active_state_count_balance_error'] == 0)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), cell=cell,
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            budget=dict(target_feedback_slots=replay.target_received,
                training_fresh_outcomes=replay.target_received if args.arm == 'fresh' else 0,
                policy_updates=counts()['policy'], critic_updates=counts()['critic'],
                value_updates=counts()['value'], actor_updates=counts()['actor'], alpha_updates=counts()['alpha'],
                packets=budget.index, native_seconds=stream.now,
                world_updates=0, trust_updates=0, counterfactual_queries=0,
                training_stopped_at=freeze_clock),
            receipts=budget.receipts, replay=replay.summary(), runtime_contract=runtime,
            target_audit=learner.target_audit(), actor_schedule_audit=learner.actor_schedule_audit(),
            cql_audit=learner.cql_audit(),
            policy_drift=policy_probe.summary(),
            clock_audit=clock_audit, action_audit=action_audit,
            post_budget_counts=dict(before=frozen_counts, after=counts()),
            factual_trace_rows=trace.observed_issues, issue_audit=audit.summary(),
            cost=dict(departed_passengers=departed, accrued_wait_seconds=ledger.accrued_wait_seconds,
                accrued_in_vehicle_seconds=ledger.accrued_in_vehicle_seconds,
                generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost/departed),
            scenario=hook.metadata(), metrics_path=str(out/'metrics.jsonl'),
            wall_seconds=time.monotonic()-started, execution_wall=dict(pid=os.getpid(),
                native_started_wall=native_started_wall, native_finished_wall=native_finished_wall))
        if branch is None:
            raise RuntimeError('No native tail fork was reached')
        policy_arm=ARMS[branch]
        actual_metrics=[dict(row,departed_passengers=departed) for row in hook.metrics]
        points,windows=cost_windows(actual_metrics,departed)
        diagonal=diagonal_checks(summary,reference,actual_metrics,reference_metrics,trace,policy_arm==args.arm)
        service_result=service.finish()
        summary.update(cell=dict(seed=args.seed,scenario='stable',entry_arm=args.arm,policy_arm=policy_arm),
            switch_time=switch_time,baseline18300=points['18300'],endpoints=points,cost_windows=windows,
            final_waiting=points['24000']['waiting_passengers'],zero_suffix_training=counts()==frozen_counts,
            diagonal_reference=diagonal,
            service=dict(resident_service_path=service_result['resident_service_path'],
                resident_summary_path=service_result['resident_summary_path'],
                final_waiting=service_result['final']['waiting_unique'],
                milestone_counts={key:dict(actual_waiting=value['waiting_unique'],ledger_waiting=value['ledger_waiting'],
                    waiting_inventory_complete=value['waiting_inventory_complete'],waiting_matches_ledger=value['waiting_matches_ledger'])
                    for key,value in service_result['milestones'].items()},
                missing_observations=service_result['missing_observations']))
        checks.update(suffix_zero_training=counts()==frozen_counts,
            complete_actual_endpoints=set(points)=={'18300','20100','24000'},
            unchanged_diagonal=policy_arm!=args.arm or all(diagonal.values()),
            full_final_waiting_inventory=service_result['final']['waiting_inventory_complete'])
        summary['status']='complete' if all(checks.values()) else 'integrity_failed'
        summary['integrity']=dict(all_pass=all(checks.values()),checks=checks)
        write(out/'summary.json', summary)
        write(out/'gate.json', {key: summary[key] for key in ('status', 'integrity', 'cell', 'budget')})
        progress(summary['status'])
        if not all(checks.values()):
            raise RuntimeError('v70 actual control integrity failed')
        service.close();trace.close();history.close();hook.close();env.close()
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    except PrefixComplete as completed:
        prefix_checks=dict(full_prefix_exact=prefix_trace_rows==prefix_matched_rows and trace.comparison_exact,
            same_v70_target_path=learner.target_audit()==reference['target_audit'],
            same_v70_actor_schedule=learner.actor_schedule_audit()==reference['actor_schedule_audit'],
            exact_restoration_budget=counts()==COUNTS and replay.target_received==285,
            frozen_parent_preserved=completed.receipt['all_pass'],
            fork_at_declared_boundary=18300<=switch_time<24000)
        result=dict(status='complete' if all(prefix_checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(prefix_checks.values()),checks=prefix_checks),
            cell=dict(seed=args.seed,scenario='stable',entry_arm=args.arm),prefix_clock=switch_time,
            trace_rows=prefix_trace_rows,matched_prefix_rows=prefix_matched_rows,
            budget=dict(restoration_feedback_slots=replay.target_received,native_prefix_seconds=switch_time,
                **{'restoration_'+key+'_updates':counts()[key] for key in ('critic','value','actor','alpha')}),
            actor_final_path=str(root_out/'actor_final.pt'),fork_path=str(root_out/'suffixes/live_fork.json'))
        write(root_out/'prefix_summary.json',result)
        if not all(prefix_checks.values()):
            write(root_out/'failure.json',dict(type='PrefixIntegrityError',failed_checks=[key for key,value in prefix_checks.items() if not value],consumed_budget=result['budget']))
            raise RuntimeError('Restored prefix audit failed')
        return result
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None,
            consumed_budget=dict(feedback_slots=replay.target_received if replay is not None else 0,
                policy_updates=counts()['policy'], critic_updates=counts()['critic'],
                value_updates=counts()['value'], actor_updates=counts()['actor'], alpha_updates=counts()['alpha'],
                packets=budget.index if budget is not None else 0,
                world_updates=0, trust_updates=0, counterfactual_queries=0),
            wall_seconds=time.monotonic()-started,
            native_started_wall=native_started_wall, native_finished_wall=native_finished_wall))
        if branch is not None:
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if service is not None: service.close()
        if history is not None: history.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()
        if branch is not None: os._exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable',), required=True)
    parser.add_argument('--arm', choices=('fresh', 'no_fresh'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external-seed', type=int)
    args=parser.parse_args()
    if args.seed not in (6201,6203): raise ValueError('Only declared stable development seeds')
    run(args)
