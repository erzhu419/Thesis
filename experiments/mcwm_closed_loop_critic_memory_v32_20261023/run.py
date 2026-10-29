"""Exact ordinary-day replay with causal context coverage; candidate stays untrained."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from bootstrap import HERE, formal, write
import numpy as np
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory, MemoryEventStream
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.probes import ZeroInitProbe, coverage
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng
from stream import create_target_env
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.seed not in p['training_seeds'] or args.arm not in p['arms']:
        raise ValueError('Cell outside the fixed v32 precheck')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cell = dict(seed=args.seed, scenario=args.scenario, arm=args.arm)
    env = hook = history = memory = trace = stream = None
    reference = HERE.parents[1]/p['reference_roots'][str(args.seed)][args.scenario]
    original = json.loads((reference/'summary.json').read_text())
    original_updates = [json.loads(line) for line in (reference/'updates.jsonl').read_text().splitlines()]
    updates_exact = True
    admitted_rows = []
    probes = None

    def progress(phase, **values):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None else 0., total=p['horizon_seconds'],
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started, **values)), flush=True)

    try:
        progress('load_initializer')
        learner, replay, contract, stations, initializer = load_learner(args.seed, arm=args.arm)
        initial_updates = learner.total_updates
        transform = learner.action_transform
        budget = formal.PacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl', reference=reference/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        frozen_counts, freeze_clock = None, None
        last_proposal = None

        def counts():
            return dict(policy=learner.total_updates-initial_updates, world=0, trust=0)

        def policy(state):
            nonlocal last_proposal
            event_clock = float(env._current_events[0].sim_time)
            expected = np.float32(np.float32(event_clock)/p['horizon_seconds'])
            clock_audit['actor_real_clock'] &= bool(state[0] == expected)
            clock_audit['max_actor_clock_error_seconds'] = max(
                clock_audit['max_actor_clock_error_seconds'],
                abs(float(state[0])-float(expected))*p['horizon_seconds'])
            clock_audit['native_policy_issues'] += 1
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
            action_audit['max_shared_hold_gap_seconds'] = max(action_audit['max_shared_hold_gap_seconds'], shared_gap)
            action_audit['max_raw_legacy_hold_gap_seconds'] = max(action_audit['max_raw_legacy_hold_gap_seconds'], raw_gap)
            action_audit['raw_legacy_disagreements'] += raw_gap > 1e-4
            action_audit['shared_rule_exact'] &= shared_gap <= 1e-6
            # Diagnostic-only factual GPS/service history is not a learner input.
            snapshot = history.snapshot(event) if str(event['station_id']) == UPSTREAM_STOP else None
            audit.observe(event, last_proposal, snapshot)

        runtime = dict(**p['runtime_contract'], td_slots=dict(source=128, historical_target=64,
            fresh_target=64 if args.arm == 'fresh' else 0,
            historical_replacement=64 if args.arm == 'no_fresh' else 0))
        write(out/'protocol.json', dict(cell=cell, protocol=p, runtime_contract=runtime,
            policy_config=asdict(learner.config), initializer=str(initializer)))
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.seed+p['target_seed_offset'])
        stream = MemoryEventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=issue)
        stream.reset()
        history = NativeRunHistory(env._bridge)
        saved_rng = _rng()
        memory = ExecutionMemory(env._bridge, window_seconds=p['memory_window_seconds'])
        stream.memory = memory
        probes = ZeroInitProbe(learner, memory.memory_dim)
        if not _equal(saved_rng, _rng()):
            raise RuntimeError('Memory/probe initialization consumed main RNG')
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        native_started_wall = time.time()
        last_progress = time.monotonic()
        with (out/'updates.jsonl').open('x') as updates, (out/'memory_training_rows.jsonl').open('x') as contexts:
            while not stream.done:
                budget.ingest(stream.step(policy))
                # Release same-car records; none are admitted as extra labels.
                stream.release_intervals()
                while budget.due(stream.now):
                    packet, _ = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    replay.add_target(packet)
                    for row in packet:
                        admitted_rows.append(row)
                        contexts.write(json.dumps(dict(sequence=row.sequence, line_id=row.line_id,
                            issue=row.started_at, completed=row.completed_at, available=row.available_at,
                            dt=row.dt, memory_s=row.memory_s, memory_ns=row.memory_ns), allow_nan=False)+'\n')
                    contexts.flush()
                    for step in range(p['policy_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if step == 0:
                            diagnostic = metrics['td_diagnostic']
                            if diagnostic['real_scale'] != 1 or diagnostic['model_scale'] != 0:
                                raise RuntimeError('The matched real-only objective changed')
                            update_row = dict(packet=budget.index, processed_at=stream.now, **metrics)
                            updates_exact &= update_row == original_updates[budget.index-1]
                            if not updates_exact:
                                raise RuntimeError('Ordinary main update differs from unchanged v28 day')
                            updates.write(json.dumps(update_row, allow_nan=False)+'\n')
                    updates.flush()
                    probes.observe(learner, packet)
                    progress('packet_training', packets=budget.index,
                        feedback_slots=replay.target_received, policy_updates=counts()['policy'])
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    frozen_counts, freeze_clock = counts(), stream.now
                if time.monotonic()-last_progress >= 45:
                    progress('native_day', packets=budget.index, policy_updates=counts()['policy'])
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
        original_metrics = [json.loads(line) for line in (reference/'metrics.jsonl').read_text().splitlines()]
        cost_record = dict(departed_passengers=departed, accrued_wait_seconds=ledger.accrued_wait_seconds,
            accrued_in_vehicle_seconds=ledger.accrued_in_vehicle_seconds,
            generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost/departed)
        exact = dict(trace=trace.comparison_exact and trace.reference.readline()==''
            and trace.matched_issues==trace.observed_issues,
            updates=updates_exact and budget.index==len(original_updates),
            receipts=budget.receipts==original['receipts'], cost=cost_record==original['cost'],
            metrics=[dict(row,departed_passengers=departed) for row in hook.metrics]==original_metrics)
        memory_summary = memory.summary()
        coverage_summary = coverage(admitted_rows)
        feedback_ids = [seq for receipt in budget.receipts for seq in receipt['sequence_ids']]
        audit_summary = audit.summary()
        checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_unchanged=len(replay.history['s']) == p['history_rows']
                and len(replay.source['s']) == p['source_rows'],
            factual_feedback_285=replay.target_received == p['target_record_budget'],
            unique_factual_labels=len(feedback_ids) == len(set(feedback_ids)) == p['target_record_budget'],
            full_packets=budget.index == 57 and all(r['records'] == 5 for r in budget.receipts),
            no_early_feedback=all(r['max_available_at'] <= r['deadline'] <= r['processed_at'] for r in budget.receipts),
            policy_updates_456=counts()['policy'] == p['policy_update_budget'],
            no_world=learner.world is None,
            no_fresh_outcomes_retained=(args.arm == 'fresh' or all(set(row) == {'s', 'a'} for row in replay.recent)),
            actor_real_clock=clock_audit['actor_real_clock'],
            shared_execution_rule=action_audit['shared_rule_exact'],
            full_native_day=stream.done and stream.now == p['horizon_seconds'],
            trace_complete=trace.observed_issues == stream.issued_count,
            post_budget_zero_training=frozen_counts is not None and counts() == frozen_counts,
            passenger_ledger_balanced=passengers['active_state_count_balance_error'] == 0,
            full_trace_exact=exact['trace'], updates_exact=exact['updates'], receipts_exact=exact['receipts'],
            cost_exact=exact['cost'], metrics_exact=exact['metrics'],
            memory_causal=memory_summary['integrity']['all_pass'],
            admitted_context_complete=coverage_summary['rows_with_both_contexts']==285,
            admitted_only_original285=len(admitted_rows)==285 and len({r.sequence for r in admitted_rows})==285,
            candidate_zero_init_exact=probes.summary()['all_pass'],
            no_candidate_optimizer=True, no_diagnostic_truth_loaded=True)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), cell=cell,
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            budget=dict(target_feedback_slots=replay.target_received,
                training_fresh_outcomes=replay.target_received if args.arm == 'fresh' else 0,
                policy_updates=counts()['policy'], packets=budget.index, native_seconds=stream.now,
                world_updates=0, trust_updates=0, counterfactual_queries=0,
                training_stopped_at=freeze_clock),
            receipts=budget.receipts, replay=replay.summary(), runtime_contract=runtime,
            replay_exact=exact, context=memory_summary, coverage=coverage_summary,
            zero_init_probes=probes.summary(),
            diagnostic_budget=dict(new_training_labels=0,new_critic_updates=0,native_suffixes=0),
            clock_audit=clock_audit, action_audit=action_audit,
            post_budget_counts=dict(before=frozen_counts, after=counts()),
            factual_trace_rows=trace.observed_issues, issue_audit=audit_summary,
            cost=dict(departed_passengers=departed, accrued_wait_seconds=ledger.accrued_wait_seconds,
                accrued_in_vehicle_seconds=ledger.accrued_in_vehicle_seconds,
                generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost/departed),
            scenario=hook.metadata(), metrics_path=str(out/'metrics.jsonl'),
            wall_seconds=time.monotonic()-started, execution_wall=dict(pid=os.getpid(),
                native_started_wall=native_started_wall, native_finished_wall=native_finished_wall))
        write(out/'summary.json', summary)
        write(out/'gate.json', {k:summary[k] for k in ('status', 'integrity', 'cell', 'budget')})
        progress(summary['status'])
        if not all(checks.values()):
            raise RuntimeError('v32 ordinary replay/context integrity failed')
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None))
        raise
    finally:
        if memory is not None: memory.close()
        if history is not None: history.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh',), required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
