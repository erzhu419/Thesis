"""Original fresh-V control with eight critic/value then one actor/alpha update per packet."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal, write
from experiments.mcwm_closed_loop_delayed_actor_v67_20261026.learning import (
    FreshOutcomeReplay, load_learner)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_delayed_actor_v67_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import EventStream, create_target_env
import numpy as np




def protocol():
    base = HERE.parent/'mcwm_closed_loop_fresh_attribution_v25_20261023/protocol.json'
    p = json.loads(base.read_text())
    candidate = json.loads((HERE/'protocol.json').read_text())
    if (candidate['horizon_seconds'] != 24000 or candidate['initializer_updates'] != 50000
            or candidate['feedback_slots'] != 285 or candidate['policy_updates'] != 57
            or candidate['critic_updates'] != 456 or candidate['value_updates'] != 456):
        raise ValueError('The actual-control candidate changed its fixed native/update budget')
    p.update(name=candidate['name'], candidate='delayed_actor',
        training_seeds=candidate['development_seeds'],
        target_seed_offset=candidate['external_seed_offset'],
        candidate_change=candidate['candidate_change'], original_replay_window=285,
        source_history_target='original .1V+.9SAC', fresh_target='reward_scale*r+discounts*(1-terminal)*V(ns)',
        scheduler_priority='high', policy_update_budget=57, policy_updates_per_packet=1,
        critic_update_budget=456, value_update_budget=456, critic_updates_per_packet=8,
        actor_update_budget=57, alpha_update_budget=57, actor_schedule=candidate['actor_schedule'])
    p['runtime_contract'] = dict(p['runtime_contract'],
        optimizer_schedule=candidate['runtime_optimizer_schedule'])
    return p


def target_audit_exact(learner, arm):
    values = learner.target_audit()
    fresh = arm == 'fresh'
    return (values['parent_mixed_target_calls'] == 456
        and values['parent_mixed_target_rows'] == 456*256
        and values['fresh_value_replacement_calls'] == (456 if fresh else 0)
        and values['fresh_value_replacement_rows'] == (456*64 if fresh else 0)
        and values['no_fresh_unchanged_returns'] == (0 if fresh else 456)
        and values['fresh_rows_start'] == 64 and values['fresh_rows_stop'] == 128)


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
        candidate='delayed_actor', external_seed=external_seed)
    env = hook = history = trace = stream = learner = replay = budget = None
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
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        frozen_counts, freeze_clock = None, None
        last_proposal = None

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
            actor_alpha_updates_57=counts()['policy'] == counts()['actor'] == counts()['alpha'] == 57,
            no_world=learner.world is None,
            no_fresh_outcomes_retained=(args.arm == 'fresh' or all(
                set(row) == {'s', 'a'} for row in replay.recent)),
            current_covariates_285=len(replay.recent) == 285 and replay.recent.maxlen == 285,
            original_replay_and_sampler=type(replay) is FreshOutcomeReplay,
            fresh_value_target_scope=target_audit_exact(learner, args.arm),
            delayed_actor_schedule=learner.actor_schedule_audit()['packets'] == 57
                and learner.actor_schedule_audit()['actor_update_after_critic_steps'] == [8]*57
                and learner.actor_schedule_audit()['optimizer_order_exact'],
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
        write(out/'summary.json', summary)
        write(out/'gate.json', {key: summary[key] for key in ('status', 'integrity', 'cell', 'budget')})
        progress(summary['status'])
        if not all(checks.values()):
            raise RuntimeError('v67 actual control integrity failed')
        return summary
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
        raise
    finally:
        if history is not None: history.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh', 'no_fresh'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external-seed', type=int)
    run(parser.parse_args())
