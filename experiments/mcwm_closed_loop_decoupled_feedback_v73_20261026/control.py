"""Separate v70 policy covariates from capacity-covered complete critic feedback."""
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
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.learning import (
    FreshOutcomeReplay, load_learner, cql_checks, recency_checks, target_checks)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.preflight import native_preflight
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
    DecoupledPacketBudget, summarize_selection, selection_checks)
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.service_trace import TailServiceTrace
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
            or candidate['feedback_slots'] != 285 or candidate['policy_updates'] != 456
            or candidate['critic_updates'] != 456 or candidate['value_updates'] != 456):
        raise ValueError('The actual-control candidate changed its fixed native/update budget')
    p.update(name=candidate['name'], candidate='executed_cql',
        training_seeds=candidate['development_seeds'],
        target_seed_offset=candidate['external_seed_offset'],
        candidate_change=candidate['candidate_change'], original_replay_window=285,
        critic_fresh_outcome_window=30, outcome_seed_offset=600000,
        source_history_target='original .1V+.9SAC', fresh_target='original .1V+.9SAC',
        scheduler_priority='high', policy_update_budget=456, policy_updates_per_packet=8,
        critic_update_budget=456, value_update_budget=456, critic_updates_per_packet=8, feedback_selection=candidate['feedback_selection'],
        recipe=candidate['recipe'], actor_update_budget=456, alpha_update_budget=456, actor_schedule=candidate['actor_schedule'])
    p.update(information_contract=candidate['information_contract'],
        target_record_budget_scope='Legacy285 counter means admitted latest5 covariate s/a rows; '
            'complete outcome slots and distinct learner record union are accounted separately.',
        data_rights='Each day admits285 own-trajectory latest5 state/action rows for actor,V,alpha,CQL. '
            'Fresh separately admits285 complete risk-selected factual outcomes for recent30 critic TD; '
            'their actual selected overlap and union are reported, with at most342 unique learner '
            'target records. No_fresh learns from285 state/action rows only, retains zero complete '
            'target outcomes, and uses complete historical critic replacements. Risk outcome '
            'selection metadata is passive for no_fresh. No counterfactual queries.',
        data_permissions=dict(covariate_slots_per_day=285,
            complete_outcome_slots_per_fresh_day=285,
            complete_outcome_slots_per_no_fresh_day=0,
            maximum_unique_learner_target_records_per_fresh_day=342,
            maximum_unique_learner_target_records_per_no_fresh_day=285,
            maximum_selected_union_per_day=342,
            counterfactual_queries=0, paired_training_labels=0,
            fresh=dict(actor_value_temperature_cql='own-trajectory original latest5 s,a only',
                critic='recent30 independently selected complete factual outcome records'),
            no_fresh=dict(actor_value_temperature_cql='own-trajectory original latest5 s,a only',
                critic='complete historical replacement; no target outcome fields copied or retained'),
            diagnostic_native_history='Passive service logs remain resident; no diagnostic truth '
                'or passenger identity enters selection or learning.'),
        compute_budget=dict(new_main_days=5, fresh_main_days=4, no_fresh_regression_days=1,
            cached_v70_no_fresh_comparison_days=4, new_main_sim_seconds=120000,
            covariate_slots=1425, fresh_supervised_outcomes=1140,
            updates_per_optimizer=2280, counterfactual_branches=0, heldout_queries=0,
            world_updates=0, trust_updates=0, extra_short_preflights=0,
            initializer_retraining_updates=0),
        stages=dict(first_pair='6201stable fresh and no_fresh full days run concurrently; '
                'first8 updates are reused technical preflight, not extra days.',
            remaining='Only after native preflight and complete exact v70 no_fresh regression pass, '
                'run the remaining three fresh full days in parallel, independently of cost sign.',
            analysis='Compare four new fresh days against four immutable v70 no_fresh days; '
                'retain v70/v72 fresh descriptive references; no automatic confirmation or retries.'),
        interpretation='Known development seeds6201/6203 in stable and shock; four new fresh days '
            'plus one no_fresh full-day regression, with four cached v70 no_fresh cost comparisons. '
            'Expanded information contract is285 covariate selections plus285 complete fresh '
            'outcome labels with actual union reported, at most342 distinct learner records per '
            'fresh day. This is not independent confirmation and is not a285-total-record trial.',
        control_comparison='Each fresh arm controls its own trajectory under matched seed, scenario, '
            'initializer, timing,F,trust and optimizer budgets; compare to cached v70 no_fresh. '
            'Current covariates are never imported from cached paths. Complete outcome feedback '
            'and state/action record selection have separate explicit information budgets.',
        primary_endpoint=candidate['primary_metric'])
    return p


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
        candidate='executed_cql', recipe=p['recipe'], external_seed=external_seed)
    env = hook = history = trace = stream = learner = replay = budget = service = None
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
        budget = DecoupledPacketBudget(p)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        frozen_counts, freeze_clock = None, None
        pre_update_prefix = None
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
        service = TailServiceTrace(env._bridge, out/'service_trace.jsonl', detail_from=0.)
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
                    packet, outcome_packet = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    if budget.index == 1:
                        trace.writer.flush()
                        pre_update_prefix = dict(trace_path=str(trace.path),
                            rows=trace.observed_issues, clock=stream.now,
                            initial_updates=learner.total_updates)
                    replay.add_target(packet, outcome_packet)
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
                            runtime_contract=p['runtime_contract'],
                            prefix_before_first_update=pre_update_prefix, outcome_packet=outcome_packet))
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
        service_summary = service.finish()
        # The reusable observer's default label mentions16200; this run requested
        # all-day inventories explicitly and keeps those larger records resident.
        service_summary['service_sample_scope'] = ('All native service events and physical '
            'queue/onboard inventories from0 seconds; all passengers eligible; no extra simulation steps.')
        write(out/'service_summary.json', service_summary)
        ledger = env._bridge.passenger_time_ledger.snapshot()
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        passengers = summarize_passengers_v5(env._bridge, ledger)
        departed = passengers['departed_passengers']
        cost = 2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds
        with (out/'metrics.jsonl').open('x') as metrics:
            for row in hook.metrics:
                metrics.write(json.dumps(dict(row, departed_passengers=departed))+'\n')
        feedback_ids = [seq for receipt in budget.receipts for seq in receipt['sequence_ids']]
        capacity_selection = summarize_selection(budget.receipts)
        capacity_checks = selection_checks(budget.receipts)
        checks = dict(capacity_feedback_selection=all(capacity_checks.values()),
            unchanged_v70_learner=learner.__class__.__module__ ==
                'experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.learning',
            initializer_50000=initial_updates == p['initializer_updates'],
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
            covariates_only_state_action=all(set(row) == {'s', 'a'} for row in replay.recent),
            complete_private_outcomes=(len(replay.fresh_outcomes) == (285 if args.arm == 'fresh' else 0)
                and all(set(row) == {'s','a','r','ns','dt','terminal'} for row in replay.fresh_outcomes)),
            separate_pool_information_budget=(replay.summary()['covariate_unique_records'] == 285
                and replay.summary()['outcome_unique_records'] == (285 if args.arm == 'fresh' else 0)
                and replay.summary()['distinct_target_records_used'] <= (342 if args.arm == 'fresh' else 285)
                and replay.summary()['selected_target_record_union'] == capacity_selection['target_record_union']),
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
                training_fresh_outcomes=len(replay.outcome_ids),
                current_state_action_slots=len(replay.covariate_ids),
                covariate_slots=len(replay.covariate_ids),
                complete_outcome_slots=len(replay.outcome_ids),
                distinct_target_records_used=replay.summary()['distinct_target_records_used'],
                unique_learner_target_records=replay.summary()['distinct_target_records_used'],
                selected_target_record_union=capacity_selection['target_record_union'],
                selected_union_count=capacity_selection['target_record_union'],
                distinct_target_record_bound=342 if args.arm == 'fresh' else 285,
                policy_updates=counts()['policy'], critic_updates=counts()['critic'],
                value_updates=counts()['value'], actor_updates=counts()['actor'], alpha_updates=counts()['alpha'],
                packets=budget.index, native_seconds=stream.now,
                world_updates=0, trust_updates=0, counterfactual_queries=0,
                training_stopped_at=freeze_clock),
            receipts=budget.receipts, replay=replay.summary(), runtime_contract=runtime,
            capacity_selection=capacity_selection, selection_checks=capacity_checks,
            passengers=passengers, service_summary_path=str(out/'service_summary.json'),
            service_trace_path=str(out/'service_trace.jsonl'), service_detail_from_seconds=0.,
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
        write(out/'summary.json', summary)
        write(out/'gate.json', {key: summary[key] for key in ('status', 'integrity', 'cell', 'budget')})
        progress(summary['status'])
        if not all(checks.values()):
            raise RuntimeError('v73 decoupled-feedback actual control integrity failed')
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
        if service is not None: service.close()
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
