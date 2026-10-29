"""Budget-matched actual control with singleton-support TD gradients blocked only for theta."""
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
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import ContextReplay, ContextTrainer
from experiments.mcwm_closed_loop_support_td_v83_20261027.learning import (
    make_learner as load_learner, singleton_td_audit)
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.learning import (
    cql_checks, recency_checks, target_checks)
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.stream import (
    ContextEventStream, NativeContextReader)
from experiments.mcwm_closed_loop_support_td_v83_20261027.context_audit import (
    ContextAudit, FirstPacketAudit, native_preflight)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.policy_probe import PacketPolicyProbe
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
    DecoupledPacketBudget, summarize_selection, selection_checks)
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.service_trace import TailServiceTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import create_target_env
import numpy as np
import torch
from copy import deepcopy
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _state, _rng




def protocol():
    from experiments.mcwm_closed_loop_capacity_control_v81_20261027.control import protocol as previous
    p = previous()
    candidate = json.loads((HERE/'protocol.json').read_text())
    if (candidate['horizon_seconds'] != 24000 or candidate['initializer_updates'] != 50000
            or candidate['feedback_slots'] != 285
            or candidate['development_seeds'] != [6207, 6208]
            or any(candidate[key] != 456 for key in
                ('policy_updates','critic_updates','value_updates','actor_updates','alpha_updates'))):
        raise ValueError('v83 fixed cohort or budget changed')
    p.update(name=candidate['name'], training_seeds=candidate['development_seeds'],
        preflight_seed=6208, candidate_change='Only singleton-F factual TD gradient into theta is stopped; baseQ and all other learning paths unchanged',
        scheduler_priority='high',
        compute_budget=dict(new_main_days=8, fresh_main_days=4, no_fresh_main_days=4,
            cached_comparison_days=16, new_main_sim_seconds=192000,
            covariate_slots=2280, fresh_supervised_outcomes=1140,
            updates_per_optimizer=3648, counterfactual_branches=0, heldout_queries=0,
            world_updates=0, trust_updates=0, extra_short_preflights=0,
            initializer_retraining_updates=0),
        stages=dict(first_pair='6208stable concurrent fresh/no_fresh; first8 updates reused',
            remaining='Six remaining full days after first-pair technical and common-prefix gates',
            analysis='Full-day primary, stable tails, worst cell; preserve old v75/v81 references'),
        interpretation='Known6207/6208 developer conditions, not independent confirmation. '
            'Only singleton-support TD theta gradient differs from v81. Actor/V/F observations unchanged. '
            'Expanded context information is declared separately; slots/updates unchanged.',
        initialization_budget=dict(new_initialization_updates=0,
            reused_v75_seed_archives=[6207,6208]),
        information_contract=candidate['information_contract'],
        context_contract=candidate['context_contract'], seed_scope=candidate['seed_scope'],
        validation_cost_gate=candidate['cost_gate'], primary_endpoint=candidate['primary_metric'],
        rng_scope=candidate['rng_scope'],
        control_comparison='Both arms use matched v75 initializer and native seed, v80 critic '
            'context and their own actual trajectories. Compare to frozen v75/v81 absolute costs.',
        old_v73_gate_conclusion='closed: two stable tails failed',
        td_gradient_gate_contract=candidate['td_gradient_gate_contract'],
        v81_gate_conclusion='closed: fixed development cost gates failed')
    return p


def target_audit_exact(learner, arm):
    return all(target_checks(learner.target_audit(), arm, calls=456).values())


def save_final_learner(learner, path, *, cell, clock, counts):
    """One resident freeze snapshot; no replay arrays or new optimizer step."""
    torch.save(dict(state=_state(learner), global_rng=_rng(),
        outcome_rng=deepcopy(learner.replay_buffer.outcome_rng.bit_generator.state),
        config=asdict(learner.config), cell=cell, frozen_at=clock, counts=counts,
        context_source='experiments.mcwm_closed_loop_context_wiring_v80_20261027',
        learning_source='experiments.mcwm_closed_loop_support_td_v83_20261027',
        td_gradient_gate=singleton_td_audit(learner),
        replay_stored=False), path)


def run(args):
    p = protocol()
    if args.arm not in p['arms'] or args.seed not in p['training_seeds']:
        raise ValueError('Unsupported actual-control candidate or arm')
    # Fixed new training-seed cohort; native costs never select a seed or initializer.
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
    env = hook = history = trace = stream = learner = replay = budget = service = context_audit = None
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
        context_audit = ContextAudit(out, args.arm)
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
            context_audit.observe_issue(event)
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
        native_context_reader = NativeContextReader(env)
        stream = ContextEventStream(env, stations, contract, domain='target',
            context_reader=lambda issued: context_audit.read_context(native_context_reader, issued),
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
                        context_audit.issue_log.flush()
                        pre_update_prefix = dict(trace_path=str(trace.path),
                            rows=trace.observed_issues, clock=stream.now,
                            initial_updates=learner.total_updates,
                            context_trace_path=str(out/'context_issues.jsonl'),
                            context_rows=context_audit.issues['total'])
                    replay.add_target(packet, outcome_packet)
                    context_audit.record_packet(packet, outcome_packet, packet=budget.index,
                        processed_at=stream.now, replay=replay)
                    first_audit = FirstPacketAudit(learner) if budget.index == 1 else None
                    learner.begin_actor_packet()
                    for step in range(p['critic_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if first_audit is not None:
                            first_audit.record_update()
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
                        context_audit.first_packet = first_audit.finish(out)
                        first_preflight = native_preflight(learner, replay, contract,
                            initializer, packet, budget.receipts[-1], clock_audit, action_audit,
                            seed=args.seed, scenario=args.scenario, arm=args.arm,
                            runtime_contract=p['runtime_contract'],
                            prefix_before_first_update=pre_update_prefix, outcome_packet=outcome_packet,
                            context=context_audit, first_packet=context_audit.first_packet)
                        write(out/'preflight.json', first_preflight)
                        if not first_preflight['technical_gate_pass']:
                            raise RuntimeError('v83 reused first8 technical preflight failed')
                    progress('packet_training', packets=budget.index,
                        feedback_slots=replay.target_received, critic_updates=counts()['critic'],
                        actor_updates=counts()['actor'], alpha_updates=counts()['alpha'])
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    frozen_counts, freeze_clock = counts(), stream.now
                    save_final_learner(learner, out/'final_learner.pt', cell=cell,
                        clock=freeze_clock, counts=frozen_counts)
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
        context_summary = context_audit.summary(replay, stream.issued_count, learner=learner)
        write(out/'context_summary.json', context_summary)
        td_gradient_gate = context_summary['td_gradient_gate']
        checks = dict(td_gradient_gate=td_gradient_gate['all_pass'],
            context_contract=context_summary['integrity']['all_pass'], capacity_feedback_selection=all(capacity_checks.values()),
            context_critic_trainer=type(learner) is ContextTrainer,
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
            fixed_recent30_context_replay=type(replay) is ContextReplay,
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
            context_summary=context_summary, context_summary_path=str(out/'context_summary.json'),
            td_gradient_gate=td_gradient_gate,
            final_learner_path=str(out/'final_learner.pt'),
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
            raise RuntimeError('v83 support-TD actual control integrity failed')
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
        if context_audit is not None: context_audit.close()
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
