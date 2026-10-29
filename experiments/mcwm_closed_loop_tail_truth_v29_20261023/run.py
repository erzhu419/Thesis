"""Two fixed native action pairs while preserving the complete v28 parent day."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import sys

from bootstrap import HERE, formal, write
import numpy as np
from stream import create_target_env
from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from experiments.mcwm_closed_loop_cumulative_shadow_v28_20261023.shadow import CumulativeShadow
from experiments.mcwm_closed_loop_tail_truth_v29_20261023.suffix import NativeTruthPair, selected_state, plain


def fixed_probes(reference, protocol):
    """Old factual states are evaluation inputs only; no outcomes are loaded."""
    chosen = {float(d): {} for d in protocol['bank_deadlines']}
    with (reference/'prefix_trace.jsonl').open() as handle:
        for text in handle:
            row = json.loads(text)
            line, now = row['line_id'], row['started_at']
            if line not in protocol['probe_lines']:
                continue
            for deadline, rows in chosen.items():
                start, stop = protocol['shadow_probe_windows'][str(int(deadline))]
                if start <= now < stop and line not in rows:
                    rows[line] = dict(s=np.asarray(row['s'],np.float32), line_id=line,
                        station_id=row['station_id'], sequence=row['sequence'],
                        started_at=now, issue_time=now,
                        role='tail_heldout_state' if deadline==18000 else 'future_reference_state')
    return chosen


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    if args.seed not in p['training_seeds'] or args.arm not in p['arms']:
        raise ValueError('Cell outside the fixed v29 diagnostic')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cell = dict(seed=args.seed, scenario=args.scenario, origin_arm=args.arm)
    reference = HERE.parents[1]/p['reference_roots'][str(args.seed)][args.scenario]
    original = json.loads((reference/'summary.json').read_text())
    original_updates = [json.loads(text) for text in (reference/'updates.jsonl').read_text().splitlines()]
    original_shadows = [json.loads(t) for t in (reference/'shadows.jsonl').read_text().splitlines()]
    chosen = json.loads((HERE/'selected_reference.json').read_text())[args.scenario]
    reference_selected = {key:chosen['selected'][key] for key in ('sequence','started_at','line_id','vehicle_id','trip_id','station_id','s','raw_s','event','issue_ledger')}
    probes = fixed_probes(reference,p)
    shadow_records = []
    bank_rows, missing_bank = [], []
    for deadline in p['bank_deadlines']:
        for line in p['probe_lines']:
            row = probes[float(deadline)].get(line)
            if row is None:
                missing_bank.append(dict(bank_deadline=deadline,line_id=line))
            else:
                bank_rows.append(dict(row,bank_id=f'{deadline}/{line}',bank_deadline=deadline))
    bank_states = np.stack([row['s'] for row in bank_rows])
    bank_metadata = [{k:v for k,v in row.items() if k!='s'} for row in bank_rows]
    pair = None
    truth_pair = None
    pre_recorded_sequence = None
    prediction_evidence = None
    updates_exact = True
    env = hook = history = trace = stream = None

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
        trace = NativePrefixTrace(out/'prefix_trace.jsonl',reference=reference/'prefix_trace.jsonl')
        audit = IssueAudit(p, transform)
        clock_audit = dict(actor_real_clock=True, max_actor_clock_error_seconds=0.,
            first_feedback_applied_at=None, native_policy_issues=0)
        action_audit = dict(shared_rule_exact=True, compared_issues=0,
            max_shared_hold_gap_seconds=0., raw_legacy_disagreements=0,
            max_raw_legacy_hold_gap_seconds=0.)
        frozen_counts, freeze_clock = None, None
        frozen_shadow_counts = None
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

        def all_counts():
            shadow = pair.update_counts() if pair else dict(fresh=0,no_fresh=0)
            return dict(**counts(),feedback=replay.target_received,
                fresh_shadow=shadow['fresh'],no_fresh_shadow=shadow['no_fresh'])

        def selector(event):
            nonlocal truth_pair,pre_recorded_sequence,prediction_evidence
            if event['sequence'] != reference_selected['sequence']:
                return None
            if truth_pair is not None:
                raise RuntimeError('The predeclared native state was selected twice')
            if selected_state(event,stream._snapshot()) != reference_selected:
                raise RuntimeError('The complete predeclared state/ledger differs from v28')
            if (budget.index != 57 or counts()['policy'] != 456
                    or pair.update_counts() != dict(fresh=248,no_fresh=248)):
                raise RuntimeError('The selected tail state is not after the unchanged training budget')
            current = next(row for row in pair._scores() if row['bank_id']==p['selected_bank_id'])
            if current != chosen['probe']:
                raise RuntimeError('Saved v28 full-precision candidate predictions did not reproduce')
            prediction_evidence = {arm:dict(min_q_low=current['q_min'][arm][3],
                min_q_high=current['q_min'][arm][4]) for arm in ('fresh','no_fresh')}
            actions = dict(low=[current['executed_action'][3]],high=[current['executed_action'][4]])
            if current['holding_seconds'][3:] != [0.,.9375]:
                raise RuntimeError('Saved candidate physical holdings changed')
            event['proposed_a'] = last_proposal.copy()
            trace.observe(event,stream._snapshot())
            pre_recorded_sequence = event['sequence']
            truth_pair = NativeTruthPair(out/'branches',dict(selected=reference_selected,
                actions=actions),all_counts,args.seed+p['target_seed_offset'])

            def prepare_child(arm):
                updates.close()
                shadows.close()

            return truth_pair.fork(event,stream,trace,prepare_child=prepare_child)

        def issue(event):
            event['proposed_a'] = last_proposal.copy()
            if truth_pair is not None and truth_pair.in_child:
                truth_pair.observe_issue(event,stream)
                return
            if event['sequence'] != pre_recorded_sequence:
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
        stream = ForkEventStream(env, stations, contract, domain='target',
            selector=selector,fork_duration_seconds=300.,
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=issue)
        stream.reset()
        history = NativeRunHistory(env._bridge)
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        native_started_wall = time.time()
        last_progress = time.monotonic()
        with (out/'updates.jsonl').open('x') as updates, (out/'shadows.jsonl').open('x') as shadows:
            while not stream.done:
                released = stream.step(policy)
                if truth_pair is not None and truth_pair.in_child:
                    stream.release_intervals()
                    truth_pair.observe_progress(stream)
                    if stream.cutoff_reached:
                        truth_pair.finish_child(stream)
                    continue
                budget.ingest(released)
                # Release same-car records; none are admitted as extra labels.
                stream.release_intervals()
                while budget.due(stream.now):
                    deadline = budget.deadlines[budget.index]
                    packet, _ = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    replay.add_target(packet)
                    shadow_record = None
                    if deadline in p['shadow_deadlines']:
                        if pair is None:
                            pair = CumulativeShadow(learner,bank_states,bank_metadata)
                        metadata = dict(cell=cell,seed=args.seed,scenario=args.scenario,
                            deadline=deadline,processed_at=stream.now,
                            missing_probes=missing_bank,packet_sequence_ids=[row.sequence for row in packet],
                            packet_issue_range=[min(row.started_at for row in packet),max(row.started_at for row in packet)],
                            packet_completed_range=[min(row.completed_at for row in packet),max(row.completed_at for row in packet)],
                            packet_available_range=[min(row.available_at for row in packet),max(row.available_at for row in packet)])
                        shadow_record = pair.advance(learner,packet,metadata)
                    for step in range(p['policy_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if step == 0:
                            diagnostic = metrics['td_diagnostic']
                            if diagnostic['real_scale'] != 1 or diagnostic['model_scale'] != 0:
                                raise RuntimeError('The matched real-only objective changed')
                            update_row = dict(packet=budget.index, processed_at=stream.now, **metrics)
                            updates_exact &= update_row == original_updates[budget.index-1]
                            if not updates_exact:
                                raise RuntimeError('Main update differs from the immutable original day')
                            updates.write(json.dumps(update_row, allow_nan=False)+'\n')
                    if shadow_record is not None:
                        comparison = pair.compare_after_main(learner)
                        shadow_record['fresh_main_replay'] = comparison
                        shadow_records.append(shadow_record)
                        shadows.write(json.dumps(shadow_record,allow_nan=False)+'\n')
                        shadows.flush()
                        if not comparison['exact'] or not all(shadow_record['integrity'].values()):
                            raise RuntimeError('Eight-step shadow or unchanged-main verification failed')
                        if shadow_record != original_shadows[len(shadow_records)-1]:
                            raise RuntimeError('Cumulative v28 diagnostics or predictions differ')
                    updates.flush()
                    progress('packet_training', packets=budget.index,
                        feedback_slots=replay.target_received, policy_updates=counts()['policy'],
                        diagnostic_updates=sum(pair.update_counts().values()) if pair else 0)
                if budget.index == len(budget.deadlines) and frozen_counts is None:
                    frozen_counts, freeze_clock = counts(), stream.now
                    frozen_shadow_counts = pair.update_counts()
                if time.monotonic()-last_progress >= 45:
                    progress('native_day', packets=budget.index, policy_updates=counts()['policy'],
                        diagnostic_updates=sum(pair.update_counts().values()) if pair else 0)
                    last_progress = time.monotonic()
        native_finished_wall = time.time()
        hook.capture_final()
        ledger = env._bridge.passenger_time_ledger.snapshot()
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        passengers = summarize_passengers_v5(env._bridge, ledger)
        departed = passengers['departed_passengers']
        cost = 2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds
        metric_rows = [dict(row,departed_passengers=departed) for row in hook.metrics]
        with (out/'metrics.jsonl').open('x') as metrics:
            for row in metric_rows:
                metrics.write(json.dumps(row)+'\n')
        original_metrics = [json.loads(text) for text in (reference/'metrics.jsonl').read_text().splitlines()]
        cost_record = dict(departed_passengers=departed,accrued_wait_seconds=ledger.accrued_wait_seconds,
            accrued_in_vehicle_seconds=ledger.accrued_in_vehicle_seconds,
            generalized_passenger_seconds=cost,generalized_seconds_per_departed_passenger=cost/departed)
        exact = dict(trace=trace.comparison_exact and trace.reference.readline()==''
            and trace.matched_issues==trace.observed_issues,
            updates=updates_exact and budget.index==len(original_updates),
            receipts=budget.receipts==original['receipts'],cost=cost_record==original['cost'],
            metrics=metric_rows==original_metrics)
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
            full_trace_exact=exact['trace'],updates_exact=exact['updates'],receipts_exact=exact['receipts'],
            cost_exact=exact['cost'],metrics_exact=exact['metrics'],
            fresh_shadow_exact_main=all(row['fresh_main_replay']['exact'] for row in shadow_records),
            shadow_origin_and_rng_unchanged=all(all(row['integrity'].values()) for row in shadow_records),
            anchors_complete=[row['metadata']['deadline'] for row in shadow_records]==p['shadow_deadlines'],
            extra_probes_not_training=replay.target_received==p['target_record_budget'],
            cumulative_shadow_updates=pair.update_counts()==dict(fresh=248,no_fresh=248),
            post_budget_zero_shadow_training=pair.update_counts()==frozen_shadow_counts,
            cumulative_v28_records_exact=shadow_records==original_shadows,
            fixed_state_exact=truth_pair is not None and truth_pair.values is not None
                and all(v['selected']==reference_selected for v in truth_pair.values.values()),
            min_q_prediction_exact=prediction_evidence is not None,
            parallel_native_pair=truth_pair is not None and truth_pair.receipt is not None
                and truth_pair.receipt['all_pass'])
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), cell=cell,
            initializer=dict(path=str(initializer), initial_updates=initial_updates,
                history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
            budget=dict(native_replay_seconds=stream.now,feedback_slots=replay.target_received,
                main_updates=counts()['policy'],shadow_packets=len(shadow_records),shadow_arms=2,
                shadow_updates=sum(pair.update_counts().values()),
                extra_target_labels=0,real_control_branches=2,
                main_feedback_slots=replay.target_received,cumulative_shadow_updates=sum(pair.update_counts().values()),
                prefix_replay_seconds=reference_selected['started_at'],
                diagnostic_suffix_seconds=600,new_training_labels=0,truth_labels_enter_training=0),
            selected=reference_selected,prediction=prediction_evidence,
            external_seed=args.seed+p['target_seed_offset'],
            native_pair=truth_pair.receipt if truth_pair else None,
            main_training_stopped_at=freeze_clock,world_updates=0,trust_updates=0,
            receipts=budget.receipts, replay=replay.summary(), runtime_contract=runtime,
            clock_audit=clock_audit, action_audit=action_audit,
            post_budget_counts=dict(before=frozen_counts['policy'] if frozen_counts else None, after=counts()['policy']),
            replay_exact=exact,reference_path=str(reference),
            post_budget_shadow_counts=dict(before=frozen_shadow_counts,after=pair.update_counts()),
            missing_bank_states=missing_bank,fixed_bank_metadata=bank_metadata,
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
            raise RuntimeError('v29 parent or native branch integrity failed')
        return summary
    except Exception as error:
        if truth_pair is not None and truth_pair.in_child:
            write(truth_pair.out/truth_pair.arm/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),native_time=stream.now))
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None))
        raise
    finally:
        if truth_pair is not None and truth_pair.in_child:
            os._exit(1)
        if history is not None: history.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh',), default='fresh')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
