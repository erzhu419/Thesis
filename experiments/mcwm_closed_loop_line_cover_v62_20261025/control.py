"""Actual full-day control: metadata-only line coverage in the original feedback pool."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal, write
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshOutcomeTrainer,
    TransitTrainerConfig, load_table, resident)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import EventStream, create_target_env
import numpy as np
import torch




FOCUS_LINES = ('7X', '102X', '705X')


def selected_sequence_ids(metadata, count=5):
    """Choose paid IDs solely from available issue/availability metadata."""
    ranked = sorted(metadata, key=lambda row: (row['available_at'], row['sequence']), reverse=True)
    chosen = []
    for line in FOCUS_LINES:
        latest = next((row for row in ranked if row['line_id'] == line), None)
        if latest is not None:
            chosen.append(latest['sequence'])
    chosen_set = set(chosen)
    for row in ranked:
        if len(chosen) == count:
            break
        if row['sequence'] not in chosen_set:
            chosen.append(row['sequence'])
            chosen_set.add(row['sequence'])
    return chosen


class LineCoveragePacketBudget(formal.PacketBudget):
    """Original ready/window/positive-dt pool; only the selected IDs differ."""
    def __init__(self, p):
        super().__init__(p)
        self.eligible_lines, self.selected_lines = Counter(), Counter()
        self.available_focus_packets, self.covered_focus_packets = Counter(), Counter()
        self.age_count, self.age_sum, self.age_min, self.age_max = 0, 0., None, None
        self.selection_exact = True

    def take(self, now):
        deadline = self.deadlines[self.index]
        if now < deadline:
            raise ValueError('Feedback deadline has not arrived')
        lower = deadline-self.p['feedback_interval_seconds']
        ready = [row for row in self.rows if row.available_at <= deadline]
        candidates = [row for row in ready if row.available_at > lower and row.dt > 0 and not row.terminal]
        n = self.p['feedback_records_per_packet']
        if len(candidates) < n:
            raise RuntimeError(f'Insufficient feedback at {deadline}: {len(candidates)} eligible, {n} required')
        # Reward, next-state and action values never enter the ID-selection helper.
        metadata = [dict(sequence=row.sequence, line_id=row.line_id,
            started_at=row.started_at, available_at=row.available_at) for row in candidates]
        selected_ids = selected_sequence_ids(metadata, n)
        by_id = {row.sequence: row for row in candidates}
        selected = [by_id[sequence] for sequence in selected_ids]
        selected.sort(key=lambda row: (row.available_at, row.sequence))
        selected_ids = {row.sequence for row in selected}
        discarded = [row.sequence for row in ready if row.sequence not in selected_ids]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        self.receipts.append(dict(deadline=deadline, processed_at=now, records=n,
            sequence_ids=[row.sequence for row in selected],
            max_available_at=max(row.available_at for row in selected)))
        self.index += 1
        pool_lines = Counter(row['line_id'] for row in metadata)
        chosen_metadata = [row for row in metadata if row['sequence'] in selected_ids]
        chosen_lines = Counter(row['line_id'] for row in chosen_metadata)
        available_focus = set(pool_lines).intersection(FOCUS_LINES)
        covered_focus = set(chosen_lines).intersection(FOCUS_LINES)
        focus_latest_ids = {line: max((row for row in metadata if row['line_id'] == line),
            key=lambda row: (row['available_at'], row['sequence']))['sequence'] for line in available_focus}
        self.selection_exact &= (len(selected) == len(selected_ids) == n
            and available_focus <= covered_focus and set(focus_latest_ids.values()) <= selected_ids)
        self.receipts[-1]['line_cover'] = dict(eligible_records=len(candidates),
            focus_latest_ids=focus_latest_ids, selected_line_counts=dict(chosen_lines))
        self.eligible_lines.update(pool_lines)
        self.selected_lines.update(chosen_lines)
        self.available_focus_packets.update(available_focus)
        self.covered_focus_packets.update(covered_focus)
        for row in chosen_metadata:
            age = float(now-row['started_at'])
            self.age_count += 1
            self.age_sum += age
            self.age_min = age if self.age_min is None else min(self.age_min, age)
            self.age_max = age if self.age_max is None else max(self.age_max, age)
        return selected, discarded

    def selection_checks_pass(self):
        return (self.selection_exact and self.age_count == self.index*self.p['feedback_records_per_packet']
            and self.available_focus_packets == self.covered_focus_packets)

    def summary(self):
        return dict(rule='At deadline, one latest eligible record per present focus line, '
                'then latest remaining records to five; original ready/time-window/dt/terminal rules.',
            focus_lines=list(FOCUS_LINES), selection_time='deadline using already available metadata',
            absent_lines_required=False, selection_reads_rewards_or_next_states=False,
            eligible_record_counts_by_line=dict(self.eligible_lines),
            selected_record_counts_by_line=dict(self.selected_lines),
            focus_available_packets=dict(self.available_focus_packets),
            focus_covered_packets=dict(self.covered_focus_packets),
            label_age_seconds=dict(count=self.age_count, minimum=self.age_min, maximum=self.age_max,
                mean=self.age_sum/self.age_count if self.age_count else None))


def protocol():
    base = HERE.parent/'mcwm_closed_loop_fresh_attribution_v25_20261023/protocol.json'
    p = json.loads(base.read_text())
    campaign = json.loads((HERE/'protocol.json').read_text())
    if (campaign['horizon_seconds'] != 24000 or campaign['initializer_updates'] != 50000
            or campaign['feedback_slots'] != 285 or campaign['policy_updates'] != 456):
        raise ValueError('The control campaign changed its fixed native/update budget')
    p.update(name='thesis-mcwm-line-cover-v62-20261025',
        training_seeds=campaign['development_seeds']+campaign['confirmation_seeds'],
        target_seed_offset=campaign['external_seed_offset'], candidate='line_cover',
        focus_lines=campaign['focus_lines'],
        candidate_change='Only metadata-based line coverage within the original five-slot eligible pool.',
        original_backup='v25 native single-step .1V+.9SAC; original replay285 and four draws; '
            'no recent30, M8, memory, trace or world.', scheduler_priority='high')
    return p


def load_learner(seed, *, arm):
    """Use the real seed archive and original replay; never substitute a seed."""
    torch.set_num_threads(1)
    inputs = resident.INPUTS
    metadata = json.loads((inputs/'history_metadata.json').read_text())
    with np.load(inputs/'history.npz') as archive:
        history = {key: archive[key] for key in (*FIELDS.values(), 'sample_weight')}
    source = load_table(inputs/'source_9101.npz', 4096)
    contract = resident.formal.TransitMDPConfigV7(**metadata['contract'])
    stations = np.asarray(metadata['station_max_by_line'], np.float32)
    initializer = inputs/f'h2oplus_seed{seed}.pt'
    payload = torch.load(initializer, map_location='cpu', weights_only=False)
    if (payload['total_updates'] != 50000 or payload['online_critic_updates'] != 0
            or payload['extra']['source_training_seed'] != seed):
        raise ValueError('Wrong read-only initializer')
    config = replace(TransitTrainerConfig(**payload['source_config']), use_ratio_correction=False)
    if config.batch_size != 256:
        raise ValueError('Actual control requires the original128/64/64 quotas')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, window=285)
    learner = FreshOutcomeTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, contract, stations, initializer


def run(args):
    p = protocol()
    if (args.candidate != 'line_cover' or args.arm not in p['arms']
            or args.seed not in p['training_seeds']):
        raise ValueError('Unsupported actual-control candidate or arm')
    # Development seeds6201/6202 are followed only by explicitly chosen seed files.
    # load_learner validates every new seed against its real50000-step archive.
    external_seed = getattr(args, 'external_seed', None)
    external_seed = args.seed+p['target_seed_offset'] if external_seed is None else external_seed
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cell = dict(seed=args.seed, scenario=args.scenario, arm=args.arm,
        candidate=args.candidate, external_seed=external_seed)
    env = hook = history = trace = stream = learner = replay = budget = None
    initial_updates = None
    native_started_wall = native_finished_wall = None

    def counts():
        return dict(policy=(learner.total_updates-initial_updates if learner is not None else 0),
            world=0, trust=0)

    def progress(phase, **values):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None else 0., total=p['horizon_seconds'],
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started, **values)), flush=True)

    try:
        progress('load_initializer')
        learner, replay, contract, stations, initializer = load_learner(args.seed, arm=args.arm)
        initial_updates = learner.total_updates
        transform = learner.action_transform
        budget = LineCoveragePacketBudget(p)
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
        with (out/'updates.jsonl').open('x') as updates:
            while not stream.done:
                budget.ingest(stream.step(policy))
                stream.release_intervals()
                while budget.due(stream.now):
                    packet, _ = budget.take(stream.now)
                    if clock_audit['first_feedback_applied_at'] is None:
                        clock_audit['first_feedback_applied_at'] = stream.now
                    replay.add_target(packet)
                    for step in range(p['policy_updates_per_packet']):
                        metrics = learner.continue_step(collect_diagnostics=step == 0)
                        if step == 0:
                            diagnostic = metrics['td_diagnostic']
                            if diagnostic['real_scale'] != 1 or diagnostic['model_scale'] != 0:
                                raise RuntimeError('The original real-only objective changed')
                            updates.write(json.dumps(dict(packet=budget.index,
                                processed_at=stream.now, **metrics), allow_nan=False)+'\n')
                    updates.flush()
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
        feedback_ids = [seq for receipt in budget.receipts for seq in receipt['sequence_ids']]
        checks = dict(initializer_50000=initial_updates == p['initializer_updates'],
            history_source_unchanged=len(replay.history['s']) == p['history_rows']
                and len(replay.source['s']) == p['source_rows'],
            factual_feedback_285=replay.target_received == p['target_record_budget'],
            unique_factual_labels=len(feedback_ids) == len(set(feedback_ids)) == p['target_record_budget'],
            full_packets=budget.index == 57 and all(r['records'] == 5 for r in budget.receipts),
            no_early_feedback=all(r['max_available_at'] <= r['deadline'] <= r['processed_at']
                for r in budget.receipts),
            policy_updates_456=counts()['policy'] == p['policy_update_budget'],
            no_world=learner.world is None,
            no_fresh_outcomes_retained=(args.arm == 'fresh' or all(
                set(row) == {'s', 'a'} for row in replay.recent)),
            current_covariates_285=len(replay.recent) == 285 and replay.recent.maxlen == 285,
            original_replay_and_sampler=type(replay) is FreshOutcomeReplay,
            line_selection_complete=budget.selection_checks_pass(),
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
                policy_updates=counts()['policy'], packets=budget.index, native_seconds=stream.now,
                world_updates=0, trust_updates=0, counterfactual_queries=0,
                training_stopped_at=freeze_clock),
            receipts=budget.receipts, replay=replay.summary(), runtime_contract=runtime,
            feedback_selection=budget.summary(),
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
            raise RuntimeError('v62 actual control integrity failed')
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None,
            consumed_budget=dict(feedback_slots=replay.target_received if replay is not None else 0,
                policy_updates=counts()['policy'], packets=budget.index if budget is not None else 0,
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
    parser.add_argument('--candidate', choices=('line_cover',), default='line_cover')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh', 'no_fresh'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external-seed', type=int)
    run(parser.parse_args())
