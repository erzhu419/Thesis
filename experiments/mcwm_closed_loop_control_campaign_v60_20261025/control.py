"""Actual full-day control: change only the fresh critic outcome age window."""
from __future__ import annotations

import argparse
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
    TransitTrainerConfig, _batch, load_table, resident)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import IssueAudit
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, UPSTREAM_STOP
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import NativeRunHistory
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from stream import EventStream, create_target_env
import numpy as np
import torch


class RecentOutcomeReplay(FreshOutcomeReplay):
    """Keep285 current s/a; critic fresh outcomes use the latest30 paid labels.

    The resident four draws and their RNG remain untouched. A private fifth draw
    advances in both arms; only fresh replaces its outcome quarter with those
    complete transitions. The no_fresh batch is exactly the resident batch.
    """

    def __init__(self, *args, outcome_seed, **kwargs):
        super().__init__(*args, window=285, **kwargs)
        self.outcome_rng = np.random.default_rng(outcome_seed)
        self.outcome_seed = int(outcome_seed)
        self.outcome_sample_calls = 0
        self.maximum_sampled_label_age = 0
        self.last_outcome_sampling = None

    def sample_mixed(self, size, online_fraction, *, rng):
        target, source = super().sample_mixed(size, online_fraction, rng=rng)
        count = size // 4
        window = min(30, len(self.recent))
        local_ids = self.outcome_rng.integers(window, size=count)
        self.outcome_sample_calls += 1
        self.last_outcome_sampling = dict(window=window, local_indices=local_ids.tolist())
        if self.arm == 'fresh':
            tail = list(self.recent)[-window:]
            table = {key: np.asarray([row[key] for row in tail]) for key in FIELDS.values()}
            outcomes = _batch(table, local_ids, self.device)
            target = {key: torch.cat((value[:count], outcomes[key])) for key, value in target.items()}
            self.maximum_sampled_label_age = max(self.maximum_sampled_label_age,
                int(np.max(window-1-local_ids)))
        return target, source

    def summary(self):
        return dict(super().summary(), covariate_window=285,
            critic_fresh_outcome_window=30 if self.arm == 'fresh' else 0,
            independent_outcome_sampler_seed=self.outcome_seed,
            independent_outcome_sampler_calls=self.outcome_sample_calls,
            maximum_fresh_outcome_age_in_admitted_labels=(
                self.maximum_sampled_label_age if self.arm == 'fresh' else None),
            no_fresh_original_batch_unchanged=self.arm == 'no_fresh',
            recency_semantics='Latest30 admitted complete outcomes; all285 current s/a retained. '
                'Independent outcome draw advances in both arms; original four draws unchanged.')


def protocol():
    base = HERE.parent/'mcwm_closed_loop_fresh_attribution_v25_20261023/protocol.json'
    p = json.loads(base.read_text())
    campaign = json.loads((HERE/'protocol.json').read_text())
    if (campaign['horizon_seconds'] != 24000 or campaign['initializer_updates'] != 50000
            or campaign['feedback_slots'] != 285 or campaign['policy_updates'] != 456):
        raise ValueError('The control campaign changed its fixed native/update budget')
    p.update(name='thesis-mcwm-actual-control-recent30-v60-20261025',
        training_seeds=campaign['development_seeds']+campaign['confirmation_seeds'],
        target_seed_offset=campaign['recent30']['external_seed_offset'], candidate='recent30',
        candidate_change='Critic fresh-outcome window30 only; current s/a window285 in both arms.',
        original_backup='v25 native single-step .1V+.9SAC; no M8, memory, trace or world.',
        scheduler_priority='high')
    return p


def load_learner(seed, *, arm, outcome_seed):
    """Use the actual seed's archived initializer; never substitute another seed."""
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
    replay = RecentOutcomeReplay(history, source, contract, stations, arm=arm,
        outcome_seed=outcome_seed)
    learner = FreshOutcomeTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, contract, stations, initializer


def run(args):
    p = protocol()
    if (args.candidate != 'recent30' or args.arm not in p['arms']
            or args.seed not in p['training_seeds']):
        raise ValueError('Unsupported actual-control candidate or arm')
    # Development seeds6201/6202 are followed only by explicitly chosen seed files.
    # load_learner validates every new seed against its real50000-step archive.
    external_seed = getattr(args, 'external_seed', None)
    external_seed = args.seed+p['target_seed_offset'] if external_seed is None else external_seed
    outcome_seed = getattr(args, 'outcome_seed', None)
    outcome_seed = args.seed+600000 if outcome_seed is None else outcome_seed
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
        learner, replay, contract, stations, initializer = load_learner(args.seed,
            arm=args.arm, outcome_seed=outcome_seed)
        initial_updates = learner.total_updates
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
            policy_config=asdict(learner.config), initializer=str(initializer),
            outcome_sampler_seed=outcome_seed))
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
            fresh_critic_outcome_age_at_most29=(args.arm != 'fresh' or
                replay.maximum_sampled_label_age <= 29),
            independent_outcome_sampler_456=replay.outcome_sample_calls == 456,
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
            raise RuntimeError('v60 actual control integrity failed')
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
    parser.add_argument('--candidate', choices=('recent30',), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh', 'no_fresh'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external-seed', type=int)
    parser.add_argument('--outcome-seed', type=int)
    run(parser.parse_args())
