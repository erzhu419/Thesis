"""One packet37 reference prefix and two real remaining-day action suffixes."""
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
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.control import protocol as source_protocol
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.learning import load_learner
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import DecoupledPacketBudget
from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027.support import snapshot, strip_profiler
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state
from experiments.mcwm_closed_loop_v1.stream import (
    EventStream, create_target_env, pack_event_batch_v7,
    policy_action_to_hold_seconds_v7, terminal_raw_state_v7)
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.suffix import selected_state
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027 import contract, physical
import numpy as np
import torch


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


class PrefixFinished(Exception):
    pass


class RemainingDayStream(EventStream):
    """Original event path with a root selector; later issues see actual actions.

    Unlike the old300-second fork stream, later controls in the root batch
    also evaluate the reference actor on their actual updated canonical state.
    The environment's real terminal event supplies the only endpoint.
    """
    def __init__(self, *args, selector, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = selector

    def step(self, policy):
        if self.episode_id < 0:
            self.reset()
        if self.done:
            return self.release()
        batch, commands = self._batch(), {}
        for event in batch:
            self._close_physical(event)
            if event['terminal']:
                commands[event['agent']] = [0., 1.]
        controls = [event for event in batch if not event['terminal']]
        if controls:
            point = self._snapshot(controls[0]['agent'])
            local = np.stack([event['event'] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, point, event['vehicle_id'])
                action = np.asarray(policy(state), np.float32).reshape(-1)
                if action.shape != (1,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.00001):
                    raise ValueError('Policy must return one executed action')
                action = np.clip(action, -1., 1.)
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=action.copy(),
                    proposed_a=action.copy(), started_at=float(point.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                replacement = self.selector(issued)
                if replacement is not None:
                    action = np.asarray(replacement, np.float32).reshape(1)
                    issued['a'] = action.copy()
                if self.on_issue is not None:
                    self.on_issue(issued)
                self._pending = dict(issued, snapshot=point)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                prior[index] = action[0]
                commands[event['agent']] = [float(policy_action_to_hold_seconds_v7(action[0], self.contract)), 1.]
                self.issued_count += 1
        self.env.step_to_event(commands)
        self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError('Native episode ended before24000; no endpoint fabricated')
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending['raw_s'], self.final_snapshot),
                            self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        return self.release()


def run(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root_out = out
    p = source_protocol()
    seed, scenario, external_seed = 6207, 'stable', 106207
    source = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'/(
        f'runs/executed_cql/validation/seed{seed}/{scenario}/fresh')
    reference = json.loads((source/'summary.json').read_text())
    source_updates = (source/'updates.jsonl').open()
    env = hook = stream = trace = probe = learner = None
    updates = None
    branch = root = freeze = frozen_state = None
    frozen_counts = None
    restore_rows = matched_rows = update_rows = 0
    rejection_reasons = {}
    root_commands = root_holds = native_info = None
    root_interventions = 0
    last_proposal = None
    clock_exact = True
    continuation_exact = True
    native_started_wall = None
    started = time.monotonic()

    def counts():
        if learner is None:
            return {}
        audit = learner.actor_schedule_audit()
        return {key: audit[key+'_updates'] for key in contract.COUNTS}

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None else 0., total=24000,
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started)), flush=True)

    def policy(state):
        nonlocal last_proposal, clock_exact
        event_clock = float(env._current_events[0].sim_time)
        expected = np.float32(np.float32(event_clock)/24000)
        clock_exact &= bool(state[0] == expected)
        last_proposal = learner.act(state[None], deterministic=True)[0]
        return learner.action_transform.numpy(state, last_proposal)

    def selector(event):
        nonlocal branch, out, trace, probe, root, root_commands, root_holds, native_info
        nonlocal native_started_wall, root_interventions
        if branch is not None or freeze is None:
            return None
        if not (12000 <= event['started_at'] < 12600) or event['line_id'] != '102X':
            return None
        actions = [learner.action_transform.numpy(event['s'], np.asarray([u], np.float32))
                   for u in (-1., 1.)]
        holds = [float(policy_action_to_hold_seconds_v7(float(a[0]), mdp)) for a in actions]
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
        elif holds[1]-holds[0] < 2*max(step, action_step):
            reason = 'singleton_or_insufficient_physical_gap'
        else:
            reason = None
        if reason:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0)+1
            return None
        root = selected_state(event, stream._snapshot())
        root_commands = dict(zip(('low', 'high'), [a.tolist() for a in actions]))
        root_holds = dict(zip(('low', 'high'), holds))
        native_info = dict(sim_step_seconds=step, action_step_seconds=action_step, at_stop=at_stop)
        write(root_out/'root.json', dict(root=root, reference=freeze, actions=root_commands,
            holds=root_holds, native_support=native_info, rejected=rejection_reasons))
        updates.flush()

        def prepare_child(key):
            nonlocal branch, out, trace, probe, native_started_wall
            branch, out = key, root_out/'suffixes'/key
            old_rows = trace.observed_issues
            trace.close()
            trace = NativePrefixTrace(out/'trace.jsonl')
            trace.observed_issues = old_rows
            updates.close()
            source_updates.close()
            probe = physical.install(env._bridge, root)
            native_started_wall = time.time()

        result = fork_children(root_out/'suffixes', trace, lambda: stream.now,
            counts, prepare_child, timeout_seconds=3600)
        if isinstance(result, str):
            root_interventions += 1
            return np.asarray(root_commands[branch], np.float32)
        arms = {key: json.loads((root_out/'suffixes'/key/'summary.json').read_text())
                for key in ('low', 'high')}
        gate = contract.pair_gate(arms['low'], arms['high'])
        checks = dict(gate['checks'], native_fork=result['all_pass'],
            exact_original_prefix=restore_rows == matched_rows and restore_rows > 0,
            exact_update_log=update_rows == 74,
            original_receipts=budget.receipts == reference['receipts'][:37],
            reference_budget=counts() == contract.COUNTS and replay.target_received == 185
                and len(replay.outcome_ids) == 185,
            reference_state_frozen=_equal(frozen_state, _state(learner)),
            next_same_vehicle_arrivals_observed=all(
                row['physical']['coverage']['next_arrival_observed'] for row in arms.values()),
            labels_not_mature_in_episode=not contract.label_available(24030, 24000),
            no_replacement_root=True)
        before, after = arms['low']['remaining'], arms['high']['remaining']
        result_value = dict(status='pass' if all(checks.values()) else 'hold',
            all_pass=all(checks.values()), checks=checks,
            root={k:root[k] for k in ('sequence','started_at','line_id','trip_id','station_id')},
            reference=freeze, source_prefix_rows=restore_rows, source_update_rows=update_rows,
            support=dict(holds=root_holds, **native_info, rejection_reasons=rejection_reasons),
            physical={key:arms[key]['physical'] for key in arms},
            remaining={key:arms[key]['remaining'] for key in arms},
            high_minus_low={key:after[key]-before[key] for key in ('wait','ride','cost')},
            available_at=24030, consumer_clock=24000, admitted_labels=0,
            fork=dict(overlap_seconds=result['overlap_seconds'],
                same_rng_at_fork=result['parent_rng_unchanged'],
                route_positions_independent=result['route_streams']['parent_offsets_unchanged'],
                note='Common initial external RNG; divergent event consumption is not stepwise paired randomness.'),
            budget=dict(new_initialization_updates=0, restored_covariates=185,
                restored_complete_outcomes=185, restored_updates_per_optimizer=296,
                prefixes=1, suffixes=2, new_remaining_day_labels=2,
                native_prefix_seconds=root['started_at'],
                suffix_native_seconds=2*(24000-root['started_at']),
                total_native_seconds=48000-root['started_at'], suffix_updates=0,
                labels_fed_back=0, failed_work_separate=True),
            next_stage='Decision required after this preflight; no automatic collection or training')
        write(root_out/'preflight.json', result_value)
        raise PrefixFinished()

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = last_proposal.copy()
        if branch is not None and event['sequence'] != root['sequence']:
            expected = learner.action_transform.numpy(event['s'], last_proposal)
            continuation_exact &= bool(np.array_equal(event['a'], expected))
        trace.observe(event, stream._snapshot())

    try:
        progress('load_original_initializer')
        learner, replay, mdp, stations, initializer = load_learner(seed, arm='fresh')
        budget = DecoupledPacketBudget(p)
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl', source/'prefix_trace.jsonl')
        updates = (root_out/'updates.jsonl').open('x')
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=external_seed)
        stream = RemainingDayStream(env, stations, mdp, domain='target', selector=selector,
            delay_seconds=30, execute_transform=False, on_issue=issue)
        stream.reset()
        hook = install_scenario(env, scenario, shock_at=9000, factor=.65, metric_every=300)
        last_progress = time.monotonic()
        while not stream.done:
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
                    if index in (0, 7):
                        row = dict(packet=budget.index, step_in_packet=index+1,
                            processed_at=stream.now, **metrics)
                        source_line = source_updates.readline()
                        if not source_line or strip_profiler(row) != strip_profiler(json.loads(source_line)):
                            raise RuntimeError('Packet37 restoration differs from original update log')
                        update_rows += 1
                        updates.write(json.dumps(row, allow_nan=False)+'\n')
                updates.flush()
                progress('restore_original_prefix')
            if freeze is None and budget.index == 37:
                if counts() != contract.COUNTS or budget.receipts != reference['receipts'][:37]:
                    raise RuntimeError('Reference snapshot budget or original receipts changed')
                restore_rows, matched_rows = trace.observed_issues, trace.matched_issues
                if not trace.comparison_exact or restore_rows != matched_rows:
                    raise RuntimeError('Original v75 prefix did not reproduce')
                freeze = contract.source_snapshot_metadata(seed=seed, scenario=scenario,
                    published_clock=stream.now,
                    max_available_at=max(row['max_available_at'] for row in budget.receipts))
                frozen_counts = counts()
                frozen_state = deepcopy(_state(learner))
                torch.save(dict(metadata=freeze, actor=learner.actor.state_dict()),
                           root_out/'reference_actor.pt')
                write(root_out/'reference.json', freeze)
                trace.reference.close()
                trace.reference = None
            if branch is None and freeze is not None and stream.now >= 12600:
                write(root_out/'preflight.json', dict(status='unsupported', all_pass=False,
                    reason='No first supported102X issue in predeclared window',
                    rejection_reasons=rejection_reasons,
                    budget=dict(native_prefix_seconds=stream.now, suffixes=0,
                        restored_updates_per_optimizer=counts(), new_labels=0)))
                return
            if time.monotonic()-last_progress >= 45:
                progress('remaining_day_suffix' if branch else 'prefix')
                last_progress = time.monotonic()
        if branch is None:
            raise RuntimeError('No supported root before episode ended')
        final = snapshot(stream.final_snapshot)
        remaining = contract.remaining_cost(root['issue_ledger'], final)
        observation = probe.summary(endpoint_reached=True)
        applied = observation['native_action']
        checks = dict(real_terminal=stream.now == 24000 and stream.done,
            physical_action_applied_once=applied is not None and root_interventions == 1,
            actual_command_exact=applied is not None
                and abs(applied['applied_hold_seconds']-root_holds[branch]) <= 1e-5,
            own_policy_continuation=continuation_exact, real_actor_clock=clock_exact,
            parameters_adam_frozen=_equal(frozen_state, _state(learner)),
            no_updates=counts() == frozen_counts,
            no_admission=replay.target_received == 185 and len(replay.outcome_ids) == 185,
            real_nonzero_horizon=remaining['dt'] > 0,
            label_not_released=not contract.label_available(remaining['available_at'], stream.now),
            all_issues_logged=trace.observed_issues == stream.issued_count)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), root=root,
            reference=freeze, external_seed=external_seed,
            executed_hold_seconds=root_holds[branch], **native_info,
            departure_clock=observation['departure']['clock_seconds'],
            final_clock=stream.now, suffix_updates=0, suffix_admitted_labels=0,
            remaining=remaining, final_ledger=final, physical=observation,
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started_wall,
                native_finished_wall=time.time()))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('Native suffix integrity failed')
        progress('complete')
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    except PrefixFinished:
        progress('preflight_complete_no_followup')
    except Exception as error:
        write(out/'failure.json', dict(type=type(error).__name__, message=str(error),
            native_time=stream.now if stream is not None else None, branch=branch,
            counts=counts(), elapsed_wall_seconds=time.monotonic()-started,
            labels_fed_back=0, original_results_untouched=True))
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        raise
    finally:
        source_updates.close()
        if updates is not None:
            updates.close()
        if probe is not None:
            probe.close()
        if trace is not None:
            trace.close()
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
