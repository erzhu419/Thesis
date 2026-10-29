"""Frozen paid-response readout versus matched Q and the native MC-WM actor."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
from experiments.mcwm_closed_loop_matched_response_v59_20261025.analyze import fit_readouts
from experiments.mcwm_closed_loop_matched_response_v59_20261025.data import load_dataset
from experiments.mcwm_closed_loop_matched_response_v59_20261025.difference import LINES, issue_features
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze, optimizer_steps
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import critic_states
from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import executable_candidates
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory, MemoryEventStream
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import (
    create_target_env, pack_event_batch_v7, policy_action_to_hold_seconds_v7, terminal_raw_state_v7)
import numpy as np
import torch

START = 18300.
HORIZON = 24000.
BIN_SECONDS = 300.
HOLDS = (0., 7.5)
SOURCE = HERE.parent/'mcwm_closed_loop_matched_response_v59_20261025'


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def fit_paid_readout(collection, model_name):
    """Open only paid train records and their causal outcome-free receipts."""
    admitted = json.loads((collection/'admitted_pair_records.json').read_text())
    dataset = load_dataset(collection/'covariates_dataset.json', arm='no_fresh',
        scenario=collection.name, seed=6202)
    models, log = fit_readouts(admitted, dataset)
    model = models[model_name]
    if len(admitted) != 9 or any(sum(row['line_id'] == line for row in admitted) != 3 for line in LINES):
        raise ValueError('Deployment requires the declared nine paid three-line training pairs')
    stations = {line: sorted({str(row['selected']['station_id']) for row in admitted
        if row['line_id'] == line}) for line in LINES}
    proof = dict(training_pairs=9, inherited_feedback_slots=285, inherited_policy_updates=456,
        paid_pair_slots=18, ordinary_slots=267, pair_ids=[row['id'] for row in admitted],
        fitting_packets=log, heldout_truth_opened=False, training_stations=stations,
        response_model=model.summary(), source_files=['admitted_pair_records.json', 'covariates_dataset.json'])
    return model, stations, proof


def load_frozen_base(scenario):
    checkpoint = SOURCE/'runs/learning'/scenario/'no_fresh/post456_model.pt'
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    learner = value['learner']
    if (value['scenario'] != scenario or value['arm'] != 'no_fresh'
            or value['feedback_slots'] != 285 or value['updates'] != 456
            or learner.total_updates != 50456 or learner.replay_buffer.target_received != 285):
        raise ValueError('Require the common v59 no_fresh post456 model')
    if learner.world is not None:
        raise ValueError('Frozen deployment may not attach a world update')
    freeze(learner)
    return learner, value['contract'], value['stations'], str(checkpoint)


class InterventionSelector:
    """Same issue-visible executable slots for both selectors and actor census."""
    def __init__(self, learner, readout, *, arm, api, training_stations):
        self.learner, self.readout, self.arm, self.api = learner, readout, arm, api
        self.training_stations = training_stations
        self.used = set()
        self.counts = Counter()
        self.by_line = {line: Counter() for line in LINES}
        self.rows = []

    def _count(self, line, name):
        self.counts[name] += 1
        self.by_line[line][name] += 1

    @torch.no_grad()
    def __call__(self, event):
        line, now = str(event['line_id']), float(event['started_at'])
        if line not in LINES or not START <= now < HORIZON:
            return None
        index = int((now-START)//BIN_SECONDS)
        self._count(line, 'issues_in_deployment_window')
        if (line, index) in self.used:
            self._count(line, 'skipped_used_bin')
            return None
        choices = executable_candidates(self.learner.action_transform, event['s'], HOLDS)
        if not choices['physically_distinct'] or not all(choices['supported']):
            self._count(line, 'skipped_F_unsupported')
            return None
        try:
            stopped = bool(self.api.vehicle.isAtBusStop(event['trip_id']))
            step = float(self.api.simulation.getDeltaT())
            action_step = float(self.api.vehicle.getActionStepLength(event['trip_id']))
        except Exception:
            self._count(line, 'skipped_native_observation_missing')
            return None
        if not stopped or HOLDS[1]-HOLDS[0] < 1.5*max(step, action_step):
            self._count(line, 'skipped_native_stop_or_gap_unsupported')
            return None
        self.used.add((line, index))
        self._count(line, 'eligible_first_issues')
        in_support = str(event['station_id']) in self.training_stations[line]
        self._count(line, 'training_station_seen' if in_support else 'training_station_unseen')
        row = dict(sequence=int(event['sequence']), issue=now, line_id=line,
            station_id=str(event['station_id']), trip_id=str(event['trip_id']), bin=index,
            bin_start=START+BIN_SECONDS*index, native_stopped=stopped,
            native_step_seconds=step, action_step_seconds=action_step,
            requested_holds_seconds=list(HOLDS), choices=choices,
            training_station_seen=in_support, memory_observed_at=event['memory_s']['observed_at'],
            actor_executed_action=float(event['a'][0]), selected=False)
        high = None
        if self.arm == 'readout':
            prediction = self.readout.predict(issue_features(event))
            row['readout'] = prediction
            if prediction['status'] == 'supported':
                high = int(prediction['gain']['cost'] > 0.)
                if prediction['headway_extrapolation']:
                    self._count(line, 'headway_extrapolation')
            else:
                self._count(line, 'skipped_readout_unsupported')
        elif self.arm == 'q_selector':
            learner = self.learner
            state = torch.as_tensor(event['s'], dtype=torch.float32, device=learner.device).reshape(1, 229)
            context = torch.as_tensor(event['memory_s']['vector'], dtype=state.dtype,
                device=state.device).reshape(1, 84)
            explicit = critic_states(learner, state, context).repeat(2, 1)
            actions = state.new_tensor(choices['executed_actions']).reshape(2, 1)
            heads = learner.critic(explicit, actions)
            minimum = heads.min(0).values
            high = int(minimum.argmax())
            row['q'] = dict(heads=heads.tolist(), minimum=minimum.tolist(),
                high_minus_low=float(minimum[1]-minimum[0]), uses_current_issue_memory=True)
        elif self.arm != 'actor_mcwm':
            raise ValueError('Unknown frozen-control arm')
        self.rows.append(row)
        if high is None:
            return None
        row.update(selected=True, choice=high, holding_seconds=HOLDS[high])
        self._count(line, 'selected')
        self._count(line, 'selected_high' if high else 'selected_zero')
        event['intervention_row'] = row
        return dict(action=np.asarray([choices['executed_actions'][high]], np.float32),
            proposal=np.asarray([choices['proposal_actions'][high]], np.float32))

    def summary(self):
        return dict(counts=dict(self.counts), by_line={key: dict(value) for key, value in self.by_line.items()},
            used_bins=[dict(line_id=line, index=index) for line, index in sorted(self.used)],
            maximum_per_line=19, maximum_total=57, intervention_start_seconds=START,
            interval_seconds=BIN_SECONDS, focus_lines=list(LINES), all_native_stations=True,
            station_support_is_descriptive=True, choices_seconds=list(HOLDS),
            original_actor_used_elsewhere=True)


class ControlEventStream(MemoryEventStream):
    """Native full-day advancement with final replacement commands in each batch.

    This retains ordinary EventStream closure and terminal semantics. It has no
    fork, artificial cutoff, training or look-ahead outcome access.
    """
    def __init__(self, *args, selector, **kwargs):
        if kwargs.get('execute_transform', True):
            raise ValueError('The controller returns commands already passed through F')
        self.selector = selector
        self.actual_execution_checks = 0
        self.maximum_execution_gap_seconds = 0.
        super().__init__(*args, **kwargs)

    def _observe_issue(self, event):
        snapshot = self.memory.snapshot(event['started_at'])
        event['memory_s'] = snapshot
        self._memory_issues[event['sequence']] = snapshot
        replacement = self.selector(event)
        if replacement is not None:
            event['a'] = replacement['action'].copy()
            event['proposed_a'] = replacement['proposal'].copy()
            event['intervened'] = True
        if self._user_on_issue is not None:
            self._user_on_issue(event)

    def step(self, policy):
        if self.episode_id < 0:
            self.reset()
        if self.done:
            return self.release()
        batch, commands = self._batch(), {}
        native_events = {str(event.bus_id): event for event in self.env._current_events}
        selected = []
        for event in batch:
            self._close_physical(event)
            if event['terminal']:
                commands[event['agent']] = [0., 1.]
        controls = [event for event in batch if not event['terminal']]
        if controls:
            snapshot = self._snapshot(controls[0]['agent'])
            local = np.stack([event['event'] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, snapshot, event['vehicle_id'])
                action = np.asarray(policy(state), np.float32).reshape(-1)
                if action.shape != (1,) or not np.isfinite(action).all() or np.any(np.abs(action)>1.00001):
                    raise ValueError('Policy must return one finite executed holding action')
                action = np.clip(action, -1., 1.)
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=action.copy(),
                    proposed_a=action.copy(), started_at=float(snapshot.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                self.on_issue(issued)
                action = np.asarray(issued['a'], np.float32).copy()
                self._pending = dict(issued, snapshot=snapshot)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                prior[index] = action[0]
                commands[event['agent']] = [float(policy_action_to_hold_seconds_v7(action[0], self.contract)), 1.]
                if issued.get('intervened'):
                    selected.append((issued['intervention_row'], native_events[str(event['trip_id'])]))
                self.issued_count += 1
        self.env.step_to_event(commands)
        for row, native in selected:
            applied = native.metadata.get('applied_additional_hold_seconds')
            row['native_applied_hold_seconds'] = applied
            if applied is None:
                raise RuntimeError('Selected native action did not record an applied holding amount')
            gap = abs(float(applied)-row['holding_seconds'])
            self.actual_execution_checks += 1
            self.maximum_execution_gap_seconds = max(self.maximum_execution_gap_seconds, gap)
            if gap > 1.e-5:
                raise RuntimeError('Selected0/7.5-second action was not applied by the native simulator')
        self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError('Native day stopped before the real configured horizon')
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending['raw_s'], self.final_snapshot),
                    self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        return self.release()


def cost_values(wait, ride, departed):
    return dict(departed_passengers=int(departed), accrued_wait_seconds=float(wait),
        accrued_in_vehicle_seconds=float(ride), generalized_passenger_seconds=float(2*wait+ride),
        generalized_seconds_per_departed_passenger=float((2*wait+ride)/departed))


def run(args):
    campaign = json.loads((HERE/'protocol.json').read_text())
    allowed = campaign['paired_readout']['development_external_seeds']+campaign['paired_readout']['confirmation_external_seeds']
    if args.external_seed not in allowed:
        raise ValueError('External seed was not declared for this frozen campaign')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cell = dict(candidate='paired_readout', external_seed=args.external_seed, scenario=args.scenario,
        arm=args.arm, model=args.model)
    env = stream = memory = trace = hook = learner = None
    native_started_wall = None
    try:
        torch.set_num_threads(1)
        learner, contract, stations, checkpoint = load_frozen_base(args.scenario)
        readout, training_stations, training = fit_paid_readout(SOURCE/'runs/collection'/args.scenario, args.model)
        before_steps, before_updates = optimizer_steps(learner), learner.total_updates
        last_proposal = None
        clock_audit = dict(actor_real_clock=True, issues=0)
        F_audit = dict(compared=0, exact=True, maximum_gap_seconds=0.)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=int(HORIZON), seed=args.external_seed)
        # The api is the actual simulator API used by this bridge instance.
        stream = ControlEventStream(env, stations, contract, domain='target', selector=lambda event: None,
            delay_seconds=30., execute_transform=False)
        stream.reset()
        memory = ExecutionMemory(env._bridge)
        stream.memory = memory
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        selector = InterventionSelector(learner, readout, arm=args.arm, api=api,
            training_stations=training_stations)
        stream.selector = selector
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        hook = install_scenario(env, args.scenario, shock_at=9000., factor=.65, metric_every=300.)
        write(out/'protocol.json', dict(cell=cell, campaign=campaign, source_checkpoint=checkpoint,
            training=training, deployment_updates=0, continuation='common frozen v59 no_fresh post456 actor; '
                'response trained under mu14 and transferred without refitting'))

        @torch.no_grad()
        def policy(state):
            nonlocal last_proposal
            real = float(env._current_events[0].sim_time)
            clock_audit['actor_real_clock'] &= bool(state[0] == np.float32(np.float32(real)/HORIZON))
            clock_audit['issues'] += 1
            states = torch.as_tensor(state, dtype=torch.float32, device=learner.device).reshape(1, 229)
            last_proposal = learner.actor(states, deterministic=True)[0][0].cpu().numpy()
            return learner.action_transform.numpy(state, last_proposal)

        with (out/'issue_memory.jsonl').open('x') as memories, (out/'interventions.jsonl').open('x') as interventions:
            def issue(event):
                if not event.get('intervened'):
                    event['proposed_a'] = last_proposal.copy()
                event['actor_proposed_a'] = last_proposal.copy()
                expected = learner.action_transform.numpy(event['s'], event['proposed_a'])
                gap = abs(float(expected[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
                F_audit['compared'] += 1
                F_audit['maximum_gap_seconds'] = max(F_audit['maximum_gap_seconds'], gap)
                F_audit['exact'] &= gap <= 1.e-6
                trace.observe(event, stream._snapshot())
                memories.write(json.dumps(dict(sequence=event['sequence'], issue=event['started_at'],
                    memory_s=event['memory_s']), separators=(',', ':'), allow_nan=False)+'\n')
            stream._user_on_issue = issue
            native_started_wall = time.time()
            progress_at, written = time.monotonic(), 0
            while not stream.done:
                stream.step(policy)
                stream.release_intervals()
                for row in selector.rows[written:]:
                    interventions.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                written = len(selector.rows)
                if time.monotonic()-progress_at >= 45:
                    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v60_frozen_control',
                        current=stream.now, total=HORIZON, unit='sim_seconds',
                        selected=selector.counts['selected'], arm=args.arm)), flush=True)
                    interventions.flush()
                    memories.flush()
                    progress_at = time.monotonic()
        native_finished_wall = time.time()
        hook.capture_final()
        ledger = env._bridge.passenger_time_ledger.snapshot()
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        passengers = summarize_passengers_v5(env._bridge, ledger)
        departed = passengers['departed_passengers']
        full = cost_values(ledger.accrued_wait_seconds, ledger.accrued_in_vehicle_seconds, departed)
        tail_start = next((row for row in hook.metrics if row['time_seconds'] == START), None)
        tail = None if tail_start is None else cost_values(
            ledger.accrued_wait_seconds-tail_start['accrued_wait_seconds'],
            ledger.accrued_in_vehicle_seconds-tail_start['accrued_in_vehicle_seconds'], departed)
        with (out/'metrics.jsonl').open('x') as output:
            for row in hook.metrics:
                output.write(json.dumps(dict(row, departed_passengers=departed), separators=(',', ':'))+'\n')
        memory_summary = memory.summary()
        checks = dict(full_native_day=stream.done and stream.now == HORIZON,
            frozen_base_post456=learner.total_updates == before_updates == 50456,
            zero_deployment_updates=optimizer_steps(learner) == before_steps,
            no_world=learner.world is None,
            actor_real_clock=clock_audit['actor_real_clock'], unchanged_F=F_audit['exact'],
            memory_causal=memory_summary['integrity']['all_pass'],
            complete_prefix=trace.observed_issues == stream.issued_count,
            physical_actions_executed=stream.actual_execution_checks == selector.counts['selected']
                and stream.maximum_execution_gap_seconds <= 1.e-5,
            bounded_interventions=len(selector.used) <= 57,
            late_ledger_present=tail_start is not None,
            passenger_ledger_balanced=passengers['active_state_count_balance_error'] == 0,
            no_deployment_feedback_ingested=learner.replay_buffer.target_received == 285,
            no_heldout_truth_fitting=training['heldout_truth_opened'] is False)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), cell=cell,
            initializer=dict(source_checkpoint=checkpoint, initial_updates=50000,
                frozen_post_updates=50456, training_seed=6202),
            budget=dict(inherited_feedback_slots=285, inherited_policy_updates=456,
                new_feedback_slots=0, new_policy_updates=0, native_seconds=stream.now,
                counterfactual_queries=0, world_updates=0, trust_updates=0,
                training_counterfactual_capability='v59:267 ordinary+18 paired slots; simulator-trained '
                    'finite300 readout; not ordinary single-trajectory dripping'),
            cost=full, cost_windows=dict(full_day=full, post_budget=tail),
            intervention=selector.summary(), memory=memory_summary,
            action_audit=F_audit, clock_audit=clock_audit,
            actual_execution=dict(checks=stream.actual_execution_checks,
                maximum_gap_seconds=stream.maximum_execution_gap_seconds),
            source_training=training, scenario=hook.metadata(),
            wall_seconds=time.monotonic()-started, execution_wall=dict(pid=os.getpid(),
                native_started_wall=native_started_wall, native_finished_wall=native_finished_wall),
            continuation='Frozen post456 no_fresh actor; finite300 readout transferred from mu14 training')
        if len(json.dumps(summary, separators=(',', ':'), allow_nan=False).encode()) > campaign['summary_max_bytes']:
            raise RuntimeError('Frozen control summary exceeds its declared small-summary budget')
        write(out/'summary.json', summary)
        write(out/'gate.json', {key: summary[key] for key in ('status', 'integrity', 'cell', 'budget')})
        if not all(checks.values()):
            raise RuntimeError('Frozen actual control integrity failed')
        return summary
    except Exception as error:
        write(out/'failure.json', dict(status='failed', cell=cell, type=type(error).__name__,
            message=str(error), native_time=stream.now if stream is not None else None,
            consumed_budget=dict(new_feedback_slots=0, new_policy_updates=0,
                native_seconds=stream.now if stream is not None else 0, counterfactual_queries=0),
            inherited_feedback_slots=285, inherited_policy_updates=456,
            wall_seconds=time.monotonic()-started, native_started_wall=native_started_wall))
        raise
    finally:
        if memory is not None: memory.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('readout', 'q_selector', 'actor_mcwm'), required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--external-seed', type=int, required=True)
    parser.add_argument('--model', choices=('readout_line', 'readout_lowdim'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
