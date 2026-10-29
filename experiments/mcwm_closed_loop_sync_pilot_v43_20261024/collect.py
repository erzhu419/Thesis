"""Prospective native one-step pilot with a real, issue-visible post24 actor."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
import numpy as np
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import make_learner, initial_preflight
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.collect import (
    ContextForkStream, action_choices, transition_dict as native_transition_dict,
)
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, selected_state, plain
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import optimizer_steps
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import create_target_env, Transition
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters

ARMS = ('low', 'high')
LINES = ('7X', '102X', '705X')


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(plain(value), output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def transition_dict(row):
    value = native_transition_dict(row)
    value['behavior_s'] = plain(row.behavior_s)
    if hasattr(row, 'feedback_id'):
        value['feedback_id'] = row.feedback_id
    return value


def decode_transition(value):
    extra = ('memory_s', 'memory_ns', 'behavior_s', 'feedback_id')
    native = {key: item for key, item in value.items() if key not in extra}
    for key in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
        if native[key] is not None:
            native[key] = np.asarray(native[key], np.float32)
    row = Transition(**native)
    for key in extra:
        if key in value:
            setattr(row, key, deepcopy(value[key]))
    return row


class StochasticController:
    """Native actor sampling on a private torch stream, never the SGD stream."""
    def __init__(self, learner, *, seed):
        self.seed, self.draws, self.snapshot = seed, 0, 'initial50000'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rng = torch.get_rng_state().clone()
        self.reference = self._copy(learner)
        self.last = None

    @staticmethod
    def _copy(learner):
        actor = deepcopy(learner.actor).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        return SimpleNamespace(actor=actor, device=learner.device,
            action_transform=deepcopy(learner.action_transform), total_updates=learner.total_updates)

    def switch_post24(self, learner):
        if learner.total_updates != 50024:
            raise RuntimeError('Behavior snapshot must follow the actual24th update')
        self.reference = self._copy(learner)
        self.snapshot = 'actual_post24'

    @torch.no_grad()
    def __call__(self, state):
        state = np.asarray(state, np.float32).reshape(229)
        parameters = actor_parameters(self.reference, state)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng)
            proposal = self.reference.actor(torch.as_tensor(state, device=self.reference.device)[None],
                deterministic=False)[0][0].cpu().numpy().reshape(1)
            self.rng = torch.get_rng_state().clone()
        executed = self.reference.action_transform.numpy(state, proposal)
        self.last = dict(policy_input=state.tolist(), execution_input=state.tolist(),
            parameters=parameters, proposal=proposal.tolist(), nominal_executed_action=executed.tolist(),
            actor_snapshot=self.snapshot,
            action_rng=dict(stream_seed=self.seed, draw_ordinal=self.draws))
        self.draws += 1
        return executed


class ChainComplete(Exception):
    pass


def install_release_clock(bridge, stream, is_child, respond):
    """Run9630 admission after a genuine native tick, including between events."""
    original = bridge._advance_one_step

    def advance(*values, **keywords):
        result = original(*values, **keywords)
        if not is_child() and stream.now == 9630.:
            respond()
        return result

    bridge._advance_one_step = advance


class ResponseStream(ContextForkStream):
    """Close45 actual events, and stop before issuing a46th control."""
    def __init__(self, *args, **kwargs):
        self.response_rows = None
        self.response_file = None
        self.response_line = self.response_arm = None
        self.response_finished = False
        super().__init__(*args, **kwargs)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending, queued = self._pending, len(self._queue)
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if pending is None or len(self._queue) == queued:
            return
        row = self._queue[-1]
        row.behavior_s = deepcopy(pending['behavior_s'])
        if self.response_rows is None:
            return
        row.feedback_id = f'response/{self.response_line}/{self.response_arm}/{len(self.response_rows)}'
        self.response_rows.append(row)
        self.response_file.write(json.dumps(dict(transition=transition_dict(row)),
            separators=(',', ':'), allow_nan=False)+'\n')
        self.response_file.flush()
        if len(self.response_rows) == 45:
            # This exception exits the existing atomic loop immediately after
            # its genuine close hook. No endpoint action or fake ns is added.
            self.response_finished = True
            self.cutoff_snapshot = snapshot
            self.cutoff_reached = True
            raise ChainComplete()

    def step(self, policy):
        try:
            return super().step(policy)
        except ChainComplete:
            return self.release()


class ResponsePair(NativeTruthPair):
    """Two concurrent native continuations; only their45 real TDs are labels."""
    def fork(self, event, stream, trace, prepare_child=None):
        if selected_state(event, stream._snapshot()) != self.expected['selected']:
            raise RuntimeError('Selected native issue/context differs before fork')
        self.before = deepcopy(self.counts())

        def prepare(arm):
            self.in_child, self.arm, self.started_wall = True, arm, time.time()
            trace.writer.close()
            if trace.reference is not None:
                trace.reference.close()
            trace.path = self.out/arm/'suffix_trace.jsonl'
            trace.writer = trace.path.open('x')
            trace.reference = None
            self._install_native_observer(stream)
            stream.response_rows = []
            stream.response_line, stream.response_arm = event['line_id'], arm
            stream.response_file = (self.out/arm/'events.jsonl').open('x')
            if prepare_child is not None:
                prepare_child(arm)

        result = fork_children(self.out, trace, lambda: stream.now, self.counts, prepare)
        if isinstance(result, str):
            return np.asarray(self.expected['actions'][result], np.float32)
        self.receipt = result
        self.values = {arm: json.loads((self.out/arm/'summary.json').read_text()) for arm in ARMS}
        low, high = self.values['low'], self.values['high']
        if low['selected'] != high['selected'] or low['native']['other_batch_actions'] != high['native']['other_batch_actions']:
            raise RuntimeError('Paired issue or first-batch other controls differ')
        return None

    def build_summary(self, stream):
        rows = stream.response_rows
        actual = self.native_actions[0] if len(self.native_actions) == 1 else None
        requested = self.expected['holds'][self.arm]
        self.execution['native_applied_hold_seconds'] = None if actual is None else actual['applied_hold_seconds']
        self.execution['native_application'] = actual
        continuous = all(left.completed_at == right.started_at
            and left.sequence+1 == right.sequence and np.array_equal(left.ns, right.s)
            and left.memory_ns == right.memory_s for left, right in zip(rows, rows[1:]))
        checks = dict(exact45=len(rows) == 45 and stream.response_finished,
            first_issue=bool(rows) and rows[0].started_at == self.expected['selected']['started_at'],
            continuous_real_events=continuous,
            actual_action=actual is not None and abs(actual['applied_hold_seconds']-requested) <= 1e-5,
            native_clock_application=actual is not None and actual['real_clock_before'] == actual['real_clock_after'] == rows[0].started_at,
            suffix_zero_training=self.counts() == self.before,
            cap300=stream.now <= self.expected['selected']['started_at']+300.,
            zero_fake_terminal=not stream.done and all(not row.terminal for row in rows),
            complete_behavior_context=all(row.behavior_s['actor_snapshot'] == 'actual_post24' for row in rows),
            endpoint_not_issued=bool(rows) and stream.issued_count == rows[-1].sequence+1,
            feedback_release=all(row.available_at == row.completed_at+30. for row in rows))
        # Physical observer's own summary retains actual departure/arrival
        # coverage and its native tick count without inventing a300s outcome.
        physical = self.physical.summary(False)
        checks['physical_ticks'] = physical['native_tick_count'] == stream.native_suffix_steps
        return plain(dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            selected=self.selected, external_seed=self.external_seed, arm=self.arm,
            execution=self.execution, physical=physical,
            native=dict(other_batch_actions=stream.fork_batch_other_controls,
                suffix_steps=stream.native_suffix_steps),
            events_path=str(self.out/self.arm/'events.jsonl'),
            rows=len(rows), zero_duration_rows=sum(row.dt == 0. for row in rows),
            available_at=max((row.available_at for row in rows), default=None),
            stop_clock=stream.now, physical_span=stream.now-self.expected['selected']['started_at'],
            counters=dict(before=self.before, after=self.counts()),
            budget=dict(native_seconds=stream.native_suffix_steps, new_feedback_rows=len(rows), optimizer_updates=0),
            execution_wall=dict(pid=os.getpid(), native_started_wall=self.started_wall,
                native_finished_wall=time.time())))


def branch_rows(branches, *, release_clock):
    """Read only the newly collected45-event physical prefixes, once each."""
    by_line = {}
    for branch in branches:
        records = [json.loads(line)['transition'] for line in Path(branch['events_path']).read_text().splitlines()]
        if len(records) != 45 or any(row['available_at'] > release_clock for row in records):
            raise RuntimeError('A required45-event prefix is short or unreleased at9630')
        by_line[(branch['line_id'], branch['arm'])] = records
    if set(by_line) != {(line, arm) for line in LINES for arm in ARMS}:
        raise RuntimeError('The prescribed three-line paired coverage is incomplete')
    # Preserve v40 final release-plan ordering: all six roots, then every
    # subsequent event ordinal with line and arm order fixed before outcomes.
    records = [by_line[(line, arm)][event] for event in range(45) for line in LINES for arm in ARMS]
    if len({row['feedback_id'] for row in records}) != 270:
        raise RuntimeError('Response feedback identities repeat')
    return [decode_transition(row) for row in records]


def observed_branch_budget(out):
    """Retain real completed and interrupted suffix work when a pilot fails."""
    rows = []
    for line in LINES:
        for arm in ARMS:
            root = Path(out)/'branches'/line/arm
            if not root.exists():
                continue
            summary, failure = root/'summary.json', root/'failure.json'
            if summary.exists():
                value = json.loads(summary.read_text())
                rows.append(dict(line_id=line, arm=arm, status=value['status'],
                    native_seconds=value['budget']['native_seconds'],
                    real_closed_rows=value['rows'], exact=True))
            elif failure.exists():
                value = json.loads(failure.read_text())
                rows.append(dict(line_id=line, arm=arm, status='failed',
                    native_seconds=value['observed_clock']-value['issue_clock'],
                    real_closed_rows=value['completed_rows'], exact=True))
            elif (root/'events.jsonl').exists():
                events = [json.loads(record)['transition'] for record in (root/'events.jsonl').read_text().splitlines()]
                rows.append(dict(line_id=line, arm=arm, status='interrupted',
                    native_seconds=events[-1]['completed_at']-events[0]['started_at'] if events else None,
                    real_closed_rows=len(events), exact=False))
            else:
                rows.append(dict(line_id=line, arm=arm, status='fork_started_without_clock_receipt',
                    native_seconds=None, real_closed_rows=0, exact=False))
    return dict(observed_suffixes=len(rows), arms=rows,
        observed_suffix_native_seconds=sum(row['native_seconds'] or 0. for row in rows),
        real_closed_rows=sum(row['real_closed_rows'] for row in rows),
        unknown_or_lower_bound_suffixes=sum(not row['exact'] for row in rows))


def collect(args, protocol, *, on_warmup_packet, on_response):
    """Callbacks preserve original native one-step updates and expose the pilot.

    warmup(learner,replay,rows5,receipt) admits5 and executes8 updates.
    response(learner,replay,rows270,receipt,reference) admits all270 before432
    updates. The reference exposes the frozen actual post24 actor and F.
    """
    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = stream = trace = memory = hook = pair = None
    started = time.time()
    branches, selections, warmup_receipts = [], {}, []
    response_receipt = None
    try:
        learner, replay, contract, stations, initializer = make_learner(args.seed, arm='shared_level')
        parity = initial_preflight(learner)
        if learner.total_updates != 50000 or replay.target_received:
            raise RuntimeError('Pilot requires the original50000 initialization with no new feedback')
        controller = StochasticController(learner, seed=protocol['action_seed'])
        warmup_protocol = dict(feedback_first_deadline=1200, feedback_last_deadline=1800,
            feedback_interval_seconds=300, feedback_records_per_packet=5)
        budget = formal.PacketBudget(warmup_protocol)
        behavior_writer = (out/'behavior_prefix.jsonl').open('x')
        torch.save(dict(seed=controller.seed, state=controller.rng, algorithm='native_Hurdle_forward'),
            out/'action_rng_initial.pt')
        pre_recorded = set()
        response_done = False
        post24_clock = None
        post_update_counts = None

        def counts():
            return dict(policy_updates=learner.total_updates-50000, feedback_slots=replay.target_received,
                optimizer_steps=optimizer_steps(learner))

        def respond():
            nonlocal response_done, response_receipt, post_update_counts
            if response_done or (pair is not None and pair.in_child):
                return
            if stream.now != 9630. or controller.snapshot != 'actual_post24':
                raise RuntimeError('Response learning must occur on the real9630 clock after actual post24 save')
            rows = branch_rows(branches, release_clock=stream.now)
            response_receipt = dict(processed_at=stream.now, records=len(rows),
                all270_admitted_before_updates=True, max_available_at=max(row.available_at for row in rows),
                feedback_ids=[row.feedback_id for row in rows], branches=branches,
                selected=selections, behavior_saved_at=post24_clock,
                pairs=[dict(line_id=line, selected=selections[line], branches={
                    arm: next(branch for branch in branches if branch['line_id'] == line and branch['arm'] == arm)
                    for arm in ARMS}) for line in LINES])
            if learner.total_updates != 50024 or replay.target_received != 15:
                raise RuntimeError('Training or feedback appeared outside the15-slot warmup')
            on_response(learner, replay, rows, response_receipt, controller.reference)
            if learner.total_updates != 50456 or replay.target_received != 285:
                raise RuntimeError('Response callback must preserve285 slots and456 total updates')
            response_done = True
            post_update_counts = counts()
            write(out/'response_receipt.json', response_receipt)

        def selector(event):
            nonlocal pair
            if pair is not None and pair.in_child:
                return None
            line, clock = event['line_id'], event['started_at']
            if line not in LINES or line in selections or not 9000. <= clock < 9300.:
                return None
            if controller.snapshot != 'actual_post24':
                raise RuntimeError('A response issue preceded the real behavior snapshot')
            choices = action_choices(learner.action_transform, event['s'])
            api = importlib.import_module(env._bridge.__class__.__module__).traci
            stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
            delta = max(float(api.simulation.getDeltaT()), float(api.vehicle.getActionStepLength(event['trip_id'])))
            if choices is None or not stop or 1.875 < 1.5*delta:
                return None
            event['proposed_a'] = np.asarray(controller.last['proposal'], np.float32)
            event['behavior_s'] = deepcopy(controller.last)
            trace.observe(event, stream._snapshot())
            pre_recorded.add(event['sequence'])
            selected = selected_state(event, stream._snapshot())
            selections[line] = selected
            pair_out = out/'branches'/line
            expected = dict(selected=selected, actions={arm: choices[arm]['action'].tolist() for arm in ARMS},
                holds={arm: choices[arm]['hold'] for arm in ARMS})
            pair = ResponsePair(pair_out, expected, counts, protocol['native_external_seed'])
            # Children close their inherited copy. It must contain no buffered
            # prefix rows that could be flushed a second time after fork.
            behavior_writer.flush()
            replacement = pair.fork(event, stream, trace)
            if pair.in_child:
                behavior_writer.close()
                return replacement
            for arm in ARMS:
                summary = pair.values[arm]
                branches.append(dict(line_id=line, arm=arm, selected=selected,
                    execution=summary['execution'], events_path=summary['events_path'],
                    summary_path=str(pair_out/arm/'summary.json'), rows=summary['rows'],
                    zero_duration_rows=summary['zero_duration_rows'],
                    physical_span=summary['physical_span'], available_at=summary['available_at'],
                    native_seconds=summary['budget']['native_seconds'], parallel_receipt=str(pair_out/'live_fork.json')))
            if len(branches) == 2:
                first_checks = dict(actual_parallel=pair.receipt['all_pass'],
                    **{arm: pair.values[arm]['integrity']['all_pass'] for arm in ARMS})
                write(out/'preflight_gate.json', dict(all_pass=all(first_checks.values()), checks=first_checks,
                    reused_in_full_run=True,
                    first_line=line, actual_parallel=pair.receipt['all_pass'], duplicate_prefixes=0,
                    actual_holds=[pair.values[arm]['execution']['native_applied_hold_seconds'] for arm in ARMS]))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v43_real_response_pair',
                current=stream.now, total=24000, unit='sim_seconds', line_id=line, branches=len(branches))), flush=True)
            return None

        def issue(event):
            event['proposed_a'] = np.asarray(controller.last['proposal'], np.float32)
            event['behavior_s'] = deepcopy(controller.last)
            event['behavior_s']['intervention'] = bool(pair is not None and pair.in_child
                and event['sequence'] == pair.expected['selected']['sequence'])
            event['behavior_s']['executed_action'] = np.asarray(event['a']).tolist()
            if pair is not None and pair.in_child:
                pair.observe_issue(event, stream)
                trace.observe(event, stream._snapshot())
            else:
                if event['sequence'] not in pre_recorded:
                    trace.observe(event, stream._snapshot())
                behavior_writer.write(json.dumps(dict(sequence=event['sequence'], started_at=event['started_at'],
                    line_id=event['line_id'], behavior_s=event['behavior_s']), separators=(',', ':'))+'\n')

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=protocol['native_external_seed'])
        stream = ResponseStream(env, stations, contract, domain='target', selector=selector,
            on_issue=issue, delay_seconds=30., execute_transform=False, fork_duration_seconds=300.)
        stream.reset()
        memory = ExecutionMemory(env._bridge)
        stream.memory = memory
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        hook = install_scenario(env, args.scenario, shock_at=9000., factor=protocol.get('speed_factor', .65),
            metric_every=300.)
        install_release_clock(env._bridge, stream, lambda: pair is not None and pair.in_child, respond)
        last_progress = time.monotonic()
        native_started = time.time()
        while not stream.done:
            released = stream.step(controller)
            stream.release_intervals()
            if pair is not None and pair.in_child:
                pair.observe_progress(stream)
                if stream.cutoff_reached:
                    pair.finish_child(stream)
                continue
            if budget.index < 3:
                budget.ingest(released)
                while budget.due(stream.now):
                    rows, _ = budget.take(stream.now)
                    receipt = deepcopy(budget.receipts[-1])
                    for row in rows:
                        row.feedback_id = f'ordinary/{row.sequence}'
                    on_warmup_packet(learner, replay, rows, receipt)
                    if learner.total_updates != 50000+8*budget.index or replay.target_received != 5*budget.index:
                        raise RuntimeError('Warmup callback must admit exactly5 and execute8 native updates per packet')
                    receipt['feedback_ids'] = [row.feedback_id for row in rows]
                    warmup_receipts.append(receipt)
                    write(out/f'warmup_packet_{budget.index}.json', dict(receipt=receipt,
                        rows=[transition_dict(row) for row in rows]))
                    if budget.index == 3:
                        post24_clock = stream.now
                        if post24_clock >= 9000.:
                            raise RuntimeError('Actual warmup finished after the response issue window began')
                        with (out/'actual_post24_model.pt').open('xb') as target:
                            torch.save(dict(learner=learner, contract=contract, stations=stations,
                                seed=args.seed, scenario=args.scenario, saved_at=post24_clock,
                                optimizer_steps=optimizer_steps(learner), total_updates=learner.total_updates,
                                feedback_slots=replay.target_received, action_rng=controller.rng.clone(),
                                action_draws=controller.draws, initializer=str(initializer)), target)
                        controller.switch_post24(learner)
                        write(out/'behavior_snapshot.json', dict(status='saved_actual_post24', clock=post24_clock,
                            updates=24, total_updates=learner.total_updates, feedback_slots=15,
                            resident_checkpoint=str(out/'actual_post24_model.pt'), optimizer_steps=optimizer_steps(learner),
                            behavior_seed=controller.seed, action_draws=controller.draws))
            if time.monotonic()-last_progress >= 45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v43_common_prefix', current=stream.now,
                    total=24000, unit='sim_seconds', warmup_packets=budget.index, response_branches=len(branches))), flush=True)
                last_progress = time.monotonic()
        native_finished = time.time()
        behavior_writer.close()
        checks = dict(initial_parity=parity['all_pass'], full_target_day=stream.now == 24000.,
            complete_prefix_trace=trace.observed_issues == stream.issued_count,
            warmup15=len(warmup_receipts) == 3 and sum(row['records'] for row in warmup_receipts) == 15,
            real_post24=post24_clock is not None and post24_clock < 9000.,
            three_lines_six_branches=set(selections) == set(LINES) and len(branches) == 6,
            labels285=replay.target_received == 285, native_updates456=learner.total_updates == 50456,
            real9630=response_receipt is not None and response_receipt['processed_at'] == 9630.,
            post_response_zero_training=response_done and counts() == post_update_counts,
            memory_causal=memory.summary()['integrity']['all_pass'],
            no_world=learner.world is None)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario,
            initializer=str(initializer), initializer_updates=50000, arm='shared_level_native_one_step',
            behavior_snapshot='actual_post24_model.pt', behavior_saved_at=post24_clock,
            warmup=warmup_receipts, response=response_receipt,
            budget=dict(new_native_prefixes=1, prefix_native_seconds=stream.now, suffixes=len(branches),
                suffix_native_seconds=sum(row['native_seconds'] for row in branches), ordinary_labels=15,
                response_labels=270, feedback_slots=replay.target_received, optimizer_updates=learner.total_updates-50000,
                warmup_updates=24, response_updates=432, duplicate_prefixes=0, diagnostic_truth_reused=0),
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started,
                native_finished_wall=native_finished, wall_seconds=time.time()-started))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('Prospective pilot native integrity failed')
        return summary
    except BaseException as error:
        if pair is not None and pair.in_child:
            write(pair.out/pair.arm/'failure.json', dict(status='failed', type=type(error).__name__,
                message=str(error), observed_clock=stream.now if stream is not None else None,
                issue_clock=pair.expected['selected']['started_at'],
                completed_rows=len(stream.response_rows or []) if stream is not None else 0))
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            budget=dict(prefix_native_seconds=stream.now if stream is not None else 0.,
                completed_suffixes=len(branches),
                completed_suffix_native_seconds=sum(row['native_seconds'] for row in branches),
                optimizer_updates=learner.total_updates-50000 if 'learner' in locals() else 0,
                feedback_slots=replay.target_received if 'replay' in locals() else 0,
                suffix_work=observed_branch_budget(out))))
        raise
    finally:
        if pair is None or not pair.in_child:
            if memory is not None:
                memory.close()
            if trace is not None:
                trace.close()
            if hook is not None:
                hook.close()
            if env is not None:
                env.close()
