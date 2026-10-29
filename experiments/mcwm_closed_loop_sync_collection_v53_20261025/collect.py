"""One new factual prefix: 14 ordinary labels, 24 updates, 271 native TDs."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
import numpy as np
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import make_learner, initial_preflight
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.collect import (
    StochasticController, transition_dict, write,
)
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import objects, steps, plain
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory, MemoryEventStream
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import create_target_env, policy_action_to_hold_seconds_v7

LINES = ('7X', '102X', '705X')
SNAPSHOT = 'new_post24_from14'
CAPACITY = 271


class WarmupBudget(formal.PacketBudget):
    """Cancel ordinary entitlement 14 before selecting the third packet."""
    def take(self, now):
        self.p['feedback_records_per_packet'] = (5, 5, 4)[self.index]
        return super().take(now)


def flow_coverage(rows, endpoint):
    """Use actual adjacent TD endpoints; a missing 300-second state stays missing."""
    states = [dict(sequence=row.sequence, started_at=row.started_at, line_id=row.line_id)
              for row in rows]
    if endpoint is not None:
        states.append(endpoint)
    coverage = {}
    for line in LINES:
        index = next((i for i, row in enumerate(rows)
            if row.line_id == line and 9000 <= row.started_at < 9300), None)
        root = rows[index] if index is not None else None
        end = next((state for state in states[index+1:]
            if state['started_at'] >= root.started_at+300), None) if root is not None else None
        coverage[line] = dict(root_sequence=root.sequence if root is not None else None,
            root_clock=root.started_at if root is not None else None,
            first_real_300_endpoint_sequence=end['sequence'] if end else None,
            first_real_300_endpoint_clock=end['started_at'] if end else None,
            final_endpoint_span=endpoint['started_at']-root.started_at if endpoint and root is not None else None)
    continuous = all(left.sequence+1 == right.sequence and left.completed_at == right.started_at
        and np.array_equal(left.ns, right.s) and left.memory_ns == right.memory_s
        for left, right in zip(rows, rows[1:]))
    endpoint_exact = bool(rows) and endpoint is not None and (
        endpoint['sequence'] == rows[-1].sequence+1 and
        endpoint['started_at'] == rows[-1].completed_at and
        np.array_equal(endpoint['s'], rows[-1].ns) and endpoint['memory_s'] == rows[-1].memory_ns)
    checks = dict(exact271=len(rows) == CAPACITY, continuous_real_events=continuous,
        one_native_episode=bool(rows) and len({row.episode_id for row in rows}) == 1,
        real272nd_issue=endpoint_exact, zero_fake_terminal=all(not row.terminal for row in rows),
        three_lines=all(value['root_sequence'] is not None for value in coverage.values()),
        all_three_reach300=all(value['first_real_300_endpoint_clock'] is not None for value in coverage.values()),
        endpoint_by9600=endpoint is not None and endpoint['started_at'] <= 9600,
        released_by9630=bool(rows) and max(row.available_at for row in rows) <= 9630,
        unique_feedback=len({row.feedback_id for row in rows}) == len(rows))
    return dict(all_pass=all(checks.values()), checks=checks, root_coverage=coverage,
        rows=len(rows), zero_duration_rows=sum(row.dt == 0 for row in rows),
        endpoint_sequence=endpoint['sequence'] if endpoint else None,
        endpoint_clock=endpoint['started_at'] if endpoint else None,
        missing_roots=[line for line, value in coverage.items() if value['root_sequence'] is None],
        missing300=[line for line, value in coverage.items() if value['first_real_300_endpoint_clock'] is None])


class ContinuousStream(MemoryEventStream):
    def __init__(self, *args, on_response_row, response_file, **kwargs):
        self.response_start = None
        self.response_rows = []
        self.response_endpoint = None
        self.on_response_row, self.response_file = on_response_row, response_file
        super().__init__(*args, **kwargs)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending, queued = self._pending, len(self._queue)
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if pending is None or len(self._queue) == queued:
            return
        row = self._queue[-1]
        row.behavior_s = deepcopy(pending['behavior_s'])
        if self.response_start is not None and self.response_start <= row.sequence < self.response_start+CAPACITY:
            row.feedback_id = f'continuous/{len(self.response_rows)}'
            self.response_rows.append(row)
            self.response_file.write(json.dumps(dict(transition=transition_dict(row)),
                separators=(',', ':'), allow_nan=False)+'\n')
            self.response_file.flush()
            self.on_response_row(row)


def collect(args, protocol):
    from .evaluator import FrozenIssueEvaluator

    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = stream = memory = trace = hook = None
    files = []
    started = time.time()
    warmup, issue_audits, applications = [], [], {}
    evaluator = controller = None
    preflight = admission = first_response_score = None
    try:
        learner, replay, contract, stations, initializer = make_learner(args.seed, arm='shared_level')
        parity = initial_preflight(learner)
        if learner.total_updates != 50000 or replay.target_received:
            raise RuntimeError('Require unchanged 50000-step initializer and empty target feedback')
        identities = objects(learner)
        controller = StochasticController(learner, seed=protocol['future_private_action_seed'])
        budget = WarmupBudget(dict(feedback_first_deadline=1200, feedback_last_deadline=1800,
            feedback_interval_seconds=300, feedback_records_per_packet=5))
        behavior_writer = (out/'behavior_prefix.jsonl').open('x')
        response_writer = (out/'continuous_feedback.jsonl').open('x')
        audit_writer = (out/'issue_support.jsonl').open('x')
        application_writer = (out/'native_applications.jsonl').open('x')
        files.extend((behavior_writer, response_writer, audit_writer, application_writer))
        pending_applications = {}
        snapshot_clock = None
        frozen_counts = None

        def counts():
            return dict(actual_updates=learner.total_updates-50000, admitted_slots=replay.target_received,
                optimizer_steps=steps(learner))

        def save_model(name):
            with (out/name).open('xb') as target:
                torch.save(dict(learner=learner, contract=contract, stations=stations,
                    initializer=str(initializer), seed=args.seed, scenario=args.scenario,
                    saved_at=stream.now, total_updates=learner.total_updates,
                    feedback_slots=replay.target_received, optimizer_steps=steps(learner),
                    torch_rng=torch.get_rng_state(), learner_rng=deepcopy(learner.rng.bit_generator.state),
                    diagnostic_rng=deepcopy(learner.diagnostic_rng.bit_generator.state),
                    action_rng=controller.rng.clone(), action_draws=controller.draws,
                    frozen_reference=controller.reference), target)

        def finish_preflight():
            nonlocal preflight
            application = applications.get(stream.response_start)
            if preflight is None and first_response_score is not None and application is not None:
                preflight = dict(all_pass=first_response_score['all_pass']
                    and all(v['all_pass'] for v in issue_audits) and application['all_pass'],
                    real_first_response=first_response_score, first_issue=issue_audits[0],
                    first_native_application=application,
                    reused_in_collection=True, duplicate_prefixes=0, duplicate_warmup_updates=0,
                    actual_updates=24, warmup_labels=14, reserved_response_updates=432,
                    atomic_batch_closed_rows_at_gate=len(stream.response_rows))
                write(out/'native_preflight.json', preflight)
                if not preflight['all_pass']:
                    raise RuntimeError('First actual response closure failed synchronized preflight')

        def on_row(row):
            nonlocal first_response_score
            if len(stream.response_rows) == 1:
                first_response_score = evaluator.score(learner, row, seed=536253)
                finish_preflight()

        def issue(event):
            event['proposed_a'] = np.asarray(controller.last['proposal'], np.float32)
            event['behavior_s'] = dict(deepcopy(controller.last), intervention=False,
                executed_action=np.asarray(event['a']).tolist())
            trace.observe(event, stream._snapshot())
            behavior_writer.write(json.dumps(dict(sequence=event['sequence'], started_at=event['started_at'],
                line_id=event['line_id'], behavior_s=event['behavior_s']), separators=(',', ':'))+'\n')
            if stream.response_start is None and 9000 <= event['started_at'] < 9300:
                if evaluator is None or snapshot_clock >= event['started_at']:
                    raise RuntimeError('A response issue preceded its immutable evaluator snapshot')
                stream.response_start = event['sequence']
            if stream.response_start is not None and event['sequence'] <= stream.response_start+CAPACITY:
                audit = evaluator.issue(event['behavior_s'], issue_clock=event['started_at'])
                audit['checks']['actual_issue_actor_input_exact'] = np.array_equal(
                    event['s'], event['behavior_s']['policy_input'])
                audit['checks']['actual_issue_action_exact'] = np.array_equal(
                    event['a'], event['behavior_s']['executed_action'])
                audit['checks']['new_post24_snapshot_tag'] = event['behavior_s']['actor_snapshot'] == SNAPSHOT
                audit['all_pass'] = all(audit['checks'].values())
                audit.update(sequence=event['sequence'], line_id=event['line_id'], started_at=event['started_at'])
                audit_writer.write(json.dumps(plain(audit), separators=(',', ':'), allow_nan=False)+'\n')
                if not audit['all_pass']:
                    raise RuntimeError('Actual issue input, F or synchronized continuation support failed')
                if event['sequence'] < stream.response_start+CAPACITY:
                    issue_audits.append(audit)
                    pending_applications[(event['trip_id'], event['station_id'])] = dict(
                        sequence=event['sequence'], issue_clock=event['started_at'],
                        expected_hold=float(policy_action_to_hold_seconds_v7(event['a'][0], contract)))
                else:
                    stream.response_endpoint = deepcopy(event)

        def admit():
            nonlocal admission
            if admission is not None:
                return
            coverage = flow_coverage(stream.response_rows, stream.response_endpoint)
            write(out/'flow_coverage.json', coverage)
            if not coverage['all_pass'] or preflight is None or not preflight['all_pass']:
                raise RuntimeError('The fixed 271-TD flow did not reach all real 300-second endpoints')
            if len(applications) != CAPACITY or not all(value['all_pass'] for value in applications.values()):
                raise RuntimeError('Native applied holding differs from the recorded executed F action')
            if replay.target_received != 14 or learner.total_updates != 50024 or counts() != frozen_counts:
                raise RuntimeError('Response training or feedback appeared before the declared admission')
            replay.add_target(stream.response_rows)
            admission = dict(processed_at=stream.now, ordinary_slots=14, response_slots=CAPACITY,
                total_slots=replay.target_received, actual_updates=24, reserved_response_updates=432,
                contract_total_updates=456, response_training_started=False,
                all271_available=max(row.available_at for row in stream.response_rows) <= stream.now,
                shared_suffix_references_are_not_extra_labels=True, coverage=coverage)
            write(out/'admission.json', admission)
            save_model('feedback_ready_model.pt')

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=protocol['future_external_seed'])
        stream = ContinuousStream(env, stations, contract, domain='target', on_issue=issue,
            delay_seconds=30., execute_transform=False, on_response_row=on_row, response_file=response_writer)
        stream.reset()
        memory = stream.memory = ExecutionMemory(env._bridge)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        hook = install_scenario(env, args.scenario, shock_at=9000.,
            factor=protocol.get('speed_factor', .65), metric_every=300.)
        bridge = env._bridge
        original_apply, original_advance = bridge.apply_action, bridge._advance_one_step

        def apply(event, command):
            before = stream.now
            result = original_apply(event, command)
            expected = pending_applications.pop((str(event.bus_id), str(event.stop_id)), None)
            if expected is not None:
                actual = event.metadata.get('applied_additional_hold_seconds')
                value = dict(**expected, actual_hold=actual, command_hold=float(command[0]),
                    real_clock_before=before, real_clock_after=stream.now,
                    all_pass=actual is not None and abs(actual-expected['expected_hold']) <= 1e-5
                        and before == stream.now == expected['issue_clock'])
                applications[expected['sequence']] = value
                application_writer.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
                finish_preflight()
            return result

        last_progress = time.monotonic()

        def advance(*values, **keywords):
            nonlocal last_progress
            result = original_advance(*values, **keywords)
            if stream.now == 9630:
                admit()
            if time.monotonic()-last_progress >= 45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v53_native_factual_collection',
                    current=stream.now, total=24000, unit='sim_seconds', ordinary_packets=budget.index,
                    response_rows=len(stream.response_rows), actual_updates=learner.total_updates-50000)), flush=True)
                last_progress = time.monotonic()
            return result

        bridge.apply_action, bridge._advance_one_step = apply, advance
        native_started = time.time()
        while not stream.done:
            released = stream.step(controller)
            stream.release_intervals()
            if budget.index < 3:
                budget.ingest(released)
                while budget.due(stream.now):
                    rows, _ = budget.take(stream.now)
                    receipt = deepcopy(budget.receipts[-1])
                    for row in rows:
                        row.feedback_id = f'ordinary/{row.sequence}'
                    replay.add_target(rows)
                    sampling = []
                    for step in range(8):
                        learner.continue_step(collect_diagnostics=step == 0)
                        sampling.append(deepcopy(replay.last_sampling))
                    if learner.total_updates != 50000+8*budget.index or objects(learner) != identities:
                        raise RuntimeError('Warmup must keep all native optimizer objects for exactly 8 updates')
                    receipt.update(feedback_ids=[row.feedback_id for row in rows], sampling=sampling,
                        optimizer_steps=steps(learner), actual_updates=learner.total_updates-50000,
                        torch_rng=plain(torch.get_rng_state()), learner_rng=deepcopy(learner.rng.bit_generator.state),
                        diagnostic_rng=deepcopy(learner.diagnostic_rng.bit_generator.state))
                    warmup.append(receipt)
                    write(out/f'warmup_packet_{budget.index}.json', dict(receipt=receipt,
                        rows=[transition_dict(row) for row in rows]))
                    if budget.index == 3:
                        if replay.target_received != 14 or stream.now >= 9000:
                            raise RuntimeError('The new post24 snapshot requires exactly 14 early labels')
                        snapshot_clock = stream.now
                        controller.switch_post24(learner)
                        controller.snapshot = SNAPSHOT
                        evaluator = FrozenIssueEvaluator(controller.reference, snapshot_clock)
                        warmup_proof = evaluator.score(learner, rows[-1], seed=536252)
                        warmup_proof['wiring'] = evaluator.check_wiring(learner, rows[-1], seed=536252)
                        warmup_proof['all_pass'] = warmup_proof['all_pass'] and warmup_proof['wiring']['all_pass']
                        write(out/'post24_preflight.json', warmup_proof)
                        if not warmup_proof['all_pass']:
                            raise RuntimeError('Synchronized lambda-zero native parity failed at real warmup row')
                        save_model('new_post24_from14_model.pt')
                        frozen_counts = counts()
                        write(out/'behavior_snapshot.json', dict(snapshot=SNAPSHOT, saved_at=snapshot_clock,
                            actual_updates=24, ordinary_slots=14, frozen_through=24000,
                            behavior_evaluator_same_actor=True, initializer=str(initializer)))
        hook.capture_final()
        write(out/'scenario.json', dict(metadata=hook.metadata(), metrics=hook.metrics))
        coverage = flow_coverage(stream.response_rows, stream.response_endpoint)
        checks = dict(initial_parity=parity['all_pass'], full_day=stream.now == 24000,
            complete_trace=trace.observed_issues == stream.issued_count,
            ordinary14=[row['records'] for row in warmup] == [5, 5, 4],
            preflight_reused=preflight is not None and preflight['all_pass'],
            native_coverage=coverage['all_pass'], admission9630=admission is not None and admission['processed_at'] == 9630,
            total285=replay.target_received == 285, only24_actual_updates=learner.total_updates == 50024,
            no_response_training=steps(learner) == frozen_counts['optimizer_steps'],
            actual_applications=len(applications) == CAPACITY and all(row['all_pass'] for row in applications.values()),
            actual_issue_support=len(issue_audits) == CAPACITY and all(row['all_pass'] for row in issue_audits),
            prescribed_scenario=hook.applied_at == (9000. if args.scenario == 'local_speed_drop' else None),
            memory_causal=memory.summary()['integrity']['all_pass'], no_world=learner.world is None)
        overlap = {}
        for line in (*LINES, 'all'):
            values = [value for value in issue_audits if line == 'all' or value['line_id'] == line]
            actions = [row.a[0] for row in stream.response_rows if line == 'all' or row.line_id == line]
            overlap[line] = dict(issues=len(values), executed_zero=sum(action == -1 for action in actions),
                executed_positive=sum(action > -1 for action in actions),
                pointwise_c_min=min((value['ratio']['c'] for value in values), default=None),
                conditional_overlap_min=min((value['overlap']['total_overlap'] for value in values), default=None),
                all_pass=bool(values) and all(value['all_pass'] for value in values))
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario,
            initializer=str(initializer), initializer_seed=args.seed, initializer_updates=50000,
            external_seed=protocol['future_external_seed'], private_action_seed=controller.seed,
            scenario_actual_shock_at=hook.applied_at,
            snapshot_clock=snapshot_clock, post24_preflight=warmup_proof,
            native_preflight=preflight, coverage=coverage,
            overlap=overlap,
            budget=dict(ordinary_labels=14, response_labels=len(stream.response_rows), admitted_slots=replay.target_received,
                actual_updates=24, warmup_updates=24, response_updates=0, reserved_response_updates=432,
                contract_total_updates=456, new_native_prefixes=1, new_native_branches=0,
                prefix_native_seconds=stream.now, duplicate_prefixes=0, diagnostic_truth_reused=0),
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started,
                native_finished_wall=time.time(), wall_seconds=time.time()-started), response_training_started=False)
        write(out/'summary.json', summary)
        if not summary['integrity']['all_pass']:
            raise RuntimeError('Native collection integrity failed')
        return summary
    except BaseException as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            budget=dict(prefix_native_seconds=stream.now if stream is not None else 0,
                ordinary_labels=sum(row['records'] for row in warmup),
                response_closed_rows=len(stream.response_rows) if stream is not None else 0,
                feedback_slots=replay.target_received if 'replay' in locals() else 0,
                actual_updates=learner.total_updates-50000 if 'learner' in locals() else 0,
                reserved_response_updates=432, response_updates=0, duplicate_prefixes=0)))
        raise
    finally:
        for handle in files:
            handle.close()
        if trace is not None:
            trace.close()
        if hook is not None:
            hook.close()
        if memory is not None:
            memory.close()
        if env is not None:
            env.close()
