"""Replay v37's unchanged factual prefix and collect legal native credit chains."""
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
if __package__:
    from .chain_stream import ChainRecorder, CreditEventStream, cost_between, ledger
else:
    from chain_stream import ChainRecorder, CreditEventStream, cost_between, ledger

import numpy as np
import torch

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.collect import action_choices, transition_dict
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze, optimizer_steps
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, selected_state
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import create_target_env

ARMS = ('low', 'high')
LINES = ('7X', '102X', '705X')


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


class CreditPair(NativeTruthPair):
    def __init__(self, *args, **kwargs):
        self.recorder = None
        super().__init__(*args, **kwargs)

    def _install_native_observer(self, stream, *, traci=None):
        super()._install_native_observer(stream, traci=traci)
        advance = stream.env._bridge._advance_one_step

        def observed_advance(*args, **kwargs):
            result = advance(*args, **kwargs)
            if self.recorder is not None:
                self.recorder.observe_tick(stream)
            return result

        stream.env._bridge._advance_one_step = observed_advance

    def observe_issue(self, event, stream):
        super().observe_issue(event, stream)
        if self.recorder is None:
            self.recorder = ChainRecorder(self.out/self.arm, self.selected)
            stream.recorder = self.recorder
        self.recorder.observe_tick(stream)

    def build_summary(self, stream):
        result = super().build_summary(stream)
        elapsed = float(stream.stop_snapshot.time_seconds-stream.fork_snapshot.time_seconds)
        checks = result['integrity']['checks']
        checks['native_suffix_exact'] = (stream.cutoff_reached and
            stream.native_suffix_steps == elapsed and 300. <= elapsed <= 420.)
        chain = self.recorder.summary()
        checks['causal_real_event_chain'] = chain['integrity']['all_pass']
        checks['exact300_ledger_available'] = (stream.boundary300 is not None
            and stream.boundary300['clock'] == self.selected['started_at']+300.)
        checks['real_endpoint_not_issued'] = (stream.real_endpoint is not None
            and stream.real_endpoint['issued'] is False and stream.real_endpoint['executed'] is False)
        checks['ledger_tick_coverage'] = chain['ledger_tick_count'] == stream.native_suffix_steps+1
        result['global_transition'] = transition_dict(stream.fork_transition)
        result['chain'], result['boundary300'] = chain, stream.boundary300
        result['real_endpoint'] = stream.real_endpoint
        result['pending_partial'] = stream.endpoint_pending_partial
        result['cost300_to_endpoint'] = (cost_between(stream.boundary300['ledger'],
            stream.real_endpoint['ledger']) if stream.real_endpoint['status'] == 'observed' else None)
        result['budget'].update(actual_suffix_native_seconds=elapsed,
            overrun_native_seconds=elapsed-300., exact300_native_seconds=300.,
            diagnostic_truth_outcomes=1, new_training_labels=0, new_optimizer_updates=0)
        result['integrity']['all_pass'] = all(checks.values())
        result['status'] = 'complete' if result['integrity']['all_pass'] else 'integrity_failed'
        return result


def declared_slots(old_pairs, protocol):
    old = {record['line_id']: record for record in old_pairs
           if record['role'] == 'heldout' and record['window_name'] == 'early'}
    if set(old) != set(LINES):
        raise RuntimeError('All three original early issues must exist before credit replay')
    result = []
    for line in LINES:
        result.append(dict(slot=line+'/early', line_id=line, window='early',
            sequence=old[line]['source_sequence'], selected=None, attempts=0,
            reasons=Counter(), reference=old[line]))
        result.append(dict(slot=line+'/late', line_id=line, window='late',
            bounds=protocol['late_window_seconds'], selected=None, attempts=0,
            reasons=Counter(), reference=None))
    return result


def compare_old_pair(record, pair):
    """Only predeclared old issue, exact next-event row and exact300 cost."""
    checks = dict(selected_issue_exact=pair.values['low']['selected'] == record['selected'],
        selected_both_arms_exact=pair.values['low']['selected'] == pair.values['high']['selected'])
    for arm in ARMS:
        expected, actual = record['outcomes'][arm], pair.values[arm]
        checks[arm+'_first_transition_exact'] = actual['global_transition'] == expected['transition']
        checks[arm+'_fixed300_exact'] = actual['truth']['fixed300'] == expected['truth']
        checks[arm+'_actual_action_exact'] = (actual['execution']['executed_action'] == record['actions'][arm]
            and actual['execution']['native_applied_hold_seconds'] == record['holds'][ARMS.index(arm)])
    return dict(all_pass=all(checks.values()), checks=checks)


def failure_budget(out, stream, pair):
    """Keep completed and observed partial work; never fill a killed arm to300."""
    clock = float(stream.now) if stream is not None and stream.initial_snapshot is not None else 0.
    if pair is not None and pair.in_child:
        elapsed = max(0., clock-float(pair.expected['selected']['started_at']))
        return dict(scope='one_suffix', actual_suffix_native_seconds=elapsed,
            overrun_native_seconds=max(0., elapsed-300.), diagnostic_truth_outcomes=0,
            new_training_labels=0, new_optimizer_updates=0)
    arms = []
    for branch in sorted((Path(out)/'states').glob('*')):
        for arm in ARMS:
            root = branch/arm
            summary_path, failure_path, ticks_path = root/'summary.json', root/'failure.json', root/'ticks.jsonl'
            if summary_path.exists():
                value = json.loads(summary_path.read_text())
                seconds = value['budget']['actual_suffix_native_seconds']
                arms.append(dict(path=str(root), status=value['status'],
                    observed_native_seconds=seconds, exact_duration=True,
                    diagnostic_truth_outcomes=1))
            elif failure_path.exists():
                value = json.loads(failure_path.read_text())
                seconds = value['budget']['actual_suffix_native_seconds']
                arms.append(dict(path=str(root), status='failed',
                    observed_native_seconds=seconds, exact_duration=True,
                    diagnostic_truth_outcomes=0))
            elif ticks_path.exists():
                last = None
                with ticks_path.open() as handle:
                    for line in handle:
                        last = json.loads(line)
                arms.append(dict(path=str(root), status='interrupted',
                    observed_native_seconds=last['relative_seconds'] if last is not None else 0.,
                    exact_duration=False, diagnostic_truth_outcomes=0))
            elif (root/'fork_prepare_failure.txt').exists():
                arms.append(dict(path=str(root), status='fork_prepare_failed',
                    observed_native_seconds=0., exact_duration=True,
                    diagnostic_truth_outcomes=0))
            elif (branch/(arm+'.log')).exists():
                arms.append(dict(path=str(root), status='started_without_native_receipt',
                    observed_native_seconds=None, exact_duration=False,
                    diagnostic_truth_outcomes=0))
    return dict(scope='failed_collection', new_native_prefixes=int(stream is not None),
        prefix_native_seconds=clock, observed_suffixes=len(arms),
        completed_suffixes=sum(value['status'] == 'complete' for value in arms),
        observed_suffix_native_seconds=sum(value['observed_native_seconds'] or 0. for value in arms),
        observed_overrun_native_seconds=sum(max(0., (value['observed_native_seconds'] or 0.)-300.) for value in arms),
        unknown_or_lower_bound_suffixes=sum(not value['exact_duration'] for value in arms),
        diagnostic_truth_outcomes=sum(value['diagnostic_truth_outcomes'] for value in arms),
        new_training_labels=0, new_optimizer_updates=0, arms=arms)


def collect(args, protocol):
    torch.set_num_threads(1)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = HERE.parent/protocol['source_v37_directory']/'runs/collection'/args.scenario
    env = stream = trace = memory = hook = pair = None
    started = time.time()
    try:
        old_pairs = json.loads((source/'pair_truth.json').read_text())
        schedule = declared_slots(old_pairs, protocol)
        learner, replay, contract, stations, initializer = load_learner(args.seed, arm='fresh')
        if learner.total_updates != 50000 or replay.target_received:
            raise RuntimeError('The unchanged frozen50000 prefix controller is required')
        freeze(learner)
        initial_steps = optimizer_steps(learner)
        records, pre_recorded = [], set()
        last_proposal = None
        native_checks = []

        def counts():
            return dict(total_updates=learner.total_updates, native_feedback=replay.target_received,
                new_optimizer_updates=0, new_training_labels=0, world_updates=0, trust_updates=0)

        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return learner.action_transform.numpy(state, last_proposal)

        def selector(event):
            nonlocal pair
            if pair is not None and pair.in_child:
                return None
            slot = next((value for value in schedule if value['selected'] is None
                and value['line_id'] == event['line_id'] and
                (event['sequence'] == value['sequence'] if value['window'] == 'early' else
                 value['bounds'][0] <= event['started_at'] < value['bounds'][1])), None)
            if slot is None:
                return None
            slot['attempts'] += 1
            choices = action_choices(learner.action_transform, event['s'])
            api = importlib.import_module(env._bridge.__class__.__module__).traci
            try:
                stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
                step = float(api.simulation.getDeltaT())
                action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            except Exception:
                slot['reasons']['native_execution_observation_missing'] += 1
                if slot['window'] == 'early':
                    raise RuntimeError('The fixed old early issue lacks its native execution observation')
                return None
            feasible = choices is not None and stop and 1.875 >= 1.5*max(step, action_step)
            if not feasible:
                slot['reasons']['F_or_native_stop_or_step_unsupported'] += 1
                if slot['window'] == 'early':
                    raise RuntimeError('The fixed old early issue no longer has native action support')
                return None
            event['proposed_a'] = last_proposal.copy()
            selected = selected_state(event, stream._snapshot())
            if slot['window'] == 'early' and selected != slot['reference']['selected']:
                raise RuntimeError('The old early issue state/memory differs from its exact reference')
            slot['selected'] = dict(sequence=event['sequence'], issue=event['started_at'])
            native = dict(at_stop=stop, simulation_step=step, action_step=action_step,
                action_holds_seconds=[0., 1.875], feasible=True)
            native_checks.append(dict(slot=slot['slot'], **native))
            trace.observe(event, stream._snapshot())
            pre_recorded.add(event['sequence'])
            state_out = out/'states'/slot['slot'].replace('/', '_')
            expected = dict(selected=selected,
                actions={arm: choices[arm]['action'].tolist() for arm in ARMS},
                holds={arm: choices[arm]['hold'] for arm in ARMS})
            pair = CreditPair(state_out, expected, counts, protocol['native_external_seed'])
            replacement = pair.fork(event, stream, trace)
            if pair.in_child:
                return replacement
            old_proof = compare_old_pair(slot['reference'], pair) if slot['window'] == 'early' else None
            if old_proof is not None and not old_proof['all_pass']:
                raise RuntimeError('An original early first-transition/fixed300 truth failed exact replay')
            record = dict(slot=slot['slot'], selected=selected, actions=expected['actions'],
                holds=expected['holds'], native_feasibility=native,
                old_early_replay=old_proof, pair_receipt=pair.receipt,
                comparison=pair.comparison,
                branch_paths={arm: str(state_out/arm/'summary.json') for arm in ARMS})
            write(state_out/'state_summary.json', record)
            records.append(record)
            if len(records) == 1:
                write(out/'preflight_gate.json', dict(all_pass=True, reused_in_full_run=True,
                    prefix_runs=1, slot=slot['slot'], native_feasibility=native,
                    old_early_replay=old_proof, actual_parallel=pair.receipt['all_pass']))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='credit_native_pair',
                current=stream.now, total=24000, unit='sim_seconds', slot=slot['slot'],
                collected_pairs=len(records))), flush=True)
            return None

        def issue(event):
            event['proposed_a'] = last_proposal.copy()
            if pair is not None and pair.in_child:
                pair.observe_issue(event, stream)
                return
            if event['sequence'] not in pre_recorded:
                trace.observe(event, stream._snapshot())

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=protocol['native_external_seed'])
        stream = CreditEventStream(env, stations, contract, domain='target', selector=selector,
            on_issue=issue, delay_seconds=30., execute_transform=False, fork_duration_seconds=300.,
            max_overrun_seconds=protocol['maximum_endpoint_overrun_seconds'])
        stream.reset()
        memory = ExecutionMemory(env._bridge)
        stream.memory = memory
        trace = NativePrefixTrace(out/'prefix_trace.jsonl', reference=source/'prefix_trace.jsonl')
        hook = install_scenario(env, args.scenario, shock_at=9000., factor=.65, metric_every=300.)
        native_started = time.time()
        progress = time.monotonic()
        while not stream.done:
            stream.step(policy)
            stream.release_intervals()
            if pair is not None and pair.in_child:
                pair.observe_progress(stream)
                if stream.cutoff_reached:
                    pair.finish_child(stream)
                continue
            if time.monotonic()-progress >= 45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='exact_v37_prefix',
                    current=stream.now, total=24000, unit='sim_seconds', pairs=len(records))), flush=True)
                progress = time.monotonic()
        native_finished = time.time()
        source_end = trace.reference.readline()
        selection = {value['slot']: dict(line_id=value['line_id'], window=value['window'],
            selected=value['selected'], attempts=value['attempts'], reasons=dict(value['reasons']),
            status='selected' if value['selected'] is not None else 'missing') for value in schedule}
        summaries = [json.loads(Path(path).read_text()) for record in records
                     for path in record['branch_paths'].values()]
        checks = dict(full24000_parent=stream.done and stream.now == 24000.,
            full_v37_prefix_exact=trace.comparison_exact and trace.observed_issues == trace.matched_issues
                == stream.issued_count and not source_end,
            all_six_declared_issues_collected=len(records) == 6,
            all_later_native_feasible=sum(row['slot'].endswith('/late') for row in native_checks) == 3,
            all_three_old_early_exact=sum(record['old_early_replay'] is not None and
                record['old_early_replay']['all_pass'] for record in records) == 3,
            all_pairs_actual_parallel=all(record['pair_receipt']['all_pass'] for record in records),
            all_suffix_integrity=all(value['integrity']['all_pass'] for value in summaries),
            collector_frozen=learner.total_updates == 50000 and replay.target_received == 0
                and optimizer_steps(learner) == initial_steps,
            causal_memory=memory.summary()['integrity']['all_pass'])
        budget = dict(new_native_prefixes=1, prefix_native_seconds=24000., pairs=len(records),
            suffixes=len(summaries), fixed300_suffix_seconds=300.*len(summaries),
            actual_suffix_native_seconds=sum(value['budget']['actual_suffix_native_seconds'] for value in summaries),
            overrun_native_seconds=sum(value['budget']['overrun_native_seconds'] for value in summaries),
            missing_real_endpoints=sum(value['real_endpoint']['status'] != 'observed' for value in summaries),
            zero_duration_transitions=sum(value['chain']['zero_duration_count'] for value in summaries),
            diagnostic_truth_outcomes=len(summaries), new_training_labels=0,
            new_optimizer_updates=0, frozen_training_slots=285,
            frozen_updates_per_model=456, preflight_duplicate_prefixes=0, preflight_duplicate_suffixes=0)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            initialization_seed=args.seed, native_external_seed=protocol['native_external_seed'],
            initializer=str(initializer), scenario=args.scenario, selection=selection,
            states=[dict(slot=record['slot'], path=str(out/'states'/record['slot'].replace('/', '_')/'state_summary.json'))
                    for record in records], budget=budget,
            prefix=dict(trace_rows=trace.observed_issues, matched_rows=trace.matched_issues),
            wall=dict(native_started=native_started, native_finished=native_finished,
                total_seconds=time.time()-started))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('v38 native collection incomplete; missing states retained')
        return summary
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error),
            native_time=stream.now if stream is not None and stream.initial_snapshot is not None else None,
            budget=failure_budget(out, stream, pair))
        if pair is not None and pair.in_child:
            write(pair.out/pair.arm/'failure.json', failure)
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        write(out/'failure.json', failure)
        raise
    finally:
        if pair is not None and pair.in_child:
            os._exit(1)
        if memory is not None: memory.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=6202)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seed != 6202:
        raise ValueError('v38 retains the fixed6202 initializer')
    collect(args, json.loads((HERE/'protocol.json').read_text()))

