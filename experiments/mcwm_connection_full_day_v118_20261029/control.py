"""One exact native prefix and two fixed full-day service consequences."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
ROOT_PID = os.getpid()


class PrefixComplete(Exception):
    def __init__(self, receipt):
        self.receipt = receipt


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    anchor = json.loads((HERE/'anchor_binding.json').read_text())['action']
    root, parent_out = args.thesis_root.resolve(), args.out.resolve()
    parent_out.mkdir(parents=True, exist_ok=False)
    out = parent_out
    env = stream = hook = reference = observer = None
    branch = None
    native_started = False
    root_receipt = root_capture = prefix_check = None
    native_started_wall = native_finished_wall = None
    started = last_progress = time.monotonic()
    counts = dict(applied_commands=0, discarded_transitions=0, discarded_intervals=0, released_terminal_transitions=0)
    root_issue_count = 0
    result = dict(status='failed', integrity=dict(all_pass=False), performance_evidence=False)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None and native_started else 0,
            total=p['stop_clock'], unit='sim_seconds', branch=branch,
            elapsed_seconds=time.monotonic()-started)), flush=True)

    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        from experiments.mcwm_logged_action_observation_v116_20261029.replay import ReplayReference, clean
        from experiments.mcwm_logged_action_observation_v116_20261029.observe import capture
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env
        from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        from experiments.mcwm_connection_window_v117_20261029.support import RecordedContinuation, state_receipt
        from accounting import Accounting
        from validation import compare_nominal
        metadata = json.loads((root/p['metadata']).read_text())
        contract = TransitMDPConfigV7(**metadata['contract'])
        stations = np.asarray(metadata['station_max_by_line'], np.float32)
        old = root/p['reference_root']
        actions = [old/name for name in p['action_parts']]
        ticks = [old/name for name in p['tick_parts']]
        continuation = RecordedContinuation(actions, p['root']['sequence'], p['stop_clock'], ticks)
        reference = ReplayReference(actions, ticks, p['root']['sequence'], p['root']['clock'])

        def boundary_receipt():
            return dict(state=state_receipt(stream, counts),
                accounting=dict(ticks=observer.ticks, issues=observer.issues, commands=observer.commands,
                    people=len(observer.people), visits=len(observer.visits),
                    wait=observer.total['wait'], ride=observer.total['ride'], root_binding=observer.root_binding))

        def policy(state):
            nonlocal branch, out, root_receipt, root_capture, prefix_check
            nonlocal native_started_wall
            if branch is not None:
                return np.asarray(continuation.proposal(state, stream), np.float32)
            proposal = reference.next_action(state)
            if stream.issued_count != p['root']['sequence']:
                return np.asarray(proposal, np.float32)
            if reference.current != anchor or stream.now != p['root']['clock']:
                raise RuntimeError('Declared root record or native clock differs')
            prefix_check = continuation.close_root(reference, stream)
            write(parent_out/'prefix_check.json', prefix_check)
            observed = capture(env._bridge, anchor, diagnostic_person_id=p['root']['diagnostic_person_id'])
            if not observed['summary']['all_pass']:
                raise RuntimeError('Root current native observation did not reproduce')
            root_capture = observed['full']
            write(parent_out/'root_capture.json', root_capture)
            binding = observer.bind_root(root_capture, diagnostic_person_id=p['root']['diagnostic_person_id'])
            if len(binding['receiver_rider_ids']) != 13:
                raise RuntimeError('Fixed receiver cohort must contain13 actual riders')
            root_receipt = boundary_receipt()
            write(parent_out/'root_state.json', root_receipt)
            observer.flush()
            trace = SimpleNamespace(writer=observer.writers['actions'], reference=None)

            def prepare_child(key):
                nonlocal branch, out, native_started_wall
                branch, out = key, parent_out/'branches'/key
                observer.detach(out)
                receipt = boundary_receipt()
                write(out/'root_state.json', receipt)
                if receipt != root_receipt:
                    raise RuntimeError('Child stream/ledger/RNG differs from paused parent')
                native_started_wall = time.time()

            selected = fork_children(parent_out/'branches', trace, lambda: stream.now,
                lambda: dict(counts), prepare_child, timeout_seconds=p['runtime']['child_timeout_seconds'])
            if isinstance(selected, dict):
                raise PrefixComplete(selected)
            return np.asarray(p['actions'][branch]['proposal'], np.float32)

        def issue(event):
            nonlocal root_issue_count
            if branch is None:
                reference.check_event(event, asdict(stream._snapshot()))
            else:
                if event['sequence'] == p['root']['sequence']:
                    expected = dict(anchor, proposed_a=p['actions'][branch]['proposal'],
                        a=p['actions'][branch]['executed'])
                    actual = clean(event)
                    different = [key for key in expected if key != 'visit_index' and actual.get(key) != expected[key]]
                    if different:
                        raise RuntimeError('Actual root issue differs: '+str(different))
                    root_issue_count += 1
                if branch == 'low':
                    continuation.check_issue(event, asdict(stream._snapshot()))
                continuation.note_issue(event)
            observer.issue(event)

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'], root/p['edge_xml'],
            episode_seconds=p['episode_horizon_seconds'], seed=p['external_seed'])
        env._ensure_bridge()
        original_apply = env._bridge.apply_action

        def apply(event, action):
            counts['applied_commands'] += 1
            return original_apply(event, action)

        env._bridge.apply_action = apply
        observer = Accounting(env._bridge, out)
        stream = EventStream(env, stations, contract, domain='target', delay_seconds=30,
            execute_transform=True, on_issue=issue)
        native_started, native_started_wall = True, time.time()
        stream.reset()
        from bus_h2o.sumo_env import rl_bridge
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
        if '1.25.0' not in str(rl_bridge.traci.getVersion()[1]):
            raise RuntimeError('Frozen runtime requires SUMO1.25.0')
        hook = install_scenario(env, 'stable', **p['scenario_settings'])
        progress('recorded_prefix')
        try:
            while not stream.done:
                transitions = stream.step(policy)
                counts['discarded_transitions'] += len(transitions)
                counts['released_terminal_transitions'] += sum(row.terminal for row in transitions)
                counts['discarded_intervals'] += len(stream.release_intervals())
                if stream.now > p['stop_clock']:
                    raise RuntimeError('Native clock passed the declared full horizon')
                if time.monotonic()-last_progress >= 30:
                    progress('recorded_prefix' if branch is None else 'full_day_suffix')
                    last_progress = time.monotonic()
            if branch is None:
                raise RuntimeError('Episode ended before the declared fork')
            native_finished_wall = time.time()
            observed = observer.finish()
            low_check = continuation.check_end(asdict(stream._snapshot())) if branch == 'low' else None
            with (out/'commands.jsonl').open() as commands:
                root_commands = [row for row in map(json.loads, commands) if row['sequence'] == p['root']['sequence']]
            command = root_commands[0] if len(root_commands) == 1 else None
            with (out/'exchanges.jsonl').open() as exchanges:
                target_events = [row for row in map(json.loads, exchanges)
                    if p['root']['diagnostic_person_id'] in row['boarded_ids']+row['alighted_ids']]
            visits = [row for row in observer.visits if row['trip_id'] == p['root']['trip_id']
                and row['stop_id'] == p['root']['station_id'] and row['action_time'] == p['root']['clock']]
            root_visit = {key: visits[0][key] for key in ('arrival_time','service_ready_time',
                'action_time','command_hold_seconds','departure_time','boarded_ids','alighted_ids')} if len(visits) == 1 else None
            progress('resident_reproduction_check' if branch == 'low' else 'full_day_accounting_complete')
            reproduction = compare_nominal(parent_out, out, old) if branch == 'low' else None
            checks = dict(root_issue_exact=root_issue_count == 1,
                exact_horizon=stream.now == p['stop_clock'], natural_terminal=stream.done,
                final_snapshot=stream.final_snapshot is not None and stream.final_snapshot.time_seconds == 24000.,
                observer=observed['all_pass'], population=observed['whole_network']['departed'] == p['native_passengers'],
                actual_root_command=command is not None and command['applied_at'] == p['root']['clock']
                    and command['requested_additional_hold_seconds'] == p['actions'][branch]['hold_seconds']
                    and command['accepted'] is True,
                nominal_suffix_exact=branch != 'low' or low_check['unchanged_branch_exact'],
                nominal_full_logs_exact=branch != 'low' or reproduction['all_pass'])
            result.update(status='complete' if all(checks.values()) else 'integrity_failed', branch=branch,
                integrity=dict(all_pass=all(checks.values()), checks=checks), accounting=observed,
                nominal_suffix=low_check, nominal_reproduction=reproduction,
                root_action=p['actions'][branch], root_command=command, root_visit=root_visit,
                target_suffix_events=target_events)
        except PrefixComplete as complete:
            native_finished_wall = time.time()
            children = {key: json.loads((parent_out/'branches'/key/'summary.json').read_text())
                        for key in ('low', 'high')}
            checks = dict(live_fork=complete.receipt['all_pass'],
                children=all(row['integrity']['all_pass'] for row in children.values()),
                paused_parent=stream.now == p['root']['clock'])
            result.update(status='complete' if all(checks.values()) else 'integrity_failed',
                integrity=dict(all_pass=all(checks.values()), checks=checks), branches=children,
                fork=complete.receipt, prefix_check=prefix_check)
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import consumed
        seconds, lower = consumed(env, native_started)
        own = None if seconds is None else seconds-(p['root']['clock'] if branch else 0.)
        known = own if own is not None else 0.
        children = result.get('branches', {})
        unknown = own is None
        if branch is None and (parent_out/'branches').exists():
            # Preserve completed sibling work if the native fork fails.
            if not children:
                children = {path.parent.name: json.loads(path.read_text())
                            for path in (parent_out/'branches').glob('*/summary.json')}
            child_seconds = [row['budget']['own_native_seconds'] for row in children.values()]
            known += sum(value for value in child_seconds if value is not None)
            unknown |= len(children) != 2 or any(value is None for value in child_seconds)
        result['budget'] = dict(own_native_seconds=own,
            unique_native_seconds=None if unknown else known, known_native_seconds=known,
            native_end_clock=seconds, lower_bound=unknown,
            learner_created=False, optimizer_updates=0, feedback_admitted=0, checkpoint_loaded=False,
            world_refits=0, terminal_transitions_generated=counts['released_terminal_transitions']
                + (sum(row.terminal for row in stream._queue) if stream is not None else 0),
            full_day_completed=branch is not None and stream is not None and stream.done and seconds == 24000.)
        if branch is None:
            result['budget']['completed_suffix_days'] = sum(
                row['budget']['full_day_completed'] for row in children.values())
            result['budget']['known_child_terminal_transitions'] = sum(
                row['budget']['terminal_transitions_generated'] for row in children.values())
        result['execution_wall'] = dict(native_started_wall=native_started_wall,
            native_finished_wall=native_finished_wall, elapsed_seconds=time.monotonic()-started)
        result['counts'] = dict(counts)
        write(out/'summary.json', result)
        if hook is not None:
            hook.close()
        if observer is not None:
            observer.close()
        if reference is not None:
            reference.close()
        progress(result['status'])
        if env is not None:
            env.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    try:
        result = run(parser.parse_args())
        code = 0 if result['status'] == 'complete' else 1
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    if os.getpid() != ROOT_PID:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
