"""Replay frozen proposals through original F, inspect one native state, stop."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


class ObservedTarget(Exception):
    pass


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def run(args):
    p = json.loads((HERE/'protocol.json').read_text())
    case = next(c for c in p['cases'] if c['scenario'] == args.scenario)
    root, out = args.thesis_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = stream = hook = reference = None
    native_started = False
    started_wall = finished_wall = None
    started = last_progress = time.monotonic()
    discarded = discarded_physical = applied_calls = 0
    result = dict(status='failed', scenario=args.scenario, target=case,
        integrity=dict(all_pass=False), observation=None)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None and native_started else 0,
            total=case['stop_before_apply_clock'], unit='sim_seconds',
            elapsed_seconds=time.monotonic()-started, scenario=args.scenario,
            applied_actions=applied_calls)), flush=True)

    try:
        import numpy as np
        from replay import ReplayReference
        from observe import capture
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        metadata = json.loads((root/p['metadata']).read_text())
        contract = TransitMDPConfigV7(**metadata['contract'])
        stations = np.asarray(metadata['station_max_by_line'], np.float32)
        old = root/p['reference_root']/args.scenario
        reference = ReplayReference([old/name for name in case['reference_action_parts']],
            [old/name for name in case['reference_tick_parts']],
            case['sequence'], case['stop_before_apply_clock'])

        def policy(state):
            return np.asarray(reference.next_action(state), np.float32)

        def issue(event):
            checked = reference.check_event(event, asdict(stream._snapshot()))
            if checked['target_reached']:
                if event['trip_id'] != case['trip_id'] or event['station_id'] != case['station_id']:
                    raise RuntimeError('The frozen target identity did not match')
                before_calls = applied_calls
                observed = capture(env._bridge, event,
                    diagnostic_person_id=case['diagnostic_person_id'])
                write(out/'capture.json', observed['full'])
                result['observation'] = observed['summary']
                result['target_ledger'] = asdict(stream._snapshot())
                result['target_reference_ledger'] = checked['reference_ledger']
                result['target_apply_calls_before'] = before_calls
                result['target_apply_calls_after'] = applied_calls
                result['target_command_applied'] = applied_calls != before_calls
                raise ObservedTarget()

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'], root/p['edge_xml'],
            episode_seconds=p['execution']['horizon_seconds'], seed=p['execution']['external_seed'])
        env._ensure_bridge()
        original_apply = env._bridge.apply_action

        def apply(event, action):
            nonlocal applied_calls
            applied_calls += 1
            if event.bus_id == case['trip_id'] and event.stop_id == case['station_id']:
                # A target call would mean the observer failed to stop at issue.
                if float(env._bridge.current_time) == case['stop_before_apply_clock']:
                    raise RuntimeError('Target command must remain unapplied')
            return original_apply(event, action)

        env._bridge.apply_action = apply
        stream = EventStream(env, stations, contract, domain='target', delay_seconds=30,
            execute_transform=True, on_issue=issue)
        native_started, started_wall = True, time.time()
        stream.reset()
        from bus_h2o.sumo_env import rl_bridge
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
        version = rl_bridge.traci.getVersion()
        if '1.25.0' not in str(version[1]):
            raise RuntimeError('Expected frozen SUMO 1.25.0 runtime')
        hook = install_scenario(env, args.scenario, **p['scenario_settings'])
        result['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__, sumo=version,
            h2o_root=os.environ['THESIS_H2O_ROOT'], sumo_root=os.environ['THESIS_SUMO_ROOT'])
        progress('native_recorded_actions')
        try:
            while not stream.done:
                discarded += len(stream.step(policy))
                discarded_physical += len(stream.release_intervals())
                if stream.now > case['stop_before_apply_clock']:
                    raise RuntimeError('Passed the declared target without its issue')
                if time.monotonic()-last_progress >= 30:
                    progress('native_recorded_actions')
                    last_progress = time.monotonic()
            raise RuntimeError('Episode ended before the frozen target')
        except ObservedTarget:
            finished_wall = time.time()
            replay_check = reference.report()
            observed = result['observation']
            checks = dict(replay_exact=replay_check['all_pass'],
                observation_unchanged=observed['all_pass'],
                at_declared_clock=stream.now == case['stop_before_apply_clock'],
                before_target_apply=result['target_command_applied'] is False,
                not_terminal=stream.done is False,
                scenario_applied=(hook.applied_at == 9000. if args.scenario == 'local_speed_drop'
                                  else hook.applied_at is None))
            result.update(status='complete' if all(checks.values()) else 'integrity_failed',
                integrity=dict(all_pass=all(checks.values()), checks=checks), replay=replay_check,
                scenario_metadata=hook.metadata())
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        if reference is not None:
            result['replay'] = reference.report()
        raise
    finally:
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import consumed
        seconds, lower_bound = consumed(env, native_started)
        result['budget'] = dict(native_seconds=seconds,
            native_seconds_is_lower_bound=lower_bound and result['status'] != 'complete',
            learner_created=False, optimizer_updates=0, feedback_admitted=0, world_refits=0,
            checkpoint_loaded=False, source_collected=0, terminal_labels_created=0,
            released_transitions_discarded=discarded, physical_intervals_discarded=discarded_physical,
            completed_transitions=stream.completed_count if stream is not None else 0,
            applied_commands=applied_calls, full_day_completed=False)
        result['execution_wall'] = dict(native_started_wall=started_wall,
            native_finished_wall=finished_wall, elapsed_seconds=time.monotonic()-started)
        write(out/'summary.json', result)
        progress(result['status'])
        if reference is not None:
            reference.close()
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    value = run(parser.parse_args())
    raise SystemExit(0 if value['status'] == 'complete' else 1)
