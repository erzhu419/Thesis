"""Two authorized full-day logged-action replays with passive pre-action context."""
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


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def run(args):
    protocol = json.loads((HERE/'protocol.json').read_text())
    old_info = json.loads((HERE/'reference_summary.json').read_text())[args.scenario]
    root, out = args.thesis_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old = root/protocol['reference_root']/args.scenario/'no_fresh'
    env = stream = reference = observer = hook = contexts = None
    native_started = False
    native_started_wall = native_finished_wall = None
    started = last_progress = time.monotonic()
    context_count = missing_contexts = discarded = discarded_intervals = terminals = 0
    result = dict(status='failed', scenario=args.scenario, integrity=dict(all_pass=False),
                  performance_evidence=False)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if native_started and stream is not None else 0,
            total=24000, unit='sim_seconds', scenario=args.scenario,
            contexts=context_count, elapsed_seconds=time.monotonic()-started)), flush=True)

    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        from experiments.mcwm_logged_action_observation_v116_20261029.replay import ReplayReference, LEDGER_FIELDS
        from experiments.mcwm_behavior_replay_v113_20261029.observer import Observer
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        from capture import capture_context
        from validation import compare_full_day
        from samples import select_and_label
        metadata = json.loads((root/protocol['metadata']).read_text())
        contract = TransitMDPConfigV7(**metadata['contract'])
        stations = np.asarray(metadata['station_max_by_line'], np.float32)
        last = old_info['last_action']
        reference = ReplayReference([old/'actions.jsonl'], [old/'ticks.jsonl'],
                                    last['sequence'], last['started_at'])
        contexts = (out/'contexts.jsonl').open('x')

        def policy(state):
            nonlocal context_count, missing_contexts
            proposal = reference.next_action(state)
            context = capture_context(env._bridge, reference.current,
                                      source_group='v113/no_fresh/'+args.scenario)
            contexts.write(json.dumps(context, separators=(',', ':'), allow_nan=False)+'\n')
            context_count += 1
            missing_contexts += context['capture']['status'] != 'complete'
            return np.asarray(proposal, np.float32)

        def issue(event):
            reference.check_event(event, asdict(stream._snapshot()))
            observer.issue(event)

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'], root/protocol['edge_xml'],
            episode_seconds=24000, seed=protocol['external_seed'])
        env._ensure_bridge()
        observer = Observer(env._bridge, out)
        stream = EventStream(env, stations, contract, domain='target', delay_seconds=30,
                             execute_transform=True, on_issue=issue)
        native_started, native_started_wall = True, time.time()
        stream.reset()
        from bus_h2o.sumo_env import rl_bridge
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
        if '1.25.0' not in str(rl_bridge.traci.getVersion()[1]):
            raise RuntimeError('Frozen runtime requires SUMO1.25.0')
        hook = install_scenario(env, args.scenario, **protocol['scenario_settings'])
        progress('recorded_actions_and_service_context')
        while not stream.done:
            rows = stream.step(policy)
            discarded += len(rows)
            terminals += sum(bool(row.terminal) for row in rows)
            discarded_intervals += len(stream.release_intervals())
            if time.monotonic()-last_progress >= 30:
                contexts.flush()
                progress('recorded_actions_and_service_context')
                last_progress = time.monotonic()
        native_finished_wall = time.time()
        contexts.flush()
        expected = reference.ledger_at(24000)
        actual = asdict(stream.final_snapshot)
        final_ledger_exact = all(actual.get(key) == expected[key] for key in LEDGER_FIELDS)
        actions_eof = next(reference.actions, None) is None
        ticks_eof = next(reference.ticks, None) is None
        observed = observer.finish()
        for writer in observer.writers.values():
            writer.flush()
        progress('full_day_reproduction')
        reproduction = compare_full_day(out, old)
        checks = dict(recorded_issues=reference.report()['all_pass'], natural_end=stream.done,
            exact_horizon=stream.now == 24000 and actual['time_seconds'] == 24000,
            final_ledger_exact=final_ledger_exact, reference_actions_exhausted=actions_eof,
            reference_ticks_exhausted=ticks_eof, complete_native_logs=reproduction['all_pass'],
            every_issue_has_pre_action_context=context_count == old_info['observer']['issues'],
            scenario_applied=(hook.applied_at == 9000. if args.scenario == 'local_speed_drop'
                              else hook.applied_at is None))
        result.update(integrity=dict(all_pass=all(checks.values()), checks=checks),
            replay=reference.report(), reproduction=reproduction,
            observation=dict(contexts=context_count, incomplete_contexts=missing_contexts),
            final_ledger=actual, scenario_metadata=hook.metadata())
        if not all(checks.values()):
            result['status'] = 'integrity_failed'
        else:
            progress('mature_service_record_selection')
            selected = select_and_label(out, protocol['feedback_contract'],
                                        horizon=24000, window_seconds=60)
            result.update(status='complete' if selected['all_pass'] else 'service_data_hold',
                          service_records=selected, service_data_qualified=selected['all_pass'])
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        if reference is not None:
            result['replay'] = reference.report()
        raise
    finally:
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import consumed
        seconds, lower = consumed(env, native_started)
        queued_terminals = sum(bool(row.terminal) for row in getattr(stream, '_queue', []))
        result['budget'] = dict(native_seconds=seconds,
            native_seconds_is_lower_bound=lower and result['status'] != 'complete',
            learner_created=False, optimizer_updates=0, feedback_admitted=0,
            world_refits=0, checkpoint_loaded=False, source_collected=0,
            terminal_transitions_generated=terminals+queued_terminals,
            released_transitions_discarded=discarded,
            physical_intervals_discarded=discarded_intervals,
            context_records=context_count,
            full_day_completed=bool(stream is not None and stream.done and seconds == 24000))
        result['execution_wall'] = dict(native_started_wall=native_started_wall,
            native_finished_wall=native_finished_wall, elapsed_seconds=time.monotonic()-started)
        write(out/'summary.json', result)
        progress(result['status'])
        if contexts is not None: contexts.close()
        if reference is not None: reference.close()
        if hook is not None: hook.close()
        if observer is not None: observer.close()
        if env is not None: env.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    value = run(parser.parse_args())
    raise SystemExit(0 if value['status'] == 'complete' else 1)
