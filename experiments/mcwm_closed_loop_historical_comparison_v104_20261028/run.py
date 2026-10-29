"""One frozen historical MC-WM actor on one declared v75 environment day."""
from __future__ import annotations

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


def periods(metrics, final, departed, boundaries=None):
    """Use exact cumulative ledger endpoints; never a nearest/interpolated bin."""
    points = {float(row['time_seconds']): row for row in metrics}
    points[float(final['time_seconds'])] = final
    result = {}
    boundaries = boundaries or dict(full_day=(0., 24000.), early=(9000., 12000.),
                                    post_budget=(18300., 24000.))
    for name, (start, end) in boundaries.items():
        low = ({'accrued_wait_seconds': 0., 'accrued_in_vehicle_seconds': 0.}
               if name == 'full_day' else points.get(start))
        high = points.get(end)
        complete = low is not None and high is not None and departed > 0
        wait = high['accrued_wait_seconds']-low['accrued_wait_seconds'] if complete else None
        ride = high['accrued_in_vehicle_seconds']-low['accrued_in_vehicle_seconds'] if complete else None
        cost = 2*wait+ride if complete else None
        result[name] = dict(status='complete' if complete else 'missing', start=start, end=end,
            wait=wait, ride=ride, cost=cost, departed=departed,
            cost_per_passenger=cost/departed if complete else None)
    return result


def artifact_checks(extra, seed, contract):
    return dict(training_seed=extra.get('training_seed') == seed,
        original_contract=extra.get('contract_config') == contract,
        finalized_historical_actor=extra.get('arm') == 'fixed0250'
            and extra.get('scenario') == 'formal_calibrated_three_demand'
            and extra.get('updates') == 70000 and extra.get('smoke') is False)


def scenario_checks(metadata, name, p, lanes):
    shock = name == 'local_speed_drop'
    return dict(scenario_contract=metadata['name'] == name
        and metadata['scheduled_shock_at_seconds'] == p['shock_at_seconds']
        and metadata['speed_factor'] == p['speed_factor'],
        native_shock_clock=metadata['actual_shock_at_seconds'] == (p['shock_at_seconds'] if shock else None),
        affected_lanes=[row['lane_id'] for row in metadata['lane_caps']] == (list(lanes) if shock else []))


def consumed(env, native_started):
    if not native_started:
        return 0., False
    try:
        return float(env._bridge.passenger_time_ledger.snapshot().time_seconds), True
    except Exception:
        return None, True


def preflight(args):
    """Load all five saved actors and forward once; never construct an environment."""
    p = json.loads(args.protocol.read_text())
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='failed', native_seconds=0, optimizer_updates=0, actors=[])
    try:
        sys.path.insert(0, os.environ['THESIS_H2O_ROOT'])
        import numpy as np
        import torch
        from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        torch.set_num_threads(1)
        metadata = json.loads((ROOT/p['metadata_path']).read_text())
        contract = asdict(TransitMDPConfigV7(**metadata['contract']))
        stations = np.asarray(metadata['station_max_by_line'], np.float32)
        if stations.shape != (12,) or not np.isfinite(stations).all():
            raise ValueError('Historical station normalization is not the twelve-line v7 schema')
        state = torch.zeros((1, 229)); state[0, 5] = 1.
        for seed in p['historical_training_seeds']:
            path = ROOT/p['actor_path_template'].format(seed=seed)
            actor, extra = load_actor_checkpoint_v7(path, 'cpu', require_policy_artifact=True)
            with torch.inference_mode(): action = actor(state, deterministic=True)[0]
            checks = artifact_checks(extra, seed, contract)
            checks['finite_v7_forward'] = bool(action.shape == (1, 1)
                and torch.isfinite(action).all() and (action.abs() <= 1).all())
            result['actors'].append(dict(actor_seed=seed, actor_path=str(path), checks=checks,
                all_pass=all(checks.values()), action=action.tolist()))
            if not all(checks.values()): raise ValueError('Historical actor preflight failed: '+str(checks))
        result.update(status='complete', all_pass=True, station_max_by_line=stations.tolist(), contract=contract)
    except Exception as error:
        result.update(all_pass=False, error=dict(type=type(error).__name__, message=str(error)))
    write(out/'preflight.json', result)
    return result


def run(args):
    p = json.loads(args.protocol.read_text())
    if (args.training_seed not in p['historical_training_seeds']
            or args.external_seed not in p['external_seeds'] or args.scenario not in p['scenarios']):
        raise ValueError('Case is outside the frozen historical-actor/environment panel')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = hook = stream = None
    native_started = False
    native_started_wall = native_finished_wall = None
    started = time.monotonic()
    identity = dict(actor_seed=args.training_seed, external_seed=args.external_seed, scenario=args.scenario)
    budget = dict(native_seconds=None, native_seconds_is_lower_bound=False,
        optimizer_updates=0, target_training_labels=0, world_updates=0, trust_updates=0)
    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None and native_started else 0., total=24000,
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started, **identity)), flush=True)
    try:
        sys.path.insert(0, os.environ['THESIS_H2O_ROOT'])
        import numpy as np
        import torch
        from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env, TransitMDPConfigV7
        from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, LANE_IDS
        from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
        torch.set_num_threads(1)
        metadata_path = ROOT/p['metadata_path']
        metadata = json.loads(metadata_path.read_text())
        contract = TransitMDPConfigV7(**metadata['contract'])
        stations = np.asarray(metadata['station_max_by_line'], np.float32)
        actor_path = ROOT/p['actor_path_template'].format(seed=args.training_seed)
        with torch.random.fork_rng(devices=[]):
            actor, extra = load_actor_checkpoint_v7(actor_path, 'cpu', require_policy_artifact=True)
        binding = artifact_checks(extra, args.training_seed, asdict(contract))
        if not all(binding.values()):
            raise ValueError('Historical actor binding failed: '+str(binding))
        actor.eval()
        for parameter in actor.parameters(): parameter.requires_grad_(False)
        initial = {key:value.detach().clone() for key,value in actor.state_dict().items()}
        transform = ExecutionAction(contract)
        audit = dict(issues=0, raw_execution_exact=True, canonical_observation_exact=True,
            actor_real_clock=True, canonical_different_events=0, max_canonical_hold_gap_seconds=0.,
            canonical_difference_threshold_seconds=1e-4)
        @torch.inference_mode()
        def policy(state):
            return actor(torch.as_tensor(state, dtype=torch.float32)[None], deterministic=True)[0][0].numpy()
        def issue(event):
            raw = completion_safety_action(event['raw_s'], event['proposed_a'], 'quarter-load-sixty', contract)
            canonical = transform.numpy(event['s'], event['proposed_a'])
            gap = abs(float(canonical[0])-float(event['a'][0]))*.5*contract.max_hold_seconds
            audit['issues'] += 1
            audit['raw_execution_exact'] &= bool(np.array_equal(raw, event['a']))
            audit['canonical_observation_exact'] &= bool(np.array_equal(event['s'],
                raw_state_to_canonical_v7(event['raw_s'], stations, contract)))
            clock = float(env._current_events[0].sim_time)
            audit['actor_real_clock'] &= bool(event['s'][0] == np.float32(np.float32(clock)/24000))
            audit['canonical_different_events'] += gap > audit['canonical_difference_threshold_seconds']
            audit['max_canonical_hold_gap_seconds'] = max(audit['max_canonical_hold_gap_seconds'], gap)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'], seed=args.external_seed)
        stream = EventStream(env, stations, contract, domain='target', execute_transform=True, on_issue=issue)
        native_started = True
        native_started_wall = time.time()
        stream.reset()
        # reset configures the backend and adds the resident online_control package
        # root before rl_bridge imports its sim_obj passenger/line/bus modules.
        from bus_h2o.sumo_env import rl_bridge
        version = rl_bridge.traci.getVersion()
        if '1.25.0' not in str(version[1]):
            raise RuntimeError('Native runtime differs from frozen SUMO 1.25.0: '+str(version))
        hook = install_scenario(env, args.scenario, shock_at=p['shock_at_seconds'],
            factor=p['speed_factor'], metric_every=p['metric_interval_seconds'])
        progress('native_day')
        last_progress = time.monotonic()
        while not stream.done:
            stream.step(policy)
            stream.release_intervals()
            if time.monotonic()-last_progress >= 45:
                progress('native_day'); last_progress = time.monotonic()
        native_finished_wall = time.time()
        hook.capture_final()
        final = asdict(stream._snapshot())
        passengers = summarize_passengers_v5(env._bridge, stream._snapshot())
        departed = passengers['departed_passengers']
        costs = periods(hook.metrics, final, departed, p['periods'])
        scene = hook.metadata()
        checks = dict(**binding, **scenario_checks(scene, args.scenario, p, LANE_IDS),
            full_day=stream.done and final['time_seconds'] == 24000,
            sumo_version='1.25.0' in str(version[1]),
            departed_cohort=departed == p['full_day_departed_passengers'],
            ledger_balanced=passengers['active_state_count_balance_error'] == 0,
            exact_cost_windows=all(row['status'] == 'complete' for row in costs.values()),
            actor_frozen=all(torch.equal(value, initial[key]) for key,value in actor.state_dict().items()),
            raw_execution_exact=audit['raw_execution_exact'],
            canonical_observation_exact=audit['canonical_observation_exact'],
            actor_real_clock=audit['actor_real_clock'], all_issues_audited=audit['issues'] == stream.issued_count)
        budget.update(native_seconds=float(final['time_seconds']))
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed', **identity,
            integrity=dict(all_pass=all(checks.values()), checks=checks,
                failed_checks=[key for key,value in checks.items() if not value]),
            periods=costs, passenger_denominator=departed, passengers=passengers, ledger=final,
            budget=budget, action_audit=audit, scenario_metadata=scene,
            source=dict(actor_path=str(actor_path), metadata_path=str(metadata_path), actor_metadata=extra),
            runtime=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                sumo=version, h2o_root=os.environ['THESIS_H2O_ROOT'],
                sumo_root=os.environ['THESIS_SUMO_ROOT']),
            execution_wall=dict(native_started_wall=native_started_wall, native_finished_wall=native_finished_wall),
            wall_seconds=time.monotonic()-started)
        write(out/'summary.json', summary)
        progress(summary['status'])
        return summary
    except Exception as error:
        seconds, lower_bound = consumed(env, native_started)
        budget.update(native_seconds=seconds, native_seconds_is_lower_bound=lower_bound)
        write(out/'summary.json', dict(status='failed', **identity, budget=budget, periods={},
            integrity=dict(all_pass=False, checks={}, failed_checks=['runtime_error']),
            error=dict(type=type(error).__name__, message=str(error)),
            execution_wall=dict(native_started_wall=native_started_wall, native_finished_wall=native_finished_wall),
            wall_seconds=time.monotonic()-started))
        raise
    finally:
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--external-seed', type=int)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    result = preflight(args) if args.preflight_only else run(args)
    if not (result['all_pass'] if args.preflight_only else result['integrity']['all_pass']):
        raise SystemExit(1)
