"""Frozen native observation to16200 seconds; no feedback admission or training."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import control as source
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state
import numpy as np


STOP_SECONDS = 16200.
WINDOW = (12000., STOP_SECONDS)
SEEDS = (6208, 6207)


class DiagnosticLimitReached(Exception):
    """The completed real tick reached the separately budgeted diagnostic limit."""


class NativeTimeLimit:
    def __init__(self, bridge, stop=STOP_SECONDS):
        self.bridge, self.stop = bridge, float(stop)
        self.original = bridge._advance_one_step
        self.reached = False
        bridge._advance_one_step = self.advance

    def advance(self):
        self.original()
        now = float(self.bridge.passenger_time_ledger.snapshot().time_seconds)
        if now > self.stop:
            raise RuntimeError('Native observation exceeded its fixed simulation budget')
        if now == self.stop:
            self.reached = True
            raise DiagnosticLimitReached()

    def close(self):
        self.bridge._advance_one_step = self.original


def native_event(env, issue):
    matches = [event for event in env._current_events
        if (str(event.bus_id), str(event.line_id), str(event.stop_id)) ==
           (str(issue['trip_id']), str(issue['line_id']), str(issue['station_id']))]
    if len(matches) != 1:
        raise RuntimeError('Issue does not uniquely identify its current native event')
    return matches[0]


def random_snapshot(learner):
    return dict(global_state=_rng(), learner=deepcopy(learner.rng.bit_generator.state),
        outcome=deepcopy(learner.replay_buffer.outcome_rng.bit_generator.state))


def write_small(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
    if len(encoded.encode()) > 48000:
        raise ValueError('Observation summary exceeds48KB')
    path.write_text(encoded)


def run(args):
    if args.seed not in SEEDS:
        raise ValueError('The predeclared observation seeds are6208 and6207')
    from experiments.mcwm_closed_loop_native_context_v79_20261027.observer import NativeContextObserver
    p, started = source.protocol(), time.monotonic()
    if p['horizon_seconds'] != 24000:
        raise ValueError('The original actor/F clock must retain its24000-second horizon')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    cell = dict(seed=args.seed, scenario='stable', external_seed=args.seed+p['target_seed_offset'])
    env = stream = observer = limit = learner = replay = None
    native_started_wall = native_finished_wall = None
    discarded = dict(chronological=0, physical=0)
    control = dict(issues=0, actor_real_clock=True, max_actor_clock_error_seconds=0.,
        shared_execution_rule_exact=True, max_execution_gap=0.)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream is not None else 0., total=STOP_SECONDS,
            unit='sim_seconds', elapsed_seconds=time.monotonic()-started,
            observed_issues=control['issues'], training_updates=0, feedback_slots=0)), flush=True)

    try:
        progress('load_frozen_initializer')
        learner, replay, contract, stations, initializer = source.load_learner(args.seed, arm='no_fresh')
        frozen = deepcopy(_state(learner))
        source_config = deepcopy(learner.config)
        outcome_rng = deepcopy(replay.outcome_rng.bit_generator.state)
        module_objects = {key: getattr(learner, key) for key in
            ('actor', 'critic', 'target_critic', 'value', 'action_transform',
             'actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer')}
        last_proposal = last_executed = None

        def policy(state):
            nonlocal last_proposal, last_executed
            event_clock = float(env._current_events[0].sim_time)
            expected = np.float32(np.float32(event_clock)/24000.)
            control['actor_real_clock'] &= bool(state[0] == expected)
            control['max_actor_clock_error_seconds'] = max(control['max_actor_clock_error_seconds'],
                abs(float(state[0])-float(expected))*24000.)
            last_proposal = learner.act(state[None], deterministic=True)[0]
            last_executed = learner.action_transform.numpy(state, last_proposal)
            return last_executed

        def on_issue(issue):
            issue['proposed_a'] = last_proposal.copy()
            gap = float(np.max(np.abs(np.asarray(issue['a'])-last_executed)))
            control['shared_execution_rule_exact'] &= gap == 0.
            control['max_execution_gap'] = max(control['max_execution_gap'], gap)
            control['issues'] += 1
            observer.capture(issue, native_event(env, issue))

        env = source.create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=cell['external_seed'])
        stream = source.EventStream(env, stations, contract, domain='target',
            delay_seconds=p['feedback_delay_seconds'], execute_transform=False, on_issue=on_issue)
        native_started_wall = time.time()
        stream.reset()
        limit = NativeTimeLimit(env._bridge)
        observer = NativeContextObserver(env._bridge, out/'context_trace.jsonl', window=WINDOW,
            state_probe=lambda: random_snapshot(learner), state_equal=_equal)
        progress('frozen_observation')
        last_progress = time.monotonic()
        try:
            while not stream.done:
                discarded['chronological'] += len(stream.step(policy))
                discarded['physical'] += len(stream.release_intervals())
                stream.physical_intervals.clear()
                if time.monotonic()-last_progress >= 45:
                    progress('frozen_observation')
                    last_progress = time.monotonic()
        except DiagnosticLimitReached:
            pass
        native_finished_wall = time.time()
        # Only already completed records may be discarded at this actual cutoff.
        discarded['chronological'] += len(stream.release())
        discarded['physical'] += len(stream.release_intervals())
        endpoint = env._bridge.passenger_time_ledger.snapshot()
        observation = observer.summary()
        optimizer_counts = learner.actor_schedule_audit()
        checks = dict(exact_native_limit=limit.reached and stream.now == STOP_SECONDS,
            original_horizon=contract.horizon_seconds == 24000,
            no_terminal_fabrication=not stream.done and stream.final_snapshot is None,
            frozen_initializer=learner.total_updates == 50000 and learner.online_critic_updates == 0,
            no_optimizer_updates=all(optimizer_counts[key+'_updates'] == 0 for key in ('critic','value','actor','alpha')),
            native_modules_and_Adam_unchanged=_equal(frozen, _state(learner)),
            learner_objects_retained=all(getattr(learner, key) is obj for key, obj in module_objects.items()),
            no_admitted_feedback=replay.target_received == len(replay.recent) == len(replay.fresh_outcomes) == 0,
            no_private_outcome_draws=_equal(outcome_rng, replay.outcome_rng.bit_generator.state),
            source_learning_config_unchanged=learner.config == source_config,
            actor_uses_real_clock=control['actor_real_clock'],
            one_original_execution_transform=control['shared_execution_rule_exact'],
            issue_count_exact=control['issues'] == stream.issued_count,
            observation_integrity=observation['all_pass'])
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), cell=cell,
            initializer=dict(path=str(initializer), updates=50000),
            original_learning_contract=dict(covariate_slots=285, fresh_complete_outcomes=285,
                updates_per_family=456, horizon_seconds=24000, F_unchanged=True,
                trust_unchanged=True, policy_sampler_unchanged=True, learning_budget_consumed=False),
            budget=dict(actual_simulated_seconds=stream.now, training_updates=0, feedback_slots=0,
                covariate_slots=0, complete_outcome_slots=0, initializer_updates=0, counterfactual_branches=0),
            endpoint={key: float(getattr(endpoint, key)) for key in ('time_seconds', 'waiting_passengers',
                'in_vehicle_passengers', 'accrued_wait_seconds', 'accrued_in_vehicle_seconds')},
            native_observation=observation, control=control, discarded_feedback=discarded,
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started_wall,
                native_finished_wall=native_finished_wall), wall_seconds=time.monotonic()-started,
            claim_scope='Frozen observation only; no response-learning or control-improvement evidence.')
        write_small(out/'summary.json', summary)
        progress(summary['status'])
        if not all(checks.values()):
            raise RuntimeError('Frozen native observation integrity failed')
        return summary
    except Exception as error:
        write_small(out/'failure.json', dict(status='failed', cell=cell, error=type(error).__name__,
            message=str(error), actual_simulated_seconds=stream.now if stream is not None else 0.,
            training_updates=0, feedback_slots=0, native_started_wall=native_started_wall,
            native_finished_wall=native_finished_wall, wall_seconds=time.monotonic()-started))
        raise
    finally:
        if observer is not None:
            observer.close()
        if limit is not None:
            limit.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
