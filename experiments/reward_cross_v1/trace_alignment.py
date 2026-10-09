"""Trace real bus executors and replay timing without changing training code."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from cross_common import HERE, EXPERIMENTS, write
from rewards import canonical_headway_reward, ridge_reward


HOLDS = (0., 7., 23., 60.)


def legacy_trace(original):
    root = EXPERIMENTS.parent.parent / "RE-SAC" if original else EXPERIMENTS / "legacy_bus/runtime/BAPR"
    sys.path.insert(0, str(root))
    from env.sim import env_bus
    from env.bus import Bus

    random.seed(64001)
    np.random.seed(64001)
    env = env_bus(str(root / "env"), **({} if original else {"fixed_mode": "normal"}))
    env.enable_plot = False
    counts, examples, chosen, emitted, latched, released = Counter(), [], {}, {}, {}, {}
    drive = Bus.drive

    def traced_drive(bus, current_time, action, *args, **kwargs):
        waiting = (bus.state.name == "WAITING_ACTION" if original else
                   bus.holding and bus.held and not bus.dwelling)
        dwelling = (bus.state.name == "DWELLING" if original else bus.dwelling)
        identity = (int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id))
        drive(bus, current_time, action, *args, **kwargs)
        if waiting:
            record = dict(time=float(current_time), hold=float(np.asarray(bus.dwelling_time).reshape(-1)[0])
                          if bus.dwelling_time is not None else 0., identity=list(identity))
            latched[identity] = record
            if identity in chosen:
                counts["controlled_latches"] += 1
                counts["latch_action_mismatches"] += abs(record["hold"]-chosen[identity]) > 1e-5
            else:
                counts["latches_without_emitted_decision"] += 1
                counts["positive_latches_without_decision"] += record["hold"] > 0
        if dwelling and not bus.in_station and identity in latched:
            elapsed = current_time-latched[identity]["time"]
            expected = max(1., np.ceil(latched[identity]["hold"]))
            counts["verified_releases"] += 1
            counts["release_duration_mismatches"] += abs(elapsed-expected) > 1e-5
            released[identity] = float(elapsed)
        if bus.obs:
            counts["emitted_events"] += 1
            emitted[tuple(bus.obs)] = dict(bus=int(bus.bus_id), trip=int(bus.trip_id),
                station=int(bus.last_station.station_id), time=float(current_time),
                reward=float(bus.reward), direction=int(bus.direction),
                forward_present=bool(bus.forward_bus), backward_present=bool(bus.backward_bus))

    Bus.drive = traced_drive
    started = time.monotonic()
    env.reset()
    state, rewards, done = env.initialize_state()
    actions = {key: None for key in range(env.max_agent_num)}

    def choose(key, observation):
        event = emitted[tuple(observation)]
        hold = HOLDS[(key*3+int(observation[1])+event["trip"]) % len(HOLDS)]
        chosen[(key, event["trip"], int(observation[1]))] = hold
        return np.array([hold], np.float32)

    # Same event consumption order as the original and frozen training loops.
    while not done and env.current_time < 60000:
        for key in state:
            states = state[key]
            if len(states) == 1 and actions[key] is None:
                actions[key] = choose(key, states[0])
            elif len(states) == 2:
                previous, following = emitted[tuple(states[0])], emitted[tuple(states[1])]
                counts["cross_trip_suppressed_rows"] += (previous["trip"] != following["trip"] and states[0][1] == states[1][1])
                if states[0][1] != states[1][1]:
                    counts["replay_rows"] += 1
                    identity = (key, previous["trip"], int(states[0][1]))
                    counts["wrong_vehicle_rows"] += previous["bus"] != following["bus"] or states[0][0] != key
                    counts["replay_action_mismatches"] += (identity not in latched or
                        abs(float(actions[key][0])-latched[identity]["hold"]) > 1e-5)
                    counts["reward_event_mismatches"] += abs(float(rewards[key])-following["reward"]) > 1e-5
                    analytic = ridge_reward(states[1][4], states[1][5],
                        forward_present=following["forward_present"], backward_present=following["backward_present"])
                    counts["reward_formula_mismatches"] += abs(float(rewards[key])-float(analytic)) > 1e-3
                    cross_trip = previous["trip"] != following["trip"]
                    counts["cross_trip_rows"] += cross_trip
                    counts["same_trip_nonadjacent_station_rows"] += (not cross_trip and abs(previous["station"]-following["station"]) != 1)
                    if len(examples) < 3 or (cross_trip and sum(x["cross_trip"] for x in examples) < 3):
                        examples.append(dict(state_event=previous, next_event=following,
                            replay_hold=float(actions[key][0]), executed=latched.get(identity),
                            release_elapsed=released.get(identity), reward=float(rewards[key]), cross_trip=cross_trip))
                state[key] = states[1:]
                actions[key] = choose(key, state[key][0])
        state, rewards, done = env.step(actions)
    return dict(runtime="original_RE-SAC" if original else "frozen_thesis_RE-SAC",
        seed=64001, policy="recognizable per-vehicle/trip/station 0,7,23,60s commands",
        completed=bool(done), simulation_seconds=float(env.current_time),
        wall_seconds=time.monotonic()-started, counts={k:int(v) for k,v in counts.items()}, examples=examples)


def setup_runtime(phase):
    runtime = json.loads((HERE / "node_runtime.json").read_text())[phase]
    os.environ.update(runtime["env"])
    sys.path.insert(0, runtime["env"]["THESIS_H2O_ROOT"])
    return runtime["env"]


def offline_alignment(path):
    import h5py
    counts = Counter()
    with h5py.File(path, "r") as data:
        for start in np.linspace(0, len(data["actions"])-513, 64, dtype=int):
            stop = start+512
            states, next_states = data["observations"][start:stop], data["next_observations"][start:stop]
            actions = data["actions"][start:stop].reshape(-1)
            index = np.rint(states[:, 2]*15).astype(int)
            ns_index = np.rint(next_states[:, 2]*15).astype(int)
            same_batch = (data["delta_t_seconds"][start:stop] == 0) & (data["terminals"][start:stop] < .5)
            next_slots = next_states[:, 5:].reshape(-1, 16, 14)
            counts["sampled_rows"] += len(states)
            counts["invalid_action_rows"] += np.count_nonzero(~np.isfinite(actions) | (abs(actions) > 1.00001))
            counts["same_batch_rows"] += same_batch.sum()
            counts["same_batch_index_mismatches"] += np.count_nonzero(same_batch & (ns_index != index+1))
            stored_prior = next_slots[np.arange(len(states)), index, -1]
            counts["same_batch_action_mismatches"] += np.count_nonzero(same_batch & (abs(actions-stored_prior) > 1e-5))
            raw_slots = data["raw_observations"][start:stop, 5:].reshape(-1, 16, 15)
            local = raw_slots[np.arange(len(states)), index]
            counts["event_line_metadata_mismatches"] += np.count_nonzero(local[:, 1] != data["event_line_index"][start:stop])
            counts["event_station_metadata_mismatches"] += np.count_nonzero(local[:, 2] != data["event_station_index"][start:stop])
    return {k:int(v) for k,v in counts.items()}


def source_trace():
    paths = setup_runtime("train")
    import h5py
    from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
    from sim_core.bus import Bus
    from bus_h2o.mdp_contract_v7 import (TransitMDPConfigV7, pack_event_batch_v7,
        raw_state_to_canonical_v7, event_batch_v7)
    from SimpleSAC.transit_v7.sampler import AllLinesTransitSamplerV7, _active_event_batch

    with h5py.File(paths["THESIS_DATASET"], "r") as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
        config = TransitMDPConfigV7(**json.loads(data.attrs["config"]))
    random.seed(64001)
    np.random.seed(64001)
    env = AllLinesSimEnv(paths["THESIS_SOURCE_ENV"], observation_schema="v7", observation_speed_seed=64001)
    counts, examples, expected, latch_times, policy_holds = Counter(), [], {}, {}, {}
    start_dwelling, process_dwelling = Bus._start_dwelling, Bus._process_dwelling

    def latch(bus, hold, speed=1.):
        key = (bus.line_id_str, bus.bus_id)
        start_dwelling(bus, hold, speed)
        if key in expected:
            counts["controlled_latches"] += 1
            counts["latch_action_mismatches"] += abs(float(bus.dwelling_time)-expected.pop(key)) > 1e-4
            latch_times[key] = (env.line_map[key[0]].current_time, float(bus.dwelling_time))

    def release(bus, current_time):
        process_dwelling(bus, current_time)
        key = (bus.line_id_str, bus.bus_id)
        if not bus.in_station and key in latch_times:
            when, hold = latch_times.pop(key)
            counts["verified_releases"] += 1
            counts["release_duration_mismatches"] += abs(current_time-when-max(1., np.ceil(hold))) > 1e-4

    Bus._start_dwelling, Bus._process_dwelling = latch, release
    step_to_event = env.step_to_event

    def step(actions, **kwargs):
        active = _active_event_batch(env.state)
        for key, observation in active:
            line_env = env.line_map[key[0]]
            bus = next(b for b in line_env.bus_all if b.bus_id == key[1])
            counts["observation_event_mismatches"] += (observation[1] != bus.last_station.station_id or
                int(bus.obs[1]) != bus.sumo_trip_index or int(observation[0]) != line_env.line_idx)
            if key in actions:
                expected[key] = float(actions[key][0])
                counts["issued_actions"] += 1
                counts["policy_to_executor_action_mismatches"] += abs(expected[key]-policy_holds[key]) > 1e-4
        result = step_to_event(actions, **kwargs)
        counts["actions_not_latched"] += len(expected)
        expected.clear()
        return result

    env.step_to_event = step
    class Replay:
        online_size = 0

        def append(self, state, action, reward, next_state, discount, **metadata):
            self.online_size += 1
            counts["replay_rows"] += 1
            previous = sampler.pending.raw_observation
            following = closing[0]
            p = event_batch_v7(previous)[int(previous[2])]
            n = event_batch_v7(following)[int(following[2])]
            counts["different_vehicle_next_rows"] += pending_key[0] != closing[1]
            counts["different_line_next_rows"] += pending_key[0][0] != closing[1][0]
            same_batch = metadata["delta_t_seconds"] == 0 and not metadata["terminal"]
            counts["same_batch_pre_execution_rows"] += same_batch
            if same_batch:
                reward_new = canonical_headway_reward(next_state)
                counts["same_batch_nonzero_ridge_rows"] += reward_new != 0
                slots = next_state[5:].reshape(16, 14)
                counts["same_batch_prior_action_mismatches"] += abs(slots[int(previous[2]), -1]-float(action[0])) > 1e-5
                if len(examples) < 3:
                    examples.append(dict(time=float(previous[0]), previous_vehicle=list(pending_key[0]),
                        next_vehicle=list(closing[1]), previous_station=float(p[1]), next_station=float(n[1]),
                        hold=float((action[0]+1)*30),
                        passenger_reward=float(reward), ridge_reward=float(reward_new),
                        physical_action_executed_before_row_closed=False))

    replay, closing, pending_key = Replay(), [None, None], [None]
    sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=stations, contract_config=config)
    append_pending = sampler._append_pending

    def append(raw_next, passenger_end, **kwargs):
        closing[0] = raw_next
        closing[1] = _active_event_batch(env.state)[int(raw_next[2])][0]
        return append_pending(raw_next, passenger_end, **kwargs)

    sampler._append_pending = append
    policy_calls = [0]

    def policy(states, deterministic=False):
        hold = HOLDS[policy_calls[0] % 4]
        policy_calls[0] += 1
        key = _active_event_batch(env.state)[int(round(float(states[0, 2])*15))][0]
        policy_holds[key] = hold
        pending_key[0] = key
        return np.array([[hold/30.-1.]], np.float32)

    started = time.monotonic()
    collected = sampler.collect(policy, 1500)
    Bus._start_dwelling, Bus._process_dwelling = start_dwelling, process_dwelling
    env.step_to_event = step_to_event

    # Freeze one real simultaneous batch, then change only one bus command.
    while len(_active_event_batch(env.state)) < 2:
        env.step_to_event({key: [0., 1.] for key, _ in _active_event_batch(env.state)})
    active = _active_event_batch(env.state)
    local = np.stack([obs for _, obs in active])
    zero = np.full(len(active), -1., np.float32)
    other = zero.copy()
    other[0] = 1.
    next_zero = raw_state_to_canonical_v7(pack_event_batch_v7(local, 1, zero, config), stations, config)
    next_hold = raw_state_to_canonical_v7(pack_event_batch_v7(local, 1, other, config), stations, config)
    counterfactual = dict(controlled_vehicle=list(active[0][0]), next_vehicle=list(active[1][0]),
        same_batch_ridge_0s=canonical_headway_reward(next_zero),
        same_batch_ridge_60s=canonical_headway_reward(next_hold),
        changed_state_columns=np.flatnonzero(next_zero != next_hold).tolist())
    chosen_key, initial_station = active[0][0], float(active[0][1][1])
    snapshot, rng, py_rng = deepcopy(env), np.random.get_state(), random.getstate()
    outcomes, first_states = [], []
    for hold in (0., 60.):
        branch = deepcopy(snapshot)
        np.random.set_state(rng)
        random.setstate(py_rng)
        branch.step_to_event({key: [hold if key == chosen_key else 0., 1.] for key, _ in active})
        first = _active_event_batch(branch.state)
        first_local = np.stack([obs for _, obs in first])
        first_state = raw_state_to_canonical_v7(pack_event_batch_v7(first_local, 0, None, config), stations, config)
        first_states.append(first_state)
        outcome = dict(hold=hold, next_global_vehicle=list(first[0][0]),
            next_global_ridge=canonical_headway_reward(first_state), next_global_time=float(first[0][1][2]))
        for _ in range(3000):
            batch = _active_event_batch(branch.state)
            own = [obs for key, obs in batch if key == chosen_key and obs[1] != initial_station]
            if own:
                packed = pack_event_batch_v7(np.stack(own), 0, None, config)
                outcome.update(same_bus_next_time=float(own[0][2]),
                    same_bus_next_station=float(own[0][1]), same_bus_next_forward=float(own[0][3]),
                    same_bus_next_ridge=canonical_headway_reward(raw_state_to_canonical_v7(packed, stations, config)))
                break
            if branch.done:
                break
            branch.step_to_event({key: [0., 1.] for key, _ in batch})
        outcomes.append(outcome)
    counterfactual["physical_outcomes"] = outcomes
    counterfactual["next_physical_global_changed_state_columns"] = np.flatnonzero(first_states[0] != first_states[1]).tolist()
    return dict(runtime="H2O_MC_source_v7", collected=collected, counts={k:int(v) for k,v in counts.items()},
        examples=examples, counterfactual=counterfactual,
        offline_alignment=offline_alignment(paths["THESIS_DATASET"]), wall_seconds=time.monotonic()-started)


def sumo_trace():
    paths = setup_runtime("evaluate")
    from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
    root = Path(paths["THESIS_SUMO_ROOT"])
    env = AllLinesSumoEnv(str(root / "online_control"),
        str(Path(paths["THESIS_H2O_ROOT"]) / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
        episode_seconds=2400, sumo_seed=64001, observation_schema="v5", sumo_backend="libsumo")
    counts, records, issued = Counter(), [], {}
    started = time.monotonic()
    try:
        env.reset()
        bridge = env._bridge
        traci = sys.modules[type(bridge).__module__].traci
        apply = bridge.apply_action

        def traced_apply(event, action):
            key = (event.bus_id, event.stop_id)
            apply(event, action)
            actual = float(traci.vehicle.getStopParameter(event.bus_id, 0, "duration"))
            counts["applied_actions"] += 1
            counts["vehicle_action_mismatches"] += abs(float(event.assigned_action)-issued[key]) > 1e-4
            counts["sumo_stop_parameter_mismatches"] += abs(actual-issued[key]) > 1e-4
            records.append(dict(bus=event.bus_id, stop=event.stop_id, hold=issued[key],
                service_completion=float(event.metadata["service_completion_time"]),
                application_time=float(bridge.current_time), sumo_stop_duration=actual))

        bridge.apply_action = traced_apply
        env.step_to_event({})
        while not env.done:
            actions = {}
            for event in env._current_events:
                hold = HOLDS[counts["issued_actions"] % 4]
                counts["issued_actions"] += 1
                issued[(event.bus_id, event.stop_id)] = hold
                actions[event.bus_id] = [hold, 1.]
            env.step_to_event(actions)
        departures = []
        for row in records:
            departure = bridge.bus_obj_dic[row["bus"]].depart_stop_time_d.get(row["stop"])
            if departure is not None:
                elapsed = float(departure)-row["service_completion"]
                counts["observed_departures"] += 1
                counts["departed_before_requested_hold"] += elapsed+1e-5 < row["hold"]
                departures.append(dict(row, departure=float(departure), elapsed_after_service=elapsed))
        return dict(runtime="SUMO_v7_executor", simulation_seconds=float(bridge.current_time),
            wall_seconds=time.monotonic()-started, counts={k:int(v) for k,v in counts.items()},
            examples=departures[:8], mean_release_delay_after_requested_hold=float(np.mean(
                [row["elapsed_after_service"]-row["hold"] for row in departures])))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("original", "frozen", "source", "sumo"))
    args = parser.parse_args()
    report = (source_trace() if args.mode == "source" else sumo_trace() if args.mode == "sumo"
              else legacy_trace(args.mode == "original"))
    write(HERE / f"alignment_{args.mode}.json", report)
    print(json.dumps(report, indent=2), flush=True)
