"""Full-horizon SUMO evaluation with the same direct action map as training."""

import json
import sys
import time

import h5py
import numpy as np
import torch

from station_common import (P, EXPERIMENTS, policy_folder, HERE, action_dim, observation,
                    neutral_action, physical_action, state_headway_reward, data_folder, write)
from dataset import create_sumo
from networks import Actor


def evaluate_policy(args):
    from collect_policy.collect_transit_v7 import raw_event_observation, event_sort_key, is_terminal_event
    learned = args.arm in P["arms"]
    out = (policy_folder(args.mode, args.objective, args.scenario, args.arm, args.seed, args.smoke)
           if learned else HERE / ("rule_smoke" if args.smoke else "rules") / args.arm)
    path = out / "evaluation" / f"case{args.case:02d}.json"
    if path.exists():
        return
    with h5py.File(data_folder(args.mode, args.smoke) / "dataset.h5") as data:
        stations = np.array(json.loads(data.attrs["stations"]), np.float32)
    actor = None
    if learned:
        payload = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
        assert payload["mode"] == args.mode and payload["objective"] == args.objective
        actor = Actor((payload["hidden_dim"],)*2, dim=action_dim(args.mode))
        actor.load_state_dict(payload["actor"])
        actor.eval()
    sys.path.insert(0, str(EXPERIMENTS / "daganzo_baseline_v1"))
    from rules import hold_seconds
    block = P["test_blocks"][args.case % 5]
    scale = P["evaluation_demand_scales"][args.case // 5]
    horizon = 1800 if args.smoke else P["horizon_seconds"]
    env, helper = create_sumo(out, scale, block["demand_seed"], block["sumo_seed"], horizon)
    last, records = {}, []
    started = time.monotonic()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            commands = {}
            for event in sorted(env._current_events, key=event_sort_key):
                if is_terminal_event(env, event):
                    commands[event.bus_id] = [0., 1.]
                    continue
                raw = raw_event_observation(env, event.bus_id)
                key = (event.line_id, event.bus_id)
                state = observation(raw, last.get(key, neutral_action(args.mode)), stations)
                if learned:
                    with torch.inference_mode():
                        action = actor(torch.from_numpy(state[None]), deterministic=True)[0].numpy()[0]
                else:
                    hold = 0. if args.arm == "zero" else hold_seconds("daganzo_duet", raw[3], raw[4], raw[5], raw[6], raw[11], raw[12])
                    action = neutral_action(args.mode)
                    action[0] = hold/30.-1
                last[key] = action.copy()
                command = physical_action(action)
                commands[event.bus_id] = command
                deviations = [abs(raw[j]-raw[j+2]) for j, flag in ((3, raw[11]), (4, raw[12])) if flag > .5]
                records.append((float(state_headway_reward(state[None])[0]),
                    float(np.mean(deviations)) if deviations else None,
                    abs(float(raw[3]-raw[4])) if raw[11] > .5 and raw[12] > .5 else None,
                    command[0], command[1]))
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        snapshot = env._bridge.passenger_time_ledger.snapshot()
        passengers = helper.summarize_passengers_v5(env._bridge, snapshot)
        if snapshot.time_seconds != horizon or passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("incomplete SUMO horizon or inconsistent passenger ledger")
        cost = 2*snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds
        result = dict(mode=args.mode, objective=args.objective, arm=args.arm, training_seed=args.seed if learned else None,
            source_scene=args.scenario if learned else None, case=args.case, demand_scale=scale,
            split="test", block=block, horizon_seconds=horizon, decision_events=len(records),
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            generalized_passenger_seconds=cost, wait_passenger_seconds=snapshot.accrued_wait_seconds,
            in_vehicle_passenger_seconds=snapshot.accrued_in_vehicle_seconds, passengers=passengers,
            headway_ridge_mean=float(np.mean([r[0] for r in records])),
            headway_plan_mae_seconds=float(np.mean([r[1] for r in records if r[1] is not None])),
            headway_imbalance_seconds=float(np.mean([r[2] for r in records if r[2] is not None])),
            mean_hold_seconds=float(np.mean([r[3] for r in records])),
            mean_speed_ratio=float(np.mean([r[4] for r in records])),
            wall_seconds=time.monotonic()-started, execution="direct matched control, no holding shrink or rule floors")
        write(path, result)
        print(result, flush=True)
    finally:
        env.close()
