"""Same-policy execution attribution on the fixed SUMO test cases."""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

from repair_common import EXPERIMENTS, P, mc_folder, write

sys.path.insert(0, str(EXPERIMENTS / "bus_control_v2"))
from station_common import P as ORIGINAL, observation, neutral_action, physical_action, state_headway_reward, ROOT
from dataset import create_sumo
from networks import Actor
from collect_policy.collect_transit_v7 import raw_event_observation, event_sort_key, is_terminal_event
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7, pack_event_batch_v7, raw_state_to_canonical_v7
from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
sys.path.insert(0, str(EXPERIMENTS / "daganzo_baseline_v1"))
from rules import hold_seconds


def execute_hold(raw_batch, index, prior, proposed, layer, config):
    command = np.asarray(proposed, np.float32).copy()
    if layer == "guarded":
        raw = pack_event_batch_v7(raw_batch, index, prior, config)
        command[0] = completion_safety_action(raw, command[:1], "quarter-load-sixty", config)[0]
    prior[index] = command[0]
    return command


def evaluate(args):
    out = mc_folder(args)
    target = out / f"case{args.case:02d}.json"
    if target.exists():
        return
    with h5py.File(os.environ["THESIS_DATASET"]) as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
        config = TransitMDPConfigV7(**json.loads(data.attrs["config"]))
    actor = None
    if args.branch == "old":
        policy = EXPERIMENTS / "mcwm_seed_confirmation_v13/runs" / args.scene / f"{args.arm}_seed{args.seed}" / "policy.pt"
        actor, _ = load_actor_checkpoint_v7(policy, "cpu", require_policy_artifact=True)
    elif args.branch == "new":
        policy = EXPERIMENTS / "bus_control_v2/runs" / args.mode / args.objective / args.scene / f"{args.arm}_seed{args.seed}" / "policy.pt"
        payload = torch.load(policy, map_location="cpu", weights_only=False)
        actor = Actor((payload["hidden_dim"],)*2, dim=2 if args.mode == "dual" else 1)
        actor.load_state_dict(payload["actor"])
    if actor is not None:
        actor.eval()
    block = ORIGINAL["test_blocks"][args.case % 5]
    scale = ORIGINAL["evaluation_demand_scales"][args.case // 5]
    horizon = 1800 if args.smoke else ORIGINAL["horizon_seconds"]
    env, helper = create_sumo(out, scale, block["demand_seed"], block["sumo_seed"], horizon)
    last, records = {}, []
    started = time.monotonic()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            commands, controls = {}, []
            for event in sorted(env._current_events, key=event_sort_key):
                if is_terminal_event(env, event):
                    commands[event.bus_id] = [0., 1.]
                else:
                    controls.append((event, raw_event_observation(env, event.bus_id)))
            if controls:
                raw_batch = np.stack([raw for _, raw in controls])
                prior = np.full(len(controls), -1., np.float32)
                for index, (event, raw) in enumerate(controls):
                    key = (event.line_id, event.bus_id)
                    mode = args.mode if args.branch == "new" else "holding"
                    state = observation(raw, last.get(key, neutral_action(mode)), stations)
                    if args.branch == "old":
                        global_state = pack_event_batch_v7(raw_batch, index, prior, config)
                        canonical = raw_state_to_canonical_v7(global_state, stations, config)
                        with torch.inference_mode():
                            proposed = actor(torch.from_numpy(canonical[None]), deterministic=True)[0].numpy()[0]
                    elif args.branch == "new":
                        with torch.inference_mode():
                            proposed = actor(torch.from_numpy(state[None]), deterministic=True)[0].numpy()[0]
                    else:
                        hold = (hold_seconds("daganzo_duet", raw[3], raw[4], raw[5], raw[6], raw[11], raw[12])
                                if args.arm == "daganzo_duet" else 0.)
                        proposed = np.array([hold / 30 - 1], np.float32)
                    action = execute_hold(raw_batch, index, prior, proposed, args.layer, config)
                    last[key] = action.copy()
                    command = physical_action(action)
                    if args.branch == "rule" and args.arm.startswith("speed"):
                        command[1] = float(args.arm.removeprefix("speed"))
                    commands[event.bus_id] = command
                    plan = [abs(raw[j]-raw[j+2]) for j, flag in ((3, raw[11]), (4, raw[12])) if flag > .5]
                    records.append((float(state_headway_reward(state[None])[0]),
                        float(np.mean(plan)) if plan else None,
                        abs(float(raw[3]-raw[4])) if raw[11] > .5 and raw[12] > .5 else None,
                        command[0], command[1], 30 * (float(proposed[0]) + 1),
                        abs(float(action[0]-proposed[0])) > 1e-5))
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        snapshot = env._bridge.passenger_time_ledger.snapshot()
        passengers = helper.summarize_passengers_v5(env._bridge, snapshot)
        if snapshot.time_seconds != horizon or passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("Incomplete SUMO evaluation or passenger ledger")
        cost = 2 * snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds
        write(target, dict(branch=args.branch, mode=args.mode, objective=args.objective, arm=args.arm,
            training_seed=args.seed, source_scene=args.scene, execution_layer=args.layer, case=args.case,
            demand_scale=scale, block=block, horizon_seconds=horizon,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            wait_passenger_seconds=snapshot.accrued_wait_seconds,
            in_vehicle_passenger_seconds=snapshot.accrued_in_vehicle_seconds, passengers=passengers,
            headway_ridge_mean=float(np.mean([r[0] for r in records])),
            headway_plan_mae_seconds=float(np.mean([r[1] for r in records if r[1] is not None])),
            headway_imbalance_seconds=float(np.mean([r[2] for r in records if r[2] is not None])),
            mean_hold_seconds=float(np.mean([r[3] for r in records])),
            mean_speed_ratio=float(np.mean([r[4] for r in records])),
            mean_proposed_hold_seconds=float(np.mean([r[5] for r in records])),
            guard_changed_decisions=sum(r[6] for r in records), decision_events=len(records),
            wall_seconds=time.monotonic()-started,
            interpretation="Frozen-policy execution attribution, not matched-executor retraining"))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("old", "new", "rule"), required=True)
    parser.add_argument("--mode", choices=("holding", "dual"), default="holding")
    parser.add_argument("--objective", choices=("headway", "passenger"), default="passenger")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--scene", default="calibrated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", choices=("direct", "guarded"), required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    evaluate(parser.parse_args())
