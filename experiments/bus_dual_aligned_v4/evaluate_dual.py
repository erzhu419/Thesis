"""Full-horizon paired SUMO endpoints; objectives do not change evaluation."""

import json
import os
import time

import h5py
import numpy as np
import torch

from aligned_common import (P,HERE,CONFIG,data_folder,policy_folder,pack_event_batch_v7,
                            physical_action,state,ridge_of_state,write)
from chronological import create_sumo
from dual_networks import Actor
from collect_policy.collect_transit_v7 import raw_event_observation,event_sort_key,is_terminal_event
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from rules import hold_seconds


def evaluate(args):
    learned = args.arm in P["arms"]
    out = (policy_folder(args.objective,args.scene,args.arm,args.seed,args.smoke) if learned else
           HERE / ("rule_smoke" if args.smoke else "rules") / args.arm)
    out = out / args.layer
    path = out / f"case{args.case:02d}.json"
    if path.exists():
        return
    with h5py.File(os.environ["THESIS_DATASET"]) as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]),np.float32)
    actor = None
    if learned:
        payload = torch.load(out.parent / "policy.pt",map_location="cpu",weights_only=False)
        actor = Actor((payload["hidden_dim"],)*2)
        actor.load_state_dict(payload["actor"])
        actor.eval()
    from station_common import P as OLD_P
    block = OLD_P["test_blocks"][args.case%5]
    scale = OLD_P["evaluation_demand_scales"][args.case//5]
    horizon = 1800 if args.smoke else CONFIG.horizon_seconds
    env,helper = create_sumo(out,scale,block["demand_seed"],block["sumo_seed"],horizon)
    records,started = [],time.monotonic()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            commands,controls = {},[]
            for event in sorted(env._current_events,key=event_sort_key):
                if is_terminal_event(env,event):
                    commands[event.bus_id] = [0.,1.]
                else:
                    controls.append((event,raw_event_observation(env,event.bus_id)))
            if controls:
                raw_batch = np.stack([r for _,r in controls])
                prior,speeds = np.full(len(controls),-1.,np.float32),np.zeros(16,np.float32)
                for index,(event,raw) in enumerate(controls):
                    global_raw = pack_event_batch_v7(raw_batch,index,prior,CONFIG)
                    s = state(global_raw,speeds,stations)
                    if learned:
                        with torch.inference_mode():
                            proposed = actor(torch.from_numpy(s[None]),deterministic=True)[0].numpy()[0]
                    else:
                        hold = 0. if args.arm=="zero" else hold_seconds("daganzo_duet",raw[3],raw[4],raw[5],raw[6],raw[11],raw[12])
                        proposed = np.asarray([hold/30.-1,0.],np.float32)
                    action = proposed.copy()
                    if args.layer=="guarded":
                        action[0] = completion_safety_action(global_raw,proposed[:1],"quarter-load-sixty",CONFIG)[0]
                    prior[index],speeds[index] = action
                    command = physical_action(action)
                    commands[event.bus_id] = command
                    deviations = [abs(raw[j]-raw[j+2]) for j,flag in ((3,raw[11]),(4,raw[12])) if flag>.5]
                    records.append((float(ridge_of_state(s[None])[0]),float(np.mean(deviations)) if deviations else None,
                        abs(float(raw[3]-raw[4])) if raw[11]>.5 and raw[12]>.5 else None,
                        *command,30*(float(proposed[0])+1),abs(float(action[0]-proposed[0]))>1e-5))
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        snapshot = env._bridge.passenger_time_ledger.snapshot()
        passengers = helper.summarize_passengers_v5(env._bridge,snapshot)
        if snapshot.time_seconds!=horizon or passengers["active_state_count_balance_error"]!=0:
            raise RuntimeError("Incomplete SUMO endpoint or inconsistent passenger ledger")
        cost = 2*snapshot.accrued_wait_seconds+snapshot.accrued_in_vehicle_seconds
        write(path,dict(objective=args.objective,arm=args.arm,training_seed=args.seed if learned else None,
            source_scene=args.scene if learned else None,case=args.case,block=block,demand_scale=scale,horizon_seconds=horizon,
            execution_layer=args.layer,state_contract="global event batch plus selected speeds",action_dim=2,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            wait_passenger_seconds=snapshot.accrued_wait_seconds,in_vehicle_passenger_seconds=snapshot.accrued_in_vehicle_seconds,
            passengers=passengers,decision_events=len(records),
            headway_ridge_mean=float(np.mean([r[0] for r in records])),
            headway_plan_mae_seconds=float(np.mean([r[1] for r in records if r[1] is not None])),
            headway_imbalance_seconds=float(np.mean([r[2] for r in records if r[2] is not None])),
            mean_hold_seconds=float(np.mean([r[3] for r in records])),mean_speed_ratio=float(np.mean([r[4] for r in records])),
            mean_proposed_hold_seconds=float(np.mean([r[5] for r in records])),guard_changed_decisions=sum(r[6] for r in records),
            wall_seconds=time.monotonic()-started,target_online_training_events=0))
    finally:
        env.close()
