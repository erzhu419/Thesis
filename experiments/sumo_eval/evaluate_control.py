"""Full-horizon SUMO comparison, retaining only small service summaries."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import h5py
import numpy as np
import torch

from data import HERE, load_protocol

ROOT = Path(load_protocol()["h2o_root"])
SUMO_ROOT = Path(load_protocol().get("sumo_root", ROOT.parent / "SUMO_ruiguang"))
sys.path.insert(0, str(ROOT))
from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
from bus_h2o.mdp_contract_v7 import (TransitMDPConfigV7, pack_event_batch_v7,
                                    raw_state_to_canonical_v7, policy_action_to_hold_seconds_v7)
from collect_policy.collect_transit_v7 import event_sort_key, is_terminal_event, raw_event_observation
from SimpleSAC.evaluate_transit_v7 import _event_record
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from SimpleSAC.transit_v3.evaluation import _headway_summary, summarize_service
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
from reproducibility.build_passenger_scenarios import (
    scaled_person_elements, _write_route, _write_passenger_objects, _write_config)


def build_scenario(out, scale, seed, jitter):
    directory = out / "scenarios" / f"demand_{scale:g}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    target_config = directory / "scenario.sumocfg"
    objects = directory / "passengers.add.xml"
    if not target_config.exists():
        sumo = SUMO_ROOT
        sources = (sumo / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
                   sumo / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml")
        destinations = (directory / "direct.rou.xml", directory / "transfer.rou.xml")
        all_records = []
        for source, destination in zip(sources, destinations):
            people, records = scaled_person_elements(source, scale, seed, depart_jitter_seconds=jitter)
            _write_route(source, destination, people)
            all_records.extend(records)
        _write_passenger_objects(sumo / "online_control/initialize_obj/save_obj_passenger.add.xml",
                                 objects, all_records)
        _write_config(sumo / "online_control/control_sim_traci_period.sumocfg", target_config,
                      sources, destinations)
    return str(target_config), str(objects)


def evaluate(job):
    method, scale, block, config_path, objects, out, p = job
    out = Path(out)
    output = out / "evaluation" / f"{method}_demand{scale:g}_seed{block['sumo_seed']}.json"
    if output.exists():
        return json.loads(output.read_text())
    torch.set_num_threads(1)
    with h5py.File(ROOT / load_protocol()["dataset"], "r") as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
        config = TransitMDPConfigV7(**json.loads(data.attrs["config"]))
    actor, _ = load_actor_checkpoint_v7(out / f"{method}.pt", "cpu", require_policy_artifact=True)
    env = AllLinesSumoEnv(str(SUMO_ROOT / "online_control"),
                          str(ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
                          episode_seconds=p["evaluation_horizon_seconds"], sumo_seed=block["sumo_seed"],
                          sumo_cfg=config_path, passenger_obj_xml=objects,
                          observation_schema="v5", sumo_backend="libsumo")
    started, records, time_series = time.monotonic(), [], []
    next_bin = 300
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            events = sorted(env._current_events, key=event_sort_key)
            actions, controls = {}, []
            for event in events:
                if is_terminal_event(env, event):
                    actions[event.bus_id] = [0., 1.]
                else:
                    controls.append((event, raw_event_observation(env, event.bus_id)))
            if controls:
                local_batch = np.stack([local for _, local in controls])
                prior = np.full(len(controls), -1., np.float32)
                for index, (event, local) in enumerate(controls):
                    raw = pack_event_batch_v7(local_batch, index, prior, config)
                    state = raw_state_to_canonical_v7(raw, stations, config)
                    with torch.inference_mode():
                        action, _ = actor(torch.from_numpy(state[None]), deterministic=True)
                    action = action.numpy()[0]
                    action = (action + 1) * p["deployment_hold_scale"] - 1
                    action = completion_safety_action(raw, action, "quarter-load-sixty", config)
                    hold = float(policy_action_to_hold_seconds_v7(action[0], config))
                    prior[index] = action[0]
                    actions[event.bus_id] = [hold, 1.]
                    records.append(_event_record(event, local, action, hold))
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
            if env._bridge.current_time >= next_bin:
                snapshot = env._bridge.passenger_time_ledger.snapshot()
                time_series.append({"time_seconds": snapshot.time_seconds,
                                    "waiting": snapshot.waiting_passengers,
                                    "onboard": snapshot.in_vehicle_passengers,
                                    "wait_passenger_seconds": snapshot.accrued_wait_seconds,
                                    "ride_passenger_seconds": snapshot.accrued_in_vehicle_seconds})
                next_bin = (int(snapshot.time_seconds)//300 + 1)*300
        bridge = env._bridge
        snapshot = bridge.passenger_time_ledger.snapshot()
        passengers = summarize_passengers_v5(bridge, snapshot)
        service = summarize_service(bridge)
        import libsumo
        version = libsumo.getVersion()
        if snapshot.time_seconds != p["evaluation_horizon_seconds"]:
            raise RuntimeError("SUMO evaluation did not finish the full service horizon")
        if passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("Passenger accounting is inconsistent")
        cost = 2*snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds
        result = {"method": method, "demand_scale": scale, **block, "sumo_version": version,
                  "horizon_seconds": snapshot.time_seconds, "elapsed_seconds": time.monotonic()-started,
                  "generalized_passenger_seconds": cost,
                  "generalized_seconds_per_departed_passenger": cost/passengers["departed_passengers"],
                  "passengers": passengers, "service": service, "headway": _headway_summary(records),
                  "mean_hold_seconds": float(np.mean([r["hold_seconds"] for r in records])),
                  "decision_events": len(records), "time_series": time_series}
    finally:
        env.close()
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(out, methods, workers):
    p = json.loads((out / "protocol.json").read_text())
    methods = methods or p["methods"]
    (out / "evaluation").mkdir(exist_ok=True)
    jobs = []
    for scale in p["evaluation_demand_scales"]:
        for block in p["evaluation_blocks"]:
            cfg, objects = build_scenario(out, scale, block["demand_seed"], p["departure_jitter_seconds"])
            for method in methods:
                if not (out / f"{method}.pt").exists():
                    raise FileNotFoundError(f"Policy is not complete: {method}")
                jobs.append((method, scale, block, cfg, objects, str(out), p))
    with ProcessPoolExecutor(max_workers=workers or p["evaluation_workers"], mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({"method": result["method"], "demand": result["demand_scale"],
                              "sumo_seed": result["sumo_seed"],
                              "cost_per_passenger": result["generalized_seconds_per_departed_passenger"],
                              "completion": result["passengers"]["completion_rate"],
                              "wall_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/control_v1")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(args.out.resolve(), args.methods, args.workers)
