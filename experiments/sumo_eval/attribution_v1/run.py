"""Frozen-policy attribution, with one SUMO process per complete service day."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["THESIS_H2O_ROOT"] = str(BASE / "runtime/H2Oplus")
os.environ["THESIS_SUMO_ROOT"] = str(BASE / "assets/SUMO_ruiguang")
os.environ["THESIS_DATASET"] = str(BASE / "contract.h5")
os.environ["SUMO_HOME"] = str(BASE / "assets/SUMO_ruiguang")
sys.path.insert(0, str(BASE))

import h5py
import numpy as np
import torch
import evaluate_control as common
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as V4_DIM
from bus_h2o.mdp_contract_v7 import (
    RAW_EVENT_STATE_INDEX, current_event_v7, hold_seconds_to_policy_action_v7,
)
from SimpleSAC.evaluate_transit_v16_headway_reserve import reserved_hold_action
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4, myopic_welfare_hold_seconds_v4,
)


def cases(protocol):
    rows = []
    treatments = [(m, layer) for m in protocol["methods"]
                  for layer in protocol["execution_layers"]]
    treatments += [(m, "native") for m in protocol["native_rule_references"]]
    for block in protocol["evaluation_blocks"]:
        for scale in protocol["evaluation_demand_scales"]:
            for method, layer in treatments:
                rows.append(dict(method=method, execution_layer=layer,
                                 demand_scale=scale, **block))
    for index, row in enumerate(rows):
        row["case_id"] = f"case{index:03d}"
    return rows


def contract():
    with h5py.File(BASE / "contract.h5", "r") as data:
        return (common.TransitMDPConfigV7(**json.loads(data.attrs["config"])),
                np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32))


def rule_action(method, raw, config):
    event = current_event_v7(raw)[:V4_DIM]
    if method == "zero":
        hold = 0.0
    elif method == "headway":
        hold = headway_balance_hold_seconds_v4(event, 1.0, config)
    elif method == "welfare":
        hold = myopic_welfare_hold_seconds_v4(event, config)
    else:
        raise ValueError(method)
    return np.asarray([hold_seconds_to_policy_action_v7(hold, config)], np.float32)


def execute(raw, proposed, layer, config, scale):
    if layer == "native":
        return proposed, proposed
    scaled = (proposed + 1) * scale - 1
    parent = reserved_hold_action(raw, scaled, 0.75, config)
    if layer == "no_floor":
        action = parent
    elif layer in ("late20", "early60"):
        variant = "baseline" if layer == "late20" else "quarter-load-sixty"
        action = common.completion_safety_action(raw, scaled, variant, config)
    else:
        raise ValueError(layer)
    return action, parent


def evaluate(job):
    case, p, out_string, cfg, objects = job
    out = Path(out_string)
    output = out / "results" / f"{case['case_id']}.json"
    if output.exists():
        return json.loads(output.read_text())
    torch.set_num_threads(1)
    config, stations = contract()
    actor = None
    if case["method"] in p["policy_exports"]:
        actor, _ = common.load_actor_checkpoint_v7(
            HERE / "inputs" / f"{case['method']}.pt", "cpu", require_policy_artifact=True)
    env = common.AllLinesSumoEnv(
        str(common.SUMO_ROOT / "online_control"),
        str(common.ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
        episode_seconds=p["evaluation_horizon_seconds"], sumo_seed=case["sumo_seed"],
        sumo_cfg=cfg, passenger_obj_xml=objects, observation_schema="v5", sumo_backend="libsumo")
    started = time.monotonic()
    records, series, bins = [], [], defaultdict(lambda: defaultdict(float))
    next_bin = 300
    next_progress = 6000
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            actions, controls = {}, []
            for event in sorted(env._current_events, key=common.event_sort_key):
                if common.is_terminal_event(env, event):
                    actions[event.bus_id] = [0.0, 1.0]
                else:
                    controls.append((event, common.raw_event_observation(env, event.bus_id)))
            if controls:
                local_batch = np.stack([local for _, local in controls])
                prior = np.full(len(controls), -1.0, np.float32)
                for index, (event, local) in enumerate(controls):
                    raw = common.pack_event_batch_v7(local_batch, index, prior, config)
                    if actor is None:
                        proposed = rule_action(case["method"], raw, config)
                    else:
                        state = common.raw_state_to_canonical_v7(raw, stations, config)
                        with torch.inference_mode():
                            proposed, _ = actor(torch.from_numpy(state[None]), deterministic=True)
                        proposed = proposed.numpy()[0]
                    action, parent = execute(raw, proposed, case["execution_layer"],
                                             config, p["deployment_hold_scale"])
                    hold = float(common.policy_action_to_hold_seconds_v7(action[0], config))
                    parent_hold = float(common.policy_action_to_hold_seconds_v7(parent[0], config))
                    prior[index] = action[0]
                    actions[event.bus_id] = [hold, 1.0]
                    records.append(common._event_record(event, local, action, hold))
                    event_raw = current_event_v7(raw)
                    decision_time = event_raw[RAW_EVENT_STATE_INDEX["sim_time_seconds"]]
                    bucket = bins[str(int(decision_time) // 1800 * 1800)]
                    bucket["decisions"] += 1
                    bucket["proposed_hold_seconds"] += float(
                        common.policy_action_to_hold_seconds_v7(proposed[0], config))
                    bucket["parent_hold_seconds"] += parent_hold
                    bucket["executed_hold_seconds"] += hold
                    if hold > parent_hold + 1e-4:
                        bucket["floor_changed_decisions"] += 1
                        bucket["floor_added_hold_seconds"] += hold - parent_hold
                        bucket["onboard_at_changed_decisions"] += float(
                            event_raw[RAW_EVENT_STATE_INDEX["onboard_passengers"]])
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
            if env._bridge.current_time >= next_bin:
                snapshot = env._bridge.passenger_time_ledger.snapshot()
                series.append(dict(time_seconds=snapshot.time_seconds,
                                   waiting=snapshot.waiting_passengers,
                                   onboard=snapshot.in_vehicle_passengers,
                                   wait_passenger_seconds=snapshot.accrued_wait_seconds,
                                   ride_passenger_seconds=snapshot.accrued_in_vehicle_seconds))
                next_bin = (int(snapshot.time_seconds) // 300 + 1) * 300
                if snapshot.time_seconds >= next_progress:
                    print(json.dumps(dict(case_id=case["case_id"], stage="progress",
                                          simulated_seconds=snapshot.time_seconds,
                                          elapsed_seconds=time.monotonic() - started)), flush=True)
                    next_progress += 6000
        bridge = env._bridge
        snapshot = bridge.passenger_time_ledger.snapshot()
        passengers = common.summarize_passengers_v5(bridge, snapshot)
        if snapshot.time_seconds != p["evaluation_horizon_seconds"]:
            raise RuntimeError("Incomplete service horizon")
        if passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("Inconsistent passenger accounting")
        cost = 2 * snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds
        import libsumo
        result = dict(**case, training_seed=p["training_seed"] if actor is not None else None,
                      horizon_seconds=snapshot.time_seconds, sumo_version=libsumo.getVersion(),
                      elapsed_seconds=time.monotonic() - started,
                      generalized_passenger_seconds=cost,
                      generalized_seconds_per_departed_passenger=cost / passengers["departed_passengers"],
                      passengers=passengers, service=common.summarize_service(bridge),
                      headway=common._headway_summary(records), decision_events=len(records),
                      mean_hold_seconds=float(np.mean([r["hold_seconds"] for r in records])),
                      execution_bins=dict(bins), time_series=series)
    finally:
        env.close()
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import libsumo
    if libsumo.getVersion()[1] != "SUMO 1.25.0":
        raise RuntimeError(f"Unexpected SUMO version: {libsumo.getVersion()}")
    p = json.loads((HERE / "protocol.json").read_text())
    selected = cases(p)[args.shard::p["shards"]]
    out = HERE / "runs" / ("smoke" if args.smoke else f"shard{args.shard:02d}")
    if args.smoke:
        p["evaluation_horizon_seconds"] = 600
        selected = selected[:1]
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results/protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    jobs = []
    for case in selected:
        cfg, objects = common.build_scenario(out, case["demand_scale"], case["demand_seed"],
                                              p["departure_jitter_seconds"])
        jobs.append((case, p, str(out), cfg, objects))
    print(json.dumps(dict(stage="start", shard=args.shard, cases=len(jobs))), flush=True)
    with ProcessPoolExecutor(max_workers=min(len(jobs), p["workers_per_shard"]),
                             mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(dict(case_id=result["case_id"], stage="complete",
                                  elapsed_seconds=result["elapsed_seconds"])), flush=True)
    (out / "results/completed.json").write_text(json.dumps(dict(
        shard=args.shard, case_ids=[case["case_id"] for case in selected], smoke=args.smoke)) + "\n")


if __name__ == "__main__":
    main()
