"""Read-only reward/credit diagnostics; all large inputs stay on the server."""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from cross_common import HERE, write
from rewards import canonical_headway_reward, ridge_reward


def distribution(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()),
                p10=float(np.quantile(values, .1)),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)))


def dataset_diagnostics(path):
    with h5py.File(path, "r") as data:
        size = len(data["rewards"])
        dt = data["delta_t_seconds"][:]
        terminal = data["terminals"][:] > .5
        trajectories = data["trajectory_id"][:]
        buses = data["event_bus_code"][:]
        lines = data["event_line_index"][:]
        valid = ~terminal[:-1] & (trajectories[:-1] == trajectories[1:])
        same_bus = valid & (buses[:-1] == buses[1:]) & (lines[:-1] == lines[1:])
        order = np.lexsort((np.arange(size), buses, trajectories))
        adjacent = (trajectories[order[:-1]] == trajectories[order[1:]]) & (
            buses[order[:-1]] == buses[order[1:]])
        return_delays = (order[1:]-order[:-1])[adjacent]
        temporal = dict(scope="entire resident dataset, all original split labels",
            rows=size, trajectories=int(len(np.unique(trajectories))),
            zero_duration_fraction=float(np.mean(dt == 0)),
            same_bus_next_fraction=float(same_bus.sum()/valid.sum()),
            different_line_next_fraction=float(np.mean(lines[:-1][valid] != lines[1:][valid])),
            same_bus_return_token_delay=distribution(return_delays),
            nonterminal_discounts=distribution(data["discounts"][:][~terminal]))
        samples = defaultdict(list)
        for start in np.linspace(0, size-513, 64, dtype=int):
            stop = start+512
            ns = data["next_observations"][start:stop]
            raw = data["raw_next_observations"][start:stop]
            term = terminal[start:stop]
            reward = canonical_headway_reward(ns, term)
            raw_slots = raw[:, 5:].reshape(-1, 16, 15)
            local = raw_slots[np.arange(len(raw)), np.rint(raw[:, 2]).astype(int)]
            physical = ridge_reward(local[:, 3], local[:, 4], local[:, 5], local[:, 6],
                                    local[:, 11] > .5, local[:, 12] > .5)
            physical = np.where(term | (local[:, 0] < .5), 0., physical)
            canonical = ns[:, 5:].reshape(-1, 16, 14)
            slot = canonical[np.arange(len(ns)), np.rint(ns[:, 2]*15).astype(int)]
            for key, value in dict(reward=reward, raw_reward=physical,
                    passenger_reward=data["rewards"][start:stop], dt=dt[start:stop],
                    clip=((slot[:, 3] >= 5) & (slot[:, 11] > .5)) |
                         ((slot[:, 4] >= 5) & (slot[:, 12] > .5)),
                    both=(slot[:, 11] > .5) & (slot[:, 12] > .5),
                    one=(slot[:, 11] > .5) ^ (slot[:, 12] > .5),
                    changed=np.abs(reward-physical) > .01).items():
                samples[key].append(value)
        samples = {k:np.concatenate(v) for k,v in samples.items()}
        zero = samples["dt"] == 0
        reward = samples["reward"]
        sample = dict(rows=len(reward), headway_reward=distribution(reward),
            scaled_headway_reward=distribution(.01*reward),
            scaled_passenger_reward=distribution(.0001*samples["passenger_reward"]),
            zero_duration_fraction=float(zero.mean()),
            zero_duration_nonzero_headway_fraction=float(np.mean(reward[zero] != 0)),
            zero_duration_headway_cost_share=float(reward[zero].sum()/reward.sum()),
            raw_vs_bounded_changed_fraction=float(samples["changed"].mean()),
            both_neighbors_fraction=float(samples["both"].mean()),
            one_neighbor_fraction=float(samples["one"].mean()),
            capped_headway_fraction=float(samples["clip"].mean()))
        observations = np.concatenate([data["observations"][i:i+32]
            for i in np.linspace(0, size-33, 16, dtype=int)])
        return temporal, sample, observations


def actor_diagnostics(observations):
    runtime = json.loads((HERE / "node_runtime.json").read_text())["train"]
    os.environ.update(runtime["env"])
    sys.path.insert(0, runtime["env"]["THESIS_H2O_ROOT"])
    from SimpleSAC.transit_v7.networks import HurdleTransitActorV7
    states = torch.as_tensor(observations)
    result = []
    for objective in ("headway", "passenger"):
        prefix = HERE / "inputs" / objective / "seed64001"
        paths = [("initializer", "offline", prefix / "initializer.pt")]
        paths += [(arm, scene, HERE / "runs/mc" / objective / scene /
                   f"{arm}_seed64001/policy.pt")
                  for scene in ("calibrated", "joint75")
                  for arm in ("h2oplus", "h2o_matched", "fixed0250")]
        for arm, scene, path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload.get("config", payload.get("source_config"))
            actor = HurdleTransitActorV7((config["hidden_dim"],)*2,
                layer_norm=config["actor_layer_norm"], activation=config["activation"])
            actor.load_state_dict(payload["actor"])
            actor.eval()
            with torch.no_grad():
                gate, _, _ = actor.distribution_parameters(states)
                probability = gate.sigmoid().numpy()
                actions = actor(states, deterministic=True)[0].numpy().reshape(-1)
                holds = (actions+1)*30
            result.append(dict(objective=objective, arm=arm, source_scene=scene,
                training_seed=64001, evaluation_states="fixed 512 offline states",
                hold_probability=distribution(probability),
                deterministic_hold_seconds=distribution(holds),
                exact_zero_fraction=float(np.mean(actions == -1)),
                at_least_50s_fraction=float(np.mean(holds >= 50))))
            del actor, payload
    return result


def training_diagnostics():
    result = []
    for objective in ("headway", "passenger"):
        for scene in ("calibrated", "joint75"):
            paths = sorted((HERE / "runs/mc" / objective / scene /
                "fixed0250_seed64001/results/diagnostics").glob("update*.json"))
            selected = [p for p in paths if int(p.stem[6:]) in (51000, 55000, 60000, 65000, 70000)]
            for path in selected:
                row = json.loads(path.read_text())
                keys = ("train/hold_probability", "train/positive_action_mean",
                        "train/alpha", "train/actor_grad_norm", "train/critic_grad_norm",
                        "train/q_data_mean", "model_td_loss", "model_weight_mean")
                result.append(dict(objective=objective, source_scene=scene,
                    total_updates=row["total_updates"],
                    metrics={k:row["metrics"].get(k) for k in keys}))
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    runtime = json.loads((HERE / "node_runtime.json").read_text())["train"]
    temporal, sample, observations = dataset_diagnostics(runtime["env"]["THESIS_DATASET"])
    report = dict(analysis_location="server", dataset_temporal=temporal,
        relabel_sample=sample, actor_common_state_probe=actor_diagnostics(observations),
        model_training_trace=training_diagnostics())
    write(HERE / "headway_diagnosis.json", report)
    print(json.dumps(report, indent=2), flush=True)
