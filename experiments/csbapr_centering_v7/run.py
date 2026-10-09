"""One projection ablation in a new directory; no change to the full model."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
path = HERE.parent / "bus_algorithm_matrix_v6/run.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("centering7_matrix_runner", path)
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)
import torch
from torch import nn


def folder(item):
    return HERE / ("smoke" if item["smoke"] else "runs") / item["mode"] / item["objective"] / f"{item['arm']}_seed{item['seed']}"


def build(item, kind, runtime):
    full = matrix.original_build(dict(item, arm="constant_symbolic"), kind, runtime)
    from csbapr.networks.policy import GaussianPolicy
    dimension = 1 if item["mode"] == "holding" else 2

    class UncenteredPolicy(GaussianPolicy):
        def __init__(self):
            super().__init__(15, dimension, 64, activation="tanh")
            self.equilibrium_mean = nn.Parameter(torch.zeros(dimension))

        def forward(self, states):
            mean, log_std = super().forward(states)
            return self.equilibrium_mean + mean, log_std

        def compute_L_eff(self):
            return float("inf")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(item["seed"]+50000)
        actor = UncenteredPolicy()
    assert all(torch.equal(v, full.actor.state_dict()[k]) for k, v in actor.state_dict().items())
    full.actor = actor
    full.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=full.config.lr_actor,
                                           weight_decay=full.config.actor_weight_decay)
    return full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--family", default="cs_context")
    parser.add_argument("--mode", choices=P["modes"], required=True)
    parser.add_argument("--objective", choices=P["objectives"], required=True)
    parser.add_argument("--arm", default="symbolic_no_center")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", default="native")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    matrix.folder, matrix.reference_path, matrix.build = folder, folder, build
    torch.set_num_threads(1)
    matrix.run(args)
    print("DONE centering ablation", flush=True)
