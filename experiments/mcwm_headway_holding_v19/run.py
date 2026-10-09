"""Keep the proven holding restriction; change only the headway TD weight."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("holding19_training", HERE.parent / "mcwm_action_matrix_v18/run.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
previous.HERE, previous.P = HERE, P
load = previous.module


def module(path):
    result = load(path)
    from SimpleSAC.transit_v3.trainer import TransitH2OTrainer

    class Trainer(result.Trainer):
        def _ensemble_td_loss(self, predictions, targets, weights=None):
            observed = TransitH2OTrainer._ensemble_td_loss(self, predictions, targets, weights)
            model = getattr(self, "model_loss", None)
            if model is None:
                return observed
            return P["observed_td_coefficient"]*observed + P["model_td_coefficient"]*model

    result.Trainer = Trainer
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("contracts", "train", "evaluate"), required=True)
    parser.add_argument("--objective", choices=P["objectives"], default="headway")
    parser.add_argument("--arm", choices=P["arms"], default="mcwm")
    parser.add_argument("--scene", choices=P["scenes"], default="calibrated")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", choices=P["executors"], default="guarded")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    previous.module = module
    previous.run(args)
    if args.phase == "train":
        path = HERE / ("smoke" if args.smoke else "runs") / "headway" / args.scene / f"mcwm_seed{args.seed}" / "ready.json"
        r = json.loads(path.read_text())
        r.update(observed_td_coefficient=.75, model_td_coefficient=.125)
        path.write_text(json.dumps(r, indent=2)+"\n")
