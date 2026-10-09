"""Change only the model-TD coefficient of the completed v16 learner."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PREVIOUS = HERE.parent / "mcwm_headway_alignment_v16"
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location("td17_headway_training", PREVIOUS / "run.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
import torch
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer


def folder(scene, arm, seed, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / scene / f"{arm}_seed{seed}"


class Trainer(previous.Trainer):
    coefficient = 0.

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        observed = TransitH2OTrainer._ensemble_td_loss(self, predictions, targets, weights)
        model = getattr(self, "model_loss", None)
        if model is None:
            return observed
        return P["observed_td_coefficient"] * observed + self.coefficient * model


def run(args):
    Trainer.coefficient = P["coefficients"][args.arm]
    previous.Trainer = Trainer
    previous.folder = folder
    if args.phase == "evaluate":
        import evaluate_headway
        evaluate_headway.folder = folder
        evaluate_headway.HERE = HERE
        evaluate_headway.P = dict(evaluate_headway.P, arms=P["arms"])
    original_arm = args.arm
    previous.run(args)
    if args.phase == "train":
        path = folder(args.scene, original_arm, args.seed, args.smoke) / "ready.json"
        row = json.loads(path.read_text())
        row.update(experiment=P["name"], observed_td_coefficient=P["observed_td_coefficient"],
                   model_td_coefficient=Trainer.coefficient, model_fitting_retained=True)
        previous.write(path, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--arm", choices=P["arms"], required=True)
    parser.add_argument("--scene", choices=P["scenes"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", choices=P["executors"], default="guarded")
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(parser.parse_args())
