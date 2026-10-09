"""Independent training trajectories, releasing each fixed-budget endpoint."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("native8_dispatch", HERE.parent / "mcwm_attribution_v15/launch.py")
infra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infra)


def task(label, fields, waits=(), *, training=False, gpu=False, ram=2000):
    row = infra.task(HERE, P, label, fields, waits, training=training, ram=ram)
    if gpu:
        row["vram"] = 1600
        row["allow_gpu_over_one_third"] = True
        row.pop("allow_cpu_training", None)
        row.pop("cpu_training_justification", None)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "preflight", "submit", "collect", "watch"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mode == "stage":
        infra.stage(HERE, {})
    elif args.mode == "collect":
        r = infra.collect(HERE)
        print(json.dumps({k:r[k] for k in ("training_ready", "evaluation_ready", "complete")}))
    elif args.mode == "watch":
        infra.watch(HERE)
    elif args.mode == "preflight":
        rows = []
        for device in ("cpu", "cuda"):
            label = f"preflight/{device}"
            # Separate the two timing outputs while keeping the same random seed.
            fields = f"--phase preflight --device {device} --seed 64001"
            row = task(label, fields, training=True, gpu=device == "cuda")
            rows.append(row)
        infra.submit(HERE, P, rows, "preflight")
    else:
        rows = []
        for mode, objective, arm, seed in product(P["modes"], P["objectives"], P["arms"], P["training_seeds"]):
            label = f"runs/{mode}/{objective}/{arm}_seed{seed}"
            fields = f"--mode {mode} --objective {objective} --arm {arm} --seed {seed}"
            rows.append(task(label, f"--phase train --device {args.device} "+fields,
                             training=True, gpu=args.device == "cuda"))
            for budget, case in product(P["checkpoints"], range(18)):
                ram = 6000 if case//3 == 3 else 10000 if case//3 == 4 else 3500
                rows.append(task(label+f"/budget{budget}/eval{case}",
                    f"--phase evaluate --budget {budget} --case {case} "+fields,
                    [label+f"/budget{budget}"], ram=ram))
        rows.append(infra.watcher(HERE, P))
        infra.submit(HERE, P, rows, "campaign")
