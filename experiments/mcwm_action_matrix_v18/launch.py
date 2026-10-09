"""Parallel physical-action tests, without new data collection or IQL jobs."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("action18_dispatch", HERE.parent / "mcwm_attribution_v15/launch.py")
infra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infra)


def jobs(smoke):
    rows = []
    for objective, scene, arm, seed in product(P["objectives"], P["scenes"][:1] if smoke else P["scenes"],
                                              P["arms"], P["training_seeds"][:1] if smoke else P["training_seeds"]):
        label = f"{'smoke' if smoke else 'runs'}/{objective}/{scene}/{arm}_seed{seed}"
        fields = f"--objective {objective} --scene {scene} --arm {arm} --seed {seed}"+(" --smoke" if smoke else "")
        rows.append(infra.task(HERE, P, label, "--phase train "+fields, training=True, ram=3000))
        for layer, case in product(P["executors"], [0] if smoke else P["cases"]):
            rows.append(infra.task(HERE, P, label+f"/{layer}/eval{case}",
                f"--phase evaluate {fields} --layer {layer} --case {case}", [label], ram=2500))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "contracts", "smoke", "submit", "collect", "watch"))
    mode = parser.parse_args().mode
    if mode == "stage":
        infra.stage(HERE, json.loads((HERE.parent / "mcwm_headway_alignment_v16/node_runtime.json").read_text()))
    elif mode == "collect":
        r = infra.collect(HERE)
        print(json.dumps({k: r[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))
    elif mode == "watch":
        infra.watch(HERE)
    elif mode == "contracts":
        rows = [infra.task(HERE, P, "contracts/"+objective,
                f"--phase contracts --objective {objective} --arm mcwm --smoke", ram=3000) for objective in P["objectives"]]
        infra.submit(HERE, P, rows, "contracts")
    else:
        rows = jobs(mode == "smoke")
        if mode == "submit":
            rows.append(infra.watcher(HERE, P))
        infra.submit(HERE, P, rows, "smoke" if mode == "smoke" else "campaign")
