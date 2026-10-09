"""One new CS component intervention across the reward/action matrix."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("centering7_dispatch", HERE.parent / "mcwm_attribution_v15/launch.py")
infra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infra)


def jobs(smoke):
    rows = []
    for mode, objective, seed in product(["dual"] if smoke else P["modes"], P["objectives"],
                                        P["training_seeds"][:1] if smoke else P["training_seeds"]):
        label = f"{'smoke' if smoke else 'runs'}/{mode}/{objective}/symbolic_no_center_seed{seed}"
        fields = f"--mode {mode} --objective {objective} --seed {seed}"+(" --smoke" if smoke else "")
        rows.append(infra.task(HERE, P, label, "--phase train "+fields, training=True, ram=3200))
        for case in ([0] if smoke else range(18)):
            ram = 6000 if not smoke and case//3 == 3 else 10000 if not smoke and case//3 == 4 else 3500
            rows.append(infra.task(HERE, P, label+f"/eval{case}", f"--phase evaluate {fields} --case {case}", [label], ram=ram))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "smoke", "submit", "collect", "watch"))
    mode = parser.parse_args().mode
    if mode == "stage":
        infra.stage(HERE, {})
    elif mode == "collect":
        r = infra.collect(HERE)
        print(json.dumps({k: r[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))
    elif mode == "watch":
        infra.watch(HERE)
    else:
        rows = jobs(mode == "smoke")
        if mode == "submit":
            rows.append(infra.watcher(HERE, P))
        infra.submit(HERE, P, rows, "smoke" if mode == "smoke" else "campaign")
