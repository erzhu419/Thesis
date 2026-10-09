"""Independent learners and eighteen independent evaluations per endpoint."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("matrix6_dispatch", HERE.parent / "mcwm_attribution_v15/launch.py")
infra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infra)


def jobs(smoke):
    rows = []
    for mode, objective, arm, seed in product(["dual"] if smoke else P["modes"], P["objectives"],
            ["resac", "ensemble_sac_control"] if smoke else P["arms"], P["training_seeds"][:1] if smoke else P["training_seeds"]):
        label = f"{'smoke' if smoke else 'runs'}/{mode}/{objective}/{arm}_seed{seed}"
        fields = f"--mode {mode} --objective {objective} --arm {arm} --seed {seed}"+(" --smoke" if smoke else "")
        rows.append(infra.task(HERE, P, label, "--phase train "+fields, training=True, ram=3200))
        for case in ([0] if smoke else range(18)):
            ram = (6000 if case//3 == 3 else 10000 if case//3 == 4 else 3500) if not smoke else 2000
            rows.append(infra.task(HERE, P, label+f"/eval{case}", f"--phase evaluate {fields} --case {case}", [label], ram=ram))
    if not smoke:
        for objective, arm, seed in product(P["objectives"], ("constant_symbolic", "constant_control"), P["training_seeds"]):
            label = f"runs/holding/{objective}/{arm}_seed{seed}"
            fields = f"--family cs_context --mode holding --objective {objective} --arm {arm} --seed {seed}"
            for case in range(18):
                ram = 6000 if case//3 == 3 else 10000 if case//3 == 4 else 3500
                rows.append(infra.task(HERE, P, label+f"/eval{case}", f"--phase evaluate {fields} --case {case}", ram=ram))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "contracts", "reference-smoke", "smoke", "submit", "collect", "watch"))
    mode = parser.parse_args().mode
    if mode == "stage":
        infra.stage(HERE, {})
    elif mode == "collect":
        r = infra.collect(HERE)
        print(json.dumps({k: r[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))
    elif mode == "watch":
        infra.watch(HERE)
    elif mode == "contracts":
        infra.submit(HERE, P, [infra.task(HERE, P, "contracts", "--phase contracts", ram=3000)], "contracts")
    elif mode == "reference-smoke":
        rows = []
        for objective, arm in product(P["objectives"], ("constant_symbolic", "constant_control")):
            fields = f"--phase evaluate --family cs_context --mode holding --objective {objective} --arm {arm} --seed 64001 --case 0 --smoke"
            rows.append(infra.task(HERE, P, f"reference-smoke/{objective}/{arm}", fields, ram=2500))
        infra.submit(HERE, P, rows, "reference_smoke")
    else:
        rows = jobs(mode == "smoke")
        if mode == "submit":
            rows.append(infra.watcher(HERE, P))
        infra.submit(HERE, P, rows, "smoke" if mode == "smoke" else "campaign")
