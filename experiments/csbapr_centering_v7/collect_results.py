"""Resident centering comparisons against the full and no-symbolic policies."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def collect():
    ready, progress, paired = [], [], []
    training = evaluations = smoke_training = smoke_evaluations = 0
    for smoke in (True, False):
        for mode, objective, seed in product(["dual"] if smoke else P["modes"], P["objectives"],
                                             P["training_seeds"][:1] if smoke else P["training_seeds"]):
            label = f"{'smoke' if smoke else 'runs'}/{mode}/{objective}/symbolic_no_center_seed{seed}"
            root = HERE / label
            if (root / "ready.json").exists():
                r = json.loads((root / "ready.json").read_text())
                assert r["episodes"] == (1 if smoke else 100) and (root / "policy.pt").exists()
                ready.append(label)
                if smoke:
                    smoke_training += 1
                else:
                    training += 1
            elif (root / "progress.json").exists():
                r = json.loads((root / "progress.json").read_text())
                progress.append(dict(label=label, episode=r["episode"], wall_seconds=r["wall_seconds"]))
            for case in ([0] if smoke else range(18)):
                path = root / f"eval_{case:02d}.json"
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                assert (r["item"]["mode"], r["item"]["objective"], r["item"]["seed"], r["case"]) == (mode, objective, seed, case)
                assert r["horizon_seconds"] == (3600 if smoke else 18000)
                if smoke:
                    smoke_evaluations += 1
                    continue
                evaluations += 1
                for reference in ("constant_symbolic", "constant_control"):
                    parent = HERE.parent / "bus_alignment_followup_v5/formal/cs_context/dual" / objective / reference / str(seed) if mode == "dual" else HERE.parent / "bus_algorithm_matrix_v6/runs/holding" / objective / f"{reference}_seed{seed}"
                    path = parent / f"eval_{case:02d}.json"
                    if not path.exists():
                        continue
                    b = json.loads(path.read_text())
                    assert r["evaluation_seed"] == b["evaluation_seed"]
                    paired.append(dict(mode=mode, objective=objective, seed=seed, case=case, scene=r["scene"], reference=reference,
                        cost_delta=r["cost"]-b["cost"], reward_delta=r["native_reward"]/r["decisions"]-b["native_reward"]/b["decisions"],
                        plan_delta=r["headway_plan_mae_seconds"]-b["headway_plan_mae_seconds"],
                        imbalance_delta=r["headway_imbalance_seconds"]-b["headway_imbalance_seconds"]))
    report = dict(experiment=P["name"], analysis_location="server", ready=ready, progress=progress,
        training_ready=training, training_expected=8, evaluation_ready=evaluations, evaluation_expected=144,
        smoke_complete=smoke_training == 2 and smoke_evaluations == 2,
        complete=training == 8 and evaluations == 144 and len(paired) == 288, paired=paired)
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))


if __name__ == "__main__":
    collect()
