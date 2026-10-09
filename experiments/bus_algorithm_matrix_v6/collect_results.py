"""Paired algorithm, component and action comparisons on server storage."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PREVIOUS = HERE.parent / "bus_alignment_followup_v5"


def read(path):
    return json.loads(path.read_text())


def collect():
    ready, progress, rows = [], [], []
    training = evaluations = smoke_training = smoke_evaluations = 0
    refs = ["stabilized_mlp", "constant_symbolic", "constant_control"]
    for smoke in (True, False):
        modes = ["dual"] if smoke else P["modes"]
        objectives = P["objectives"]
        seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
        arms = ["resac", "ensemble_sac_control"] if smoke else P["arms"]+refs
        for mode, objective, arm, seed in product(modes, objectives, arms, seeds):
            label = f"{'smoke' if smoke else 'runs'}/{mode}/{objective}/{arm}_seed{seed}"
            root = HERE / label
            if arm in P["arms"]:
                if (root / "ready.json").exists():
                    r = read(root / "ready.json")
                    assert r["episodes"] == (1 if smoke else P["episodes"])
                    assert r["horizon_seconds"] == (3600 if smoke else P["horizon_seconds"])
                    assert (root / "policy.pt").exists()
                    ready.append(label)
                    if smoke:
                        smoke_training += 1
                    else:
                        training += 1
                elif (root / "progress.json").exists():
                    r = read(root / "progress.json")
                    progress.append(dict(label=label, episode=r["episode"], wall_seconds=r["wall_seconds"]))
            elif arm == "stabilized_mlp" or mode == "dual":
                family = "cs_recipe" if arm == "stabilized_mlp" else "cs_context"
                root = PREVIOUS / "formal" / family / mode / objective / arm / str(seed)
            for case in ([0] if smoke else range(18)):
                path = root / f"eval_{case:02d}.json"
                if not path.exists():
                    continue
                r = read(path)
                assert (r["item"]["mode"], r["item"]["objective"], r["item"]["arm"], r["item"]["seed"], r["case"]) == (mode, objective, arm, seed, case)
                assert r["horizon_seconds"] == (3600 if smoke else P["horizon_seconds"])
                assert r["evaluation_seed"] == P["evaluation_seeds"][case % 3]
                if smoke:
                    smoke_evaluations += 1
                else:
                    if arm in P["arms"] or mode == "holding" and arm.startswith("constant_"):
                        evaluations += 1
                    rows.append(dict(mode=mode, objective=objective, arm=arm, seed=seed, case=case, scene=r["scene"],
                        people=r["generated_passengers"], cost=r["cost"], reward=r["native_reward"]/r["decisions"],
                        plan=r["headway_plan_mae_seconds"], imbalance=r["headway_imbalance_seconds"], speed=r["mean_speed_ratio"]))
    indexed = {(r["mode"], r["objective"], r["arm"], r["seed"], r["case"]): r for r in rows}
    contrasts = []
    for r in rows:
        references = [(r["mode"], "ensemble_sac_control"), (r["mode"], "resac")]
        if r["mode"] == "dual":
            references.append(("holding", r["arm"]))
        for mode, arm in references:
            if (mode, arm) == (r["mode"], r["arm"]):
                continue
            b = indexed.get((mode, r["objective"], arm, r["seed"], r["case"]))
            if b is None:
                continue
            # Passenger arrivals use action-dependent RNG ordering in this legacy simulator.
            contrasts.append(dict(mode=r["mode"], objective=r["objective"], arm=r["arm"], seed=r["seed"], scene=r["scene"],
                case=r["case"], reference_arm=arm, reference_mode=mode, people_difference=r["people"]-b["people"],
                **{k+"_delta": r[k]-b[k] for k in ("cost", "reward", "plan", "imbalance")}))
    aggregate = []
    for mode, objective, arm, scene in product(P["modes"], P["objectives"], P["arms"]+refs, P["scenes"]):
        group = [r for r in rows if (r["mode"], r["objective"], r["arm"], r["scene"]) == (mode, objective, arm, scene)]
        if len(group) == 6:
            aggregate.append(dict(mode=mode, objective=objective, arm=arm, scene=scene, cases=6,
                                  **{k: mean(r[k] for r in group) for k in ("cost", "reward", "plan", "imbalance", "speed")}))
    report = dict(experiment=P["name"], analysis_location="server", ready=ready, progress=progress,
        training_ready=training, training_expected=32, evaluation_ready=evaluations, evaluation_expected=720,
        smoke_complete=smoke_training == 4 and smoke_evaluations == 4,
        complete=training == 32 and evaluations == 720, aggregates=aggregate, contrasts=contrasts,
        interpretation=P["interpretation"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))


if __name__ == "__main__":
    collect()
