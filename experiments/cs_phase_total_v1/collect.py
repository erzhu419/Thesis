"""Compare the new global time-bin bound against frozen paired controls."""

from itertools import product
import importlib.util
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["frozen_campaign"]
spec = importlib.util.spec_from_file_location("source_envelope_metrics", PRIOR / "collect.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("seed*/day*.json")]
    expected = set(product(P["seeds"], P["evaluation_days"], P["scenes"]))
    assert {(r["seed"], r["day"], r["scene"]) for r in rows} == expected and len(rows) == len(expected)
    assert all(r["learner_updates"] == 0 and r["rollout"]["timestamp_cost_identity_passed"] for r in rows)
    references = json.loads((PRIOR / "inputs/reference.json").read_text())
    output = dict(complete=True, evaluations=len(rows), training_runs=0, analysis_location="server",
        source_fit=json.loads((PRIOR / "fit_summary.json").read_text()), groups={}, per_seed={}, controls={},
        maximum_wall_seconds=max(r["wall_seconds"] for r in rows), maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    for scene in P["scenes"]:
        cases = sorted([r for r in rows if r["scene"] == scene], key=lambda r: (r["seed"], r["day"]))
        pairs = []
        old_variants = {name: [] for name in ["daily_total", "phase_envelope"]}
        for row in cases:
            pair = dict(seed=row["seed"], day=row["day"], cost_seconds=row["rollout"]["cost"],
                holding_seconds=row["rollout"]["mean_holding_seconds"],
                reduction_vs_unchanged=row["cost_reduction_seconds"], adaptation=row["adaptation"])
            for variant in old_variants:
                old = json.loads((PRIOR / "results" / variant / f'seed{row["seed"]}' / f'day{row["day"]}_{scene}.json').read_text())
                old_variants[variant].append(old["rollout"])
                pair["reduction_vs_" + variant] = old["rollout"]["cost"] - row["rollout"]["cost"]
            pairs.append(pair)
        originals = [references[f'{r["seed"]}/{r["day"]}/{scene}'] for r in cases]
        output["controls"]["unchanged/" + scene] = shared.metrics(originals)
        for name, values in old_variants.items():
            output["controls"][name + "/" + scene] = shared.metrics(values)
        old_root = HERE.parent / "cs_passenger_alignment_v1/results"
        observed = [json.loads((old_root / "observed_overlap" / f"seed{seed}" / f"day{day}_{scene}.json").read_text())
                    for seed, day in product(P["seeds"], P["evaluation_days"])]
        fixed = [json.loads((old_root / "fixed0/seed0" / f"day{day}_{scene}.json").read_text()) for day in P["evaluation_days"]]
        output["controls"]["observed_overlap/" + scene] = shared.metrics(observed)
        output["controls"]["fixed0/" + scene] = shared.metrics(fixed)
        for key, selected in [(scene, cases)] + [(f"seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["seeds"]]:
            selected_pairs = [pair for pair in pairs if any(pair["seed"] == r["seed"] and pair["day"] == r["day"] for r in selected)]
            value = shared.metrics([r["rollout"] for r in selected])
            value.update(unchanged_cases=sum(r["unchanged_rollout_matched"] for r in selected),
                changed_fraction=sum(r["adaptation"]["changed_decisions"] for r in selected) / sum(r["adaptation"]["decisions"] for r in selected))
            for control in ["unchanged", "daily_total", "phase_envelope"]:
                reductions = [pair["reduction_vs_" + control] for pair in selected_pairs]
                value["reduction_vs_" + control] = dict(mean_seconds=mean(reductions),
                    improved_cases=sum(v > 1e-5 for v in reductions), unchanged_cases=sum(abs(v) <= 1e-5 for v in reductions))
            if key == scene:
                value["pairs"] = pairs
                output["groups"][key] = value
            else:
                output["per_seed"][key] = value
    shared.write(HERE / "analysis_summary.json", output)
    print(json.dumps({k: output[k] for k in ["complete", "evaluations", "training_runs", "maximum_wall_seconds", "maximum_rss_mb"]}))


if __name__ == "__main__":
    collect()
