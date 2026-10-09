"""Fit source-only envelopes and summarize paired frozen-policy outcomes."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit():
    rows = [json.loads(path.read_text()) for path in (HERE / "probes").glob("*.json")]
    assert {(r["scene"], r["day"]) for r in rows} == set(product(P["source_scenes"], P["source_probe_days"]))
    assert len(rows) == 9 and not set(P["source_probe_days"]) & set(P["evaluation_days"])
    assert all(r["learner_updates"] == 0 and r["rollout"]["timestamp_cost_identity_passed"] for r in rows)
    caps = [[max(r["peaks"][phase][j] for r in rows) for j in range(13)] for phase in range(10)]
    assert all(v > 0 for phase in caps for v in phase), "An empty source phase requires inspection"
    limits = dict(bin_seconds=P["time_bin_seconds"], phase_caps=caps,
        daily_caps=[max(phase[j] for phase in caps) for j in range(13)],
        source_days=P["source_probe_days"], source_scenes=P["source_scenes"], driver="fixed0",
        threshold_rule="source-probe maximum, no target rewards or target observations")
    write(HERE / "inputs/envelope.json", limits)
    root = HERE.parent / "cs_terminal_coverage_v1/results/global_coverage"
    references = {}
    for seed, day, scene in product(P["actors"], P["evaluation_days"], P["scenes"]):
        row = json.loads((root / f"seed{seed}" / f"day{day}_{scene}.json").read_text())
        references[f"{seed}/{day}/{scene}"] = {k: row[k] for k in
            ["cost", "exact_cost", "decisions", "mean_holding_seconds", "groups"]}
    write(HERE / "inputs/reference.json", references)
    report = dict(complete=True, source_probes=len(rows), source_days=P["source_probe_days"],
        global_daily_cap_passengers=limits["daily_caps"][0] * 100000,
        phase_global_caps_passengers=[phase[0] * 100000 for phase in caps],
        all_phase_caps_positive=True, target_data_used_for_thresholds=False)
    write(HERE / "fit_summary.json", report)
    print(json.dumps(report))


def metrics(cases):
    return dict(cases=len(cases), cost_seconds=mean(r["cost"] for r in cases),
        holding_seconds=mean(r["mean_holding_seconds"] for r in cases),
        wait_seconds=mean(r["groups"]["all"]["mean_wait_seconds"] for r in cases),
        onboard_seconds=mean(r["groups"]["all"]["mean_onboard_seconds"] for r in cases),
        completion_rate=mean(r["groups"]["all"]["completion_rate"] for r in cases),
        original_window_cost_seconds=mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in cases))


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/seed*/day*.json")]
    expected = set(product(P["variants"], [int(s) for s in P["actors"]], P["evaluation_days"], P["scenes"]))
    assert {(r["variant"], r["seed"], r["day"], r["scene"]) for r in rows} == expected and len(rows) == len(expected)
    references = json.loads((HERE / "inputs/reference.json").read_text())
    grouped = defaultdict(list)
    for row in rows:
        assert row["learner_updates"] == 0 and row["rollout"]["timestamp_cost_identity_passed"]
        grouped[row["variant"], row["scene"]].append(row)
    output = dict(complete=True, evaluations=len(rows), training_runs=0, analysis_location="server",
        source_fit=json.loads((HERE / "fit_summary.json").read_text()), groups={}, per_seed={}, controls={},
        maximum_wall_seconds=max(r["wall_seconds"] for r in rows), maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    for scene in P["scenes"]:
        originals = [references[f"{seed}/{day}/{scene}"] for seed, day in product(P["actors"], P["evaluation_days"])]
        output["controls"]["unchanged/" + scene] = metrics(originals)
        old_root = HERE.parent / "cs_passenger_alignment_v1/results"
        observed = [json.loads((old_root / "observed_overlap" / f"seed{seed}" / f"day{day}_{scene}.json").read_text())
                    for seed, day in product(P["actors"], P["evaluation_days"])]
        fixed = [json.loads((old_root / "fixed0/seed0" / f"day{day}_{scene}.json").read_text()) for day in P["evaluation_days"]]
        output["controls"]["observed_overlap/" + scene] = metrics(observed)
        output["controls"]["fixed0/" + scene] = metrics(fixed)
    for (variant, scene), cases in grouped.items():
        summary = metrics([r["rollout"] for r in cases])
        n = sum(r["adaptation"]["decisions"] for r in cases)
        summary.update(changed_fraction=sum(r["adaptation"]["changed_decisions"] for r in cases) / n,
            unchanged_cases=sum(r["unchanged_rollout_matched"] for r in cases),
            improved_cases=sum(r["cost_reduction_seconds"] > 0 for r in cases),
            mean_cost_reduction_seconds=mean(r["cost_reduction_seconds"] for r in cases),
            above59_fraction=sum(r["adaptation"]["above59_fraction"] * r["adaptation"]["decisions"] for r in cases) / n,
            pairs=[dict(seed=r["seed"], day=r["day"], cost_reduction_seconds=r["cost_reduction_seconds"],
                        cost_seconds=r["rollout"]["cost"], holding_seconds=r["rollout"]["mean_holding_seconds"],
                        adaptation=r["adaptation"]) for r in sorted(cases, key=lambda r: (r["seed"], r["day"]))])
        output["groups"][variant + "/" + scene] = summary
        for seed in [int(s) for s in P["actors"]]:
            selected = [r for r in cases if r["seed"] == seed]
            value = metrics([r["rollout"] for r in selected])
            value.update(mean_cost_reduction_seconds=mean(r["cost_reduction_seconds"] for r in selected),
                         improved_cases=sum(r["cost_reduction_seconds"] > 0 for r in selected))
            output["per_seed"][f"{variant}/seed{seed}/{scene}"] = value
    write(HERE / "analysis_summary.json", output)
    print(json.dumps({k: output[k] for k in ["complete", "evaluations", "training_runs", "maximum_wall_seconds", "maximum_rss_mb"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "collect"])
    {"fit": fit, "collect": collect}[parser.parse_args().mode]()
