"""Decompose existing held-out TransitDuet costs without rerunning policies."""

import json
from pathlib import Path
from statistics import mean, stdev

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "TransitDuet/transit_duet"
RESULTS = PROJECT / "results_remote/revision3_heldout"
SEEDS = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
METHODS = ["H_timetable_v5_stable_300", "H_hiro_final",
           "H_fixed_timetable_300_final"]


def merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def config(path):
    data = yaml.safe_load(path.read_text())
    parent = data.pop("_extends", None)
    return merge(config(path.parent / parent), data) if parent else data


def components(metrics):
    wait = metrics["wait_mean"] / 10
    cv = metrics["cv_mean"]
    total = metrics["composite_mean"]
    return {"wait": wait, "resource": total - wait - cv,
            "snapshot_cv": cv, "total": total}


def main():
    rows = {}
    configs = {}
    for method in METHODS:
        configs[method] = config(PROJECT / "configs_ablation" / (method + ".yaml"))
        rows[method] = []
        for seed in SEEDS:
            payload = json.loads((RESULTS / method / f"seed{seed}.json").read_text())
            # Missing/mismatched evaluations would invalidate the paired decomposition.
            assert payload["training_seed"] == seed
            assert payload["n_test_episodes"] == 50
            assert payload["selection"]["checkpoint_episode"] == 299
            if method != METHODS[0]:
                base = rows[METHODS[0]][SEEDS.index(seed)]["payload"]
                assert payload["fleet_schedule"] == base["fleet_schedule"]
                assert payload["test_episode_seeds"] == base["test_episode_seeds"]
            parts = components(payload["metrics"])
            assert parts["resource"] >= 0
            rows[method].append({"seed": seed, "payload": payload, "cost": parts})

    matched = {}
    for key in ("env", "lower"):
        matched[key] = all(configs[m][key] == configs[METHODS[0]][key] for m in METHODS)
    assert all(matched.values()), matched
    output = {"evaluation": "existing held-out final policies; no new runs",
              "training_seeds": SEEDS, "episodes_per_seed": 50,
              "matched_config_sections": matched, "methods": {}, "contrasts": {}}
    for method, records in rows.items():
        cfg = configs[method]
        output["methods"][method] = {
            "holding_grid_seconds": cfg["lower"]["action_bins"],
            "coupling_mode": cfg["coupling"]["coupling_mode"],
            "upper_warmup_episodes": cfg["coupling"]["upper_warmup_eps"],
            "cost_mean": {k: mean(r["cost"][k] for r in records) for k in records[0]["cost"]},
            "per_seed": [{"seed": r["seed"], **r["cost"]} for r in records],
        }
    for method in METHODS[1:]:
        differences = [{k: a["cost"][k] - b["cost"][k] for k in a["cost"]}
                       for a, b in zip(rows[METHODS[0]], rows[method])]
        output["contrasts"][method] = {
            k: {"mean": mean(d[k] for d in differences),
                "seed_sd": stdev(d[k] for d in differences),
                "negative_seeds": sum(d[k] < 0 for d in differences)}
            for k in differences[0]
        }
    destination = ROOT / "manuscript/figures/source_data/transit_cost_components.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"config": matched, "contrasts": output["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
