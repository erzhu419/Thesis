"""Use corrected affected arms and unchanged controls without replacing old records."""

import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "enhancement_v1"
sys.path.insert(0, str(OLD))
from collect import P, interval, summarize_kind, write

ARMS = ("mixed_nomodel", "fixed_direct", "fixed_residual", "calibrated_selection")


def folder_for(arm, seed):
    root = HERE / "runs" if arm in ARMS else OLD / "runs/mc"
    return root / f"{arm}_seed{seed}/results"


def validate_protocol():
    assert P["deployment_hold_scale"] == .25
    for arm in P["mc_arms"]:
        for seed in P["training_seeds"]:
            folder = folder_for(arm, seed)
            for case in range(10):
                path = folder / f"eval_{case:02d}.json"
                if not path.exists():
                    continue
                row = json.loads(path.read_text())
                block = P["evaluation_blocks"][case % 5]
                assert (row["demand_seed"], row["sumo_seed"]) == (
                    block["demand_seed"], block["sumo_seed"]), path
                assert row["demand_scale"] == P["evaluation_demand_scales"][case // 5], path
                assert row["horizon_seconds"] == 24000, path
                assert row["sumo_version"][1] == "SUMO 1.25.0", path
            path = folder / "ratio_final.json"
            if arm in ARMS and path.exists():
                row = json.loads(path.read_text())
                assert row["validation_available"] and row["total_updates"] == 16144, path
                assert row["source_train_days"] == [0, 2, 3, 4], path
                assert row["source_validation_days"] == [1], path
                assert row["offline_holdout_blocks"] == [1202], path


def ratio_summary():
    protocol = json.loads((HERE.parent / "review_round3/protocol.json").read_text())["ratio"]
    names = ("joint_auc", "state_action_auc", "joint_brier", "state_action_brier",
             "weight_mean", "ess_fraction", "raw_ess_fraction", "high_clip_fraction", "low_clip_fraction")
    cells, missing = [], []
    for seed in P["training_seeds"]:
        rows = []
        for case in range(3):
            path = HERE / f"runs/mixed_nomodel_seed{seed}/heldout/case{case}/results/ratio_final.json"
            if not path.exists():
                missing.append(str(path.relative_to(HERE)))
                continue
            row = json.loads(path.read_text())
            assert row["validation_available"] and row["new_gradient_updates"] == 0, path
            assert row["source"]["seed"] == protocol["validation_seeds"][case], path
            assert row["offline_holdout_blocks"] == [1202], path
            rows.append(row)
        if len(rows) == 3:
            cells.append(dict(seed=seed, **{
                name: statistics.mean(r["metrics"]["ratio/validation_"+name] for r in rows)
                for name in names}))
    if missing:
        return dict(complete=False, missing=missing)
    return dict(complete=True, cells=cells,
                summary={name: interval([r[name] for r in cells]) for name in names})


if __name__ == "__main__":
    validate_protocol()
    historical = json.loads((OLD / "summary.json").read_text())
    current = summarize_kind("mc", folder_for)
    summary = dict(historical, mc=current)
    summary["source_split_revision"] = "day_id_before_replay_append"
    summary["ratio"] = ratio_summary()
    write(HERE / "summary.json", summary)
    print(json.dumps({k: v for k, v in current.items()
                      if k in ("complete", "available_tests", "expected_tests")}), flush=True)
    if current["complete"]:
        changes = {scene: {arm: {
            metric: row["mean"]-historical["mc"]["results"][scene]["means"][arm][metric]["mean"]
            for metric, row in metrics.items()}
            for arm, metrics in result["means"].items() if arm in ARMS}
            for scene, result in current["results"].items()}
        write(HERE / "historical_differences.json", changes)
