"""Analyze new dual and existing holding records on the server only."""

from collections import defaultdict
import json
import math
from statistics import mean

from common import HERE, P, folder, holding_folder, label, placements, write


def read(path):
    return json.loads(path.read_text())


def values(row):
    count = row["generated_passengers"]
    windows = row.get("windows", [])
    windows = windows.values() if isinstance(windows, dict) else windows
    action_count = row.get("decisions", sum(w["decisions"] for w in windows))
    return dict(cost_seconds=row["generalized_seconds_per_passenger"],
        wait_seconds=row["wait_passenger_seconds"]/count,
        ride_seconds=row["ride_passenger_seconds"]/count,
        plan_error_seconds=row["headway_plan_mae_seconds"], imbalance_seconds=row["headway_imbalance_seconds"],
        completion_percent=100*row["completion_rate"],
        recorded_reward_per_decision=row.get("native_reward", row.get("reward", 0.))/max(action_count, 1),
        mean_speed_ratio=row.get("mean_speed_ratio", 1.))


def main():
    ready, finished, progress, raw = [], [], [], []
    expected_train, expected_eval = 0, 0
    for smoke in (True, False):
        items = list(placements(smoke))
        policy_count, eval_count = defaultdict(int), defaultdict(int)
        for item in items:
            out = folder(**item)
            policy_path = out / "results/ready.json"
            if policy_path.exists():
                record = read(policy_path)
                if record["episodes"] != (1 if smoke else P["episodes"]) or record["action_dim"] != 2:
                    raise ValueError("Incomplete or wrong-action endpoint: "+str(policy_path))
                ready.append(label(item))
                finished.append(label(item)+"/train")
                policy_count[item["kind"]] += 1
            progress_path = out / ("progress.json" if item["kind"] == "resac" else "results/progress.json")
            if progress_path.exists():
                record = read(progress_path)
                progress.append(dict(item, episode=record["episode"]+(item["kind"]=="resac"),
                    elapsed_seconds=record.get("elapsed_seconds", record.get("duration_s"))))
            cases = [0] if smoke else range(9)
            if not smoke:
                expected_train += 1
                expected_eval += len(cases)
            for case in cases:
                path = out / "results" / f"eval_{case:02d}.json"
                if not path.exists():
                    continue
                row = read(path)
                horizon = 3600 if smoke else P["horizon_seconds"][item["kind"]]
                if row["horizon_seconds"] != horizon or row["action_dim"] != 2 or not all(math.isfinite(v) for v in values(row).values()):
                    raise ValueError("Invalid dual evaluation: "+str(path))
                eval_count[item["kind"]] += 1
                finished.append(label(item)+f"/eval/{case}")
                if not smoke:
                    raw.append(dict(item, case=case, scene=row["scene"], evaluation_seed=row["evaluation_seed"], **values(row)))
            if smoke:
                continue
            parent = holding_folder(item["kind"], item["objective"], item["arm"], item["seed"])
            if read(parent / "results/ready.json")["episodes"] != P["episodes"]:
                raise ValueError("Holding endpoint budget differs")
            for case in range(9):
                row = read(parent / "results" / f"eval_{case:02d}.json")
                if row["evaluation_seed"] != P["evaluation_seeds"][item["kind"]][case % 3] or row["horizon_seconds"] != P["horizon_seconds"][item["kind"]]:
                    raise ValueError("Holding endpoint uses different test days")
                raw.append(dict(item, mode="holding", case=case, scene=row["scene"], evaluation_seed=row["evaluation_seed"], **values(row)))
        for kind in P["arms"]:
            contract = HERE / "contracts" / (kind+".json")
            if contract.exists() and read(contract)["passed"]:
                finished.append("contracts/"+kind)
                if smoke and policy_count[kind] == 4 and eval_count[kind] == 4:
                    ready.append("smoke-pass/"+kind)
    rule_rows = []
    for kind in P["arms"]:
        for case in range(9):
            path = folder(kind, "dual", "headway", "fixed_speed", 0) / "results" / f"eval_{case:02d}.json"
            if path.exists():
                row = read(path)
                if row["horizon_seconds"] != P["horizon_seconds"][kind]:
                    raise ValueError("Incomplete fixed-speed rule")
                rule_rows.append(dict(kind=kind, case=case, scene=row["scene"], **values(row)))
                finished.append(f"formal/{kind}/dual/headway/fixed_speed/0/eval/{case}")
    index = {(r["kind"], r["mode"], r["objective"], r["arm"], r["seed"], r["case"]):r for r in raw}
    contrasts = defaultdict(list)
    for row in raw:
        peers = []
        if row["mode"] == "dual":
            peers.append(("holding", row["arm"], "action_extension"))
        if row["arm"] == P["arms"][row["kind"]][0]:
            peers.append((row["mode"], P["arms"][row["kind"]][1], "algorithm_control"))
        for mode, arm, contrast in peers:
            peer = index.get((row["kind"], mode, row["objective"], arm, row["seed"], row["case"]))
            if peer is None:
                continue
            if (row["scene"], row["evaluation_seed"]) != (peer["scene"], peer["evaluation_seed"]):
                raise ValueError("Nonmatching paired evaluation")
            key = (contrast, row["kind"], row["mode"], row["objective"], row["arm"], mode, arm, row["scene"], row["seed"])
            contrasts[key].append(dict(cost_delta_seconds=row["cost_seconds"]-peer["cost_seconds"],
                reference_cost_seconds=peer["cost_seconds"],
                recorded_reward_delta=row["recorded_reward_per_decision"]-peer["recorded_reward_per_decision"],
                plan_error_delta_seconds=row["plan_error_seconds"]-peer["plan_error_seconds"],
                reference_plan_error_seconds=peer["plan_error_seconds"],
                completion_delta_pp=row["completion_percent"]-peer["completion_percent"]))
    paired = []
    for key, rows in contrasts.items():
        if len(rows) == 3:
            paired.append(dict(zip(("contrast", "kind", "mode", "objective", "arm", "reference_mode", "reference_arm", "scene", "seed"), key),
                **{k:mean(r[k] for r in rows) for k in rows[0]},
                cost_change_pct=100*mean(r["cost_delta_seconds"] for r in rows)/mean(r["reference_cost_seconds"] for r in rows),
                plan_error_change_pct=100*mean(r["plan_error_delta_seconds"] for r in rows)/mean(r["reference_plan_error_seconds"] for r in rows),
                paired_cases=3))
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["kind"], row["mode"], row["objective"], row["arm"], row["scene"], row["seed"])].append(row)
    means = [dict(zip(("kind", "mode", "objective", "arm", "scene", "seed"), key), cases=len(rows),
        **{field:mean(r[field] for r in rows) for field in values_template()}) for key,rows in grouped.items()]
    report = dict(name=P["name"], analysis_location="server", ready=sorted(set(ready)), completed_labels=sorted(set(finished)),
        expected_training=expected_train, expected_evaluation=expected_eval+27,
        training_complete=sum(x.startswith("formal/") and x.endswith("/train") for x in finished),
        dual_evaluations=sum(r["mode"]=="dual" for r in raw), fixed_speed_evaluations=len(rule_rows),
        reused_holding_evaluations=sum(r["mode"]=="holding" for r in raw), progress=progress,
        seed_means=means, paired_contrasts=paired, fixed_speed_rules=rule_rows,
        complete=sum(r["mode"]=="dual" for r in raw)==expected_eval and len(rule_rows)==27
            and sum(x.startswith("formal/") and x.endswith("/train") for x in finished)==expected_train)
    write(HERE / "summary.json", report)
    print({k:report[k] for k in ("training_complete", "dual_evaluations", "fixed_speed_evaluations", "complete")}, flush=True)


def values_template():
    return ("cost_seconds", "wait_seconds", "ride_seconds", "plan_error_seconds", "imbalance_seconds", "completion_percent", "recorded_reward_per_decision", "mean_speed_ratio")


if __name__ == "__main__":
    main()
