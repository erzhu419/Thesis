"""Fixed demand-cohort accounting from existing passenger logs; no native imports."""
import argparse
import json
import math
from pathlib import Path


V113 = "experiments/mcwm_behavior_replay_v113_20261029"
V114 = "experiments/mcwm_same_state_holding_v114_20261029"
SCENES = ("stable", "local_speed_drop")
N = 15079


def read(path):
    return json.loads(path.read_text())


def people(path):
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            pid = row["passenger_id"]
            if pid in result:
                raise ValueError("Duplicate passenger identity")
            result[pid] = {key: row[key] for key in ("wait", "ride", "completed")}
    return result


def group(treatment, reference, ids):
    changes = [(pid, treatment[pid]["wait"] - reference[pid]["wait"],
                treatment[pid]["ride"] - reference[pid]["ride"]) for pid in ids]
    waits = [w for _, w, _ in changes]
    rides = [r for _, _, r in changes]
    costs = [2 * w + r for _, w, r in changes]
    def direction(values):
        return dict(increase=sum(v > 0 for v in values),
                    decrease=sum(v < 0 for v in values), equal=sum(v == 0 for v in values))
    return dict(passengers=len(changes), delta_wait=sum(waits), delta_ride=sum(rides),
        delta_cost=sum(costs), cost_per_total_passenger=sum(costs) / N,
        people_by_delta=dict(wait=direction(waits), ride=direction(rides), cost=direction(costs)),
        completed=dict(reference=sum(reference[p]["completed"] for p in ids),
            treatment=sum(treatment[p]["completed"] for p in ids),
            gained=sum(treatment[p]["completed"] and not reference[p]["completed"] for p in ids),
            lost=sum(reference[p]["completed"] and not treatment[p]["completed"] for p in ids)),
        gross_cost_increase=sum(c for c in costs if c > 0),
        gross_cost_decrease=sum(c for c in costs if c < 0))


def analyze(root):
    mains = {"v113": read(root / V113 / "review/summary.json"),
             "v114": read(root / V114 / "summary.json")}
    if not all(row["integrity"]["all_pass"] for row in mains.values()):
        raise ValueError("Original aggregate integrity is not complete")
    result, panel_ids = {}, None
    files = []
    for version, source, treatment, reference in (
            ("v113", V113, "fresh", "no_fresh"), ("v114", V114, "low", "high")):
        for scene in SCENES:
            parent = root / source / "runs" / scene
            if version == "v114":
                parent /= "branches"
            paths = {arm: parent / arm for arm in (treatment, reference)}
            rows = {arm: people(path / "passengers.jsonl") for arm, path in paths.items()}
            ids = rows[treatment].keys()
            if len(ids) != N or ids != rows[reference].keys():
                raise ValueError("The paired 15079 passenger identities differ")
            if panel_ids is None:
                panel_ids = set(ids)
            if set(ids) != panel_ids:
                raise ValueError("Passenger identities differ across the fixed comparisons")
            groups = {"planned_transfer": sorted(p for p in ids if p.startswith("transfer_")),
                      "other_demand": sorted(p for p in ids if not p.startswith("transfer_"))}
            if [len(groups[g]) for g in groups] != [868, 14211]:
                raise ValueError("Observed demand groups differ from the original demand inventory")
            groups["all"] = sorted(ids)
            summaries = {arm: read(path / "summary.json")["periods"]["full_day"]
                         for arm, path in paths.items()}
            for arm in paths:
                for metric in ("wait", "ride"):
                    if sum(r[metric] for r in rows[arm].values()) != summaries[arm][metric]:
                        raise ValueError("Passenger totals do not equal the original arm ledger")
            output = {name: group(rows[treatment], rows[reference], members)
                      for name, members in groups.items()}
            for metric in ("wait", "ride", "cost"):
                key = "delta_" + metric
                value = output["all"][key]
                if output["planned_transfer"][key] + output["other_demand"][key] != value:
                    raise ValueError("Demand-cohort decomposition does not reconcile")
                if value != summaries[treatment][metric] - summaries[reference][metric]:
                    raise ValueError("Passenger delta does not equal the original arm ledgers")
                if version == "v113":
                    expected = mains[version]["scenes"][scene]["fresh_minus_matched"]["full_day"][metric]
                    equal = value == expected
                else:
                    expected = mains[version]["cells"][scene]["comparisons"]["full_day"]["low_minus_high"][metric + "_per_passenger"]
                    equal = math.isclose(value / N, expected, rel_tol=0, abs_tol=1e-9)
                if not equal:
                    raise ValueError("Passenger delta differs from the existing main summary")
            ranked = sorted(ids, key=lambda p: (abs(2 * (rows[treatment][p]["wait"] -
                rows[reference][p]["wait"]) + rows[treatment][p]["ride"] - rows[reference][p]["ride"]), p), reverse=True)[:3]
            top = []
            for pid in ranked:
                w = rows[treatment][pid]["wait"] - rows[reference][pid]["wait"]
                r = rows[treatment][pid]["ride"] - rows[reference][pid]["ride"]
                top.append(dict(passenger_id=pid, delta_wait=w, delta_ride=r, delta_cost=2*w+r))
            result[version + "/" + scene] = dict(comparison=treatment + " minus " + reference,
                groups=output, top_three_absolute_cost_changes=top,
                exact_passenger_and_arm_ledger_reconciliation=True,
                existing_main_summary_reconciliation=True)
            files.extend(str(path.relative_to(root) / "passengers.jsonl") for path in paths.values())
    return dict(status="complete", all_pass=True, new_native_seconds=0, new_training_updates=0,
        metric="Delta cost = 2 Delta wait + Delta ride; raw passenger-seconds; normalization /15079",
        cohort_definition="Pre-intervention planned transfer demand, not realized transfer completion",
        marker_evidence=dict(passenger_objects="experiments/sumo_eval/assets/SUMO_ruiguang/online_control/initialize_obj/save_obj_passenger.add.xml",
            transfer_route="experiments/sumo_eval/assets/SUMO_ruiguang/e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml",
            locally_verified_object_count=15079, marked_with_nonempty_transfer_stop=868,
            unmarked_with_empty_transfer_stop=14211, marked_route_people_with_two_rides=868),
        comparisons=result, passenger_logs_read=files,
        interpretation="v113 describes the original fresh effect; v114 describes two selected holding interventions. Their mechanisms cannot be substituted for one another. Cohort wait includes both initial and transfer waiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.root)
    output = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(output.encode()) > 12000:
        raise ValueError("Small readout exceeds the frozen 12000-byte limit")
    print(output)
