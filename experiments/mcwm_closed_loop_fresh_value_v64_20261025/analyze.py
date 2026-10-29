"""Full-day fresh-value control evidence; confirmation retains all five seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control_analyze import (
    COST_KEYS, _overlap, _runtime, aggregate, compare_prefix, difference, read, read_metrics, window)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.analyze import _run_checks

ARMS = ("fresh", "no_fresh")
PERIODS = (("full_day", (0,24000)), ("early", (9000,12000)), ("post_budget", (18300,24000)))


def statistics(cells, protocol, *, seeds):
    def group(selected, period="full_day"):
        result = aggregate(selected, "no_fresh", period)
        if period == "full_day":
            values = [row["comparisons"]["no_fresh"][period] for row in selected
                      if row["comparisons"]["no_fresh"][period]["status"] == "complete"]
            result["win_fraction"] = result["wins"]/len(values) if values else None
            result["loss_fraction"] = result["losses"]/len(values) if values else None
            for outcome in ("win", "loss"):
                deltas = [row["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"]
                          for row in values if row["outcome"] == outcome]
                result[outcome+"_mean_difference"] = sum(deltas)/len(deltas) if deltas else None
        return result
    return dict(overall=group(cells),
        by_scenario={scenario:group([row for row in cells if row["scenario"] == scenario])
                     for scenario in protocol["scenarios"]},
        by_identity={str(seed):group([row for row in cells if row["identity"] == seed]) for seed in seeds},
        early=group(cells,"early"), post_budget=group(cells,"post_budget"))


def cost_gate(aggregates):
    return (aggregates["overall"]["cells"] > 0
        and aggregates["overall"]["mean_cost_difference"] < 0
        and aggregates["by_scenario"]["stable"]["mean_cost_difference"] is not None
        and aggregates["by_scenario"]["stable"]["mean_cost_difference"] <= 0
        and aggregates["by_scenario"]["local_speed_drop"]["mean_cost_difference"] is not None
        and aggregates["by_scenario"]["local_speed_drop"]["mean_cost_difference"] < 0)


def target_checks(summary, arm):
    """Native target proof is separate from the realized passenger-cost gate."""
    audit = summary.get("target_audit", {})
    fresh = arm == "fresh"
    return dict(target_audit_present=bool(audit),
        original_full_mixed_target_evaluated=audit.get("parent_mixed_target_calls") == 456
            and audit.get("parent_mixed_target_rows") == 456*256
            and audit.get("mixed_target_generated_before_replacement") is True,
        only_declared_fresh_rows_replaced=audit.get("fresh_rows_start") == 64
            and audit.get("fresh_rows_stop") == 128
            and audit.get("fresh_value_replacement_calls") == (456 if fresh else 0)
            and audit.get("fresh_value_replacement_rows") == (456*64 if fresh else 0),
        no_fresh_target_path_unchanged=audit.get("no_fresh_unchanged_returns") == (0 if fresh else 456),
        original_four_sampler_draws=audit.get("original_sampler_draws") == 4,
        native_target_scope_audit=summary.get("integrity", {}).get("checks", {}).get("fresh_value_target_scope") is True)


def analyze(root, stage):
    root = Path(root)
    protocol = read(root/"protocol.json")
    seeds = protocol[f"{stage}_seeds"]
    cells, missing, failures, failure_work, checks = [], [], [], [], {}
    for seed in seeds:
        for scenario in protocol["scenarios"]:
            key = f"seed{seed}/{scenario}"
            summaries, points, directories, configs = {}, {}, {}, {}
            for arm in ARMS:
                directory = root/"runs"/"fresh_value"/stage/f"seed{seed}"/scenario/arm
                summary = read(directory/"summary.json")
                if summary is None:
                    failure = read(directory/"failure.json")
                    (failures if failure else missing).append(f"{key}/{arm}")
                    if failure:
                        failure_work.append(dict(cell=f"{key}/{arm}", **{field:failure.get(field)
                            for field in ("type","native_time","consumed_budget","wall_seconds")}))
                    continue
                summaries[arm], directories[arm] = summary, directory
                configs[arm] = read(directory/"protocol.json") or {}
                run_checks = _run_checks(summary, arm)
                run_checks.update(correct_cell=summary.get("cell", {}).get("seed") == seed
                        and summary.get("cell", {}).get("scenario") == scenario
                        and summary.get("cell", {}).get("arm") == arm
                        and summary.get("cell", {}).get("candidate") == "fresh_value",
                    exact_external_seed=summary.get("cell", {}).get("external_seed") == seed+protocol["external_seed_offset"],
                    native_all_checks=bool(summary.get("integrity", {}).get("checks")) and all(
                        value is True for value in summary.get("integrity", {}).get("checks", {}).values()),
                    recorded_learner_config=isinstance(configs[arm].get("policy_config"),dict))
                run_checks.update(target_checks(summary,arm))
                points[arm] = read_metrics(directory,summary)
                final = {row["time_seconds"]:row for row in points[arm]}.get(24000,{})
                run_checks["final_metrics_match_ledger"] = all(final.get(field) == summary.get("cost",{}).get(field)
                    for field in ("accrued_wait_seconds","accrued_in_vehicle_seconds"))
                run_checks["primary_ledger_valid"] = window(summary,points[arm],0,24000)["status"] == "complete"
                checks.update({f"{key}/{arm}/{name}": value is True for name,value in run_checks.items()})
            if len(summaries) != len(ARMS):
                continue
            left,right = summaries["fresh"],summaries["no_fresh"]
            first = [row.get("clock_audit",{}).get("first_feedback_applied_at") for row in (left,right)]
            valid_first = all(isinstance(value,(int,float)) and math.isfinite(value) and value>0 for value in first)
            checks[f"{key}/first_feedback_exact"] = valid_first and first[0] == first[1]
            prefix = compare_prefix(directories["fresh"]/"prefix_trace.jsonl",directories["no_fresh"]/"prefix_trace.jsonl",
                                    min(first) if valid_first else 0.)
            parallel = _overlap(left,right)
            checks.update({f"{key}/prefix_exact":prefix["all_pass"],
                f"{key}/actual_parallel":parallel["positive_overlap"],
                f"{key}/initializer_exact":left.get("initializer") == right.get("initializer"),
                f"{key}/policy_config_exact":configs["fresh"].get("policy_config") == configs["no_fresh"].get("policy_config"),
                f"{key}/runtime_F_exact":_runtime(left) == _runtime(right)})
            baseline = None
            if stage == "development":
                reference = root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(
                    seed=seed,scenario=scenario)
                previous = read(reference/"summary.json")
                if previous is None:
                    missing.append(f"{key}/original_no_fresh_reference")
                else:
                    matches = {field:right.get("cost",{}).get(field) == previous.get("cost",{}).get(field)
                               for field in COST_KEYS}
                    baseline = dict(reference=str(reference), reused_as_control=False, fields=matches)
                    checks[f"{key}/no_fresh_matches_original_ledger"] = all(matches.values())
                    checks[f"{key}/reference_complete"] = previous.get("status") == "complete" and previous.get("integrity",{}).get("all_pass") is True
            comparison = {name:difference(window(left,points["fresh"],*bounds),window(right,points["no_fresh"],*bounds))
                          for name,bounds in PERIODS}
            cells.append(dict(identity=seed,scenario=scenario,comparisons=dict(no_fresh=comparison),
                cost_ledgers={arm:value.get("cost") for arm,value in summaries.items()},
                initializer=left.get("initializer"),pre_feedback_prefix=prefix,actual_parallel=parallel,
                original_no_fresh_exactness=baseline,
                target_audit={arm:value.get("target_audit") for arm,value in summaries.items()}))
    expected = len(seeds)*len(protocol["scenarios"])
    native_intact = not missing and not failures and len(cells)==expected and bool(checks) and all(checks.values())
    aggregates = statistics(cells,protocol,seeds=seeds)
    stage_pass = native_intact and cost_gate(aggregates)
    retained = None
    retained_intact, retained_pass = stage != "confirmation", stage != "confirmation"
    if stage == "confirmation":
        development = read(root/"results"/"fresh_value"/"development.json")
        if development is None:
            missing.append("retained_development_result")
        else:
            development_cells = development.get("cells",[])
            expected_pairs = {(seed,scene) for seed in protocol["development_seeds"] for scene in protocol["scenarios"]}
            actual_pairs = [(row["identity"],row["scenario"]) for row in development_cells]
            retained_intact = (development.get("candidate")=="fresh_value" and development.get("stage")=="development"
                and development.get("status")=="complete" and development.get("integrity",{}).get("all_pass") is True
                and len(actual_pairs)==len(expected_pairs) and set(actual_pairs)==expected_pairs
                and development.get("decision",{}).get("advance_to_confirmation") is True)
            checks["retained_development_exact_cohort_and_admission"] = retained_intact
            combined_cells = development_cells+cells
            all_seeds = protocol["development_seeds"]+protocol["confirmation_seeds"]
            combined = statistics(combined_cells,protocol,seeds=all_seeds)
            retained_pass = retained_intact and native_intact and len(combined_cells)==10 and cost_gate(combined)
            retained = dict(cells=len(combined_cells),expected_cells=10,identities=all_seeds,
                development_source="results/fresh_value/development.json",aggregates=combined,
                integrity_pass=retained_intact and native_intact,scientific_gate_pass=bool(retained_pass))
    intact = native_intact and not missing and retained_intact and all(checks.values())
    passed = intact and stage_pass and retained_pass
    return dict(status="complete" if intact else "integrity_failed" if failures or any(
        value is not True for value in checks.values()) else "pending",candidate="fresh_value",stage=stage,
        complete=intact,expected_cells=expected,available_cells=len(cells),
        integrity=dict(all_pass=intact,checked_conditions=len(checks),failed_checks=[key for key,value in checks.items() if value is not True]),
        missing=missing,failures=failures,failed_run_work=failure_work,cells=cells,
        aggregates=dict(no_fresh=aggregates),retained_all_five_seeds=retained,
        decision=dict(advance_to_confirmation=bool(passed and stage=="development"),
            confirmed_control_improvement=bool(passed and stage=="confirmation"),
            ordinary_dripping_goal_ready_for_root_review=bool(passed and stage=="confirmation"),
            ordinary_dripping_goal_achieved=False,stage_cost_gate_pass=bool(stage_pass),
            retained_all_five_seeds_gate_pass=bool(retained_pass) if stage=="confirmation" else None,
            scientific_gate_pass=bool(passed),reason="fixed_cost_gates_pass" if passed else
                "incomplete_or_integrity_failed" if not intact else "candidate_closed_cost_gate_failed"),
        metric="Full-day(2W+R)/all departed passengers, including unfinished-trip time; negative difference is beneficial",
        interpretation="Original no_fresh is rerun, not substituted. Confirmation must pass independently and with all four retained development cells; only root can complete the goal.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("development","confirmation"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    value=analyze(HERE,args.stage)
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    if len(raw.encode())>48000:
        raise ValueError("Actual-control summary exceeds48000 bytes")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as handle:handle.write(raw+"\n")
    print(json.dumps({key:value[key] for key in ("status","stage","decision")},ensure_ascii=False))


if __name__=="__main__":
    main()
