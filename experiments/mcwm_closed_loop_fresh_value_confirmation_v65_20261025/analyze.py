"""Frozen v64 confirmation: full-day gates and separate stable-tail assessment."""
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
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.analyze import target_checks

ARMS = ("fresh", "no_fresh")
PERIODS = (("full_day",(0,24000)),("early",(9000,12000)),("post_budget",(18300,24000)))


def cost_view(value):
    if value.get("status") != "complete":
        return dict(status=value.get("status","missing"))
    return dict(wait=value["accrued_wait_seconds"],ride=value["accrued_in_vehicle_seconds"],
        cost=value["generalized_passenger_seconds"],departed=value["departed_passengers"],
        cost_per_passenger=value["generalized_seconds_per_departed_passenger"])


def populate_periods(cell,summaries,points):
    cell["period_arms"],cell["comparisons"] = {},dict(no_fresh={})
    for name,bounds in PERIODS:
        arms = {arm:window(summaries[arm],points[arm],*bounds) for arm in ARMS}
        cell["period_arms"][name] = arms
        cell["comparisons"]["no_fresh"][name] = difference(arms["fresh"],arms["no_fresh"])


def group(cells,period,*,expected_cells):
    result = aggregate(cells,"no_fresh",period)
    valid = [cell for cell in cells if cell["comparisons"]["no_fresh"][period]["status"] == "complete"]
    count = len(valid)
    deltas = [cell["comparisons"]["no_fresh"][period]["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"] for cell in valid]
    arms = {}
    for arm in ARMS:
        values = [cell["period_arms"][period][arm] for cell in valid]
        arms[arm] = dict(raw={label:sum(row[key] for row in values) if count else None
            for label,key in (("wait","accrued_wait_seconds"),("ride","accrued_in_vehicle_seconds"),("cost","generalized_passenger_seconds"))},
            mean_per_passenger={label:sum(row[key]/row["departed_passengers"] for row in values)/count if count else None
                for label,key in (("wait","accrued_wait_seconds"),("ride","accrued_in_vehicle_seconds"),("cost","generalized_passenger_seconds"))})
    wins,losses = [value for value in deltas if value<0],[value for value in deltas if value>0]
    return dict(expected_cells=expected_cells,complete_cells=count,missing_cells=expected_cells-count,
        mean_cost_difference=result["mean_cost_difference"],percent_change=result.get("percent_change"),
        mean_wait_difference=result.get("mean_wait_seconds_per_passenger_difference"),
        mean_ride_difference=result.get("mean_ride_seconds_per_passenger_difference"),arms=arms,
        wins=len(wins),losses=len(losses),ties=count-len(wins)-len(losses),
        win_fraction=len(wins)/count if count else None,loss_fraction=len(losses)/count if count else None,
        winning_mean_reduction=-sum(wins)/len(wins) if wins else None,
        losing_mean_increase=sum(losses)/len(losses) if losses else None)


def cohort(cells,seeds,scenarios,*,audited):
    periods = {name:dict(overall=group(cells,name,expected_cells=2*len(seeds)),by_scenario={scene:group(
        [cell for cell in cells if cell["scenario"]==scene],name,expected_cells=len(seeds)) for scene in scenarios}) for name,_ in PERIODS}
    full = periods["full_day"]
    complete = len(cells)==2*len(seeds) and full["overall"]["complete_cells"]==2*len(seeds)
    conditions = dict(overall_lower=complete and full["overall"]["mean_cost_difference"]<0,
        stable_not_higher=complete and full["by_scenario"]["stable"]["mean_cost_difference"] is not None
            and full["by_scenario"]["stable"]["mean_cost_difference"]<=0,
        shock_lower=complete and full["by_scenario"]["local_speed_drop"]["mean_cost_difference"] is not None
            and full["by_scenario"]["local_speed_drop"]["mean_cost_difference"]<0)
    return dict(seeds=list(seeds),cells=len(cells),available_cells=len(cells),expected_cells=2*len(seeds),periods=periods,
        gate=dict(all_pass=bool(audited and complete and all(conditions.values())),
                  native_and_source_audits=bool(audited),conditions=conditions))


def compact_period(cell,name):
    comparison = cell["comparisons"]["no_fresh"][name]
    arms = {arm:cost_view(value) for arm,value in cell["period_arms"][name].items()}
    if comparison["status"] != "complete":
        return dict(status=comparison["status"],arms=arms,delta=None,percent_change=None,outcome=None)
    return dict(status="complete",arms=arms,delta=dict(
        wait_per_passenger=comparison["wait_seconds_per_passenger_difference"],
        ride_per_passenger=comparison["ride_seconds_per_passenger_difference"],
        cost_per_passenger=comparison["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"],
        raw_cost=comparison["candidate_minus_baseline"]["generalized_passenger_seconds"]),
        percent_change=comparison["percent_change"],outcome=comparison["outcome"])


def tail_assessment(value):
    group = value["periods"]["post_budget"]["by_scenario"]["stable"]
    complete = group["expected_cells"]>0 and group["complete_cells"]==group["expected_cells"]
    harm = group["mean_cost_difference"]>0 if complete else None
    return dict(complete=complete,mean_cost_difference=group["mean_cost_difference"],
        losing_seeds=group["losses"],winning_seeds=group["wins"],expected_seeds=group["expected_cells"],
        stable_tail_harm_persists=harm,
        full_day_pass_with_stable_tail_harm=bool(value["gate"]["all_pass"] and harm) if complete else None)


def analyze(root=HERE):
    root = Path(root);protocol = read(root/"protocol.json")
    seeds,scenarios = protocol["confirmation_seeds"],protocol["scenarios"]
    cells,evidence,checks,missing,failures,failed_work = [],[],{},[],[],[]
    for seed in seeds:
        for scene in scenarios:
            key=f"seed{seed}/{scene}";runs,points,configs,directories = {},{},{},{}
            for arm in ARMS:
                directory=root/"runs/fresh_value/confirmation"/f"seed{seed}"/scene/arm
                summary=read(directory/"summary.json")
                if summary is None:
                    failure=read(directory/"failure.json")
                    (failures if failure else missing).append(key+"/"+arm)
                    if failure:failed_work.append(dict(cell=key+"/"+arm,**{field:failure.get(field)
                        for field in ("type","native_time","consumed_budget","wall_seconds")}))
                    continue
                runs[arm],directories[arm]=summary,directory
                configs[arm]=read(directory/"protocol.json") or {}
                points[arm]=read_metrics(directory,summary)
                audit=_run_checks(summary,arm);audit.update(target_checks(summary,arm))
                audit.update(correct_cell=summary.get("cell",{}).get("seed")==seed
                    and summary.get("cell",{}).get("scenario")==scene and summary.get("cell",{}).get("arm")==arm
                    and summary.get("cell",{}).get("candidate")=="fresh_value",
                    external_seed=summary.get("cell",{}).get("external_seed")==seed+protocol["external_seed_offset"],
                    all_native_checks=bool(summary.get("integrity",{}).get("checks")) and all(
                        value is True for value in summary.get("integrity",{}).get("checks",{}).values()),
                    learner_config_recorded=isinstance(configs[arm].get("policy_config"),dict),
                    primary_ledger_valid=window(summary,points[arm],0,24000)["status"]=="complete")
                final={row["time_seconds"]:row for row in points[arm]}.get(24000,{})
                audit["final_metrics_match_ledger"]=all(final.get(field)==summary.get("cost",{}).get(field)
                    for field in ("accrued_wait_seconds","accrued_in_vehicle_seconds"))
                checks.update({key+"/"+arm+"/"+name:value is True for name,value in audit.items()})
            if len(runs)!=2:continue
            left,right=runs["fresh"],runs["no_fresh"]
            first=[value.get("clock_audit",{}).get("first_feedback_applied_at") for value in (left,right)]
            valid=all(isinstance(value,(int,float)) and math.isfinite(value) and value>0 for value in first)
            prefix=compare_prefix(directories["fresh"]/"prefix_trace.jsonl",directories["no_fresh"]/"prefix_trace.jsonl",min(first) if valid else 0.)
            parallel=_overlap(left,right)
            checks.update({key+"/first_feedback_exact":valid and first[0]==first[1],
                key+"/prefix_exact":prefix["all_pass"],key+"/actual_parallel":parallel["positive_overlap"],
                key+"/initializer_exact":left.get("initializer")==right.get("initializer"),
                key+"/policy_config_exact":configs["fresh"].get("policy_config")==configs["no_fresh"].get("policy_config"),
                key+"/runtime_F_exact":_runtime(left)==_runtime(right)})
            reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(seed=seed,scenario=scene)
            original=read(reference/"summary.json");matches=None
            if original is None:missing.append(key+"/v63_no_fresh_reference")
            else:
                matches={field:right.get("cost",{}).get(field)==original.get("cost",{}).get(field) for field in COST_KEYS}
                checks[key+"/no_fresh_v63_exact"]=all(matches.values())
                checks[key+"/v63_reference_complete"]=original.get("status")=="complete" and original.get("integrity",{}).get("all_pass") is True
            cell=dict(identity=seed,scenario=scene,cohort="independent3")
            populate_periods(cell,runs,points);cells.append(cell)
            evidence.append(dict(seed=seed,scenario=scene,initializer=left.get("initializer"),prefix=prefix,
                actual_parallel=parallel,no_fresh_reference_exact=matches,
                target_scope={arm:{field:value.get("target_audit",{}).get(field) for field in
                    ("parent_mixed_target_calls","parent_mixed_target_rows","fresh_value_replacement_calls", "fresh_value_replacement_rows","no_fresh_unchanged_returns")}
                    for arm,value in runs.items()}))
    native_ok=not missing and not failures and len(cells)==2*len(seeds) and bool(checks) and all(checks.values())
    source_root=root.parent/protocol["development_directory"]
    development=read(source_root/protocol["development_relative"])
    old_cells,source_checks,source_missing=[],{},[]
    if development is None:
        missing.append("v64_development_result");source_missing.append("v64_development_result")
    else:
        expected={(seed,scene) for seed in protocol["development_seeds"] for scene in scenarios}
        source_cells=development.get("cells",[])
        membership=[(row["identity"],row["scenario"]) for row in source_cells]
        source_checks["v64_development_admitted_exact_cohort"]=(development.get("candidate")=="fresh_value"
            and development.get("stage")=="development" and development.get("status")=="complete"
            and development.get("integrity",{}).get("all_pass") is True
            and development.get("decision",{}).get("advance_to_confirmation") is True
            and len(membership)==len(expected) and set(membership)==expected)
        for source in source_cells:
            seed,scene=source["identity"],source["scenario"];runs,points={},{}
            for arm in ARMS:
                directory=source_root/"runs/fresh_value/development"/f"seed{seed}"/scene/arm
                summary=read(directory/"summary.json")
                if summary is None:
                    absent=f"v64/seed{seed}/{scene}/{arm}"
                    missing.append(absent);source_missing.append(absent);continue
                runs[arm],points[arm]=summary,read_metrics(directory,summary)
                source_checks[f"v64/seed{seed}/{scene}/{arm}/retained_ledger_exact"]=(summary.get("status")=="complete"
                    and summary.get("integrity",{}).get("all_pass") is True
                    and summary.get("cost")==source.get("cost_ledgers",{}).get(arm))
            if len(runs)!=2:continue
            cell=dict(identity=seed,scenario=scene,cohort="retained_v64_development")
            populate_periods(cell,runs,points);old_cells.append(cell)
    expected_old=2*len(protocol["development_seeds"])
    checks.update(source_checks)
    source_ok=development is not None and len(old_cells)==expected_old and not source_missing and all(source_checks.values())
    independent=cohort(cells,seeds,scenarios,audited=native_ok)
    combined=cohort(old_cells+cells,protocol["development_seeds"]+seeds,scenarios,audited=native_ok and source_ok)
    development_cohort=cohort(old_cells,protocol["development_seeds"],scenarios,audited=source_ok)
    intact=native_ok and source_ok and not missing and not failures and all(checks.values())
    confirmed=intact and independent["gate"]["all_pass"] and combined["gate"]["all_pass"]
    new_tail,all_tail,old_tail=map(tail_assessment,(independent,combined,development_cohort))
    deployment=bool(confirmed and new_tail["complete"] and new_tail["mean_cost_difference"]<=0)
    return dict(status="complete" if intact else "integrity_failed" if failures or any(
        value is not True for value in checks.values()) else "pending",candidate="fresh_value",stage="confirmation",
        complete=intact,integrity=dict(all_pass=intact,checked_conditions=len(checks),
            failed_checks=[key for key,value in checks.items() if value is not True]),
        missing=missing,failures=failures,failed_run_work=failed_work,
        data_sources=dict(new_root=str(root),new_native_days=12,retained_development_root=str(source_root),
            reused_development_days=8,no_fresh_reference_root=str(root.parent/protocol["original_baseline_directory"]),
            old_development_is_new_data=False,reference_substitutes_new_control=False),
        independent3=independent,retained5=combined,development2=development_cohort,
        cells=[dict(identity=cell["identity"],scenario=cell["scenario"],cohort=cell["cohort"],
            periods={name:compact_period(cell,name) for name,_ in PERIODS}) for cell in old_cells+cells],
        new_native_evidence=evidence,
        stable_tail_by_seed=[dict(seed=cell["identity"],cohort=cell["cohort"],**compact_period(cell,"post_budget"))
            for cell in old_cells+cells if cell["scenario"]=="stable"],
        secondary_assessment=dict(independent3=new_tail,retained5=all_tail,old_development2=old_tail,
            windows_are_secondary=True,full_day_cost_gate_unchanged=True),
        decision=dict(confirmed_control_improvement=bool(confirmed),scientific_gate_pass=bool(confirmed),
            independent3_gate_pass=independent["gate"]["all_pass"],retained5_gate_pass=combined["gate"]["all_pass"],
            stable_tail_harm_persists=new_tail["stable_tail_harm_persists"],deployment_admissible=deployment,
            ordinary_dripping_goal_ready_for_root_review=bool(confirmed),ordinary_dripping_goal_achieved=False,
            reason="full_day_cost_gates_pass" if confirmed else "incomplete_or_integrity_failed" if not intact
                else "fixed_full_day_cost_gate_failed"),
        units=dict(wait="raw waiting passenger-seconds",ride="raw riding passenger-seconds",
            cost="2*wait+ride weighted passenger-seconds",mean_per_passenger="each arm uses its full-day departed cohort; equal weight per fixed cell"),
        interpretation="Full-day confirmation and retained-cohort gates are unchanged. Stable-tail deployment assessment is separate; only root can complete the goal.")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();value=analyze(HERE)
    raw=json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    if len(raw.encode())>48000:raise ValueError("Confirmation summary exceeds48000 bytes")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as handle:handle.write(raw+"\n")
    print(json.dumps({key:value[key] for key in ("status","decision")},ensure_ascii=False))


if __name__=="__main__":main()
