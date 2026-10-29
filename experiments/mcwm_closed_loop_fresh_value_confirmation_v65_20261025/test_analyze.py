"""Frozen confirmation keeps primary gates separate from stable-tail harm."""
import copy
import json

import pytest

from experiments.mcwm_closed_loop_fresh_value_confirmation_v65_20261025.analyze import HERE,analyze
from experiments.mcwm_closed_loop_fresh_value_v64_20261025 import analyze as source_analysis
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_analyze import build_stage,jsonl,write


def harm_stable_tail(paths):
    for directory in paths:
        value=json.loads((directory/"summary.json").read_text())
        if value["cell"]["scenario"]!="stable" or value["cell"]["arm"]!="fresh":continue
        path=directory/"metrics.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["time_seconds"]==18300:row["accrued_in_vehicle_seconds"]-=240.
        jsonl(path,rows)


def populate(tmp_path,*,new_deltas=None,tail_harm=False):
    root=tmp_path/"v65";protocol=json.loads((HERE/"protocol.json").read_text())
    write(root/"protocol.json",protocol)
    old=root.parent/protocol["development_directory"]
    old_protocol=json.loads((source_analysis.HERE/"protocol.json").read_text())
    write(old/"protocol.json",old_protocol)
    old_paths=build_stage(old,old_protocol,"development")
    new_paths=build_stage(root,protocol,"confirmation",new_deltas)
    if tail_harm:harm_stable_tail(old_paths+new_paths)
    write(old/protocol["development_relative"],source_analysis.analyze(old,"development"))
    for path in new_paths:
        value=json.loads((path/"summary.json").read_text())
        if value["cell"]["arm"]!="no_fresh":continue
        reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(
            seed=value["cell"]["seed"],scenario=value["cell"]["scenario"])
        previous=copy.deepcopy(value);previous["cell"]["candidate"]="recent30"
        write(reference/"summary.json",previous)
    return root,old,new_paths,old_paths


def test_old_development_is_read_only_reference_and_new_twelve_days_are_distinct(tmp_path):
    root,old,_,_=populate(tmp_path)
    before=(old/"results/fresh_value/development.json").read_text()
    value=analyze(root)
    assert value["status"]=="complete" and value["integrity"]["all_pass"]
    assert value["decision"]["confirmed_control_improvement"]
    assert value["decision"]["deployment_admissible"]
    assert value["decision"]["ordinary_dripping_goal_achieved"] is False
    assert value["independent3"]["cells"]==6 and value["retained5"]["cells"]==10
    assert value["data_sources"]["new_native_days"]==12
    assert value["data_sources"]["reused_development_days"]==8
    assert value["data_sources"]["old_development_is_new_data"] is False
    assert value["data_sources"]["reference_substitutes_new_control"] is False
    assert len([row for row in value["cells"] if row["cohort"]=="retained_v64_development"])==4
    assert (old/"results/fresh_value/development.json").read_text()==before


def test_all_three_windows_report_absolute_wait_ride_cost_and_win_loss(tmp_path):
    root,_,_,_=populate(tmp_path,new_deltas=[-2,-2,1,1,-2,-2])
    value=analyze(root)
    for cohort in ("independent3","retained5"):
        for name in ("full_day","early","post_budget"):
            for scene in ("stable","local_speed_drop"):
                stats=value[cohort]["periods"][name]["by_scenario"][scene]
                assert stats["complete_cells"]==stats["expected_cells"]
                for arm in ("fresh","no_fresh"):
                    assert set(stats["arms"][arm]["raw"])=={"wait","ride","cost"}
                    assert stats["arms"][arm]["raw"]["cost"]==2*stats["arms"][arm]["raw"]["wait"]+stats["arms"][arm]["raw"]["ride"]
                assert stats["win_fraction"]+stats["loss_fraction"]==1.
    overall=value["independent3"]["periods"]["full_day"]["overall"]
    assert overall["wins"]==4 and overall["losses"]==2
    assert overall["winning_mean_reduction"]==2. and overall["losing_mean_increase"]==1.


def test_full_day_can_pass_while_stable_tail_harm_persists_and_deployment_fails(tmp_path):
    root,_,_,_=populate(tmp_path,tail_harm=True)
    value=analyze(root)
    assert value["decision"]["confirmed_control_improvement"]
    assert value["decision"]["stable_tail_harm_persists"]
    assert not value["decision"]["deployment_admissible"]
    assert value["secondary_assessment"]["full_day_cost_gate_unchanged"]
    assert value["secondary_assessment"]["independent3"]["mean_cost_difference"]==pytest.approx(2.)
    assert value["secondary_assessment"]["old_development2"]["mean_cost_difference"]==pytest.approx(2.)
    assert len(value["stable_tail_by_seed"])==5
    assert all(row["delta"]["cost_per_passenger"]>0 for row in value["stable_tail_by_seed"])


def test_shock_harm_fails_original_gate_even_if_retained_five_are_good(tmp_path):
    root,_,_,_=populate(tmp_path,new_deltas=[-.1,.01,-.1,.01,-.1,.01])
    value=analyze(root)
    assert value["status"]=="complete"
    assert value["retained5"]["gate"]["all_pass"]
    assert not value["independent3"]["gate"]["all_pass"]
    assert not value["decision"]["confirmed_control_improvement"]


def test_missing_secondary_endpoint_is_unknown_not_zero_or_a_new_primary_gate(tmp_path):
    root,_,paths,_=populate(tmp_path)
    path=paths[0]/"metrics.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    jsonl(path,[row for row in rows if row["time_seconds"]!=18300])
    value=analyze(root)
    assert value["status"]=="complete" and value["decision"]["confirmed_control_improvement"]
    assert value["decision"]["stable_tail_harm_persists"] is None
    assert not value["decision"]["deployment_admissible"]
    stats=value["independent3"]["periods"]["post_budget"]["by_scenario"]["stable"]
    assert stats["expected_cells"]==3 and stats["complete_cells"]==2 and stats["missing_cells"]==1
    row=next(row for row in value["stable_tail_by_seed"] if row["seed"]==6203)
    assert row["status"]=="missing_or_invalid_endpoint" and row["delta"] is None


@pytest.mark.parametrize("failure",("baseline","target","budget","leak","parallel","prefix","config"))
def test_native_or_reference_failures_cannot_become_scientific_failures(tmp_path,failure):
    root,_,paths,_=populate(tmp_path);path=paths[1];value=json.loads((path/"summary.json").read_text())
    if failure=="baseline":
        protocol=json.loads((root/"protocol.json").read_text())
        reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(seed=6203,scenario="stable")
        old=json.loads((reference/"summary.json").read_text());old["cost"]["accrued_wait_seconds"]+=1
        write(reference/"summary.json",old)
    elif failure=="target":value["target_audit"]["parent_mixed_target_rows"]-=1
    elif failure=="budget":value["budget"]["policy_updates"]=455
    elif failure=="leak":value["replay"]["fresh_outcomes_stored"]=1
    elif failure=="parallel":value["execution_wall"]=dict(native_started_wall=100.,native_finished_wall=200.)
    elif failure=="prefix":jsonl(path/"prefix_trace.jsonl",[dict(started_at=1000,a=7.5)])
    else:write(path/"protocol.json",dict(policy_config=dict(batch_size=512)))
    write(path/"summary.json",value)
    result=analyze(root)
    assert result["status"]=="integrity_failed"
    assert result["decision"]["reason"]=="incomplete_or_integrity_failed"


def test_missing_new_run_or_old_cohort_is_not_zero_or_omitted_success(tmp_path):
    root,old,paths,_=populate(tmp_path);(paths[0]/"summary.json").unlink()
    value=analyze(root)
    assert value["status"]=="pending" and value["missing"]
    assert not value["decision"]["confirmed_control_improvement"]
    assert value["independent3"]["cells"]==5 and not value["independent3"]["gate"]["all_pass"]
    assert value["independent3"]["expected_cells"]==6
    assert value["independent3"]["available_cells"]==5
    tail=value["independent3"]["periods"]["post_budget"]["by_scenario"]["stable"]
    assert tail["expected_cells"]==3 and tail["complete_cells"]==2 and tail["missing_cells"]==1
    assert value["secondary_assessment"]["independent3"]["complete"] is False
    assert value["decision"]["stable_tail_harm_persists"] is None
    assert value["decision"]["deployment_admissible"] is False
    assert value["development2"]["gate"]["all_pass"]
    source=old/"results/fresh_value/development.json";source.unlink()
    value=analyze(root)
    assert "v64_development_result" in value["missing"]
    assert not value["retained5"]["gate"]["all_pass"]
    assert value["development2"]["periods"]["post_budget"]["by_scenario"]["stable"]["expected_cells"]==2
    assert value["secondary_assessment"]["old_development2"]["complete"] is False


def test_native_sized_confirmation_and_window_decompositions_stay_small(tmp_path):
    root,old,new_paths,old_paths=populate(tmp_path)
    protocol=json.loads((root/"protocol.json").read_text())
    for directory in new_paths+old_paths:
        path=directory/"summary.json";value=json.loads(path.read_text());seed=value["cell"]["seed"];arm=value["cell"]["arm"]
        wait=8438598+100*(seed-6201);ride=16361600+347*(seed-6201)-(1200 if arm=="fresh" else 0);total=2*wait+ride
        value["cost"].update(departed_passengers=15079,accrued_wait_seconds=wait,accrued_in_vehicle_seconds=ride,
            generalized_passenger_seconds=total,generalized_seconds_per_departed_passenger=total/15079)
        write(path,value);jsonl(directory/"metrics.jsonl",[dict(time_seconds=t,
            accrued_wait_seconds=wait*f,accrued_in_vehicle_seconds=ride*f)
            for t,f in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
        if arm=="no_fresh":
            if seed in protocol["development_seeds"]:
                previous_protocol=json.loads((old/"protocol.json").read_text())
                reference=old.parent/previous_protocol["original_baseline_directory"]/previous_protocol["original_baseline_relative"].format(seed=seed,scenario=value["cell"]["scenario"])
            else:
                reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(seed=seed,scenario=value["cell"]["scenario"])
            original=json.loads((reference/"summary.json").read_text());original["cost"]=value["cost"]
            write(reference/"summary.json",original)
    write(old/protocol["development_relative"],source_analysis.analyze(old,"development"))
    value=analyze(root)
    assert value["integrity"]["all_pass"] and value["decision"]["confirmed_control_improvement"]
    assert len(json.dumps(value,separators=(",",":")).encode())<48000
