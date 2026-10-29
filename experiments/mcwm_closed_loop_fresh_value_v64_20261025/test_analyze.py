"""Realized-control gates, frozen original baseline and retained-cohort tests."""
import copy
import json

import pytest

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.analyze import HERE, analyze


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def summary(seed,scene,arm,cost):
    quota=dict(source=128,historical_target=64,fresh_target=64 if arm=="fresh" else 0,
               historical_replacement=64 if arm=="no_fresh" else 0)
    receipts=[dict(deadline=1200+300*i,processed_at=1202+300*i,records=5,
        sequence_ids=list(range(5*i,5*i+5)),max_available_at=1199+300*i) for i in range(57)]
    return dict(status="complete",cell=dict(seed=seed,scenario=scene,arm=arm,candidate="fresh_value",external_seed=100000+seed),
        integrity=dict(all_pass=True,checks=dict(no_fresh_outcomes_retained=True,fresh_value_target_scope=True)),
        initializer=dict(path=f"/server/h2oplus_seed{seed}.pt",initial_updates=50000,history_rows=32768,source_rows=4096),
        budget=dict(target_feedback_slots=285,packets=57,policy_updates=456,native_seconds=24000,
            world_updates=0,trust_updates=0,counterfactual_queries=0,training_fresh_outcomes=285 if arm=="fresh" else 0),
        replay=dict(feedback_slots_received=285,recent_covariates=285,fresh_outcomes_stored=285 if arm=="fresh" else 0,
            td_slots=quota,fresh_current_s_a_use=["IQL value","SAC actor","temperature"],
            fresh_outcome_use=["critic factual Bellman target"] if arm=="fresh" else []),
        runtime_contract=dict(action_transform="ExecutionAction",real_td_coefficient=1,world_enabled=False,
            ratio_correction=False,td_slots=quota,actor_value_covariates="fresh_current_s_a",temperature_states="current"),
        action_audit=dict(shared_rule_exact=True),clock_audit=dict(actor_real_clock=True,first_feedback_applied_at=1202.),
        post_budget_counts=dict(before=dict(policy=456,world=0,trust=0),after=dict(policy=456,world=0,trust=0)),
        receipts=receipts,cost=dict(departed_passengers=100,accrued_wait_seconds=2000.,
            accrued_in_vehicle_seconds=100*cost-4000.,generalized_passenger_seconds=100*cost,
            generalized_seconds_per_departed_passenger=cost),
        target_audit=dict(parent_mixed_target_calls=456,parent_mixed_target_rows=116736,
            fresh_value_replacement_calls=456 if arm=="fresh" else 0,
            fresh_value_replacement_rows=29184 if arm=="fresh" else 0,
            no_fresh_unchanged_returns=0 if arm=="fresh" else 456,
            fresh_rows_start=64,fresh_rows_stop=128,mixed_target_generated_before_replacement=True,
            original_sampler_draws=4),execution_wall=dict(native_started_wall=0.,native_finished_wall=100.))


def build_stage(root,protocol,stage,deltas=None):
    paths=[]
    for index,seed in enumerate(protocol[f"{stage}_seeds"]):
        for offset,scene in enumerate(protocol["scenarios"]):
            delta=-2. if deltas is None else deltas[2*index+offset]
            for arm in ("fresh","no_fresh"):
                directory=root/"runs"/"fresh_value"/stage/f"seed{seed}"/scene/arm
                value=summary(seed,scene,arm,100.+delta if arm=="fresh" else 100.)
                write(directory/"summary.json",value)
                write(directory/"protocol.json",dict(policy_config=dict(batch_size=256,online_fraction=.5)))
                cost=value["cost"]
                jsonl(directory/"metrics.jsonl",[dict(time_seconds=t,
                    accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
                    accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
                    for t,f in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
                jsonl(directory/"prefix_trace.jsonl",[dict(started_at=1000,a=0),dict(started_at=1300,a=arm)])
                paths.append(directory)
                if stage=="development" and arm=="no_fresh":
                    reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(seed=seed,scenario=scene)
                    old=copy.deepcopy(value);old["cell"]["candidate"]="recent30"
                    write(reference/"summary.json",old)
    return paths


def populate(tmp_path,stage="development",deltas=None):
    root=tmp_path/"v64"
    protocol=json.loads((HERE/"protocol.json").read_text())
    write(root/"protocol.json",protocol)
    if stage=="confirmation":
        build_stage(root,protocol,"development")
        write(root/"results/fresh_value/development.json",analyze(root,"development"))
    return root,build_stage(root,protocol,stage,deltas)


def test_development_gate_uses_real_ledger_without_zero_metric_and_keeps_original_exactness(tmp_path):
    root,_=populate(tmp_path)
    result=analyze(root,"development")
    assert result["integrity"]["all_pass"]
    assert result["decision"]["advance_to_confirmation"]
    assert result["aggregates"]["no_fresh"]["overall"]["mean_cost_difference"]==-2.
    assert result["aggregates"]["no_fresh"]["post_budget"]["mean_cost_difference"]==pytest.approx(-.4)
    assert all(all(row["original_no_fresh_exactness"]["fields"].values()) for row in result["cells"])
    assert all(row["original_no_fresh_exactness"]["reused_as_control"] is False for row in result["cells"])


def test_all_fixed_cells_retained_and_win_loss_magnitudes_explicit(tmp_path):
    root,_=populate(tmp_path,deltas=[-2,-2,1,1])
    result=analyze(root,"development");overall=result["aggregates"]["no_fresh"]["overall"]
    assert result["decision"]["advance_to_confirmation"]
    assert overall["cells"]==4 and overall["wins"]==overall["losses"]==2
    assert overall["win_fraction"]==overall["loss_fraction"]==.5
    assert overall["win_mean_difference"]==-2. and overall["loss_mean_difference"]==1.


def test_overall_improvement_cannot_override_shock_or_stable_harm(tmp_path):
    root,_=populate(tmp_path,deltas=[-4,1,-4,1])
    result=analyze(root,"development")
    assert result["status"]=="complete" and result["aggregates"]["no_fresh"]["overall"]["mean_cost_difference"]<0
    assert result["decision"]["reason"]=="candidate_closed_cost_gate_failed"


@pytest.mark.parametrize("failure",("target_scope","parent_rng_path","budget","leak","parallel","prefix","config","original_ledger"))
def test_technical_or_provenance_failures_block_cost_gate_without_scientific_verdict(tmp_path,failure):
    root,paths=populate(tmp_path);path=paths[1];value=json.loads((path/"summary.json").read_text())
    if failure=="target_scope":value["target_audit"]["fresh_rows_start"]=0
    elif failure=="parent_rng_path":value["target_audit"]["parent_mixed_target_rows"]-=1
    elif failure=="budget":value["budget"]["policy_updates"]=455
    elif failure=="leak":value["replay"]["fresh_outcomes_stored"]=1
    elif failure=="parallel":value["execution_wall"]=dict(native_started_wall=100.,native_finished_wall=200.)
    elif failure=="prefix":jsonl(path/"prefix_trace.jsonl",[dict(started_at=1000,a=7.5)])
    elif failure=="config":write(path/"protocol.json",dict(policy_config=dict(batch_size=512)))
    else:
        protocol=json.loads((root/"protocol.json").read_text())
        reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(seed=6201,scenario="stable")
        old=json.loads((reference/"summary.json").read_text());old["cost"]["accrued_wait_seconds"]+=1
        write(reference/"summary.json",old)
    write(path/"summary.json",value)
    result=analyze(root,"development")
    assert result["status"]=="integrity_failed"
    assert result["decision"]["reason"]=="incomplete_or_integrity_failed"
    assert not result["decision"]["advance_to_confirmation"]


def test_missing_runs_and_failed_budget_are_never_zero_costs(tmp_path):
    root,paths=populate(tmp_path);path=paths[0];(path/"summary.json").unlink()
    result=analyze(root,"development")
    assert result["status"]=="pending" and result["available_cells"]==3
    assert result["decision"]["advance_to_confirmation"] is False
    write(path/"failure.json",dict(type="NativeError",consumed_budget=dict(policy_updates=16)))
    result=analyze(root,"development")
    assert result["status"]=="integrity_failed"
    assert result["failed_run_work"][0]["consumed_budget"]["policy_updates"]==16


def test_confirmation_reports_six_new_and_ten_retained_cells_without_completing_goal(tmp_path):
    root,_=populate(tmp_path,stage="confirmation")
    result=analyze(root,"confirmation")
    assert result["integrity"]["all_pass"] and result["decision"]["confirmed_control_improvement"]
    assert result["available_cells"]==6
    assert result["retained_all_five_seeds"]["cells"]==10
    assert result["retained_all_five_seeds"]["aggregates"]["overall"]["cells"]==10
    assert result["decision"]["ordinary_dripping_goal_ready_for_root_review"]
    assert result["decision"]["ordinary_dripping_goal_achieved"] is False
    assert len(json.dumps(result,separators=(",",":")).encode())<48000


def test_confirmation_cannot_ignore_missing_or_truncated_development_cohort(tmp_path):
    root,_=populate(tmp_path,stage="confirmation")
    path=root/"results/fresh_value/development.json"
    previous=json.loads(path.read_text());path.unlink()
    result=analyze(root,"confirmation")
    assert result["status"]=="pending"
    assert "retained_development_result" in result["missing"]
    write(path,dict(previous,cells=previous["cells"][:-1]))
    result=analyze(root,"confirmation")
    assert result["status"]=="integrity_failed"
    assert not result["decision"]["confirmed_control_improvement"]


def test_confirmation_harm_fails_even_when_retained_aggregate_is_good(tmp_path):
    root,_=populate(tmp_path,stage="confirmation",deltas=[-.1,.01,-.1,.01,-.1,.01])
    result=analyze(root,"confirmation")
    assert result["status"]=="complete"
    assert result["retained_all_five_seeds"]["scientific_gate_pass"]
    assert not result["decision"]["stage_cost_gate_pass"]
    assert not result["decision"]["confirmed_control_improvement"]


def test_native_sized_six_cell_confirmation_and_retained_ten_stay_under48kb(tmp_path):
    root,_=populate(tmp_path,stage="confirmation")
    protocol=json.loads((root/"protocol.json").read_text())
    for path in (root/"runs/fresh_value").rglob("summary.json"):
        value=json.loads(path.read_text());seed=value["cell"]["seed"];arm=value["cell"]["arm"]
        waiting=8438598+100*(seed-6201)
        riding=16361600+347*(seed-6201)-(1200 if arm=="fresh" else 0)
        total=2*waiting+riding
        value["cost"].update(departed_passengers=15079,accrued_wait_seconds=waiting,
            accrued_in_vehicle_seconds=riding,generalized_passenger_seconds=total,
            generalized_seconds_per_departed_passenger=total/15079)
        write(path,value)
        jsonl(path.parent/"metrics.jsonl",[dict(time_seconds=t,accrued_wait_seconds=waiting*f,
            accrued_in_vehicle_seconds=riding*f) for t,f in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
        if seed in protocol["development_seeds"] and arm=="no_fresh":
            reference=root.parent/protocol["original_baseline_directory"]/protocol["original_baseline_relative"].format(
                seed=seed,scenario=value["cell"]["scenario"])
            old=json.loads((reference/"summary.json").read_text());old["cost"]=value["cost"]
            write(reference/"summary.json",old)
    write(root/"results/fresh_value/development.json",analyze(root,"development"))
    result=analyze(root,"confirmation")
    assert result["integrity"]["all_pass"] and result["decision"]["confirmed_control_improvement"]
    assert len(json.dumps(result,separators=(",",":")).encode())<48000
