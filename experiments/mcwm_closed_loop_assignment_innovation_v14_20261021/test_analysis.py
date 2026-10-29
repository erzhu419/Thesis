"""Scientific comparison gates, explicit missing outcomes, and bounded resident summary."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v14_analysis_test", HERE/"analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.fixture
def protocol():
    return dict(training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
        target_lines=["7X", "102X", "705X"], target_record_budget=285,
        preflight=dict(seed=6201, scenario="stable", target_record_budget=50))


def truth(action, missing_same_car=False):
    high = action == "high"
    value = dict(global_next=dict(status="complete", dt=0., wait=0., ride=0., cost=0.),
        same_vehicle_next=dict(status="complete", dt=100.+7.5*high, wait=100., ride=300.+50.*high, cost=500.+50.*high),
        fixed300=dict(status="complete", dt=300., wait=2000., ride=6000.-10.*high, cost=10000.-10.*high))
    if missing_same_car:
        value["same_vehicle_next"] = dict(status="missing", reason="beyond_native_suffix")
    return value


def diagnostic():
    zero = dict(dt=0., wait=0., ride=0., cost=0.)
    value = dict(state_values=dict(global_issue_cost_rate_per_1000=2.3), endpoints={})
    for endpoint in analysis.ENDPOINTS:
        baseline, holding = {**zero, "cost": 10.}, {**zero, "cost": 15.}
        interactions = {f"holding_x_state_{i}": {**zero, "cost": 0.} for i in range(5)}
        interactions["holding_x_global_issue_cost_rate_per_1000"] = {**zero, "cost": 5.}
        total = {**zero, "cost": 30.}
        value["endpoints"][endpoint] = dict(status="supported", baseline=baseline, level=deepcopy(zero),
            line_holding=holding, state_interactions=interactions, raw_total=deepcopy(total),
            total=deepcopy(total), analytic_adjustment=deepcopy(zero), analytic_zero_duration=False,
            fixed_weight_no_state_interactions={**zero,"cost":25.},
            raw_fixed_weight_no_state_interactions={**zero,"cost":25.},
            fixed_weight_no_global_cost_rate={**zero,"cost":25.},
            raw_fixed_weight_no_global_cost_rate={**zero,"cost":25.})
    return value


def prediction():
    result = dict(global_comparator=dict(arms={}))
    for model in ("base_analytic", "global58"):
        result["global_comparator"]["arms"][model] = dict(low=dict(dt=0.,cost=0.), high=dict(dt=0.,cost=0.),
            cost_gain_low_minus_high=0., raw_cost_gain_low_minus_high=2.)
    for model in (*analysis.MODELS, "legacy", "residual"):
        low = dict(global_next=dict(dt=0.,wait=0.,ride=0.,cost=0.),
            same_vehicle_next=dict(dt=100.,wait=100.,ride=300.,cost=500.),
            fixed300=dict(dt=300.,wait=2000.,ride=6000.,cost=10000.))
        high=deepcopy(low)
        high["same_vehicle_next"].update(ride=350.,cost=550.)
        high["fixed300"].update(ride=5970.,cost=9970.)
        result[model] = dict(low=dict(endpoints=low),high=dict(endpoints=high),
            gains=dict(global_next=0.,same_vehicle_next=-50.,fixed300=30.), raw_global_next_gain=20.)
    for model in analysis.DIAGNOSTIC_MODELS:
        result[model+"_diag"] = diagnostic()
    return result


def anchor(line="7X", index=0):
    return dict(selected=dict(sequence=4+index,line_id=line,started_at=18001.+index,s=[1.,2.],raw_s=[9.]),
        prefix=dict(external_seed=83201,counters=dict(policy_updates=456),receipts=[dict(deadline=18000)]),
        prediction=prediction())


def branch(action,line="7X",index=0,missing_same_car=False):
    return dict(status="complete",integrity=dict(all_pass=True,checks=dict(full_native_prefix_exact=True,
        v13_reference_truth_exact=True)),**anchor(line,index),truth=truth(action,missing_same_car),
        execution=dict(requested_hold_seconds=7.5 if action=="high" else 0.,executed_hold_seconds=7.5 if action=="high" else 0.),
        suffix=dict(seconds=300.,training_delta=dict(policy_updates=0,world_updates=0,trust_updates=0,separated_labels=0)),
        budget=dict(prefix_sim_seconds=18001.,suffix_sim_seconds=300.,replay_target_labels=285,truth_labels_enter_training=0),
        native=dict(other_batch_actions=[],prefix_matched_issues=5))


def day(p,preflight=False):
    rows = [dict(line_id=line,role=role,hold_seconds=0. if action=="low" else 7.5,
        prediction=prediction(),truth=truth(action)) for line in p["target_lines"]
        for role in ("train","heldout") for action in ("low","high")]
    return dict(status="complete",integrity=dict(all_pass=True,checks=dict(prefix_trace_recorded=True,
        v13_common_flow_exact=True,v13_old_predictions_exact=True)),budget=dict(target_labels=50 if preflight else 285),
        initializer=dict(updates=50000),coverage=dict(train={},heldout={}),
        anchors={line:[anchor(line,i) for i in range(2)] for line in p["target_lines"]},evaluations=rows,
        interaction_model=dict(design=dict(rows=55,columns=19,rank=19),coefficients={name:[[999.,999.,999.] for _ in range(19)] for name in analysis.ENDPOINTS}),
        separated_model=dict(design=dict(rows=55,columns=21,rank=21),readouts=[[999.]],
            nuisance_diagnostics={name:dict(status="fitted",fold_counts={"0":25,"1":30},fold_models=[dict(coefficients=[[999.]])],
                global_cost_rate_action=dict(original_norm=3.,residualized_norm=2.,action_coefficient_physical=dict(cost=100.)))
                for name in analysis.ENDPOINTS}))


def populate(root,p,preflight=False):
    days,pairs=analysis.expected_paths(root,p,preflight=preflight)
    for _,path in days:
        path.mkdir(parents=True)
        (path/"summary.json").write_text(json.dumps(day(p,preflight)))
    for cell,path in pairs:
        path.mkdir(parents=True)
        (path/"summary.json").write_text(json.dumps(branch(cell["branch"],cell["target_line"],cell["anchor_index"])))


def test_comparators_and_matching_never_impute_missing_labels():
    rows=[dict(hold_seconds=0.,prediction=prediction(),truth=truth("low")) for _ in range(2)]
    rows[1]["prediction"]["separated"]["low"]["endpoints"]["same_vehicle_next"]=None
    value=analysis.summarize_rows(rows)["endpoints"]["same_vehicle_next"]
    assert set(value["matched_errors"])==set(analysis.MODELS)
    assert value["available"]["innovation"]==2 and value["available"]["separated"]==1
    assert all(item["n"]==1 for item in value["matched_errors"].values())
    assert analysis.prediction_endpoint(prediction(),"global58","fixed300","low") is None
    rows[0]["truth"]["same_vehicle_next"]=dict(status="missing")
    value=analysis.summarize_rows(rows)["endpoints"]["same_vehicle_next"]
    assert value["missing_truth"]==1 and all(item["n"]==0 for item in value["matched_errors"].values())


def test_regret_tie_and_wrong_sign():
    assert analysis.effect_metrics(10.,-5.)["regret"]==5.
    assert analysis.effect_metrics(0.,12.)["regret"]==12.
    assert analysis.effect_metrics(0.,12.)["selected_action"]=="low"
    assert analysis.effect_metrics(None,3.) is None


def test_full_prefix_and_all_previous_state_truth_are_required():
    cell=dict(seed=6201,scenario="stable",line="7X",anchor_index=1)
    low,high=branch("low",index=1),branch("high",index=1)
    assert analysis.analyze_pair(cell,low,high,anchor(index=1))["integrity"]["all_pass"]
    high["integrity"]["checks"]["v13_reference_truth_exact"]=False
    high["prefix"]["receipts"][0]["deadline"]+=1
    high["suffix"]["training_delta"]["separated_labels"]=1
    pair=analysis.analyze_pair(cell,low,high,anchor(index=1))
    assert not pair["integrity"]["all_pass"]
    assert not pair["integrity"]["checks"]["v13_reference_truth_exact"]
    assert not pair["integrity"]["checks"]["common_prefix"]
    assert not pair["integrity"]["checks"]["suffix_zero_training"]
    assert analysis.summarize_effects([pair])["fixed300"]["matched_n"]==0


def test_native_commands_and_missing_samecar_are_distinct():
    low,high=branch("low"),branch("high",missing_same_car=True)
    cell=dict(seed=6201,scenario="stable",line="7X")
    pair=analysis.analyze_pair(cell,low,high,anchor())
    assert pair["integrity"]["all_pass"] and pair["truth"]["same_vehicle_next"]["true_gain"] is None
    assert pair["effects"]["same_vehicle_next"]["separated"] is None
    high["execution"]["executed_hold_seconds"]=7.
    assert not analysis.analyze_pair(cell,low,high,anchor())["integrity"]["checks"]["actual_low_high_commands"]


def test_four_days_twelve_states_budget_and_compact_no_raw_arrays(tmp_path,protocol):
    populate(tmp_path,protocol)
    value=analysis.analyze(tmp_path,protocol)
    assert value["status"]=="complete" and value["integrity"]["all_pass"]
    assert len(value["cells"])==4 and len(value["pairs"])==12
    assert value["budget"]["main_unique_training_slots"]==1140
    extra=value["budget"]["extra_native_branch_budget"]
    assert extra["replay_target_labels"]==6840 and extra["suffix_sim_seconds"]==7200.
    assert extra["truth_labels_enter_training"]==0
    assert value["independent_confirmation"]["status"]=="not_submitted"
    payload=json.dumps(value,allow_nan=False)
    assert len(payload.encode())<analysis.MAX_BYTES
    assert all(name not in payload for name in ('"raw_s"','"receipts"','"coefficients"','"fold_models"','"readouts"'))
    assert "nuisance_diagnostics" not in value["cells"][0]["separated_model"]


@pytest.mark.parametrize("check",["prefix_trace_recorded","v13_common_flow_exact","v13_old_predictions_exact"])
def test_preflight_full_flow_gates_cannot_be_omitted(tmp_path,protocol,check):
    populate(tmp_path,protocol,True)
    path=tmp_path/"day"/"summary.json"
    value=json.loads(path.read_text());del value["integrity"]["checks"][check];path.write_text(json.dumps(value))
    result=analysis.analyze(tmp_path,protocol,preflight=True)
    assert not result["integrity"]["all_pass"] and len(json.dumps(result).encode())<20000
    assert "truth" not in result["pairs"][0]


def test_missing_state_remains_terminal_without_fabricated_zero_truth(tmp_path,protocol):
    populate(tmp_path,protocol)
    _,pairs=analysis.expected_paths(tmp_path,protocol)
    (pairs[0][1]/"summary.json").write_text('{"status":"selection_missing","integrity":{"all_pass":false}}')
    value=analysis.analyze(tmp_path,protocol)
    assert value["status"]=="complete" and not value["integrity"]["all_pass"]
    assert value["missing"]["branch_statuses"]["selection_missing"]==1
    assert any(pair["truth"]["fixed300"]["true_gain"] is None for pair in value["pairs"])
    assert not value["independent_confirmation"]["review_ready"]


def test_analytic_clock_error_blocks_preflight(tmp_path,protocol):
    populate(tmp_path,protocol,True)
    path=tmp_path/"day"/"summary.json";value=json.loads(path.read_text())
    value["evaluations"][0]["prediction"]["separated"]["low"]["endpoints"]["fixed300"]["dt"]=299.
    path.write_text(json.dumps(value))
    assert not analysis.analyze(tmp_path,protocol,preflight=True)["integrity"]["checks"]["fixed300_residual_exact"]


def test_raw_zero_duration_costrate_and_analytic_cancellation_stay_separate():
    pred=prediction();endpoint=pred["innovation_diag"]["endpoints"]["global_next"]
    endpoint["analytic_zero_duration"]=True;endpoint["analytic_adjustment"]["cost"]=-30.;endpoint["total"]["cost"]=0.
    value=analysis.interaction_diagnostic(pred)["endpoints"]["global_next"]
    assert value["raw_total"]==30. and value["total"]==0.
    assert value["raw_fixed_weight_no_global_cost_rate"]==25. and value["fixed_weight_no_global_cost_rate"]==0.
    assert value["reconstruction_error"]==value["analytic_reconstruction_error"]==0.
    endpoint["level"]["cost"]=1.
    assert analysis.interaction_diagnostic(pred)["endpoints"]["global_next"]["reconstruction_error"]==1.


def test_model_numeric_noise_is_zero_but_actual_gain_never_thresholded():
    pairs=[]
    for index in (0,1):
        low,high=branch("low",index=index),branch("high",index=index)
        for value in (low,high):
            value["prediction"]["innovation"]["gains"]["fixed300"]+=1e-10*index
            value["prediction"]["separated"]["gains"]["fixed300"]+=7.*index
        high["truth"]["fixed300"]["cost"]-=5.*index
        pairs.append(analysis.analyze_pair(dict(seed=6201,scenario="stable",line="7X",anchor_index=index),
            low,high,{key:low[key] for key in ("selected","prefix","prediction")}))
    value=analysis.state_changes(pairs)[0]["endpoints"]["fixed300"]
    assert value["true_gain_change"]==5.
    assert value["models"]["innovation"]["predicted_gain_change"]==0.
    assert not value["models"]["innovation"]["change_direction_correct"]
    assert value["models"]["separated"]["predicted_gain_change"]==7.
    assert value["models"]["separated"]["change_direction_correct"]


def test_fixed_weight_costrate_deletion_and_line_refit_are_both_reported(tmp_path,protocol):
    populate(tmp_path,protocol)
    result=analysis.analyze(tmp_path,protocol)
    value=result["fixed_weight_ablation"]["overall"]["fixed300"]["separated"]
    assert value["matched_n"]==12
    assert value["models"]["separated"][1]==20.
    assert value["models"]["fixed_weight_no_global_cost_rate"][1]==15.
    assert value["models"]["innovation_line"][1]==20.
    assert all("innovation_line" in row["endpoints"]["fixed300"] for row in result["stable_degradation"])


def test_failed_preflight_work_is_accounted_without_inventing_budget(tmp_path,protocol):
    populate(tmp_path,protocol)
    path=tmp_path/"preflight"/"day";path.mkdir(parents=True)
    (path/"failure.json").write_text('{"status":"failed","time":1234,"message":"native failure"}')
    result=analysis.analyze(tmp_path,protocol)["preflight_compute_budget"]
    assert result[0]["runs"][0]["native_seconds"]==1234
    assert result[0]["runs"][0]["target_labels"] is None




def test_innovation_audit_retains_counts_coefficients_and_excludes_individual_fit_ids():
    metric = dict(dt=0., wait=2., ride=3., cost=7.)
    source = dict(line_only=False, innovation_diagnostics={"fixed300": dict(
        status="fitted", first_rows=23, second_rows=22, cold_start_first_rows=1,
        snapshot_scope_valid=True, fit_sequence_ids=[11, 12], first_sequence_ids=[11], second_sequence_ids=[12],
        first_coverage={"7X": dict(n=8, zero_holding=3, high_holding=5)},
        action_design=dict(columns=9, rank=8), innovation_error_rmse=deepcopy(metric),
        score_error_rmse=deepcopy(metric), action_coefficient_physical={"line7X": deepcopy(metric)})})
    cell = {key: analysis.compact_innovation_model(deepcopy(source))
            for key in ("innovation_model", "innovation_line_model")}
    schema = analysis.compact_innovation_audits([cell])
    row = cell["innovation_model"]["innovation_diagnostics"]["fixed300"]
    assert row["first_rows"] == 23 and row["second_rows"] == 22
    assert row["first_coverage"]["7X"] == [8, 3, 5]
    assert row["action_coefficient_physical"] == [[0., 2., 3., 7.]]
    assert row["score_error_rmse"] == [0., 2., 3., 7.]
    assert schema["action_feature_order"] == ["line7X"]
    assert "sequence_ids" not in json.dumps(cell)


def test_actual_v13_cohorts_and_expanded_innovation_audits_fit_small_budget(tmp_path, protocol):
    source = HERE.parent / "mcwm_closed_loop_v13_review_20261021" / "summary.json"
    if not source.is_file():
        pytest.skip("Actual small v13 review summary is not locally available")
    previous = json.loads(source.read_text())
    mapping = {"interaction": "innovation", "separated_no_cost": "innovation_line"}
    def rename(value):
        if isinstance(value, dict):
            return {mapping.get(key, key): rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value
    value = rename(previous)
    populate(tmp_path, protocol)
    synthetic = analysis.analyze(tmp_path, protocol)
    # Keep every real phase/line/scenario/zero-duration cohort, physical numeric
    # value, and paired decomposition; only the additional audit shape is synthetic.
    value["fixed_weight_ablation"] = synthetic["fixed_weight_ablation"]
    fields = previous["schema"]["action_diagnostic_order"]
    interactions = previous["schema"]["action_interaction_order"]
    metric = dict(dt=123.45678, wait=12345.678, ride=23456.789, cost=48148.145)
    coverage = {line: dict(n=17, zero_holding=8, high_holding=9) for line in protocol["target_lines"]}
    design = dict(rows=26, columns=9, rank=9, active_columns=9, ridge=1., ridge_precision_condition=123.45678)
    audit = dict(status="fitted", background_rows=53, first_rows=26, second_rows=27,
        first_coverage=deepcopy(coverage), cold_start_first_rows=1, snapshot_scope_valid=True,
        action_design=deepcopy(design), innovation_error_rmse=deepcopy(metric),
        score_error_rmse=deepcopy(metric), action_coefficient_physical={
            f"line_{i}": deepcopy(metric) for i in range(3)} | {name: deepcopy(metric) for name in interactions},
        cold_start_action_coefficient_physical={
            f"line_{i}": deepcopy(metric) for i in range(3)} | {name: deepcopy(metric) for name in interactions},
        cold_warm_beta_reconstruction_error=1.234e-12)
    model = dict(design={**design, "columns":21, "rank":21},
        endpoint_designs={endpoint:{**design, "columns":21,"rank":21,"learned_targets":["dt","wait","ride"]}
            for endpoint in analysis.ENDPOINTS},
        endpoint_coverage={endpoint:deepcopy(coverage) for endpoint in analysis.ENDPOINTS},
        innovation_diagnostics={endpoint:deepcopy(audit) for endpoint in analysis.ENDPOINTS})
    for cell in value["cells"]:
        cell.pop("interaction_model", None)
        cell.pop("joint_cost_rate_coefficient", None)
        cell["separated_model"] = {key:item for key,item in cell["separated_model"].items()
                                   if key in ("design","endpoint_designs")}
        for key in ("innovation_model", "innovation_line_model"):
            cell[key] = deepcopy(model)
        count = dict(records=53, by_line={line:dict(first=dict(low=8,high=9),second=dict(low=9,high=8))
                                        for line in protocol["target_lines"]})
        cell["assignment_audit"] = dict(issued=deepcopy(count),selected=deepcopy(count),
            not_selected=deepcopy(count),snapshot_scope_valid=True,cold_start_first_by_endpoint={
                endpoint:1 for endpoint in analysis.ENDPOINTS})
        cell["action_difference"] = deepcopy(synthetic["cells"][0]["action_difference"])
    value["schema"].pop("nuisance", None)
    value["schema"]["innovation"] = analysis.compact_innovation_audits(value["cells"])
    # Decode the previous shared n/mae/mse rows before applying the v14 matrix schema.
    for row in value["strata"]:
        for endpoint in row["endpoints"].values():
            endpoint["matched_errors"] = {name:dict(zip(("n","mae","mse"), metrics))
                for name, metrics in endpoint["matched_errors"].items()}
    value["schema"]["phase_tables"] = analysis.compact_phase_tables(value["strata"])
    payload = json.dumps(analysis.compact(value), ensure_ascii=False,separators=(",",":"),allow_nan=False)
    print("actual_v13_expansion_bytes",len(payload.encode()))
    assert len(value["pairs"]) == 12 and len(value["strata"]) == len(previous["strata"])
    assert len(payload.encode()) < analysis.MAX_BYTES
    assert all(name not in payload for name in ('"raw_s"', '"receipts"', '"coefficients"', '"fit_sequence_ids"'))


def test_finalizer_exception_is_saved_for_continuation_driver(tmp_path):
    import subprocess
    import sys
    # A real resident parse failure must terminate polling just like a cap failure.
    root = tmp_path/'preflight'
    (root/'day').mkdir(parents=True)
    (root/'day'/'summary.json').write_text('{broken_json')
    out = tmp_path/'review'/'gate.json'
    result = subprocess.run([sys.executable,str(HERE/'analyze.py'),'--root',str(root),
                             '--out',str(out),'--preflight'],capture_output=True,text=True)
    failure = json.loads((out.parent/'summary_failure.json').read_text())
    assert result.returncode != 0 and not out.exists()
    assert failure['status'] == 'failed' and failure['error_type'] == 'JSONDecodeError'
