"""Verify truth signs, paired-cost accounting and exclusion of invalid evidence."""
import json

from experiments.mcwm_closed_loop_physical_truth_v55_20261025 import analyze as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def fixture(tmp_path):
    root = tmp_path/"runs"
    protocol = dict(scenarios=["stable", "local_speed_drop"], focus_lines=["102X", "705X"],
        windows=dict(early=[9900, 10200], late=[19200, 19500]), native_external_seed=106255,
        compute_budget=dict(maximum_heldout_states=8, maximum_native_suffixes=16,
            maximum_suffix_native_seconds=4800, source_feedback_slots=285, source_updates=456))
    write(tmp_path/"protocol.json", protocol)
    for scenario in protocol["scenarios"]:
        case = root/scenario
        parent = dict(integrity=dict(all_pass=True, checks=dict(frozen_models_exact=True)),
            native_external_seed=106255, initializer_seed=6202, private_action_seed=526255,
            controller_snapshot="v53_actual_post24_from14",
            frozen_model_gate=dict(continuation_mu14_gate=dict(all_pass=True,
                shared_frozen_actor_exact=True, F_contract_exact=True, actor_updates=50024)), selection={}, states=[],
            budget=dict(native_prefix_seconds=24000, states=4, suffix_sim_seconds=2400,
                new_training_labels=0, new_prefix_optimizer_updates=0, suffix_optimizer_updates=0))
        for index, (line, window) in enumerate((line, window) for line in protocol["focus_lines"] for window in protocol["windows"]):
            slot = line+"/"+window
            clock = float(protocol["windows"][window][0]+index)
            selected = dict(sequence=6000+index, started_at=clock, trip_id=line+"_49", line_id=line)
            parent["selection"][slot] = dict(selected=dict(sequence=6000+index, issue=clock), reasons={})
            directory = case/"states"/slot.replace("/", "_")
            parent["states"].append(dict(slot=slot, path=str(directory/"state_summary.json")))
            actions = dict(low=[-1.], high=[-.75])
            separation = dict(status=["separable", "nonseparable", "physical_unknown", "separable"][index],
                departure_gap_seconds=None if index == 2 else 0. if index == 1 else 8.,
                next_same_vehicle_arrival_gap_seconds=None if index == 2 else 0. if index == 1 else 6.,
                low_departure_status="observed", high_departure_status="observed",
                low_next_arrival_status="observed", high_next_arrival_status="truncated" if index == 2 else "observed")
            record = dict(slot=slot, selected=selected, holds=dict(low=0., high=7.5), actions=actions,
                pair_receipt=dict(all_pass=True, overlap_seconds=100.), comparison=dict(physical_separation=separation),
                branch_paths={arm: str(directory/arm/"summary.json") for arm in ("low", "high")}, prediction={})
            for model in module.MODELS:
                # Opposite policy choices expose sign mistakes in regret/cost deltas.
                delta = 1. if model == "trace" else -1.
                record["prediction"][model] = dict(q_min=[-100., -100.+delta], q_mean=[-99., -99.+delta],
                    predicted_choice=int(delta > 0), executed_actions=[-1., -.75], actor_holding_seconds=1.)
            for arm in ("low", "high"):
                high = arm == "high"
                # index0: benefit [-1, 3, 1], high is better; index1: benefit[-2, 2,-2], low is better.
                wait = 30000.+(index+1)*high
                ride = 60000.+(index-3)*high
                fixed = dict(status="complete", dt=300., wait=wait, ride=ride, cost=2*wait+ride)
                event = dict(status="complete", dt=0. if index == 0 else 1.+high,
                    wait=0. if index == 0 else 30., ride=0. if index == 0 else 60.,
                    cost=0. if index == 0 else 120., zero_duration=index == 0)
                hold = 7.5 if high else 0.
                branch = dict(integrity=dict(all_pass=True, checks=dict(suffix_zero_training=True)),
                    selected=selected, external_seed=106255,
                    execution=dict(executed_action=actions[arm], executed_hold_seconds=hold,
                        native_applied_hold_seconds=hold, registered_hold_seconds=None if index == 2 else hold),
                    truth=dict(next_event=event, fixed300=fixed), global_transition=dict(r=-int(high)))
                write(directory/arm/"summary.json", branch)
            write(directory/"state_summary.json", record)
        write(case/"summary.json", parent)
        (case/"prefix_trace.jsonl").write_text("\n".join(json.dumps(row) for row in (
            dict(sequence=1, started_at=1., a=[-1.], proposed_a=[-.5]),
            dict(sequence=2, started_at=8999., a=[-.75], proposed_a=[0.]),
            dict(sequence=3, started_at=9000., a=[-.75], proposed_a=[0.])) )+"\n")
    return root


def test_pair_direction_regret_stable_cost_and_budget(tmp_path):
    result = module.analyze(fixture(tmp_path))
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    assert len(result["states"]) == 8
    first = result["states"][0]
    assert first[10] == [-1., 3., 1.]
    assert first[11][0][3] == 1. and first[11][1][3] == 0.
    assert first[11][1][6] == [1., -3., -1.]
    assert first[12] == -1.
    assert first[9] == [0., 0., [0., 0., 0.], -1.]
    assert result["budget"]["native_prefix_seconds"] == 48000
    assert result["budget"]["suffix_native_seconds"] == 4800
    assert result["budget"]["new_truth_labels"] == 16
    assert result["budget"]["new_optimizer_updates"] == 0
    assert result["cases"]["stable"]["overall"][4] == [2, 1, 1]
    assert len(result["zero_duration_next_events"]) == 2
    assert len(result["nonseparable_or_unknown"]) == 4
    assert len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) < 24000


def test_truncated300_and_missing_branch_never_score_as_zero(tmp_path):
    root = fixture(tmp_path)
    path = root/"stable"/"states"/"102X_early"/"high"/"summary.json"
    branch = module.read(path)
    branch["truth"]["fixed300"] = dict(status="truncated", dt=200., wait=None, ride=None, cost=None)
    write(path, branch)
    result = module.analyze(root)
    first = result["states"][0]
    assert result["status"] == "partial"
    assert first[5] == "missing300" and first[10] is None and first[12] is None
    assert first[11][1][3] is None and first[11][1][6] is None
    assert result["cases"]["stable"]["overall"][2] == 3
    path.unlink()
    result = module.analyze(root)
    assert not result["integrity"]["all_pass"]
    assert result["states"][0][5] == "missing_branch" and result["states"][0][12] is None


def test_wrong_actual_execution_parallel_or_training_excludes_truth(tmp_path):
    root = fixture(tmp_path)
    path = root/"stable"/"states"/"102X_early"/"high"/"summary.json"
    branch = module.read(path)
    branch["execution"]["native_applied_hold_seconds"] = 0.
    branch["integrity"]["checks"]["suffix_zero_training"] = False
    write(path, branch)
    record_path = path.parent.parent/"state_summary.json"
    record = module.read(record_path)
    record["pair_receipt"]["overlap_seconds"] = 0.
    write(record_path, record)
    result = module.analyze(root)
    first = result["states"][0]
    assert first[5] == "integrity_failed" and first[12] is None
    assert first[11][0][3] is None and first[11][1][6] is None
    failures = result["cases"]["stable"]["state_checks"]["102X/early"]["failed"]
    assert {"actual_pair_parallel", "high_actual_holding", "high_suffix_zero_training"} <= set(failures)


def test_unselected_unsupported_slot_is_retained(tmp_path):
    root = fixture(tmp_path)
    path = root/"stable"/"summary.json"
    parent = module.read(path)
    parent["selection"]["102X/early"] = dict(selected=None, reasons=dict(F_not_supported=4))
    parent["states"] = [state for state in parent["states"] if state["slot"] != "102X/early"]
    parent["budget"].update(states=3, suffix_sim_seconds=1800)
    write(path, parent)
    result = module.analyze(root)
    assert result["integrity"]["all_pass"] and result["status"] == "partial"
    assert len(result["states"]) == 8 and result["states"][0][5] == "unselected"
    assert result["cases"]["stable"]["unsupported_visible_attempts"] == {"102X/early": {"F_not_supported": 4}}


def test_failed_prefix_preserves_spent_budget_and_leaves_total_unknown(tmp_path):
    root = fixture(tmp_path)
    (root/"stable"/"summary.json").unlink()
    write(root/"stable"/"failure.json", dict(type="RuntimeError", message="native interruption",
        failure_budget=dict(native_prefix_seconds=10100, suffix_sim_seconds=600, states=1)))
    result = module.analyze(root)
    assert result["budget"]["native_prefix_seconds"] is None
    assert result["budget"]["suffix_native_seconds"] is None
    assert result["budget"]["failed_or_incomplete_prefix_costs"]["stable"] == dict(
        native_prefix_seconds=10100, suffix_sim_seconds=600, states=1)
    assert len(result["missing_or_partial"]) == 4


def test_native_prefix_mismatch_missing_or_short_never_pass(tmp_path):
    root = fixture(tmp_path)
    result = module.analyze(root)
    assert result["common_prefix_before9000"]["rows"] == 2
    assert result["common_prefix_before9000"]["exact"] is True
    path = root/"local_speed_drop"/"prefix_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["a"] = [-1.]
    path.write_text("\n".join(map(json.dumps, rows))+"\n")
    result = module.analyze(root)
    assert result["common_prefix_before9000"]["status"] == "different"
    assert not result["integrity"]["all_pass"]
    path.write_text(json.dumps(rows[0])+"\n")
    assert module.common_prefix(root)["status"] == "short_before_boundary"
    path.unlink()
    assert module.common_prefix(root)["exact"] is None


def test_identical_trace_does_not_substitute_for_controller_or_seed_proof(tmp_path):
    root = fixture(tmp_path)
    path = root/"local_speed_drop"/"summary.json"
    parent = module.read(path)
    parent["private_action_seed"] += 1
    parent["frozen_model_gate"]["continuation_mu14_gate"]["shared_frozen_actor_exact"] = False
    write(path, parent)
    result = module.analyze(root)
    assert result["common_prefix_before9000"]["exact"] is True
    assert result["integrity"]["checks"]["local_speed_drop_initializer_and_action_seed"] is False
    assert result["integrity"]["checks"]["local_speed_drop_same_frozen_mu14_controller"] is False
