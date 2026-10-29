"""Paired truth signs, missing/zero endpoints, integrity and full-size export."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from experiments.mcwm_closed_loop_memory_truth_v34_20261024 import analyze as module


FIELDS = ["action_step_length_seconds", "is_at_bus_stop", "lane_id", "lane_position_m",
          "last_action_time_seconds", "position_xy", "speed_mps", "stop_state"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def fixture(tmp_path):
    root = tmp_path/"runs"/"seed6202"
    protocol = dict(scenarios=["stable", "local_speed_drop"], focus_lines=["7X", "102X", "705X"],
        windows=dict(early=[9300, 9900], tail=[18300, 18900]), native_external_seed=106203,
        compute_budget=dict(frozen_model_reconstruction_updates_total=1824,
            old_feedback_slots_per_model=285, old_updates_per_model=456, new_native_prefixes=2,
            prefix_native_seconds=48000, maximum_heldout_states=12, maximum_native_suffixes=24,
            maximum_suffix_native_seconds=7200, new_training_labels=0,
            new_prefix_optimizer_updates=0, suffix_optimizer_updates=0))
    write(tmp_path/"protocol.json", protocol)
    for scenario in protocol["scenarios"]:
        case = root/scenario
        parent = dict(integrity=dict(all_pass=True, checks=dict(v33_reconstruction_exact=True,
            models_unchanged=True, new_prefix_and_suffix_zero_training=True,
            all_pairs_parallel_and_parent_preserved=True)), native_external_seed=106203,
            selection={}, states=[], budget=dict(reconstruction_updates=912,
                native_prefix_seconds=24000, states=6, suffix_sim_seconds=3600))
        for index, (line, window) in enumerate((line, window) for line in protocol["focus_lines"] for window in protocol["windows"]):
            slot = line+"/"+window
            clock = protocol["windows"][window][0]+index*11.
            selected = dict(sequence=12000+index, started_at=clock, trip_id=line+"_49",
                line_id=line, station_id=line+"_station", memory_s=dict(vector=[.1]*84))
            parent["selection"][slot] = dict(selected=dict(sequence=selected["sequence"], issue=clock),
                attempted=1, reasons={}, status="selected")
            directory = case/"states"/slot.replace("/", "_")
            parent["states"].append(dict(slot=slot, path=str(directory/"state_summary.json")))
            comparison = dict(physical_separation=dict(status=["separable", "nonseparable", "physical_unknown"][index%3],
                departure_gap_seconds=1. if index%3 == 0 else 0. if index%3 == 1 else None,
                next_same_vehicle_arrival_gap_seconds=2. if index%3 == 0 else 0. if index%3 == 1 else None,
                low_departure_status="observed", high_departure_status="observed",
                low_next_arrival_status="observed", high_next_arrival_status="truncated" if index%3 == 2 else "observed"))
            actions = dict(low=[-1.], high=[-.9375])
            record = dict(slot=slot, selected=selected, pair_receipt=dict(all_pass=True), comparison=comparison,
                actions=actions, holds=dict(low=0., high=1.875), prediction={},
                branch_paths={arm:str(directory/arm/"summary.json") for arm in ("low", "high")})
            for model in module.MODELS:
                delta = (1 if (model == "memory") == (index%2 == 0) else -1)*.0000123456789123
                baseline = -123.123456789123
                record["prediction"][model] = dict(q_min=[baseline, baseline+delta],
                    q_mean=[baseline+.5123456789, baseline+.5123456789+delta*2],
                    predicted_choice=int(delta > 0), executed_actions=[-1., -.9375],
                    actor_holding_seconds=1.123456789123)
            for arm in ("low", "high"):
                wait = 30000.+(index+1)*(1 if arm == "high" else 0)
                ride = 60000.+(index-3)*(1 if arm == "high" else 0)
                fixed = dict(status="complete", dt=300., wait=wait, ride=ride, cost=2*wait+ride)
                event = dict(status="complete", dt=0. if index == 0 else 1.+(arm == "high"),
                    wait=0. if index == 0 else 30., ride=0. if index == 0 else 60.,
                    cost=0. if index == 0 else 120., zero_duration=index == 0)
                branch = dict(integrity=dict(all_pass=True), selected=selected, external_seed=106203,
                    execution=dict(executed_action=actions[arm], registered_hold_seconds=None if index == 2 else
                        (0. if arm == "low" else 1.875)),
                    physical=dict(missing={field:95+index for field in FIELDS}),
                    truth=dict(next_event=event, fixed300=fixed))
                write(directory/arm/"summary.json", branch)
            write(directory/"state_summary.json", record)
        write(case/"summary.json", parent)
    return root


def test_twelve_state_worst_case_summary_fits_and_has_correct_budget_and_regret(tmp_path):
    root = fixture(tmp_path)
    result = module.analyze(root)
    assert result["integrity"]["all_pass"] and result["status"] == "complete"
    assert len(result["states"]) == 12
    assert result["budget"]["reconstruction_updates"] == 1824
    assert result["budget"]["prefix_native_seconds"] == 48000
    assert result["budget"]["suffix_native_seconds"] == 7200
    assert result["budget"]["new_truth_labels"] == 24
    first = result["states"][0]
    assert first[10] == [-1., 3., 1.]
    assert first[11][0][3] == 1. and first[11][1][3] == 0.
    assert first[12] == -1.
    assert first[13]["zero_next_event_arms"] == ["low", "high"]
    assert first[5] == "scored"  # Zero next duration does not erase fixed300 truth.
    case = result["cases"]["stable"]
    assert case["overall"][4] == [2, 2, 2]
    assert case["separable_only"][0] == 2
    assert case["overall"][6] == 1
    size = len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode())+1
    assert size <= 20000, f"12-state worst compact summary is {size} bytes"


def test_missing300_has_no_regret_and_does_not_become_zero(tmp_path):
    root = fixture(tmp_path)
    path = root/"stable"/"states"/"7X_early"/"high"/"summary.json"
    branch = module.read(path)
    branch["truth"]["fixed300"] = dict(status="truncated", dt=200., wait=None, ride=None, cost=None)
    write(path, branch)
    result = module.analyze(root)
    first = result["states"][0]
    assert first[5] == "missing300" and first[10] is None and first[12] is None
    assert first[11][0][3] is None and first[11][1][3] is None
    assert result["cases"]["stable"]["overall"][2] == 5


def test_unselected_slot_is_coverage_missing_but_missing_branch_fails_integrity(tmp_path):
    root = fixture(tmp_path)
    parent_path = root/"stable"/"summary.json"
    parent = module.read(parent_path)
    parent["selection"]["7X/early"]["selected"] = None
    parent["states"] = [record for record in parent["states"] if record["slot"] != "7X/early"]
    write(parent_path, parent)
    result = module.analyze(root)
    assert result["integrity"]["all_pass"]
    assert result["states"][0][5] == "unselected"
    assert result["states"][0][10] is None
    (root/"stable"/"states"/"7X_tail"/"high"/"summary.json").unlink()
    result = module.analyze(root)
    assert not result["integrity"]["all_pass"]
    assert result["states"][1][5] == "missing_branch"
    assert result["states"][1][12] is None


def test_changed_selected_or_executed_action_is_not_scored(tmp_path):
    root = fixture(tmp_path)
    path = root/"stable"/"states"/"7X_early"/"high"/"summary.json"
    branch = module.read(path)
    branch["selected"]["sequence"] += 1
    branch["execution"]["executed_action"] = [-.96875]
    write(path, branch)
    result = module.analyze(root)
    assert not result["integrity"]["all_pass"]
    assert result["states"][0][5] == "integrity_failed"
    assert result["states"][0][11][0][3] is None
    assert result["cases"]["stable"]["overall"][2] == 5


def test_cli_writes_failure_receipt_so_coordinator_does_not_wait_forever(tmp_path):
    destination = tmp_path/"truth_summary.json"
    command = [sys.executable, str(Path(module.__file__)), "--root", str(tmp_path/"missing"/"runs"/"seed6202"),
               "--out", str(destination)]
    completed = subprocess.run(command, capture_output=True)
    assert completed.returncode != 0 and not destination.exists()
    assert (tmp_path/"truth_summary_failure.json").exists()
