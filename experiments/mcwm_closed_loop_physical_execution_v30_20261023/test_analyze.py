"""Three-action physical separation, directional truth, and small export tests."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location("v30_analysis", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def endpoint(wait, ride, dt=300):
    return dict(status="complete", dt=dt, wait=wait, ride=ride, cost=2*wait+ride)


def state(clock, position=0., stopped=True):
    return dict(native_clock_seconds=clock, ledger_clock_seconds=clock+1,
        action_step_length_seconds=1., simulation_step_length_seconds=1., last_action_time_seconds=clock,
        is_at_bus_stop=stopped, stop_state=1 if stopped else 0,
        speed_mps=0. if stopped else 4., lane_id="road_0", lane_position_m=position,
        position_xy=[position, 1.], native_bus_state="Stop" if stopped else "Edge",
        current_service_station_id="selected-stop", next_stop_id="next-stop")


def physical(index):
    return dict(apply_before=dict(**state(18314.), registered_stop_duration_seconds=3600.),
        apply_after=dict(**state(18314.), registered_stop_duration_seconds=(0., 0.938, 1.875)[index]),
        short_tick_samples=[state(18315.+tick, 3.*(tick-index), tick <= index) for tick in range(12)],
        departure=dict(status="observed", station_id="selected-stop", clock_seconds=18316.+index,
                       observed_at=18316.+index, time_source="traci_stop_transition"),
        next_same_vehicle_arrival=dict(status="observed", station_id="next-stop", clock_seconds=18390.+index,
                       observed_at=18390.+index, time_source="traci_stop_transition"),
        native_history_departure=dict(status="observed", station_id="selected-stop", clock_seconds=18316.+index,
            observed_at=18317.+index, time_source="native_history_depart_stop_time_d"),
        native_history_next_arrival=dict(status="observed", station_id="next-stop", clock_seconds=18390.+index,
            observed_at=18391.+index, time_source="native_history_arriver_stop_time_d"),
        transitions=[state(18315.+tick, tick, tick < 2) for tick in range(32)],
        coverage=dict(actual_clock_observed=True, action_step_length_observed=True,
            issue_stop_state_observed=True, departure_observed=True, next_arrival_observed=True),
        missing={key: 0 for key in MODULE.STATE_FIELDS})


def fixture(root):
    for scenario in MODULE.SCENARIOS:
        path = root/"runs"/"seed6202"/scenario
        issue_time = 18315 if scenario == "stable" else 18311
        selected = dict(sequence=123, started_at=issue_time, line_id="7X", station_id="selected-stop")
        main = dict(status="complete", integrity=dict(all_pass=True, checks=dict(prefix=True, rng=True,
            v29_predictions_exact=True, v29_truth_exact=True)), selected=selected, external_seed=106202,
            budget=dict(main_feedback_slots=285, main_updates=456,
                cumulative_shadow_updates=496, prefix_replay_seconds=issue_time, native_replay_seconds=24000),
            prediction=dict(fresh=dict(min_q_low=-2., min_q_mid=-1., min_q_high=-3.),
                no_fresh=dict(min_q_low=-1., min_q_mid=-2., min_q_high=-3.)))
        write(path/"summary.json", main)
        clocks = {}
        for index, arm in enumerate(MODULE.ARMS):
            clocks[arm] = dict(pid=100+index, native_started_wall=100.+index, native_finished_wall=110.+index)
            truth = dict(next_event=endpoint(10+index, 10-4*index, dt=5+index),
                         fixed300=endpoint(100+index, 100-4*index))
            value = dict(status="complete", integrity=dict(all_pass=True), arm=arm,
                selected=selected, external_seed=106202,
                execution=dict(requested_hold_seconds=MODULE.HOLDING[arm], executed_hold_seconds=MODULE.HOLDING[arm],
                    native_applied_hold_seconds=MODULE.HOLDING[arm], executed_action=[-1.+index/32.],
                    native_application=dict(bus_id="7X-vehicle", station_id="selected-stop", time=18314.,
                        issue_ledger_time=issue_time, assigned_action=MODULE.HOLDING[arm],
                        applied_hold_seconds=MODULE.HOLDING[arm], real_clock_before=18314., real_clock_after=18314.,
                        predicted_departure_time_seconds=18316.+MODULE.HOLDING[arm])), native=dict(other_batch_actions=[]),
                truth=truth, physical=physical(index),
                counters=dict(before=dict(updates=456), after=dict(updates=456), delta=dict(updates=0)),
                budget=dict(suffix_sim_seconds=300, replay_target_labels=0, replay_policy_updates=0, new_truth_labels=1),
                execution_wall=clocks[arm])
            write(path/"branches"/arm/"summary.json", value)
        fork = dict(all_pass=True, children=clocks, child_exitcodes=dict.fromkeys(MODULE.ARMS, 0),
            native_before=issue_time, native_after=issue_time, counters_before=dict(updates=456), counters_after=dict(updates=456),
            parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
            route_streams=dict(parent_offsets_unchanged=True))
        write(path/"branches"/"live_fork.json", fork)


def test_three_physical_actions_and_cost_regret_are_separate(tmp_path):
    fixture(tmp_path)
    value = MODULE.analyze(tmp_path)
    assert value["status"] == "complete"
    case = value["cases"][0]
    assert case["physical"]["three_distinct_observed_responses"] is True
    assert abs(case["physical"]["arms"]["mid"]["registered_hold_minus_command_seconds"]-0.0005) < 1e-12
    assert case["physical"]["contrasts"]["high_minus_low"]["actual_departure"]["delta_seconds"] == 2
    contrast = case["endpoints"]["fixed300"]["contrasts"]["mid_minus_low"]
    assert contrast["delta"] == dict(dt=0, wait=1, ride=-4, cost=-2)
    assert contrast["reward_delta"] == 2 and contrast["preferred_action"] == "mid"
    assert case["predictions"]["fresh"]["contrasts"]["mid_minus_low"]["strict_sign_agreement"]["fixed300"] is True
    assert case["predictions"]["fresh"]["fixed300"]["realized_regret"] == 2
    assert case["predictions"]["no_fresh"]["fixed300"]["realized_regret"] == 4
    assert case["endpoints"]["next_event"]["contrasts"]["high_minus_low"]["unequal_horizons"] is True


def test_observed_tie_is_unscored_and_missing_arrival_is_not_zero(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"seed6202"/"stable"/"branches"
    for arm in MODULE.ARMS:
        file = path/arm/"summary.json"
        row = json.loads(file.read_text())
        row["truth"]["fixed300"] = endpoint(100, 100)
        row["physical"]["next_same_vehicle_arrival"] = dict(status="truncated", cutoff_clock_seconds=18615.)
        write(file, row)
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert case["integrity"]["all_pass"] is True
    assert case["physical"]["missing_next_arrival_arms"] == list(MODULE.ARMS)
    for model in MODULE.MODELS:
        assert case["predictions"][model]["fixed300"]["realized_regret"] == 0
        assert case["predictions"][model]["contrasts"]["mid_minus_low"]["strict_sign_agreement"]["fixed300"] is None
    left, right = physical(0), physical(0)
    right["departure"] = dict(status="missing")
    right["next_same_vehicle_arrival"] = dict(status="missing")
    difference = MODULE.physical_contrast(left, right)
    assert difference["actual_departure"]["delta_seconds"] is None
    assert difference["status"] == "insufficient"
    right = physical(0)
    assert MODULE.physical_contrast(left, right)["status"] == "no_separation_observed"


def test_zero_duration_truncated_and_missing_truth_remain_distinct():
    branches = {arm: dict(truth=dict(next_event=endpoint(0, 0, dt=0))) for arm in MODULE.ARMS}
    branches["mid"]["truth"]["next_event"] = dict(status="truncated", dt=2, wait=1, ride=0, cost=2)
    branches["high"]["truth"] = {}
    value = MODULE.endpoint_set(branches, "next_event")
    assert value["zero_duration_arms"] == ["low"]
    assert value["truncated_arms"] == ["mid"]
    assert value["missing_arms"] == ["high"]
    assert value["contrasts"]["high_minus_low"]["delta"] is None
    assert value["contrasts"]["mid_minus_low"]["reward_delta"] is None


def test_serial_third_arm_and_training_update_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"seed6202"/"stable"/"branches"
    file = path/"high"/"summary.json"
    row = json.loads(file.read_text())
    row["execution_wall"].update(native_started_wall=112., native_finished_wall=120.)
    row["counters"]["after"]["updates"] = 457
    row["counters"]["delta"]["updates"] = 1
    write(file, row)
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert case["integrity"]["all_pass"] is False
    assert "actual_three_arm_parallel" in case["integrity"]["failed"]
    assert "high/zero_suffix_training" in case["integrity"]["failed"]
    assert "high/counters_unchanged" in case["integrity"]["failed"]


def test_budget_separates_full_replay_prefix_and_six_suffixes_and_export_is_bounded(tmp_path):
    fixture(tmp_path)
    out = tmp_path/"small.json"
    value = MODULE.run(SimpleNamespace(root=tmp_path, out=out, wait=False))
    assert out.stat().st_size < 20000
    budget = value["budget"]
    assert budget["parent_full_day_replay_seconds"] == 48000
    assert budget["prefix_native_seconds"] == 36626
    assert budget["replayed_feedback_slots"] == 570
    assert budget["replayed_main_updates"] == 912
    assert budget["replayed_cumulative_shadow_updates"] == 992
    assert budget["observed_suffix_seconds"] == 1800
    assert budget["diagnostic_action_truths"] == 6
    assert budget["new_training_labels"] == budget["suffix_optimizer_updates"] == 0


def test_prefix_failure_is_terminal_for_waiting_finalizer(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"seed6202"/"stable"
    (path/"summary.json").unlink()
    write(path/"failure.json", dict(type="RuntimeError", message="prefix mismatch"))
    value = MODULE.run(SimpleNamespace(root=tmp_path, out=tmp_path/"failure_summary.json", wait=True))
    assert value["status"] == "failed" and value["failures"] == ["stable"]
