"""Tests of cost direction, paired truth integrity, and bounded terminal exports."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location("v29_analysis", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def endpoint(wait, ride, dt=300):
    return dict(status="complete", dt=dt, wait=wait, ride=ride, cost=2*wait+ride)


def fixture(root):
    for scenario in MODULE.SCENARIOS:
        path = root/"runs"/"seed6202"/scenario
        selected = dict(sequence=123, started_at=18400., line_id="7X", station_id="stop")
        main = dict(status="complete", integrity=dict(all_pass=True, checks=dict(prefix=True, rng=True)),
            selected=selected, external_seed=1234,
            budget=dict(main_feedback_slots=285, main_updates=456,
                cumulative_shadow_updates=496, prefix_replay_seconds=18400, native_replay_seconds=24000),
            prediction=dict(fresh=dict(min_q_low=-2., min_q_high=-1.),
                no_fresh=dict(min_q_low=-1., min_q_high=-2.)))
        write(path/"summary.json", main)
        clocks = {}
        for index, arm in enumerate(MODULE.ARMS):
            clocks[arm] = dict(pid=100+index, native_started_wall=100.+index, native_finished_wall=110.+index)
            truth = dict(next_event=endpoint(10+index, 10-4*index, dt=5+index),
                         fixed300=endpoint(100+index, 100-4*index))
            value = dict(status="complete", integrity=dict(all_pass=True), arm=arm,
                selected=selected, external_seed=1234,
                execution=dict(requested_hold_seconds=MODULE.HOLDING[arm], executed_hold_seconds=MODULE.HOLDING[arm],
                               native_applied_hold_seconds=MODULE.HOLDING[arm]), native=dict(other_batch_actions=[]),
                truth=truth, counters=dict(before=dict(updates=456), after=dict(updates=456), delta=dict(updates=0)),
                budget=dict(suffix_sim_seconds=300, replay_target_labels=0, replay_policy_updates=0, new_truth_labels=1),
                execution_wall=clocks[arm])
            write(path/"branches"/arm/"summary.json", value)
        fork = dict(all_pass=True, children=clocks, child_exitcodes=dict(low=0, high=0),
            native_before=18400., native_after=18400., counters_before=dict(updates=456), counters_after=dict(updates=456),
            parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
            route_streams=dict(parent_offsets_unchanged=True))
        write(path/"branches"/"live_fork.json", fork)


def test_realized_reward_is_opposite_cost_and_does_not_merge_wait_ride():
    value = MODULE.endpoint_pair(endpoint(100, 100), endpoint(101, 96), "fixed300")
    assert value["high_minus_low"] == dict(dt=0, wait=1, ride=-4, cost=-2)
    assert value["reward_high_minus_low"] == 2
    assert value["preferred_action"] == "high"
    q = MODULE.compare_prediction(dict(fresh=dict(min_q_low=-2, min_q_high=-1),
        no_fresh=dict(min_q_low=-1, min_q_high=-2)), {"fixed300": value})
    assert q["fresh"]["endpoint_comparison"]["fixed300"]["strict_sign_agreement"] is True
    assert q["no_fresh"]["endpoint_comparison"]["fixed300"]["strict_sign_agreement"] is False


def test_two_pairs_retain_unequal_event_horizons_and_fit_small_export(tmp_path):
    fixture(tmp_path)
    value = MODULE.analyze(tmp_path)
    assert value["status"] == "complete"
    assert value["cases"][0]["endpoints"]["next_event"]["unequal_horizons"] is True
    assert value["budget"]["replayed_feedback_slots"] == 570
    assert value["budget"]["replayed_main_updates"] == 912
    assert value["budget"]["replayed_cumulative_shadow_updates"] == 992
    assert value["budget"]["observed_suffix_seconds"] == 1200
    assert value["budget"]["parent_full_day_replay_seconds"] == 48000
    assert value["budget"]["prefix_native_seconds"] == 36800
    out = tmp_path/"small.json"
    MODULE.run(SimpleNamespace(root=tmp_path, out=out, wait=False))
    assert out.stat().st_size < 20000


def test_truncation_zero_duration_and_missing_are_not_converted_to_zero_truth():
    low = endpoint(0, 0, dt=0)
    high = dict(status="truncated", dt=298, wait=1, ride=1, cost=3)
    pair = MODULE.endpoint_pair(low, high, "fixed300")
    assert pair["high_minus_low"] is None and pair["reward_high_minus_low"] is None
    assert pair["zero_duration_arms"] == ["low"]
    assert pair["truncated_arms"] == ["high"]
    assert MODULE.endpoint_pair(None, endpoint(1, 1), "fixed300")["missing_arms"] == ["low"]


def test_serial_native_suffix_and_one_training_update_fail_pair(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"seed6202"/"stable"/"branches"
    high = json.loads((path/"high"/"summary.json").read_text())
    high["execution_wall"].update(native_started_wall=112., native_finished_wall=120.)
    high["counters"]["after"]["updates"] = 457
    high["counters"]["delta"]["updates"] = 1
    write(path/"high"/"summary.json", high)
    value = MODULE.analyze(tmp_path)
    case = value["cases"][0]
    assert value["status"] == "integrity_failed"
    assert case["integrity"]["checks"]["actual_parallel"] is False
    assert case["integrity"]["checks"]["high/zero_suffix_training"] is False
    assert case["integrity"]["checks"]["high/counters_unchanged"] is False


def test_failed_prefix_is_terminal_for_waiting_finalizer(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"seed6202"/"stable"
    (path/"summary.json").unlink()
    write(path/"failure.json", dict(type="RuntimeError", message="prefix mismatch"))
    value = MODULE.run(SimpleNamespace(root=tmp_path, out=tmp_path/"failure_summary.json", wait=True))
    assert value["status"] == "failed"
    assert value["failures"] == ["stable"]
