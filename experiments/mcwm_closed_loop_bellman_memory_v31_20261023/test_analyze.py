"""Native-shaped event alignment, delayed costs and bounded v31 export tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("v31_analysis", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def physical_state(clock, shifted=False):
    return dict(native_clock_seconds=float(clock), ledger_clock_seconds=float(clock),
        action_step_length_seconds=1., simulation_step_length_seconds=1., last_action_time_seconds=float(clock),
        is_at_bus_stop=shifted, stop_state=int(shifted), speed_mps=0. if shifted else 8.,
        lane_id="route-selected-lane_0", lane_position_m=54. if shifted else 62.,
        position_xy=[54. if shifted else 62., 100.], native_bus_state="Stop" if shifted else "Edge",
        current_service_station_id="selected-stop", next_stop_id="next-stop",
        registered_stop_duration_seconds=1.875 if shifted else 0.)


def memory(clock, arm, selected=False):
    applied = clock > 18315
    return dict(observed_at=clock, selected_trip_id="7X-trip", selected_vehicle_id="7X_fleet_6",
        selected_station_id="selected-stop", actor_is_selected_vehicle=selected,
        execution_applied=applied, executed_hold_seconds=MODULE.v30.HOLDING[arm] if applied else None,
        applied_real_clock=18315. if applied else None, applied_event_clock=18314.,
        applied_issue_ledger_clock=18315., elapsed_since_apply_seconds=clock-18315. if applied else None,
        registered_stop_duration_seconds=MODULE.v30.HOLDING[arm],
        departed_observed=clock >= (18317 if arm == "high" else 18316),
        next_same_vehicle_arrived_observed=clock >= 18386.,
        latest_physical=physical_state(clock, arm == "high" and clock == 18316))


def event(number, arm="low", kind="global_transition", shift=0, changed=False):
    start = 18315.+number*3+shift
    line = ("7X", "122S", "705S")[number % 3]
    state = [0.1*index for index in range(229)]
    ns = [0.11*index for index in range(229)]
    if changed:
        state[2] += 0.02
        ns[3] += 0.03
    reward = -5050.-(12. if changed else 0.)
    transition = dict(s=state, ns=ns, raw_s=[1.]*245, raw_ns=[2.]*245,
        a=[-1.+MODULE.v30.HOLDING[arm]/30.], r=reward, dt=2., terminal=False,
        episode_id="fixed6202", domain="target", started_at=start, completed_at=start+2,
        available_at=start+2, sequence=12775+number, line_id=line,
        vehicle_id="7X_fleet_6" if number == 0 else line+"_fleet_35",
        trip_id="7X-trip" if number == 0 else line+"-trip-"+str(number),
        station_id="selected-stop" if number == 0 else "stop-"+str(number), next_vehicle_id="122S_fleet_35")
    models = {}
    for index, model in enumerate(MODULE.MODELS):
        bootstrap = -67.-index-(0.2 if changed else 0.)
        models[model] = dict(q_heads=[-68.+number/100.]*5, q_min=-68.+number/100.+MODULE.v30.HOLDING[arm]/1000.,
            next_value=-67.5-index-(0.4 if changed else 0.), value_contribution=bootstrap/2,
            sac_contribution=bootstrap/2, bootstrap_contribution=bootstrap,
            target=reward*.001+bootstrap, td_heads=[4.]*5, target_subset=list(range(5)),
            next_actions=dict(zero_executed=-1., positive_executed=-0.9), target_path_exact=True)
    return dict(kind=kind, identity={key: transition[key] for key in ("line_id", "trip_id", "station_id", "vehicle_id")},
        transition=transition, memory_s=memory(start, arm, number == 0), memory_ns=memory(start+2, arm),
        bellman=dict(dt_seconds=2., raw_reward=reward, reward_scaled=reward*.001,
            terminal=False, zero_duration=False, diagnostic_seed=316202, models=models,
            integrity=dict(rng_restored=True, learner_rng_restored=True, target_path_exact=True),
            diagnostic_budget=dict(optimizer_updates=0, training_labels=0)))


def fixture(root):
    # The original v30 fixture supplies every supported native field, fork
    # receipt and accounting field. Add the v31 data that can exceed a small
    # unit fixture: actual-sized canonical/raw vectors and all 301 clocks.
    old_spec = importlib.util.spec_from_file_location("v30_fixture", MODULE.v30.HERE/"test_analyze.py")
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    old.fixture(root)
    for scenario in MODULE.SCENARIOS:
        for arm in MODULE.ARMS:
            branch = root/"runs"/"seed6202"/scenario/"branches"/arm
            value = json.loads((branch/"summary.json").read_text())
            value["visibility"] = dict(integrity=dict(all_pass=True,
                checks=dict(memory_observed_causally=True, no_training_memory_injection=True)),
                budget=dict(optimizer_updates=0, training_labels=0), evaluation_count=35)
            # Aliased low/mid and physically separated high, as native v30.
            if arm == "mid":
                low = json.loads((branch.parent/"low"/"summary.json").read_text())
                value["physical"] = low["physical"]
            if scenario == "local_speed_drop":
                value["physical"]["missing"].update({field: 93 if arm == "high" else 95 for field in
                    ("action_step_length_seconds", "last_action_time_seconds", "is_at_bus_stop", "stop_state",
                     "speed_mps", "lane_id", "lane_position_m", "position_xy")})
            write(branch/"summary.json", value)
            events = [event(number, arm, changed=(arm == "high" and number >= 4)) for number in range(35)]
            partial = event(35, arm, kind="cutoff_partial")
            partial["transition"].update(ns=None, raw_ns=None, r=None, completed_at=18615.)
            partial.update(bellman=None, memory_ns=None, partial_cost=dict(wait=25., ride=20., cost=70.))
            write_rows(branch/"visibility.jsonl", events+[partial])
            ticks = []
            for relative in range(301):
                extra = 1. if arm == "high" and relative >= 3 else 0.
                ticks.append(dict(clock_seconds=18315.+relative, relative_seconds=relative,
                    wait=600.*relative+extra, ride=1260.*relative+extra, cost=2460.*relative+3.*extra,
                    waiting_passengers=600., in_vehicle_passengers=1260.,
                    physical=physical_state(18315.+relative, arm == "high" and relative == 1),
                    memory=memory(18315.+relative, arm)))
            write_rows(branch/"ledger_ticks.jsonl", ticks)


def test_logical_bus_stop_matching_does_not_zip_after_insertions():
    left = [event(0), event(1), event(2)]
    right = [event(0, "high"), event(77, "high"), event(1, "high", shift=1, changed=True), event(2, "high")]
    pairs, counts = MODULE.align_events(left, right)
    assert counts == dict(matched=3, unmatched=[0, 1], identity_missing=[0, 0])
    assert [a["transition"]["trip_id"] == b["transition"]["trip_id"] for _, a, b in pairs] == [True]*3
    contrast = MODULE.visibility_contrast(left, right, [], [])
    assert contrast["alignment"]["different_event_clocks"] == 1
    assert contrast["first_witness"]["r"]["strict_same_clock"] is False


def test_physical_memory_gate_does_not_promote_aliases_or_filter_on_future(tmp_path):
    low, mid, high = event(1, "low", shift=-2), event(1, "mid", shift=-2), event(1, "high", shift=-2)
    alias = MODULE.visibility_contrast([low], [mid], [], [])
    physical = MODULE.visibility_contrast([low], [high], [], [])
    assert alias["memory"]["first_command"] is not None
    assert alias["memory"]["first_physical_other_vehicle_issue"] is None
    witness = physical["memory"]["first_physical_other_vehicle_issue"]
    assert witness["canonical_s_equal"] is True and witness["strict_same_clock"] is True
    assert "physical/speed_mps" in witness["fields"]
    fixture(tmp_path)
    for arm in MODULE.ARMS:
        file = tmp_path/"runs"/"seed6202"/"stable"/"branches"/arm/"visibility.jsonl"
        rows = MODULE.jsonl(file)
        rows[1] = event(1, arm, shift=-2)
        if arm == "high":
            rows[1]["transition"].update(completed_at=18319., dt=3., next_vehicle_id="different-next-vehicle")
        write_rows(file, rows)
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert "high_minus_low" in case["memory_gate"]["hidden_physical_memory_at_matched_other_vehicle_issue"]
    witness = case["visibility"]["high_minus_low"]["memory"]["first_physical_other_vehicle_issue"]
    assert witness["strict_same_clock"] is False and witness["same_next_vehicle"] is False


def test_common_clock_cost_and_physical_onsets_are_independent_of_event_alignment(tmp_path):
    fixture(tmp_path)
    case = MODULE.analyze(tmp_path)["cases"][0]
    contrast = case["visibility"]["high_minus_low"]
    assert contrast["common_clock_physical"]["first"]["relative_seconds"] == 1
    assert contrast["common_clock_cost"]["first"]["relative_seconds"] == 3
    assert contrast["first_differences"]["r"] == [18329., 18329.]
    assert contrast["first_differences"]["fresh/target"] == [18329., 18329.]
    assert contrast["memory"]["first_other_vehicle_issue"]["canonical_s_equal"] is True
    assert case["physical"]["contrasts"]["mid_minus_low"]["status"] == "no_separation_observed"
    assert case["counts"]["low"]["cutoff_partial"] == 1


def test_same_ns_same_draw_preserves_bootstrap_target_while_q_can_differ():
    left, right = [event(0), event(1)], [event(0, "mid"), event(1, "mid")]
    result = MODULE.visibility_contrast(left, right, [], [])
    assert "fresh/q_min" in result["first_differences"]
    assert "fresh/target" not in result["first_differences"]
    assert result["common_ns"]["fresh"]["compared"] == 2
    right[1]["bellman"]["models"]["fresh"]["target"] += 1
    result = MODULE.visibility_contrast(left, right, [], [])
    assert result["common_ns"]["fresh"]["target_violations"] == 1


def test_zero_duration_cutoff_actual_and_partial_missing_are_distinct(tmp_path):
    fixture(tmp_path)
    branch = tmp_path/"runs"/"seed6202"/"stable"/"branches"/"low"
    rows = MODULE.jsonl(branch/"visibility.jsonl")
    rows[1]["transition"]["dt"] = 0
    rows[1]["transition"]["completed_at"] = rows[1]["transition"]["started_at"]
    rows[2]["kind"] = "cutoff_actual_without_issue"
    write_rows(branch/"visibility.jsonl", rows)
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert case["counts"]["low"]["zero_duration"] == 1
    assert case["counts"]["low"]["cutoff_actual_without_issue"] == 1
    assert case["counts"]["low"]["cutoff_partial"] == 1
    assert case["counts"]["low"]["missing_next_state"] == 0
    (branch/"ledger_ticks.jsonl").unlink()
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert case["integrity"]["all_pass"] is False
    assert "low/visibility_present" in case["integrity"]["failed"]


def test_future_memory_and_bootstrap_drift_are_rejected(tmp_path):
    fixture(tmp_path)
    branch = tmp_path/"runs"/"seed6202"/"stable"/"branches"/"high"
    rows = MODULE.jsonl(branch/"visibility.jsonl")
    rows[0]["memory_s"]["applied_real_clock"] = rows[0]["memory_s"]["observed_at"]+2
    rows[0]["bellman"]["models"]["fresh"]["bootstrap_contribution"] += .5
    write_rows(branch/"visibility.jsonl", rows)
    case = MODULE.analyze(tmp_path)["cases"][0]
    assert "high/causal_memory" in case["integrity"]["failed"]
    assert "common_ns_exact_bootstrap_target" in case["integrity"]["failed"]


def test_rich_native_schema_export_stays_small_and_separates_added_budget(tmp_path):
    fixture(tmp_path)
    result = MODULE.run(SimpleNamespace(root=tmp_path, out=tmp_path/"small.json", wait=False))
    assert result["status"] == "complete"
    assert (tmp_path/"small.json").stat().st_size <= 20000
    assert result["budget"]["parent_full_day_replay_seconds"] == 48000
    assert result["budget"]["prefix_native_seconds"] == 36626
    assert result["budget"]["replayed_feedback_slots"] == 570
    assert result["budget"]["replayed_main_updates"] == 912
    assert result["budget"]["replayed_cumulative_shadow_updates"] == 992
    assert result["budget"]["observed_suffix_seconds"] == 1800
    assert result["budget"]["truth_labels_enter_training"] == 0
    assert result["budget"]["frozen_transition_probes"] == 210
    assert result["budget"]["frozen_model_probes"] == 420
    assert result["budget"]["native_target_parity_probes"] == 12


def test_parent_failure_is_terminal_to_waiting_finalizer(tmp_path):
    fixture(tmp_path)
    root = tmp_path/"runs"/"seed6202"/"stable"
    (root/"summary.json").unlink()
    write(root/"failure.json", dict(type="RuntimeError", message="old v30 prediction mismatch"))
    result = MODULE.run(SimpleNamespace(root=tmp_path, out=tmp_path/"small.json", wait=True))
    assert result["status"] == "failed" and result["failures"] == ["stable"]
