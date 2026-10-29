"""Cost attribution is scored only after native restoration/diagonal validity."""
import json

import pytest

from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026 import analyze as mod


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mutate(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    write(path, value)


def fixture(root, *, waits=(12000., 10000., 10500., 10000.)):
    for seed in mod.SEEDS:
        for entry_index, entry in enumerate(mod.ARMS):
            base = root/"runs"/f"seed{seed}"/entry
            budget = dict(restoration_feedback_slots=285, native_prefix_seconds=18301.)
            budget.update({"restoration_"+family+"_updates": 456 for family in ("critic", "value", "actor", "alpha")})
            write(base/"prefix_summary.json", dict(status="complete", cell=dict(seed=seed, scenario="stable", entry_arm=entry),
                integrity=dict(all_pass=True, checks=dict(native=True)), budget=budget, prefix_clock=18301.,
                trace_rows=100, matched_prefix_rows=100))
            clocks = {"low": dict(native_started_wall=10., native_finished_wall=30.),
                      "high": dict(native_started_wall=11., native_finished_wall=32.)}
            write(base/"suffixes/live_fork.json", dict(all_pass=True, native_before=18301., native_after=18301.,
                counters_before=dict(critic=456), counters_after=dict(critic=456),
                parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
                route_streams=dict(parent_offsets_unchanged=True, parent_offsets_before={"route": 123}, parent_offsets_after={"route": 123}),
                overlap_seconds=19., children=clocks))
            for policy_index, (branch, policy) in enumerate(mod.BRANCHES.items()):
                wait, ride = waits[entry_index*2+policy_index], 30000.
                endpoints = {}
                for clock, fraction in ((18300, 0), (20100, .5), (24000, 1)):
                    w, r = 100000.+wait*fraction, 200000.+ride*fraction
                    endpoints[str(clock)] = dict(time_seconds=clock, accrued_wait_seconds=w,
                        accrued_in_vehicle_seconds=r, accrued_generalized_passenger_seconds=2*w+r,
                        waiting_passengers=(330+entry_index) if clock == 18300 else (337 if policy == "fresh" else 331),
                        in_vehicle_passengers=0)
                costs = {}
                for period, (start, end) in mod.PERIODS.items():
                    w = endpoints[str(end)]["accrued_wait_seconds"]-endpoints[str(start)]["accrued_wait_seconds"]
                    r = endpoints[str(end)]["accrued_in_vehicle_seconds"]-endpoints[str(start)]["accrued_in_vehicle_seconds"]
                    costs[period] = dict(wait=w, ride=r, cost=2*w+r, denominator=15079, cost_per_passenger=(2*w+r)/15079)
                write(base/"suffixes"/branch/"summary.json", dict(status="complete",
                    integrity=dict(all_pass=True, checks=dict(native=True)),
                    cell=dict(seed=seed, scenario="stable", entry_arm=entry, policy_arm=policy),
                    switch_time=18301., baseline18300=endpoints["18300"], endpoints=endpoints,
                    cost_windows=costs, final_waiting=endpoints["24000"]["waiting_passengers"],
                    zero_suffix_training=True, execution_wall=clocks[branch],
                    diagonal_reference={key: True if entry == policy else None
                        for key in ("ledger_exact", "metrics_exact", "full_remaining_trace_exact")},
                    service=dict(final_waiting=337 if policy == "fresh" else 331)))


def suffix(root, seed=6201, entry="fresh", branch="low"):
    return root/"runs"/f"seed{seed}"/entry/"suffixes"/branch/"summary.json"


def test_same_entry_policy_effect_and_exact_decomposition(tmp_path):
    fixture(tmp_path)
    result, _ = mod.analyze(tmp_path)
    assert result["status"] == "complete"
    d = result["seeds"][0]["decomposition"]["tail"]
    assert d["fresh_entry_policy_fresh_minus_no_fresh"]["cost"] == 4000
    assert d["no_fresh_entry_policy_fresh_minus_no_fresh"]["cost"] == 1000
    assert d["policy_component_P"]["cost"] == 2500
    assert d["state_component_S"]["cost"] == 1500
    assert d["interaction"]["cost"] == 3000
    assert d["original_diagonal_fresh_minus_no_fresh"]["cost"] == 4000
    assert d["additive_identity_exact"] is True
    assert result["decision"]["reason"] == "may_consider_fixed_late_feedback_candidate"
    assert result["decision"]["new_training_authorized_by_diagnostic"] is False
    assert result["budget"]["restoration_updates_per_family"] == 1824
    assert result["budget"]["known_total_native_seconds"] == 118796


@pytest.mark.parametrize("kind", ["diagonal", "normalization", "suffix_training", "prefix_restore"])
def test_failed_native_contract_blocks_causal_interpretation(tmp_path, kind):
    fixture(tmp_path)
    if kind == "diagonal":
        mutate(suffix(tmp_path), lambda row: row["diagonal_reference"].update(full_remaining_trace_exact=False))
    elif kind == "normalization":
        mutate(suffix(tmp_path), lambda row: row["cost_windows"]["tail"].update(denominator=100))
    elif kind == "suffix_training":
        mutate(suffix(tmp_path), lambda row: row.update(zero_suffix_training=False))
    else:
        mutate(tmp_path/"runs/seed6201/fresh/prefix_summary.json",
               lambda row: row["budget"].update(restoration_critic_updates=455))
    result, _ = mod.analyze(tmp_path)
    assert result["status"] == "integrity_failed"
    assert result["seeds"][0]["decomposition"] is None
    assert result["decision"]["feedback_redistribution_may_be_considered"] is False
    assert result["integrity"]["failed_checks"]


def test_missing_suffix_is_unknown_not_zero(tmp_path):
    fixture(tmp_path)
    suffix(tmp_path).unlink()
    result, _ = mod.analyze(tmp_path)
    assert not result["complete"]
    assert result["missing"]
    assert result["seeds"][0]["decomposition"] is None
    assert len(result["cells"]) == 7
    assert result["decision"]["reason"] == "incomplete_or_integrity_failed"


def test_no_wall_overlap_rejects_claimed_parallel(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs/seed6201/fresh/suffixes/live_fork.json"
    mutate(path, lambda row: row["children"]["high"].update(native_started_wall=31.))
    result, _ = mod.analyze(tmp_path)
    assert "seed6201/fresh/fork/actual_parallel" in result["integrity"]["failed_checks"]
    assert result["seeds"][0]["decomposition"] is None


def test_seed_disagreement_stops_without_expansion(tmp_path):
    fixture(tmp_path)
    # Both policy effects are negative for6203: its fresh final actor is better.
    other = tmp_path/"other"
    fixture(other, waits=(8000., 10000., 9000., 10000.))
    import shutil
    shutil.rmtree(tmp_path/"runs/seed6203")
    shutil.copytree(other/"runs/seed6203", tmp_path/"runs/seed6203")
    result, _ = mod.analyze(tmp_path)
    assert result["status"] == "complete"
    assert result["decision"]["reason"] == "interaction_stop_no_seed_expansion"
    assert result["decision"]["automatic_seed_expansion"] is False


def test_no_existing_policy_rescue_directs_to_prior_service(tmp_path):
    fixture(tmp_path, waits=(8000., 10000., 9000., 10000.))
    result, _ = mod.analyze(tmp_path)
    assert result["decision"]["reason"] == "prior_state_service_diagnosis"


def test_export_stays_small_and_never_overwrites(tmp_path):
    fixture(tmp_path)
    out = tmp_path/"results/diagnostic.json"
    receipt = mod.export(tmp_path, out)
    assert receipt["status"] == "complete"
    assert set(path.name for path in out.parent.iterdir()) == {"diagnostic.json", "service.json"}
    assert all(size < 48000 for size in receipt["files"].values())
    with pytest.raises(FileExistsError):
        mod.export(tmp_path, out)


def test_waiting_id_exchange_visible_even_with_equal_counts():
    def point(people):
        return dict(waiting_inventory_complete=True, waiting_matches_ledger=True,
            waiting_passengers=[dict(passenger_id=person, station_id="stopA", passable_lines=["7X"])
                                for person in people])
    delta = mod.waiting_comparison(point(["person_common", "person_left"]), point(["person_common", "person_right"]))
    assert delta["common_waiting"] == 1
    assert delta["left_only"] == delta["right_only"] == 1
    assert delta["left_minus_right"] == 0
    assert delta["left_only_station_groups"] == [dict(station_id="stopA", passable_lines=("7X",), passengers=1)]
    assert "person_left" not in json.dumps(delta)


def test_incomplete_waiting_inventory_is_not_an_empty_set():
    result = mod.waiting_comparison(dict(waiting_inventory_complete=False), None)
    assert result["status"] == "inventory_incomplete"
    assert result["membership_difference"] is None


def test_actual_memberships_remain_valid_with_known_ledger_tick_lag():
    def point(person, count):
        return dict(waiting_inventory_complete=True, waiting_matches_ledger=False,
            ledger_waiting=count, waiting_passengers=[dict(passenger_id=person, station_id="A", passable_lines=["7X"])])
    result = mod.waiting_comparison(point("a", 2), point("b", 3))
    assert result["status"] == "complete"
    assert result["left_only"] == result["right_only"] == 1
    assert result["ledger_waiting_left_right"] == [2, 3]
    assert result["actual_unique_left_right"] == [1, 1]


def test_missing_fork_receipt_blocks_decomposition_even_with_both_costs(tmp_path):
    fixture(tmp_path)
    (tmp_path/"runs/seed6201/fresh/suffixes/live_fork.json").unlink()
    result, _ = mod.analyze(tmp_path)
    assert result["status"] == "pending"
    assert result["seeds"][0]["decomposition"] is None
