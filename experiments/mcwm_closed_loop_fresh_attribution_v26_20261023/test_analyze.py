"""New seed routing and fixed-window passenger-cost replication endpoints."""

import json

import pytest

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_analyze import fixture
from experiments.mcwm_closed_loop_fresh_attribution_v26_20261023.analyze import (
    ARMS, SCENARIOS, SEED, analyze, fixed_replication_endpoints,
)


REFERENCE = dict(seed=6201, departed_passengers=15079, source="retained v25r1 small summary",
    windows=dict(stable=dict(
        early_shock=dict(accrued_wait_seconds=-56304., accrued_in_vehicle_seconds=24970.,
                         generalized_passenger_seconds=-87638.),
        post_budget=dict(accrued_wait_seconds=11355., accrued_in_vehicle_seconds=-8981.,
                         generalized_passenger_seconds=13729.)),
        local_speed_drop=dict(
            early_shock=dict(accrued_wait_seconds=21955., accrued_in_vehicle_seconds=130122.,
                             generalized_passenger_seconds=174032.),
            post_budget=dict(accrued_wait_seconds=-16099., accrued_in_vehicle_seconds=-21676.,
                             generalized_passenger_seconds=-53874.))))


def new_seed_fixture(root):
    fixture(root)
    for scenario in SCENARIOS:
        base = root/"runs"/scenario
        (base/"seed6201").rename(base/f"seed{SEED}")
        for arm in ARMS:
            directory = base/f"seed{SEED}"/arm
            summary = json.loads((directory/"summary.json").read_text())
            summary["cell"]["seed"] = SEED
            summary["initializer"]["path"] = f"/resident/h2oplus_seed{SEED}.pt"
            summary["scenario"] = dict(installed_at_seconds=485., metric_every_seconds=300.)
            points = []
            for now in (485, *range(600, 24001, 300)):
                wait, ride = 2964.+2*(now-485), float(now-485)
                if arm == "fresh":
                    early, tail = max(min(now-9000, 3000), 0), max(now-18300, 0)
                    if scenario == "stable":
                        wait, ride = wait+early+tail, ride-.5*early-.5*tail
                    else:
                        wait, ride = wait+2*early-tail, ride+3*early-2*tail
                points.append(dict(time_seconds=now, accrued_wait_seconds=wait,
                    accrued_in_vehicle_seconds=ride, waiting_passengers=2,
                    in_vehicle_passengers=1, departed_passengers=100))
            last = points[-1]
            wait, ride = last["accrued_wait_seconds"], last["accrued_in_vehicle_seconds"]
            summary["cost"].update(accrued_wait_seconds=wait, accrued_in_vehicle_seconds=ride,
                generalized_passenger_seconds=2*wait+ride,
                generalized_seconds_per_departed_passenger=(2*wait+ride)/100)
            (directory/"summary.json").write_text(json.dumps(summary))
            (directory/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in points))
    (root/"protocol.json").write_text(json.dumps(dict(replication_reference=REFERENCE)))


def test_seed6202_complete_four_day_gate_keeps_90_checks_and_exact_windows(tmp_path):
    new_seed_fixture(tmp_path)
    result = analyze(tmp_path, "pilot")
    assert result["status"] == "complete"
    assert len(result["integrity"]["checks"]) == 90
    assert result["integrity"]["all_pass"]
    assert all(row["all_pass"] for row in result["pre_feedback_prefix"].values())
    fixed = result["fixed_replication_endpoints"]
    assert fixed["seed"] == 6202 and fixed["reference_seed"] == 6201
    early = fixed["windows"]["local_speed_drop"]["early_shock"]
    assert early["fresh_minus_no_fresh"] == dict(accrued_wait_seconds=6000.,
        accrued_in_vehicle_seconds=9000., generalized_passenger_seconds=21000.,
        generalized_seconds_per_departed_passenger=210.)
    tail = fixed["windows"]["local_speed_drop"]["post_budget"]
    assert tail["fresh_minus_no_fresh"]["accrued_wait_seconds"] == -5700.
    assert tail["fresh_minus_no_fresh"]["accrued_in_vehicle_seconds"] == -11400.
    assert tail["fresh_minus_no_fresh"]["generalized_passenger_seconds"] == -22800.
    assert fixed["comparison_columns"] == ["same_sign", "absolute_magnitude_ratio"]
    assert tail["comparison"]["generalized_passenger_seconds"][0]
    assert tail["comparison"]["generalized_passenger_seconds"][1] == pytest.approx(22800/53874)
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 30000


def test_opposite_replication_sign_and_stable_harm_do_not_change_integrity_dispatch(tmp_path):
    new_seed_fixture(tmp_path)
    pilot = analyze(tmp_path, "pilot")
    stable = analyze(tmp_path, "stable")
    early = pilot["fixed_replication_endpoints"]["windows"]["stable"]["early_shock"]
    assert early["comparison"]["generalized_passenger_seconds"][0] is False
    assert stable["absolute_cost"]["stable"]["post_budget"]["fresh_minus_no_fresh"]["generalized_passenger_seconds"] > 0
    assert stable["integrity"]["all_pass"]
    assert stable["decision"]["next_action"] == "run paired shock pilot"
    assert pilot["fixed_replication_endpoints"]["effect_used_as_gate"] is False
    assert pilot["decision"]["independent_confirmation"] == "not_authorized_by_this_gate"


def test_missing_endpoint_is_explicit_instead_of_zero_benefit():
    summaries = {(scenario, arm): dict(cost=dict(departed_passengers=100))
        for scenario in SCENARIOS for arm in ARMS}
    points = {key: [dict(time_seconds=9000, accrued_wait_seconds=2., accrued_in_vehicle_seconds=3.)]
              for key in summaries}
    result = fixed_replication_endpoints(summaries, points, REFERENCE)
    row = result["windows"]["local_speed_drop"]["early_shock"]
    assert row["status"] == "missing_endpoint"
    assert row["fresh_minus_no_fresh"] is None
    assert row["comparison"] == {}
