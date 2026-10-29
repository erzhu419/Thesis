"""Native-reset recorder clocks retain early cost without relaxing cadence."""
import json

import pytest

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_analyze import fixture
from experiments.mcwm_closed_loop_fresh_attribution_v25r1_20261023.analyze import analyze


def native_fixture(root, *, fresh_extra=0.):
    fixture(root, stages=("stable",))
    for arm in ("fresh", "no_fresh"):
        directory = root/"runs"/"stable"/"seed6201"/arm
        points = []
        for now in [485, *range(600, 24001, 300)]:
            wait = 2964. + 2*(now-485)
            ride = float(now-485)
            if arm == "fresh":
                ride += fresh_extra*max(now-1200, 0)
            points.append(dict(time_seconds=now, accrued_wait_seconds=wait,
                accrued_in_vehicle_seconds=ride, waiting_passengers=2,
                in_vehicle_passengers=1, departed_passengers=100))
        path = directory/"summary.json"
        summary = json.loads(path.read_text())
        summary["scenario"] = dict(installed_at_seconds=485., metric_every_seconds=300.)
        wait, ride = points[-1]["accrued_wait_seconds"], points[-1]["accrued_in_vehicle_seconds"]
        summary["cost"].update(accrued_wait_seconds=wait, accrued_in_vehicle_seconds=ride,
            generalized_passenger_seconds=2*wait+ride,
            generalized_seconds_per_departed_passenger=(2*wait+ride)/100)
        path.write_text(json.dumps(summary))
        (directory/"metrics.jsonl").write_text("".join(json.dumps(point)+"\n" for point in points))


def change_points(root, change):
    path = root/"runs"/"stable"/"seed6201"/"fresh"/"metrics.jsonl"
    points = [json.loads(line) for line in path.read_text().splitlines()]
    change(points)
    path.write_text("".join(json.dumps(point)+"\n" for point in points))


def test_native_485_start_keeps_early_ledger_cost_and_exact_tail(tmp_path):
    native_fixture(tmp_path)
    result = analyze(tmp_path, "stable")
    assert result["integrity"]["all_pass"]
    costs = result["absolute_cost"]["stable"]
    full = costs["full_day"]["arms"]["fresh"]
    assert full["start"] == 0
    assert full["accrued_wait_seconds"] == 2964 + 2*(24000-485)
    assert full["accrued_in_vehicle_seconds"] == 24000-485
    assert full["generalized_passenger_seconds"] == 2*full["accrued_wait_seconds"] + full["accrued_in_vehicle_seconds"]
    assert full["generalized_seconds_per_departed_passenger"] == full["generalized_passenger_seconds"]/100
    tail = costs["post_budget"]["arms"]["fresh"]
    assert tail["accrued_wait_seconds"] == 2*(24000-18300)
    assert tail["accrued_in_vehicle_seconds"] == 24000-18300
    assert costs["full_day"]["fresh_minus_no_fresh"]["generalized_passenger_seconds"] == 0


@pytest.mark.parametrize("change", [
    lambda points: points.pop(1),
    lambda points: points[0].update(time_seconds=0),
    lambda points: points[1].update(time_seconds=601),
])
def test_missing_scheduled_point_wrong_origin_and_shifted_grid_fail(tmp_path, change):
    native_fixture(tmp_path)
    change_points(tmp_path, change)
    result = analyze(tmp_path, "stable")
    assert result["status"] == "integrity_failed"
    assert result["integrity"]["checks"]["stable/fresh/all_fixed_clock_metrics"] is False


def test_stable_harm_does_not_stop_shock_on_valid_native_data(tmp_path):
    native_fixture(tmp_path, fresh_extra=3.)
    result = analyze(tmp_path, "stable")
    assert result["integrity"]["all_pass"]
    assert result["absolute_cost"]["stable"]["full_day"]["fresh_minus_no_fresh"]["generalized_passenger_seconds"] > 0
    assert result["absolute_cost"]["stable"]["post_budget"]["fresh_minus_no_fresh"]["generalized_passenger_seconds"] > 0
    assert result["decision"]["next_action"] == "run paired shock pilot"
    assert result["decision"]["cost_used_as_dispatch_gate"] is False
