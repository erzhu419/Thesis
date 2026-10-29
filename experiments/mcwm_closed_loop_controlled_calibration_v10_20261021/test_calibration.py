from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import (
    CalibrationPacketBudget, ControlledCalibration,
)


def protocol(**updates):
    return dict(feedback_first_deadline=1200, feedback_last_deadline=18000,
        feedback_interval_seconds=300, feedback_records_per_packet=5,
        feedback_delay_seconds=30, target_record_budget=285,
        horizon_seconds=24000, **updates)


class ExampleTransform:
    max_hold = 60.

    def __init__(self, reserve=0., floor=0.):
        self.reserve, self.floor = reserve, floor

    def numpy(self, state, proposal):
        parent = float(proposal[0]+1.)*.5*self.max_hold
        hold = min(parent, max(parent*.125, self.reserve))
        return np.asarray([2*max(self.floor, hold)/self.max_hold-1], np.float32)


def issued(plan, sequence=10, line="7X"):
    return dict(started_at=plan["issue_start"]+1, line_id=line,
        station_id="diagnostic", sequence=sequence, s=np.zeros(229, np.float32))


def row(sequence, available_at, dt=5.):
    return SimpleNamespace(sequence=sequence, available_at=float(available_at),
        completed_at=float(available_at)-30, dt=dt, terminal=False)


def bundle(sequence, available_at, dt=0., line="7X", assignment="low", role="train", status="complete"):
    return SimpleNamespace(sequence=sequence, available_at=float(available_at),
        completed_at=float(available_at)-30, global_next=row(sequence, 1100, dt),
        line_id=line, assignment=dict(assignment=assignment), role=role, status=status)


def test_seed_fixed_balanced_line_blocks_and_rng_independence():
    np.random.seed(321)
    expected = np.random.get_state()[1].copy()
    a = ControlledCalibration(protocol(), 6201, ExampleTransform(), station_id="diagnostic")
    b = ControlledCalibration(protocol(), 6201, ExampleTransform(), station_id="diagnostic")
    assert a.plan == b.plan
    assert np.array_equal(expected, np.random.get_state()[1])
    for role in ("train", "heldout"):
        for line in a.lines:
            holds = a.action_sequences[role][line]
            assert abs(holds.count(0.)-holds.count(7.5)) <= 1
            for index in range(0, len(holds)-1, 2):
                assert set(holds[index:index+2]) == {0., 7.5}
    assert sum(p["role"] == "train" for p in a.plan) == 57
    assert a.plan[-1]["issue_end"]+330 <= 24000


@pytest.mark.parametrize("reserve", [0., 3.14, 20.])
def test_command_is_verified_after_existing_execution_transform(reserve):
    transform = ExampleTransform(reserve=reserve)
    calibration = ControlledCalibration(protocol(), 6201, transform, station_id="diagnostic")
    for plan in calibration.plan[:6]:
        decision = calibration.choose(issued(plan, plan["window_index"]))
        assert decision is not None
        replayed = transform.numpy(np.zeros(229), decision["proposal"])
        assert np.array_equal(replayed, decision["action"])
        assert abs(decision["executed_hold_seconds"]-decision["target_hold_seconds"]) <= 1e-5
        assert calibration.choose(issued(plan, 500)) is None


def test_completion_floor_is_retained_as_collapsed_not_silently_executed():
    calibration = ControlledCalibration(protocol(), 6201, ExampleTransform(floor=60.), station_id="diagnostic")
    assert calibration.choose(issued(calibration.plan[0])) is None
    summary = calibration.summary(18000)
    assert summary["statuses"] == {"collapsed": 1}
    assert summary["issued"] == 0
    assert len(summary["missed_windows"]) == 57


def test_actual_execution_rule_reaches_requested_holds_and_preserves_floor():
    from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
    transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=360.))
    calibration = ControlledCalibration(protocol(), 6201, transform, station_id="diagnostic")
    for stage in (.1, .4, .8):
        for reserve in (0., 3.14, 20.):
            state = np.zeros(229, np.float32)
            state[0] = stage
            event = state[5:19]
            event[3:7] = [1., 1.+2.*reserve/360., 1., 1.]
            event[8], event[9], event[11], event[12] = .4, .1, 1., 1.
            for target in (0., 7.5):
                proposal, action, actual = calibration._action_for_hold(state, target)
                assert np.array_equal(action, transform.numpy(state, proposal))
                assert actual == target
    state[0], event[8], event[12] = .4, .1, 0.
    assert calibration._action_for_hold(state, 0.) is None
    assert calibration._action_for_hold(state, 7.5) is None


def test_heldout_has_its_own_prespecified_plan_and_role():
    calibration = ControlledCalibration(protocol(), 6201, ExampleTransform(), station_id="diagnostic")
    plan = next(p for p in calibration.plan if p["role"] == "heldout")
    event = issued(plan)
    assert calibration.choose(event) is None
    decision = calibration.choose(event, role="heldout")
    assert decision["role"] == "heldout"
    assert calibration.summary(24000)["coverage"]["heldout"][event["line_id"]][decision["assignment"]]["issued"] == 1


def test_empty_windows_do_not_consume_line_assignment_or_use_future_labels():
    calibration = ControlledCalibration(protocol(), 6201, ExampleTransform(), station_id="diagnostic")
    first = calibration.choose(issued(calibration.plan[0], 1, line="705X"))
    # No candidate in the intervening window. Other lines do not consume 705X's block.
    calibration.choose(issued(calibration.plan[2], 2, line="102X"))
    last = calibration.choose(issued(calibration.plan[3], 3, line="705X"))
    assert {first["assignment"], last["assignment"]} == {"low", "high"}
    assert calibration.line_successes[("train", "705X")] == 2
    missing = calibration.summary(calibration.plan[4]["issue_start"])["missed_windows"]
    assert [r["window_index"] for r in missing] == [1]
    # Response outcomes never enter the selector interface.
    assert "outcome" not in first and "cost" not in first


def test_zero_global_and_all_endpoints_consume_one_delayed_slot():
    budget = CalibrationPacketBudget(protocol())
    budget.reserve(99)
    budget.ingest([row(99, 1100, dt=0.)] + [row(i, 1150+i) for i in range(5)])
    delayed = bundle(99, 1230)
    budget.ingest_calibration([delayed])
    first, _ = budget.take(1201)
    assert len(first) == 5 and all(r.sequence != 99 for r in first)
    budget.ingest([row(10+i, 1450+i) for i in range(5)])
    second, discarded = budget.take(1500)
    assert len(second) == 5 and sum(r.sequence == 99 for r in second) == 1
    assert 10 in discarded
    assert budget.last_calibration_records == [delayed]
    assert budget.receipts[-1]["max_available_at"] == 1454
    assert budget.receipts[-1]["learned_global_records"] == 4
    assert budget.summary()["target_labels"] == 10
    assert budget.summary()["calibration_selected_by_line_action"]["7X"]["low"] == 1
    assert budget.summary()["zero_global_admitted"] == 1


def test_only_one_bundle_per_packet_and_extra_complete_bundle_remains_pending():
    budget = CalibrationPacketBudget(protocol())
    for seq in (100, 101):
        budget.reserve(seq)
    budget.ingest_calibration([bundle(100, 1100), bundle(101, 1150, assignment="high")])
    budget.ingest([row(i, 1150+i) for i in range(5)])
    selected, _ = budget.take(1200)
    assert sum(r.sequence >= 100 for r in selected) == 1
    assert budget.summary()["completed_calibration_unused"] == 1
    assert budget.summary()["pending_sequence_ids"] == [101]


@pytest.mark.parametrize("values", [dict(role="heldout"), dict(status="missing"), dict(status="pending")])
def test_heldout_or_incomplete_response_never_enters_budget(values):
    budget = CalibrationPacketBudget(protocol())
    budget.reserve(99)
    with pytest.raises(ValueError):
        budget.ingest_calibration([bundle(99, 1100, **values)])
    assert budget.summary()["target_labels"] == 0


def test_exact_original_285_labels_with_missing_calibration_is_honest():
    budget = CalibrationPacketBudget(protocol())
    budget.reserve(99999)
    for index, deadline in enumerate(budget.deadlines):
        budget.ingest([row(index*5+i, deadline-10+i) for i in range(5)])
        selected, _ = budget.take(deadline)
        assert len(selected) == 5
    assert not budget.due(24000)
    assert budget.summary()["target_labels"] == 285
    assert budget.summary()["calibration_admitted"] == 0
    assert budget.summary()["pending_sequence_ids"] == [99999]


def test_missing_ordinary_labels_fails_without_expanding_budget():
    budget = CalibrationPacketBudget(protocol())
    budget.ingest([row(i, 1100+i) for i in range(4)])
    with pytest.raises(RuntimeError, match="Insufficient feedback"):
        budget.take(1200)
    assert budget.index == 0
    assert budget.summary()["target_labels"] == 0
