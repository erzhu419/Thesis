"""Exercise action aliasing, signed cancellation and the resident transfer limit."""
from copy import deepcopy
import json

import pytest

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.analyze import (
    aggregate, analyze, interval_overlap, stats)


LINES = ["102S", "102X", "122S", "122X", "311S", "311X", "406S", "406X",
         "705S", "705X", "7S", "7X"]
FOCUS = ["7X", "102X", "705X"]


def score(sequence=0, *, memory=False):
    clock = 1100.+300.*(sequence//5)+sequence%5
    window = "before9000" if clock < 9000 else "9000-12000" if clock < 12000 else "12000-18000"
    scale = (sequence+1)/317.
    effect = (-1 if sequence%2 else 1)*scale if memory else 0.
    return dict(sequence=sequence, line_id=LINES[sequence%12], issue=clock,
        context_observed_at=clock, window=window, holding_seconds=[0., .9375, 1.875, 0.],
        executed_actions=[-1., -.96875, -.9375, -1.],
        actor_holding_seconds=.123456789123+effect*.01, target=-10.123456789+effect,
        value_contribution=-1.23456789+effect*.3, sac_contribution=-8.888888899+effect*.7,
        advantage_min=[0., -.000123456789+effect*.001, -.000246913578+effect*.002, 0.],
        zero_context_advantage_min=[0., -.000123456789, -.000246913578, 0.],
        best_fixed_action=1 if memory and sequence%2 else 0, reward_scaled=-.123456789123)


def test_alias_actions_have_no_advantage_or_ranking_support():
    left, right = score(), score(memory=True)
    for row in (left, right):
        row["executed_actions"][:3] = [-1., -1., -1.]
        row["holding_seconds"][:3] = [0., 0., 0.]
    left["best_fixed_action"], right["best_fixed_action"] = 0, 2
    result = aggregate([(left, right)])
    assert result["nonalias"] == [0, 0]
    assert result["fixed_ranking_changed"] == 0
    assert result["metrics"][4:] == [None]*4
    # When only 15s has distinct physical execution, only that comparison counts.
    for row in (left, right):
        row["executed_actions"][2] = -.9375
        row["holding_seconds"][2] = 1.875
    result = aggregate([(left, right)])
    assert result["nonalias"] == [0, 1]
    assert result["metrics"][4] is None and result["metrics"][6] is None
    assert result["metrics"][5] is not None and result["metrics"][7] is not None
    assert result["fixed_ranking_changed"] == 1


def test_absolute_summary_retains_changes_when_signed_effects_cancel():
    assert stats([-2., 2.]) == [0., 2., 2.]
    left, positive, negative = score(), score(memory=True), score(memory=True)
    positive["target"], negative["target"] = left["target"]+3., left["target"]-3.
    result = aggregate([(left, positive), (left, negative)])
    assert result["metrics"][1] == [0., 3., 3.]


def test_actual_optimizer_interval_overlap():
    assert interval_overlap([], [(0., 1.)]) == 0
    assert interval_overlap([(0., 1.)], [(1., 2.)]) == 0
    assert interval_overlap([(0., 4.), (6., 9.), (11., 12.)],
                            [(1., 2.), (3., 7.), (10., 14.)]) == 4.
    assert interval_overlap([(10.1, 10.9)], [(10.4, 11.2)]) == pytest.approx(.5)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"))+"\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":"))+"\n" for row in rows))


def resident_fixture(root):
    protocol = dict(scenarios=["stable", "local_speed_drop"], arms=["zero_memory", "memory"],
        focus_lines=FOCUS, compute_budget=dict(scenarios=2, arms_per_scenario=2,
            feedback_slots_per_arm=285, updates_per_arm=456, new_shadow_updates_total=1824,
            new_target_labels=0, new_source_labels=0, new_native_days=0,
            new_native_suffixes=0, shadow_scored_horizon_seconds=24000))
    write(root/"protocol.json", protocol)
    for scenario in protocol["scenarios"]:
        cell = root/"runs"/"seed6202"/scenario
        write(cell/"summary.json", dict(integrity=dict(all_pass=True),
            training_overlap_wall_seconds=570., budget=dict(feedback_slots_per_arm=285,
                updates_per_arm=456, new_target_labels=0, new_native_days=0)))
        for arm in protocol["arms"]:
            write(cell/arm/"summary.json", dict(integrity=dict(all_pass=True,
                checks=dict(zero_reference_updates_exact=True, zero_full_actor_trace_exact=True)),
                budget=dict(shadow_updates=456), zero_actor_issues_checked=17000))
            packets, scores, bank, seen = [], [], [], set()
            for packet in range(1, 58):
                packets.append(dict(receipt=dict(index=packet, records=5), sampling=[], rng={},
                    optimizer_objects_retained=True, update_started_wall=10.*packet+(arm == "memory"),
                    update_finished_wall=10.*packet+5.+(arm == "memory")))
                current = [score(index, memory=arm == "memory") for index in range((packet-1)*5, packet*5)]
                for row in current:
                    key = row["line_id"], row["window"]
                    if row["line_id"] in FOCUS and key not in seen:
                        seen.add(key)
                        bank.append(deepcopy(row))
                for phase, values in (("packet_before", current), ("packet_after", current), ("bank_after", bank)):
                    scores.append(dict(packet=packet, phase=phase, rows=deepcopy(values)))
            assert len(seen) == 9
            write_rows(cell/arm/"packets.jsonl", packets)
            write_rows(cell/arm/"scores.jsonl", scores)


def test_full_57_packet_12_line_three_window_summary_fits_transfer_bound(tmp_path):
    resident_fixture(tmp_path)
    result = analyze(tmp_path)
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    for case in result["cases"]:
        assert len(case["by_line"]) == 12
        assert len(case["focus_line_windows"]) == len(case["final_seen_bank"]) == 9
        assert case["by_phase"]["packet_after"]["n"] == 285
        assert case["optimizer_overlap_seconds"] == 57*4.
    size = len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode())+1
    assert size <= 20000, f"Full compact summary is {size} bytes; transfer limit is 20000"
