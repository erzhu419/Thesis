"""No native execution: exact concatenation, shared work and frozen cost gates."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import analyze
from experiments.mcwm_historical_coverage_v111_20261029 import test_analyze as original_fixture


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def put_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def panel(root):
    p = analyze.read(analyze.ROOT / "experiments/mcwm_historical_coverage_v111_20261029/protocol.json")
    p.update(v111_root="experiments/mcwm_historical_coverage_v111_20261029",
        v113_root="experiments/mcwm_behavior_replay_v113_20261029", summary_max_bytes=64000, anchors={})
    experiment = root / analyze.RELATIVE
    put(experiment / "protocol.json", p)
    anchors = []
    for scene, clock, updates in (("stable", 17235, 432), ("local_speed_drop", 14547, 360)):
        folder = experiment / "runs" / scene
        root_counts = analyze.original.expected_counts(updates)
        feedback = updates // 8 * 5
        parent = dict(status="prefix_complete", counts=root_counts,
            budget=dict(native_seconds=clock, policy_updates=updates,
                        feedback_slots=feedback, target_training_labels=feedback))
        put(folder / "summary.json", parent)
        state = dict(native_clock=clock, counts=root_counts, pending_sequences=[1, 2, 3])
        put(folder / "root_state.json", state)
        anchor = dict(scenario=scene, trip_id="trip", stop_id="stop", high=44., low=5.,
            records=dict(fresh=dict(actions=dict(sequence=2, started_at=clock, proposed_a=[.4], a=[.4]))))
        anchors.append(anchor)
        p["anchors"][scene] = dict(trip_id="trip", station_id="stop", sequence=2, clock=clock,
                                   high_hold_seconds=44., low_hold_seconds=5.)
        put(folder / "branches/live_fork.json", dict(all_pass=True, native_before=clock,
            native_after=clock, overlap_seconds=10, child_exitcodes=dict(high=0, low=0), parent_rng_unchanged=True))
        with patch.object(original_fixture, "analyze", analyze.original):
            high = original_fixture.cell(p, scene, "fresh", 106)
            low = original_fixture.cell(p, scene, "fresh", 105.5)
            nofresh = original_fixture.cell(p, scene, "no_fresh", 100)
        for run in (high, low, nofresh):
            run["ledger"] = dict(time_seconds=24000)
        put(root / p["v111_root"] / "runs" / scene / "no_fresh/summary.json", nofresh)
        put(root / p["frozen_reference_root"] / scene / "summary.json",
            dict(status="complete", actor_seed=6201, external_seed=106206, scenario=scene,
                 integrity=dict(all_pass=True), periods=original_fixture.periods(p, 105)))
        old = root / p["v113_root"] / "runs" / scene / "fresh"
        put(old / "summary.json", high)
        for name in ("actions.jsonl", "updates.jsonl"):
            put_rows(folder / name, [dict(sequence=1)])
            put_rows(old / name, [dict(sequence=1), dict(sequence=2)])
        for name in ("passengers.jsonl", "visits.jsonl"):
            put_rows(old / name, [dict(identity="full_cumulative_record")])
        for name in ("metrics.json", "packet_receipts.json"):
            put(old / name, [dict(full_day=True)])
        for arm, run in (("high", high), ("low", low)):
            child = folder / "branches" / arm
            run = copy.deepcopy(run)
            run["branch"] = dict(name=arm, root_clock=clock, root_counts=root_counts,
                root_feedback_slots=feedback, suffix_native_seconds=24000-clock,
                suffix_updates=456-updates, suffix_feedback_slots=285-feedback,
                root_issue_count=1, proposal_overrides=1 if arm == "low" else 0)
            put(child / "summary.json", run)
            put(child / "root_state.json", state)
            put(child / "root_action.json", dict(branch=arm, sequence=2, started_at=clock,
                trip_id="trip", station_id="stop", expected_hold_seconds=anchor[arm],
                actual_hold_seconds=anchor[arm], proposed_a=[.4] if arm == "high" else [-.8],
                executed_a=[.4] if arm == "high" else [-.8], raw_F_exact=True,
                root_state_exact=True, only_root_override=True))
            put_rows(child / "commands.jsonl", [dict(sequence=2, accepted=True,
                requested_additional_hold_seconds=anchor[arm], trip_id="trip", stop_id="stop", applied_at=clock)])
            for name in ("actions.jsonl", "updates.jsonl"):
                put_rows(child / name, [dict(sequence=2)])
            for name in ("passengers.jsonl", "visits.jsonl"):
                put_rows(child / name, [dict(identity="full_cumulative_record")])
            for name in ("metrics.json", "packet_receipts.json"):
                put(child / name, [dict(full_day=True)])
    put(experiment / "anchor_binding.json", dict(anchors=anchors))
    put(experiment / "protocol.json", p)
    return p, experiment


def observed_stub(folder, row, p):
    # v113 already tests the resident ledger reader. This fixture exercises the
    # new v114 fork/concatenation/reference contracts, without large raw logs.
    totals = analyze.behavior.blocks()
    total = row["periods"]["full_day"]
    totals["full_day"].update(wait=total["wait"], ride=total["ride"])
    totals["full_day"]["ride_phases"]["moving"] = total["ride"]
    return dict(all_pass=True, checks=dict(synthetic_ledger=True), totals=totals), {}, {}


class AnalyzeTests(unittest.TestCase):
    def test_v114_branch_budget_omits_obsolete_zero_counterfactual_field(self):
        p = analyze.read(analyze.ROOT / "experiments/mcwm_historical_coverage_v111_20261029/protocol.json")
        with patch.object(original_fixture, "analyze", analyze.original):
            row = original_fixture.cell(p, "stable", "fresh", 106)
        row["budget"].pop("counterfactual_queries")
        self.assertEqual(analyze.run_errors(row, p, "stable", "fresh"), [])
        row["budget"]["policy_updates"] -= 1
        self.assertEqual(analyze.run_errors(row, p, "stable", "fresh"), ["actual_budget_failed"])

    def test_high_reproduction_excludes_only_obsolete_counterfactual_declaration(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, experiment = panel(root)
            current_folder = experiment / "runs/stable"
            old_folder = root / p["v113_root"] / "runs/stable/fresh"
            current = analyze.read(current_folder / "branches/high/summary.json")
            current["budget"].pop("counterfactual_queries")
            self.assertTrue(analyze.high_reproduction(current_folder, old_folder, current)["all_pass"])
            self.assertNotIn("counterfactual_queries", current["budget"])
            self.assertEqual(analyze.read(old_folder / "summary.json")["budget"]["counterfactual_queries"], 0)
            current["budget"]["feedback_slots"] -= 1
            self.assertFalse(analyze.high_reproduction(current_folder, old_folder, current)["all_pass"])

    def test_concat_has_no_root_duplication_or_missing_event(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [dict(sequence=index, hold=index / 10) for index in range(6)]
            a, b, old = (folder / name for name in ("prefix", "suffix", "reference"))
            put_rows(a, rows[:3]); put_rows(b, rows[3:]); put_rows(old, rows)
            self.assertTrue(analyze.compare_segments([a, b], old)["all_pass"])
            put_rows(b, rows[2:])
            self.assertFalse(analyze.compare_segments([a, b], old)["all_pass"])
            put_rows(b, rows[4:])
            self.assertFalse(analyze.compare_segments([a, b], old)["all_pass"])
            b.unlink()
            self.assertFalse(analyze.compare_segments([a, b], old)["all_pass"])

    def test_shared_prefix_not_counted_twice_missing_suffix_remains_unknown(self):
        prefix = analyze.original.expected_counts(400)
        branch = dict(ledger=dict(time_seconds=24000), counts=analyze.original.expected_counts(456))
        result = analyze.unique_work(17000, prefix, dict(high=branch, low=branch), 24000)
        self.assertEqual(result["new_native_seconds"], 31000)
        self.assertEqual(result["counters"]["policy"]["total"], 512)
        result = analyze.unique_work(17000, prefix, dict(high=branch), 24000)
        self.assertIsNone(result["new_native_seconds"])
        self.assertEqual(result["known_native_seconds"], 24000)
        self.assertIsNone(result["counters"]["policy"]["total"])
        self.assertEqual(result["counters"]["policy"]["known_subtotal"], 456)

    def test_anchor_benefit_does_not_replace_original_dual_gate(self):
        cells = {scene: dict(comparisons=dict(full_day={
            "low_minus_high": dict(cost_per_passenger=-1),
            "low_minus_matched": dict(cost_per_passenger=.5),
            "low_minus_frozen": dict(cost_per_passenger=-.2)}))
            for scene in ("stable", "local_speed_drop")}
        gates = analyze.original_cost_gates(cells)
        self.assertFalse(gates["matched_mean_negative"])
        self.assertTrue(gates["each_frozen_nonpositive"])
        self.assertEqual(len(gates), 5)

    def test_complete_panel_uses_same_state_cost_gate_and_actual_shared_work(self):
        with TemporaryDirectory() as folder, patch.object(analyze.behavior, "observed_cell", observed_stub):
            root = Path(folder)
            p, experiment = panel(root)
            result = analyze.analyze(root)
            self.assertTrue(result["integrity"]["all_pass"], result["integrity"])
            self.assertEqual(result["budget"]["new_native_seconds"], 64218)
            self.assertEqual(result["budget"]["policy_updates"], 1032)
            self.assertEqual(result["budget"]["fresh_admissions"], 645)
            self.assertEqual(result["means"]["full_day"]["low_minus_high"]["cost_per_passenger"], -.5)
            self.assertFalse(result["decision"]["low_intervention_cost_gate_pass"])
            self.assertFalse(result["decision"]["method_candidate_passed"])
            self.assertLess((experiment / "summary.json").stat().st_size, p["summary_max_bytes"])

    def test_changed_root_dose_command_high_trace_and_missing_branch_hold(self):
        for change in ("state", "dose", "command", "high_trace", "missing_branch"):
            with self.subTest(change=change), TemporaryDirectory() as folder, patch.object(
                    analyze.behavior, "observed_cell", observed_stub):
                root = Path(folder)
                _, experiment = panel(root)
                child = experiment / "runs/stable/branches/low"
                if change == "state":
                    value = analyze.read(child / "root_state.json")
                    value["pending_sequences"] = [1, 3]
                    put(child / "root_state.json", value)
                elif change == "dose":
                    value = analyze.read(child / "root_action.json")
                    value["actual_hold_seconds"] = 6
                    put(child / "root_action.json", value)
                elif change == "command":
                    put_rows(child / "commands.jsonl", [dict(sequence=2, accepted=False)])
                elif change == "high_trace":
                    put_rows(experiment / "runs/stable/branches/high/actions.jsonl", [dict(sequence=3)])
                else:
                    (child / "summary.json").unlink()
                result = analyze.analyze(root)
                self.assertEqual(result["status"], "technical_hold")
                self.assertIsNone(result["cost_gates"])
                if change == "missing_branch":
                    self.assertIsNone(result["budget"]["new_native_seconds"])
                    self.assertEqual(result["budget"]["known_native_seconds"], 57453)


if __name__ == "__main__":
    unittest.main()
