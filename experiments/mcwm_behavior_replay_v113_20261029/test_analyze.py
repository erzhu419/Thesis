"""Synthetic replay mutations must stop attribution; no simulator or learner."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import analyze
from experiments.mcwm_historical_coverage_v111_20261029 import test_analyze as old_fixture


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    p = analyze.read(analyze.ROOT / "experiments/mcwm_historical_coverage_v111_20261029/protocol.json")
    p.update(v111_root="experiments/mcwm_historical_coverage_v111_20261029", summary_max_bytes=65536)
    experiment = root / analyze.RELATIVE
    put(experiment / "protocol.json", p)
    for scene in p["scenarios"]:
        for arm in p["arms"]:
            previous = old_fixture.cell(p, scene, arm, 101 if arm == "fresh" else 100)
            previous["wall_seconds"] = 1
            previous["prefix"] = dict(previous["prefix"], trace_path="old/trace", services_path="old/services")
            for base in (experiment, root / p["v111_root"]):
                folder = base / "runs" / scene / arm
                old_fixture.logs(folder)
                put(folder / "summary.json", previous)
                old_fixture.first_packet(folder, arm)
            path = experiment / "runs" / scene / arm / "summary.json"
            current = copy.deepcopy(previous)
            current["wall_seconds"] = 3
            current["execution_wall"] = dict(native_started=300, native_finished=400)
            current["prefix"].update(trace_path="new/trace", services_path="new/services")
            current["observer"] = {"new_receipt": True}
            put(path, current)
    return p, experiment


def passive_fixture(root):
    """Two identifiable people, including a passenger riding only in fresh."""
    p, experiment = fixture(root)
    p["full_day_departed_passengers"] = 2
    put(experiment / "protocol.json", p)
    put(experiment / "preflight_gate.json", dict(all_pass=True))
    put(root / p["v111_root"] / "summary.json", dict(
        decision=dict(state="CLOSE_FIXED_PILOT"), cost_gates=dict(matched_mean_negative=False)))
    for scene in p["scenarios"]:
        for arm in p["arms"]:
            fresh = arm == "fresh"
            people, visits = [], []
            for i in range(2):
                wait = (1 if fresh else 2) if i == 0 else (0 if fresh else 1)
                holding = (3 if fresh else 1) if i == 0 else 0
                moving = (1 if fresh else 2) if i == 0 else (2 if fresh else 0)
                piece = dict(wait=wait, ride=holding + moving,
                    ride_phases=dict(service=0, holding=holding, moving=moving, traffic_or_other=0))
                periods = {name: copy.deepcopy(piece) for name in analyze.WINDOWS}
                full = analyze.zero()
                for _ in analyze.PARTITION:
                    analyze.add(full, piece)
                person = dict(passenger_id="person" + str(i), departure_time=0,
                    completion_time=23999 if i == 0 or fresh else None,
                    completed=i == 0 or fresh, **full, periods=periods)
                people.append(person)
                visits.append(dict(trip_id="trip" + str(i), vehicle_id="vehicle" + str(i),
                    line_id="7X", stop_id=str(i), visit_index=1,
                    action_time=10000 if i == 0 else None,
                    command_hold_seconds=(5 if fresh else 3) if i == 0 else None,
                    command_accepted=True if i == 0 else None,
                    departure_time=11000 if i == 0 else None, **full, periods=periods))
            totals = analyze.summed_rows(people)
            cumulative, metrics = analyze.zero(), []
            def metric(time, values):
                return dict(time_seconds=time, accrued_wait_seconds=values["wait"],
                            accrued_in_vehicle_seconds=values["ride"])
            for name in analyze.PARTITION:
                if name == "12000_18300":
                    metrics.append(metric(12600, cumulative))
                    midway = copy.deepcopy(cumulative)
                    analyze.add(midway, totals["12600_12900"])
                    metrics.append(metric(12900, midway))
                analyze.add(cumulative, totals[name])
                metrics.append(metric(analyze.WINDOWS[name][1], cumulative))
            costs = {}
            for name, bounds in p["periods"].items():
                source = totals[{"full_day": "full_day", "early": "9000_12000",
                                 "post_budget": "18300_24000"}[name]]
                cost = 2 * source["wait"] + source["ride"]
                costs[name] = dict(status="complete", start=bounds[0], end=bounds[1],
                    wait=source["wait"], ride=source["ride"], cost=cost,
                    departed=2, cost_per_passenger=cost / 2)
            for base in (experiment, root / p["v111_root"]):
                folder = base / "runs" / scene / arm
                row = analyze.read(folder / "summary.json")
                row["periods"] = costs
                row["passengers"] = dict(completed_passengers=sum(x["completed"] for x in people))
                put(folder / "summary.json", row)
                put(folder / "metrics.json", metrics)
            folder = experiment / "runs" / scene / arm
            for name, rows in (("passengers", people), ("visits", visits)):
                (folder / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
            put(folder / "observer_summary.json", dict(status="complete", ledger_exact=True,
                checks=dict(ledger_exact=True, phase_totals_reconcile=True), start=0, end=24000,
                passenger_totals=totals["full_day"], errors=[]))
    return p, experiment


class ReplayTests(unittest.TestCase):
    def test_four_arms_exact_semantics_allow_only_explicit_metadata_changes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, _ = fixture(root)
            for scene in p["scenarios"]:
                for arm in p["arms"]:
                    self.assertTrue(analyze.cell_reproduction(root, p, scene, arm)["all_pass"])

    def test_changed_or_missing_old_evidence_rejects_fresh_and_nofresh(self):
        changes = [(name, operation) for name in (
            "metrics.json", "packet_receipts.json", "updates.jsonl",
            "prefix_trace.jsonl", "prefix_services.jsonl") for operation in ("change", "remove")]
        changes += [("summary.json", "cost"), ("summary.json", "counts"), ("summary.json", "mechanism")]
        for arm in ("fresh", "no_fresh"):
            for name, operation in changes:
                with self.subTest(arm=arm, file=name, change=operation), TemporaryDirectory() as folder:
                    root = Path(folder)
                    p, experiment = fixture(root)
                    path = experiment / "runs/stable" / arm / name
                    if operation == "remove":
                        path.unlink()
                    elif name.endswith(".jsonl"):
                        rows = path.read_text().splitlines()
                        rows[0] = '{"changed":true}'
                        path.write_text("\n".join(rows) + "\n")
                    else:
                        row = analyze.read(path)
                        if operation == "cost":
                            row["periods"]["full_day"]["ride"] += 1
                        elif operation == "counts":
                            row["counts"]["critic"] -= 1
                        elif operation == "mechanism":
                            row["mechanism"]["main_rng_preserved"] = False
                        else:
                            row[0]["changed"] = True
                        put(path, row)
                    self.assertFalse(analyze.cell_reproduction(root, p, "stable", arm)["all_pass"])

    def test_preflight_does_not_ignore_changed_first_update(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, experiment = fixture(root)
            path = experiment / "runs/stable/fresh/preflight.json"
            current = analyze.read(path)
            current["feedback"]["training_outcomes"] = 4
            put(path, current)
            result = analyze.preflight(root, experiment)
            self.assertFalse(result["all_pass"])
            self.assertEqual(result["cells"]["fresh"]["summary"]["differing_top_level_fields"], ["feedback"])

    def test_identified_passenger_composition_phases_and_missing_commands(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, experiment = passive_fixture(root)
            result = analyze.analyze(root)
            self.assertEqual(result["status"], "complete", result["integrity"])
            scene = result["scenes"]["stable"]
            self.assertEqual(scene["fresh_minus_matched"]["full_day"]["cost_per_passenger"], -2)
            self.assertEqual(scene["passengers"]["by_ride_group"]["both"]["delta"]["ride"], 4)
            self.assertEqual(scene["passengers"]["by_ride_group"]["fresh_only"]["delta"]["ride"], 8)
            self.assertEqual(scene["fresh_minus_matched"]["full_day"]["ride_phases"]["holding"], 8)
            commands = scene["visits"]["commands"]
            self.assertEqual((commands["known_pairs"], commands["longer"], commands["unknown"]), (1, 1, 1))
            self.assertEqual(commands["observed_delta_command_seconds"], 2)
            self.assertIsNone(scene["visits"]["commands_by_reference_action_time"]["0_9000"]["observed_delta_command_seconds"])
            self.assertEqual(result["decision"]["original_v111_decision"]["state"], "CLOSE_FIXED_PILOT")
            self.assertLessEqual((experiment / "summary.json").stat().st_size, p["summary_max_bytes"])

    def test_phase_roundoff_only_does_not_relax_exact_ledger(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, experiment = passive_fixture(root)
            cell = experiment / "runs/stable/fresh"
            rows = [json.loads(line) for line in (cell / "visits.jsonl").read_text().splitlines()]
            # Fractional command summation order may differ by <1e-6 person-sec.
            rows[0]["ride_phases"]["holding"] += 1e-8
            rows[0]["periods"]["0_9000"]["ride_phases"]["holding"] += 1e-8
            (cell / "visits.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            observed, _, _ = analyze.observed_cell(cell, analyze.read(cell / "summary.json"), p)
            self.assertTrue(observed["all_pass"])
            # A real missing person-second must fail; it cannot use phase tolerance.
            metrics = analyze.read(cell / "metrics.json")
            metrics[-1]["accrued_wait_seconds"] += 1
            put(cell / "metrics.json", metrics)
            observed, _, _ = analyze.observed_cell(cell, analyze.read(cell / "summary.json"), p)
            self.assertFalse(observed["all_pass"])
            self.assertFalse(observed["checks"]["passenger_ledger_exact"])

    def test_failed_stable_gate_finishes_with_unstarted_shock_unknown(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            _, experiment = passive_fixture(root)
            put(experiment / "preflight_gate.json", dict(all_pass=False))
            for arm in ("fresh", "no_fresh"):
                (experiment / "runs/local_speed_drop" / arm / "summary.json").unlink()
            result = analyze.analyze(root)
            self.assertEqual(result["status"], "technical_hold")
            self.assertEqual(len(result["missing"]), 2)
            self.assertIsNone(result["budget"]["new_native_seconds"])
            self.assertEqual(result["budget"]["known_native_seconds"], 48000)
            self.assertIsNone(result["scenes"]["local_speed_drop"])
            self.assertFalse(result["attribution_to_original_v111_allowed"])

    def test_skipped_stop_does_not_shift_later_visit_pairing(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            _, experiment = passive_fixture(root)
            sample = json.loads((experiment / "runs/stable/fresh/visits.jsonl").read_text().splitlines()[0])
            records = {}
            for arm, stops in (("fresh", ["A", "C"]), ("no_fresh", ["A", "B", "C"])):
                path = root / (arm + ".jsonl")
                rows = [dict(sample, trip_id="same_trip", stop_id=stop, visit_index=index)
                        for index, stop in enumerate(stops, 1)]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                records[arm] = analyze.load_records(path, False)
            result = analyze.visit_attribution(records["fresh"], records["no_fresh"], 2)
            self.assertEqual((result["matched_visits"], result["only_no_fresh_visits"]), (2, 1))
            later = next(row for row in result["largest_visit_changes"] if row["identity"]["stop_id"] == "C")
            self.assertTrue(later["matched"])
            self.assertEqual((later["fresh_visit_index"], later["no_fresh_visit_index"]), (2, 3))


if __name__ == "__main__":
    unittest.main()
