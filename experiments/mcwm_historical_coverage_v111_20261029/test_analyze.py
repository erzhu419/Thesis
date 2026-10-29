"""Exercise historical coverage and measured extra-compute gates without runs."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import analyze


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mechanism(arm, updates):
    n = updates if arm == "fresh" else 0
    return dict(mode="critic_only_fresh_historical_coverage" if arm == "fresh" else "original_no_fresh",
        auxiliary_captures=n, value_calls=n, actor_calls=n, temperature_calls=n,
        original_batch_used=dict(value=n, actor=n, temperature=n),
        critic_complete_replacements=n * 64, critic_replacement_unit="rows",
        history_coverage_updates=n, history_critic_rows_per_update=128 if arm == "fresh" else 0,
        extra_critic_rows=n * 64, extra_bootstrap_rows=n * 256, extra_bootstrap_calls=n,
        main_rng_preserved=True, common_targets_exact=True)


def periods(p, cost):
    result = {}
    for name, bounds in p["periods"].items():
        total = cost * {"full_day": 1, "early": .1, "post_budget": .2}[name] * 15079
        result[name] = dict(status="complete", start=bounds[0], end=bounds[1],
            wait=total / 4, ride=total / 2, cost=total, departed=15079,
            cost_per_passenger=total / 15079)
    return result


PREFIX = dict(trace_rows=1, service_rows=1, service_events=1, clock=1201)


def cell(p, scene, arm, cost):
    counts, evidence = analyze.expected_counts(456), mechanism(arm, 456)
    return dict(status="complete", cell=dict(training_seed=6201, external_seed=106206,
        scenario=scene, arm=arm), integrity=dict(all_pass=True), periods=periods(p, cost),
        counts=counts, mechanism=evidence, replay=dict(mechanism=copy.deepcopy(evidence)),
        prefix=PREFIX, action_audit=dict(raw_execution_exact=True, canonical_observation_exact=True),
        execution_wall=dict(native_started=100 if arm == "fresh" else 101, native_finished=200),
        post_budget_counts=dict(frozen_at=18002, last_update_clock=18002,
            post18300_observed_at=18304, before=counts, after=counts, post18300=counts),
        budget=dict(native_seconds=24000, native_seconds_is_lower_bound=False, packets=57,
            feedback_slots=285, policy_updates=456, update_attempts=456,
            partial_update_possible=False, source_collection_events=0, model_refits=0,
            offline_training_updates=0, counterfactual_queries=0,
            target_training_labels=285 if arm == "fresh" else 0))


def logs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    put(folder / "metrics.json", [dict(clock=0, wait=0), dict(clock=24000, wait=100)])
    packets = [dict(packet=i, records=5) for i in range(1, 58)]
    put(folder / "packet_receipts.json", packets)
    (folder / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in packets))
    (folder / "prefix_trace.jsonl").write_text('{"clock":1200,"action":0.2}\n')
    (folder / "prefix_services.jsonl").write_text('{"clock":1200,"service":1}\n')


def fixture(root, gate=True):
    p = analyze.read(analyze.HERE / "protocol.json")
    experiment = root / analyze.RELATIVE
    put(experiment / "protocol.json", p)
    if gate:
        put(experiment / "preflight_gate.json", dict(status="complete", all_pass=True))
    for scene in p["scenarios"]:
        put(root / p["frozen_reference_root"] / scene / "summary.json",
            dict(status="complete", actor_seed=6201, external_seed=106206, scenario=scene,
                integrity=dict(all_pass=True), periods=periods(p, 105)))
        for arm in p["arms"]:
            folder = experiment / "runs" / scene / arm
            logs(folder)
            put(folder / "summary.json", cell(p, scene, arm, 90 if arm == "fresh" else 100))
            for version, fresh_cost in (("v107", 85), ("v109", 88)):
                old = root / p[version + "_root"] / "runs" / scene / arm
                logs(old)
                previous = cell(p, scene, arm, fresh_cost if arm == "fresh" else 100)
                previous.pop("mechanism")
                previous.pop("replay")
                put(old / "summary.json", previous)
                put(old / "preflight.json", dict(status="complete", cell=previous["cell"],
                    integrity=dict(all_pass=True), prefix=PREFIX))
    return p, experiment


def first_packet(folder, arm):
    put(folder / "preflight.json", dict(status="complete", integrity=dict(all_pass=True),
        cell=dict(training_seed=6201, external_seed=106206, scenario="stable", arm=arm),
        initial_updates=70000, initial_online_critic_updates=20000,
        counts=analyze.expected_counts(8), mix=dict(real=.75, model=.25), model_batch_sizes=[128],
        feedback=dict(records_received=5, training_outcomes=5 if arm == "fresh" else 0),
        prefix=PREFIX, execution_wall=dict(native_started=100, first_packet_finished=110),
        mechanism=mechanism(arm, 8)))


class AnalyzeTests(unittest.TestCase):
    def test_exact_reproduction_both_fresh_comparisons_and_extra_compute(self):
        with TemporaryDirectory() as folder:
            p, experiment = fixture(Path(folder))
            result = analyze.analyze(Path(folder))
            self.assertEqual(result["decision"]["state"], "PASS_FIXED_PILOT")
            self.assertTrue(all(row["all_pass"] for row in result["no_fresh_reproduction"].values()))
            self.assertEqual(len(result["cost_gates"]), 5)
            delta = result["means"]["full_day"]["fresh_minus_v107_fresh"]
            self.assertEqual(delta, dict(wait_per_passenger=1.25,
                ride_per_passenger=2.5, cost_per_passenger=5.0))
            self.assertEqual(result["means"]["full_day"]["fresh_minus_v109_fresh"],
                dict(wait_per_passenger=.5, ride_per_passenger=1., cost_per_passenger=2.))
            self.assertEqual(result["additional_compute"]["actual_totals"],
                dict(extra_critic_rows=58368, extra_bootstrap_rows=233472, extra_bootstrap_calls=912))
            self.assertTrue(all(result["additional_compute"]["counters_complete"].values()))
            self.assertLessEqual((experiment / "summary.json").stat().st_size, p["summary_max_bytes"])

    def test_changed_or_missing_reproduction_is_a_technical_failure(self):
        cases = [(name, operation) for name in ("metrics.json", "updates.jsonl", "packet_receipts.json")
                 for operation in ("change", "remove")]
        cases += [("summary.json", "cost"), ("prefix_services.jsonl", "change")]
        for name, operation in cases:
            with self.subTest(name=name, operation=operation), TemporaryDirectory() as folder:
                root = Path(folder)
                p, experiment = fixture(root)
                path = experiment / "runs/stable/no_fresh" / name
                if operation == "remove":
                    path.unlink()
                elif operation == "cost":
                    value = analyze.read(path)
                    value["periods"] = periods(p, 101)
                    put(path, value)
                elif name.endswith(".jsonl"):
                    lines = path.read_text().splitlines()
                    lines[0] = '{"changed":true}'
                    path.write_text("\n".join(lines) + "\n")
                else:
                    value = analyze.read(path)
                    value[0]["changed"] = True
                    put(path, value)
                result = analyze.analyze(root)
                self.assertFalse(result["no_fresh_reproduction"]["stable"]["all_pass"])
                self.assertEqual(result["decision"]["state"], "HOLD_TECHNICAL_INCOMPLETE")
                self.assertIsNone(result["cost_gates"])

    def test_missing_or_wrong_mechanism_evidence_is_rejected(self):
        for change in ("missing", "wrong_channel", "replay_disagrees",
                       "wrong_coverage", "rng_changed", "targets_changed"):
            with self.subTest(change=change), TemporaryDirectory() as folder:
                root = Path(folder)
                _, experiment = fixture(root)
                path = experiment / "runs/stable/fresh/summary.json"
                row = analyze.read(path)
                if change == "missing":
                    row.pop("mechanism")
                elif change == "wrong_channel":
                    row["mechanism"]["original_batch_used"]["value"] = 0
                    row["replay"]["mechanism"] = copy.deepcopy(row["mechanism"])
                elif change == "replay_disagrees":
                    row["replay"]["mechanism"]["critic_complete_replacements"] -= 1
                else:
                    key, value = {"wrong_coverage": ("history_critic_rows_per_update", 64),
                        "rng_changed": ("main_rng_preserved", False),
                        "targets_changed": ("common_targets_exact", False)}[change]
                    row["mechanism"][key] = value
                    row["replay"]["mechanism"] = copy.deepcopy(row["mechanism"])
                put(path, row)
                result = analyze.analyze(root)
                self.assertIn("stable/fresh:mechanism_channel_evidence_failed", result["integrity"]["errors"])
                self.assertIsNone(result["decision"]["cost_gate_pass"])

    def test_missing_extra_counters_stay_unknown_including_no_fresh(self):
        for arm in ("fresh", "no_fresh"):
            with self.subTest(arm=arm), TemporaryDirectory() as folder:
                root = Path(folder)
                _, experiment = fixture(root)
                path = experiment / "runs/stable" / arm / "summary.json"
                row = analyze.read(path)
                row["mechanism"].pop("extra_bootstrap_rows")
                row["replay"]["mechanism"] = copy.deepcopy(row["mechanism"])
                put(path, row)
                result = analyze.analyze(root)
                self.assertEqual(result["decision"]["state"], "HOLD_TECHNICAL_INCOMPLETE")
                extra = result["additional_compute"]
                self.assertIsNone(extra["by_cell"]["stable/" + arm]["extra_bootstrap_rows"])
                self.assertIsNone(extra["actual_totals"]["extra_bootstrap_rows"])
                self.assertFalse(extra["counters_complete"]["extra_bootstrap_rows"])
                self.assertEqual(extra["recorded_subtotals"]["extra_bootstrap_rows"],
                                 116736 if arm == "fresh" else 233472)

    def test_preflight_waits_then_checks_original_prefix_and_eight_call_channels(self):
        for change in (None, "missing_old_prefix", "wrong_first_count", "missing_extra_count"):
            with self.subTest(change=change), TemporaryDirectory() as folder:
                root = Path(folder)
                p, experiment = fixture(root, gate=False)
                self.assertEqual(analyze.preflight(root, experiment)["status"], "pending")
                self.assertFalse((experiment / "preflight_gate.json").exists())
                for arm in p["arms"]:
                    first_packet(experiment / "runs/stable" / arm, arm)
                if change == "missing_old_prefix":
                    (root / p["v109_root"] / "runs/stable/no_fresh/prefix_trace.jsonl").unlink()
                elif change == "wrong_first_count":
                    path = experiment / "runs/stable/fresh/preflight.json"
                    row = analyze.read(path)
                    row["mechanism"]["value_calls"] = 456
                    put(path, row)
                elif change == "missing_extra_count":
                    path = experiment / "runs/stable/fresh/preflight.json"
                    row = analyze.read(path)
                    row["mechanism"].pop("extra_critic_rows")
                    put(path, row)
                result = analyze.preflight(root, experiment)
                self.assertEqual(result["all_pass"], change is None)


if __name__ == "__main__":
    unittest.main()
