"""Checks for cumulative-state interpretation, exact gates and bounded summaries."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("v28_analyze", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def probe(index=0, *, step=1, initial=False):
    line = analysis.LINES[index % 3]
    bank_deadline = (9000, 9300, 11700, 18000)[index // 3]
    rate = (index+1)*math.sin(step/7)*1.234567e-4
    shift = 0. if initial else step*rate
    return dict(index=index, bank_id=f"{bank_deadline}/{line}", line_id=line,
        bank_deadline=bank_deadline, role="tail_heldout_state" if bank_deadline == 18000 else "future_reference_state",
        canonical_clock=float(18300 if bank_deadline == 18000 else bank_deadline+index),
        issue_time=float(18300 if bank_deadline == 18000 else bank_deadline+index),
        proposal_seconds=[3.456789, 3.456789+shift, 3.456789-shift, 0., 7.5],
        holding_seconds=[.4320986, .4320986+shift/8, .4320986-shift/8, 0., .9375],
        q_mean={"fresh": [1.234567, 1.234567+shift, 1.234567-shift, 1., 1.+shift],
                "no_fresh": [1.234567, 1.234567-shift/2, 1.234567+shift/2, 1.+shift/3, 1.-shift]},
        q_min={"fresh": [1.134567, 1.134567+shift, 1.134567-shift, .9, .9+shift],
               "no_fresh": [1.134567, 1.134567-shift/2, 1.134567+shift/2, .9+shift/3, .9-shift]})


def record(step=1, *, seed=6201, scenario="stable", states=12):
    batches = [dict(partition=partition, n=n,
        reward_scaled=dict(mean=2.+step/17), bootstrap_contribution=dict(mean=3.+step/13),
        bellman_target=dict(mean=5.+step/17+step/13)) for partition, n in (
            ("historical_base", 64), ("fresh_outcome", 64), ("source", 128))]
    arms = {arm: dict(updates=8, training_steps=[dict(step=i+1, batches=deepcopy(batches))
        for i in range(8)]) for arm in ("fresh", "no_fresh")}
    for item in arms["no_fresh"]["training_steps"]:
        item["batches"][1]["partition"] = "historical_replacement"
    value = dict(metadata=dict(seed=seed, scenario=scenario, deadline=analysis.DEADLINES[step-1]),
        packet_index=step, cumulative_updates=8*step, candidates=analysis.CANDIDATES,
        probe_rows=[probe(i, step=step) for i in range(states)], arms=arms,
        integrity=dict(matched_sampling_draws=True, matched_actor_value_covariates=True,
                       global_rng_restored=True, common_candidate_execution=True),
        fresh_main_replay=dict(exact=True, checks=dict(global_rng=True, optimizers=True)))
    if step == 1:
        value["initial_probe_rows"] = [probe(i, step=1, initial=True) for i in range(states)]
    value["packet_feedback"] = [dict(dt_seconds=12., terminal=False) for _ in range(5)]
    return value


def complete_tree(root):
    for seed in analysis.SEEDS:
        for scenario in analysis.SCENARIOS:
            directory = root/"runs"/f"seed{seed}"/scenario
            directory.mkdir(parents=True)
            summary = dict(status="complete", cell=dict(seed=seed, scenario=scenario, origin_arm="fresh"),
                integrity=dict(all_pass=True, checks=dict(actual_F_exact=True)),
                budget=analysis.EXPECTED_BUDGET,
                replay_exact=dict(trace=True, updates=True, receipts=True, cost=True, metrics=True),
                post_budget_counts=dict(before=456, after=456))
            summary["post_budget_shadow_counts"] = dict(before=dict(fresh=248, no_fresh=248),
                                                       after=dict(fresh=248, no_fresh=248))
            (directory/"summary.json").write_text(json.dumps(summary))
            rows = [record(step, seed=seed, scenario=scenario) for step in range(1, 32)]
            (directory/"shadows.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_common_candidate_contrasts_and_baseline_levels_are_distinct():
    row = probe()
    row["proposal_seconds"] = [1., 10., 8., 0., 7.5]
    row["holding_seconds"] = [1., 3., 1., 0., 7.5]
    row["q_mean"] = dict(fresh=[10., 16., 13., 5., 9.], no_fresh=[20., 21., 25., 7., 6.])
    row["q_min"] = dict(fresh=[8., 12., 10., 4., 7.], no_fresh=[18., 19., 22., 6., 5.])
    values = analysis._metrics(row)
    assert values == [2., 2., 3., 2., -4., -3., 4., 3., -1., -1.]
    first = deepcopy(row)
    first["q_mean"]["fresh"][3] = 2.
    assert analysis._baseline_metrics(row, first)[:3] == [3., 6., 3.]
    assert analysis.persistence([None, -1., -2., 0., 3., 4., 0., 5.]) == [7, -2., 5., 5., 2, 1, 2]


def test_fixed_state_reuse_aliases_missing_line_and_zero_duration_stay_explicit():
    records = [record(step, states=2) for step in range(1, 32)]
    for value in records:
        value["probe_rows"][0]["holding_seconds"][1:3] = [0., 0.]
        value["probe_rows"][0]["dt_seconds"] = 0.
    result = analysis.summarize_bank(records)
    assert result["distinct_states"] == 2
    assert result["fixed_bank_consistent"] is True
    assert result["zero_duration_occurrences"] == 31
    assert result["state_only_occurrences"] == 62
    assert result["F_alias_counts"][0][1][:3] == [31, 31, 31]
    assert result["trajectories"][2][1] == 0
    assert all(values == [None]*12 for values in result["trajectories"][2][2])
    assert result["packet_feedback_counts"] == dict(rows=155, positive_dt=155, terminal=0)
    cancellation = record()
    cancellation["probe_rows"] = [probe(0), probe(3)]
    for row, delta in zip(cancellation["probe_rows"], (2., -2.)):
        row["holding_seconds"][1:3] = [4.+delta, 4.]
    values = analysis.summarize_bank([cancellation])["trajectories"][0][2][0]
    assert values[1] == 0. and values[10:12] == [2., 2.]
    records[-1]["probe_rows"][0]["canonical_clock"] += 1
    assert analysis.summarize_bank(records)["fixed_bank_consistent"] is False


def test_full_31_packet_four_day_gate_and_real_shaped_size(tmp_path):
    complete_tree(tmp_path)
    result = analysis.analyze(tmp_path)
    assert result["status"] == "complete"
    assert result["budget"]["diagnostic_optimizer_updates"] == 1984
    assert all(cell["integrity"]["exact_main_packets"] == 31 for cell in result["cells"])
    encoded = analysis.encode(result)
    print("v28 compact four-day bytes:", len(encoded.encode()))
    assert len(encoded.encode()) <= 50000
    padded = deepcopy(result)
    padded["test_padding"] = "x"*2400
    assert len(json.dumps(padded, separators=(",", ":")).encode()) > 50000
    compact = analysis.encode(padded)
    assert len(compact.encode()) <= 50000
    assert json.loads(compact)["summary_precision"].startswith("3 significant digits")
    assert all(len(item[2]) == 31 for cell in result["cells"] for item in cell["bank"]["trajectories"])
    path = tmp_path/"runs"/"seed6202"/"stable"/"shadows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["fresh_main_replay"]["checks"]["global_rng"] = False
    path.write_text("".join(json.dumps(row)+"\n" for row in rows[:-1]))
    assert analysis.analyze(tmp_path)["status"] == "integrity_failed"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    result = analysis.analyze(tmp_path)
    assert result["status"] == "integrity_failed"
    assert "6202/stable/fresh_main_parameters_Adam_RNG" in result["integrity"]["failed"]


def test_native_failure_does_not_wait_forever_and_bellman_uses_sample_weights(tmp_path):
    directory = tmp_path/"runs"/"seed6201"/"stable"
    directory.mkdir(parents=True)
    (directory/"failure.json").write_text('{"status":"failed"}')
    result = analysis.analyze(tmp_path)
    assert result["status"] == "integrity_failed"
    assert result["integrity"]["failed"] == ["6201/stable"]
    first, second = record(), record(2)
    extra = deepcopy(first["arms"]["fresh"]["training_steps"][0]["batches"][0])
    extra.update(n=128, reward_scaled=dict(mean=10.), bootstrap_contribution=dict(mean=20.),
                 bellman_target=dict(mean=30.))
    first["arms"]["fresh"]["training_steps"][0]["batches"].append(extra)
    value = analysis.summarize_training([first, second])
    row = next(row for row in value["rows"] if row[:2] == ["fresh", "historical_base"])
    assert row[2] == 8*64*2+128
    assert row[3][0] == pytest.approx((8*64*(2.+1/17)+8*64*(2.+2/17)+128*10)/row[2])
    assert value["bellman_nonadditive_batches"] == 0
    first["arms"]["fresh"]["training_steps"][0]["batches"][0]["bellman_target"]["mean"] = -1.
    assert analysis.summarize_training([first])["bellman_nonadditive_batches"] == 1
