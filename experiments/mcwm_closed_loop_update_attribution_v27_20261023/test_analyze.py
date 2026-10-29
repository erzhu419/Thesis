"""Aggregation failures that would change the attribution interpretation."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("v27_analyze", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def point(delta=0., *, transition=True):
    return dict(reward_scaled=2. if transition else None,
        bootstrap_contribution=3.+delta if transition else None,
        bellman_target=5.+delta if transition else None,
        q_observed_mean=1.+delta if transition else None,
        q_on_before_actor_mean=4.+delta, q_on_post_actor_mean=4.+2*delta,
        proposal_seconds=10.+delta, holding_seconds=7.+delta)


def probe(*, transition=True, role="packet_feedback", dt=12., terminal=False, line="7X"):
    return dict(role=role, line_id=line, has_transition=transition,
                dt_seconds=dt if transition else None, terminal=terminal,
                before=point(0., transition=transition),
                fresh=point(2., transition=transition),
                no_fresh=point(-1., transition=transition))


def record(deadline=9300):
    batches = [dict(partition="source", n=2,
                   **{key: dict(mean=value) for key, value in
                      zip(analysis.METRICS[:3], (2., 3., 5.))})]
    return dict(metadata=dict(seed=6201, scenario="stable", deadline=deadline),
                probe_rows=[probe()], integrity=dict(matched_sampling_draws=True),
                arms={arm: dict(updates=8, training_steps=[dict(step=i+1, batches=deepcopy(batches))
                      for i in range(8)], parameter_change={}) for arm in ("fresh", "no_fresh")})


def test_separates_in_sample_from_state_only_and_zero_terminal():
    value = record()
    value["probe_rows"].extend([
        probe(transition=False, role="tail_heldout_state"),
        probe(dt=0.), probe(terminal=True),
    ])
    summary = analysis.summarize_probes([value])
    assert len(summary["rows"]) == 4
    assert summary["counts"]["zero_dt"] == summary["counts"]["terminal"] == 1
    assert summary["counts"].get("unexpected_missing_metric_values", 0) == 0
    pure = next(row for row in summary["rows"] if row[4] == "tail_heldout_state")
    assert pure[7][:4] == [0, 0, 0, 0]
    assert pure[8][:4] == [None]*4
    assert pure[9][6] == 2.
    assert pure[10][6] == 3.
    assert pure[9][6]-pure[10][6] == -1.  # recover no_fresh minus before


def test_missing_is_not_zero_and_nonadditive_targets_are_detected():
    value = record()
    value["probe_rows"][0]["no_fresh"]["holding_seconds"] = None
    value["probe_rows"][0]["fresh"]["bellman_target"] = 99.
    summary = analysis.summarize_probes([value])
    assert summary["counts"]["unexpected_missing_metric_values"] == 1
    assert summary["rows"][0][7][-1] == 0
    assert summary["rows"][0][10][-1] is None
    assert summary["bellman_additivity"]["nonadditive"] == 1


def test_training_means_weight_actual_sample_uses_and_keep_phases():
    first, second = record(9300), record(18000)
    extra = dict(partition="source", n=6,
                 **{key: dict(mean=value) for key, value in
                    zip(analysis.METRICS[:3], (6., 9., 15.))})
    first["arms"]["fresh"]["training_steps"][0]["batches"].append(extra)
    summary = analysis.summarize_training([first, second])
    early = next(row for row in summary["rows"] if row[2:5] == ["early_labels", "fresh", "source"])
    assert early[5] == 22
    assert early[7][0] == pytest.approx((16*2+6*6)/22)
    assert {row[2] for row in summary["rows"]} == {"early_labels", "last_update"}


def _make_complete_tree(root):
    for seed in analysis.SEEDS:
        for scenario in analysis.SCENARIOS:
            directory = root/"runs"/f"seed{seed}"/scenario
            directory.mkdir(parents=True)
            summary = dict(status="complete", cell=dict(seed=seed, scenario=scenario, origin_arm="fresh"),
                integrity=dict(all_pass=True, checks=dict(actual_F_exact=True)),
                budget=dict(native_replay_seconds=24000, feedback_slots=285, main_updates=456,
                    shadow_anchors=4, shadow_arms=2, shadow_updates=64, extra_target_labels=0,
                    real_control_branches=0),
                replay_exact=dict(trace=True, updates=True, receipts=True, cost=True, metrics=True),
                post_budget_counts=dict(before=456, after=456))
            (directory/"summary.json").write_text(json.dumps(summary))
            rows = [record(deadline) for deadline in analysis.DEADLINES]
            for row in rows:
                row["metadata"].update(seed=seed, scenario=scenario)
            (directory/"shadows.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_full_gate_rejects_reference_or_budget_drift(tmp_path):
    _make_complete_tree(tmp_path)
    summary = analysis.analyze(tmp_path)
    assert summary["status"] == "complete"
    assert summary["budget"]["diagnostic_optimizer_updates"] == 256
    path = tmp_path/"runs"/"seed6202"/"stable"/"summary.json"
    run = json.loads(path.read_text())
    run["replay_exact"]["updates"] = False
    run["budget"]["main_updates"] = 457
    path.write_text(json.dumps(run))
    result = analysis.analyze(tmp_path)
    assert result["status"] == "integrity_failed"
    assert result["integrity"]["checks"]["6202/stable/full_reference_exact"] is False
    assert result["integrity"]["checks"]["6202/stable/budget"] is False
