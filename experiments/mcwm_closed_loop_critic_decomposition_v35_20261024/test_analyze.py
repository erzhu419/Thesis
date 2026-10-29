"""Full twelve-state truth join, all masked variants and compact export limit."""
import json

import pytest
import torch

from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024 import analyze as module
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.decomposition import summarize
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.coverage import ACTION_COLUMNS, SUPPORT_COLUMNS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def decomposition(index, heads=2):
    head = torch.arange(heads, dtype=torch.float32)
    low = -100.123456789+head*.04321
    zero = torch.stack((low, low+(head-.5)*.045678+index*.01), dim=1)
    base = zero+torch.stack((head*.00543, head*.00765), dim=1)
    level = torch.stack((head*-.01789, head*-.01789), dim=1)
    action = torch.stack((torch.zeros_like(head), (head+1)*.01234), dim=1)
    values = dict(zero_memory=zero, memory_base=base, level_only=base+level,
                  action_only=base+action, full=base+(level+action))
    variants = {name:summarize(value) for name, value in values.items()}
    full, masked, old = [variants[name]["delta_min"] for name in ("full", "memory_base", "zero_memory")]
    return dict(variants=variants, components=dict(level_heads=level.tolist(),
        action_delta_heads=(action[:, 1]-action[:, 0]).tolist()),
        between_learner_base_drift=dict(min_advantage_difference=masked-old),
        direct_memory_same_weights=dict(full_minus_base_min=full-masked,
            min_nonadditivity=full-variants["level_only"]["delta_min"]-variants["action_only"]["delta_min"]+masked),
        integrity=dict(all_pass=True))


def fixture(tmp_path, monkeypatch):
    protocol = module.read(module.HERE/"protocol.json")
    monkeypatch.setattr(module, "HERE", tmp_path)
    write(tmp_path/"protocol.json", protocol)
    root = tmp_path/"runs"
    expected = {}
    action = [95, 0., .123456789, 1.875, 35.123456789, 60., 60., 60., 39.987654321,
              24.123456789, 2, 3, 4, 77, 7]
    metadata = dict(admitted_target_rows=285, nonzero_memory_rows=285, source_history_rows_included=0,
        action_semantics="archived executed a converted directly to seconds; no proposal and no repeated F",
        neighborhood_seconds=.5, neighborhood_is_gate=False, low_action_upper_seconds=1.875,
        v34_training_windows=dict(early=[9300, 9900], tail=[18300, 18900]),
        design="Actual residual inputs [m,z*m], z=executed_hold/max_hold; all-zero columns omitted, constant nonzero features retained")
    design = dict(rows=285, nonzero_features=84, varying_features=72, constant_nonzero_features=12,
        all_zero_features=0, columns=168, rank=29, full_rank_condition=None,
        nonzero_subspace_condition=123.456789012345)
    for scenario in protocol["scenarios"]:
        case = root/scenario
        write(case/"summary.json", dict(integrity=dict(all_pass=True,
            checks=dict(all_state_checks=True, all_weights_unchanged=True,
                counters_and_optimizer_steps_unchanged=True, global_rng_unchanged=True))))
        selected, chunks = [], []
        for index, (line, window) in enumerate((line, window) for line in protocol["focus_lines"] for window in ("early", "tail")):
            slot = line+"/"+window
            source = tmp_path/"old_v34"/scenario/slot.replace("/", "_")/"state_summary.json"
            benefit = (-1 if index%2 else 1)*(index+1)*37.
            write(source, dict(slot=slot, selected=dict(sequence=12000+index),
                comparison=dict(high_minus_low=dict(fixed300=dict(status="complete", cost=-benefit)))))
            diagnostic = decomposition(index)
            chunks.append(dict(slot=slot, sequence=12000+index, source=str(source), decomposition=diagnostic))
            expected[scenario, slot] = benefit, diagnostic
            selected.append(dict(slot=slot, supports={key:[95, 0., 60., 1, 2, .0123456789123,
                .123456789123, True, True, 23, list(range(23))]
                for key in ("full", "9000-12000", "12000-18000")}))
        (case/"heads.jsonl").write_text("".join(json.dumps(value)+"\n" for value in chunks))
        cover = dict(selected=selected, metadata=metadata, all_actions=[285]+action[1:],
            by_line={line:action for line in protocol["focus_lines"]},
            by_line_window={line+"/"+window:action for line in protocol["focus_lines"]
                for window in ("before9000", "9000-12000", "12000-18000", "tail")},
            v34_window_training={line+"/"+window:([0]+[None]*9+[0]*5 if window == "tail" else action)
                for line in protocol["focus_lines"] for window in ("early", "tail")},
            residual_design={key:design for key in ("all_target285", *protocol["focus_lines"])},
            columns=dict(action=ACTION_COLUMNS, selected_support=SUPPORT_COLUMNS))
        write(case/"coverage.json", cover)
    return root, expected


def test_twelve_state_truth_join_and_all_five_variants(tmp_path, monkeypatch):
    root, expected = fixture(tmp_path, monkeypatch)
    result = module.analyze(root)
    assert result["integrity"]["all_pass"] and result["status"] == "complete"
    assert len(result["states"]) == 12
    for row in result["states"]:
        benefit, diagnostic = expected[row[0], row[1]]
        assert row[3] == benefit
        assert len(row[5]) == 2
        for index, name in enumerate(module.VARIANTS):
            actual, source = row[4][index], diagnostic["variants"][name]
            expected_regret = max(-benefit, 0.) if source["choice"] else max(benefit, 0.)
            assert actual[0] == pytest.approx(source["delta_min"], rel=2e-6)
            assert actual[1] == pytest.approx(source["delta_mean"], rel=2e-6)
            assert actual[2:4] == source["min_heads"]
            assert actual[4] == source["choice"]
            assert actual[5] == expected_regret
            assert actual[6] == ((source["delta_min"] > 0)-(source["delta_min"] < 0) == (1 if benefit > 0 else -1))
            assert actual[7] == pytest.approx(source["fixed_low_head_delta"], rel=2e-6)
            assert actual[8] == pytest.approx(source["selection_gap"], rel=2e-6, abs=1e-12)
        for head in range(2):
            expected_head = [diagnostic["variants"]["zero_memory"]["delta_heads"][head],
                diagnostic["variants"]["memory_base"]["delta_heads"][head],
                diagnostic["components"]["level_heads"][head][0],
                diagnostic["components"]["action_delta_heads"][head],
                diagnostic["variants"]["full"]["delta_heads"][head]]
            assert row[5][head] == pytest.approx(expected_head, rel=2e-6, abs=1e-12)
    for scenario, case in result["cases"].items():
        selected = [row for row in result["states"] if row[0] == scenario]
        for index, variant in enumerate(module.VARIANTS):
            assert case["overall"][variant]["n"] == 6
            assert case["overall"][variant]["regret"] == sum(row[4][index][5] for row in selected)
        assert case["coverage"]["selected_v34_training_windows"]["7X/tail"] == 0


def test_full_two_head_twelve_state_summary_below_20000_bytes(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    result = module.analyze(root)
    size = len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode())+1
    assert size <= 20000, f"Full12-state,2-head compact summary is {size} bytes"


def test_failed_parent_cannot_be_reported_as_completed_scientific_result(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    path = root/"stable"/"summary.json"
    value = module.read(path)
    value["integrity"]["all_pass"] = False
    value["integrity"]["checks"]["all_weights_unchanged"] = False
    write(path, value)
    result = module.analyze(root)
    assert result["status"] == "integrity_failed"
    assert not result["integrity"]["all_pass"]


def test_missing300_truth_never_becomes_zero_regret(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path/"old_v34"/"stable"/"7X_early"/"state_summary.json"
    value = module.read(path)
    value["comparison"]["high_minus_low"]["fixed300"] = dict(status="missing", cost=None)
    write(path, value)
    with pytest.raises(RuntimeError, match="300s truth is missing"):
        module.analyze(root)
