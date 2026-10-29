"""Three-arm factual contrasts, old prediction parity and development regret."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024 import analyze as module
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.decomposition import summarize


LINES = ["102S", "102X", "122S", "122X", "311S", "311X", "406S", "406X", "705S", "705X", "7S", "7X"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value)+"\n" for value in values))


def heldout(index, arm):
    low = torch.tensor([-100.123456789, -100.111111111])
    base = torch.stack((low, low+torch.tensor([.01*(index-2), -.02+index*.005])), dim=1)
    if arm == "zero_memory":
        variants, components = dict(full=summarize(base)), {}
    else:
        base = base+(torch.tensor([[.05], [-.02]]) if arm == "per_head_level" else torch.tensor([[.04], [-.01]]))
        level = torch.tensor([[-.08, -.08], [.03, .03]]) if arm == "per_head_level" else torch.tensor([[.03212345, .03212345]])
        action = torch.tensor([[0., .01234567], [0., .02345678]])
        matrices = dict(base=base, level_only=base+level, action_only=base+action, full=base+(level+action))
        variants = {name:summarize(value) for name, value in matrices.items()}
        components = dict(level_heads=level.tolist(), action_heads=action.tolist())
    prediction = dict(variants["full"], predicted_choice=variants["full"]["choice"],
        actor_holding_seconds=3.123456789123+index*.00321, executed_actions=[-1., -.9375],
        holding_seconds=[0., 1.875])
    return dict(prediction=prediction, variants=variants, components=components)


def fixture(tmp_path, monkeypatch):
    protocol = module.read(module.HERE/"protocol.json")
    monkeypatch.setattr(module, "HERE", tmp_path)
    write(tmp_path/"protocol.json", protocol)
    root = tmp_path/"runs"
    for scenario in protocol["scenarios"]:
        case = root/scenario
        write(case/"summary.json", dict(integrity=dict(all_pass=True, checks=dict(workers_complete=True,
            actual_three_way_training_overlap=True, matched_original_budgets=True)),
            training_overlap_wall_seconds=123.456789, optimizer_overlap_wall_seconds=47.123456789))
        outputs = {arm:[] for arm in module.ARMS}
        for index, (line, window) in enumerate((line, window) for line in protocol["focus_lines"] for window in ("early", "tail")):
            slot, sequence = line+"/"+window, 12000+index
            clock = 9300.+index if window == "early" else 18300.+index
            directory = tmp_path/"old_v34"/scenario/slot.replace("/", "_")
            values = {arm:heldout(index, arm) for arm in module.ARMS}
            benefit = (index+1)*37.*(-1 if index%2 else 1)
            record = dict(slot=slot, selected=dict(sequence=sequence), actions=dict(low=[-1.], high=[-.9375]),
                prediction={"zero_memory":values["zero_memory"]["prediction"], "memory":values["per_head_level"]["prediction"]},
                comparison=dict(high_minus_low=dict(fixed300=dict(status="complete", wait=-benefit/2., ride=0., cost=-benefit)),
                    physical_separation=dict(status="separable")),
                branch_paths={arm:str(directory/arm/"summary.json") for arm in ("low", "high")})
            source = directory/"state_summary.json"
            write(source, record)
            for arm in ("low", "high"):
                write(directory/arm/"summary.json", dict(truth=dict(next_event=dict(zero_duration=index%2 == 0),
                    fixed300=dict(status="complete", dt=300.)), execution=dict(registered_hold_seconds=0. if arm == "low" else 1.875),
                    physical=dict(missing={field:95+index for field in module.API_FIELDS})))
            for arm in module.ARMS:
                outputs[arm].append(dict(slot=slot, sequence=sequence, issue=clock, line_id=line, source=str(source), **values[arm]))
        for arm in module.ARMS:
            old = "zero_memory" if arm == "zero_memory" else "memory" if arm == "per_head_level" else None
            write(case/arm/"summary.json", dict(integrity=dict(all_pass=True,
                checks=dict(original_sampling_and_global_private_rng_exact=True, optimizer_objects_retained=True,
                    post_budget_zero_training=True)), initial_preflight=dict(all_pass=True),
                budget=dict(feedback_slots=285, shadow_updates=456), heldout=dict(rows=6),
                old_reference_arm=old, old_final_prediction_rows=14 if old else 0))
            chunks = []
            for packet in range(57):
                rows = []
                for offset in range(5):
                    sequence = packet*5+offset
                    window = ("before9000", "9000-12000", "12000-18000")[sequence//95]
                    contrast = (.02 if sequence%2 else -.02) if arm == "shared_level" else 0.
                    rows.append(dict(sequence=sequence, line_id=LINES[sequence%12], window=window,
                        issue=1500.+sequence*50., context_observed_at=1500.+sequence*50.,
                        holding_seconds=[0., .9375, 1.875, 3.5], executed_actions=[-1., -.96875, -.9375, -.8833333],
                        reward_scaled=-.123456789, target=-20.-sequence*.0123456789+contrast,
                        value_contribution=-2.-sequence*.000123456789+contrast*.7,
                        sac_contribution=-18.-sequence*.0122222221+contrast*.3,
                        actor_holding_seconds=3.5+contrast*.013456789, proposal_seconds=28.+contrast*.107654312))
                chunks.append(dict(packet=packet+1, phase="packet_after", rows=rows))
                # Duplicate cumulative-bank rows must not inflate factual evidence.
                chunks.append(dict(packet=packet+1, phase="bank_after", rows=deepcopy(rows)))
            write_lines(case/arm/"scores.jsonl", chunks)
            write_lines(case/arm/"heldout.jsonl", outputs[arm])
    return root


def test_three_arm_285_contrasts_all_masks_and_twelve_truth_regrets(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    result = module.analyze(root)
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    assert len(result["states"]) == 12
    for case in result["cases"].values():
        assert case["optimizer_overlap_seconds"] == pytest.approx(47.123456789)
        training = case["training"]["overall"]
        assert training["n"] == 285
        target = training["differences"][0][0]
        assert target == pytest.approx([-.02/285., .02, .02], rel=2e-6, abs=1e-12)
        assert training["changed_actor_F"] == [285, 285, 0]
    for row in result["states"]:
        benefit = row[3][2]
        for prediction in row[4]:
            assert prediction[3] == (max(-benefit, 0.) if prediction[2] else max(benefit, 0.))
        assert row[7] == row[4][2][3]-row[4][1][3]
        assert len(row[5]) == 4
        assert row[6][2:4] == [True, True]
        assert abs(row[6][0]) <= row[6][4] and abs(row[6][1]) <= row[6][4]
        assert "api_missing_counts" in row[8]
    assert result["stable_gains_losses"][1] == sum(row[7] for row in result["states"] if row[0] == "stable")


def test_complete_twelve_state_three_arm_summary_fits_transfer_limit(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    result = module.analyze(root)
    size = len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode())+1
    assert size <= 20000, f"Full v36 compact fixture is {size}bytes"


def test_old_frozen_prediction_mismatch_blocks_scientific_comparison(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    path = root/"stable"/"zero_memory"/"heldout.jsonl"
    values = module.lines(path)
    values[0]["prediction"]["q_min"][0] += 1.
    write_lines(path, values)
    with pytest.raises(RuntimeError, match="Old twelve-state frozen predictions"):
        module.analyze(root)


def test_missing300_is_none_and_parent_failure_is_not_a_scientific_result(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    path = tmp_path/"old_v34"/"stable"/"7X_early"/"state_summary.json"
    value = module.read(path)
    value["comparison"]["high_minus_low"]["fixed300"] = dict(status="truncated", wait=None, ride=None, cost=None)
    write(path, value)
    result = module.analyze(root)
    missing = next(row for row in result["states"] if row[0] == "stable" and row[1] == "7X/early")
    assert missing[3] is None and missing[7] is None
    assert all(value[3] is None for value in missing[4])
    parent = root/"stable"/"summary.json"
    value = module.read(parent)
    value["integrity"]["all_pass"] = False
    write(parent, value)
    with pytest.raises(RuntimeError, match="parent/worker failed"):
        module.analyze(root)


def test_cli_failure_receipt_prevents_coordinator_hang(tmp_path):
    command = [sys.executable, str(Path(module.__file__)), "--root", str(tmp_path/"missing"),
               "--out", str(tmp_path/"shared_level_summary.json")]
    completed = subprocess.run(command, capture_output=True)
    assert completed.returncode != 0
    assert (tmp_path/"shared_level_summary_failure.json").exists()
