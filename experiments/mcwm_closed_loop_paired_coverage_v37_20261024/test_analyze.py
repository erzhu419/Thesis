"""Resident two-scenario truth joins, finite300 rights and bounded export."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v37_analyze_test", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
ARMS = analysis.ARMS
LINES = ("7X", "102X", "705X")
SCENARIOS = ("stable", "local_speed_drop")
STAGES = ("packet_before", "packet_after", "frozen_final")


def write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value))


def record(index, line, window, benefit, *, role="train"):
    issue = (9600 + 3000*(index//3)) if role == "train" else (18400 if window == "early" else 21200)
    low = dict(status="complete", dt=300.0, wait=10000.+benefit,
        ride=20000.-benefit, cost=40000.+benefit)
    high = dict(status="complete", dt=300.0, wait=10000., ride=20000., cost=40000.)
    result = dict(id=f"train/{index}" if role == "train" else f"heldout/{line}/{window}",
        role=role, index=index, line_id=line, window_name=window, issued_at=issue,
        source_sequence=100+index, features=[float(line == value) for value in LINES]+[(index//3-1)*.25],
        holds=[0.,1.875], status="collected", selected_for_training=role == "train",
        available_at=issue+330., admitted_at=issue+600., admitted_deadline=issue+600.,
        outcomes={arm:dict(truth=truth,
            next_event=dict(status="complete", zero_duration=index % 3 == 0,
                wait=0.,ride=0.,cost=0.)) for arm,truth in (("low",low),("high",high))},
        comparison=dict(physical_separation=dict(status="separable", departure_separable=True,
            next_arrival_separable=True)), pair_receipt=dict(overlap_seconds=287.123456789))
    return result


def bellman(index, arm_index=0, phase_index=0):
    value = (index+1)*.0123456789123 + arm_index*.003456789123 + phase_index*.001234567891
    return dict(reward_scaled=0., value_contribution=value,
        sac_contribution=-value*.37123456789, target=value*(1-.37123456789))


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "HERE", tmp_path / "code")
    protocol = json.loads((HERE / "protocol.json").read_text())
    write(analysis.HERE / "protocol.json", protocol)
    root = tmp_path / "runs"
    for scenario in SCENARIOS:
        collection, shadow = root/"collection"/scenario, root/"shadow"/scenario
        budget = dict(feedback_slots=285, ordinary_slots=267, paired_slots=18,
            admitted_pairs=9, heldout_pairs=6, prefix_native_seconds=24000,
            training_suffixes=18, heldout_suffixes=12, suffix_native_seconds=9000,
            suffix_optimizer_updates=0, post18000_training=0)
        write(collection/"summary.json", dict(integrity=dict(all_pass=True), budget=budget))
        write(shadow/"summary.json", dict(integrity=dict(all_pass=True),
            optimizer_overlap_wall_seconds=76.123456789))
        train = [record(i, LINES[i%3], f"w{i//3+1}", 20*(i%3+1)+40*(i//3-1)) for i in range(9)]
        benefits = [10,-20,30,-40,50,-60] if scenario == "stable" else [-70,80,-90,100,-110,120]
        heldout = [record(i,line,window,benefits[i],role="heldout")
            for i,(line,window) in enumerate((line,window) for line in LINES for window in ("early","tail"))]
        excluded = record(0,"7X","w1",10**8)
        excluded.update(id="unadmitted/pollutant", selected_for_training=False)
        write(collection/"pair_truth.json", list(reversed(heldout))+train+[excluded])
        write(collection/"dataset.json", dict(packets=[dict(index=i+1,
            receipt=dict(pair_ids=[row["id"]],processed_at=row["admitted_at"])) for i,row in enumerate(train)]))
        for ai,arm in enumerate(ARMS):
            target_contract = dict(reward_scale=.0001, mixed_backup_lambda=.1, lambda_weights="V",
                gamma=1., rewards="raw -(2W+R)", no_finite300_target=True)
            write(shadow/arm/"summary.json", dict(integrity=dict(all_pass=True),
                budget=dict(feedback_slots=285,shadow_updates=456), target_contract=target_contract))
            scores = []
            for i,row in enumerate(heldout):
                if arm == "zero_memory":
                    delta = .123456789123
                elif arm == "per_head_level":
                    delta = -.123456789123 if scenario == "stable" else (1 if benefits[i]>0 else -1)*.123456789123
                else:
                    delta = ((1 if benefits[i]>0 else -1) if scenario == "stable" and i != 5 else
                        (1 if scenario == "stable" else -1))*.123456789123
                scores.append(dict(slot=row["id"].removeprefix("heldout/"),
                    sequence=row["source_sequence"], issue=row["issued_at"],
                    prediction=dict(delta_min=delta,min_heads=[ai%2,(ai+1)%2]),
                    bellman_high_minus_low=bellman(i,ai)))
            write(shadow/arm/"heldout.jsonl", list(reversed(scores)), jsonl=True)
            write(shadow/arm/"pair_targets.jsonl", [dict(pair_index=i,phase=phase,
                prediction=dict(delta_min=.123456789123*(i+1)+ai*.019876543219+pi*.001234567891),
                bellman_high_minus_low=bellman(i,ai,pi))
                for i in range(9) for pi,phase in enumerate(STAGES)], jsonl=True)
    return root


def test_full_join_direction_regret_stage27_and_export_limit(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    result = analysis.analyze_cases(root)
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    assert len(result["states"]) == 12 and result["missing"] == []
    for scenario in SCENARIOS:
        case = result["cases"][scenario]
        assert len(case["training_pairs"]) == 9
        assert case["finite_fit_packets"][-1][2] == 9
        assert set(case["pair_target_evolution"]) == set(ARMS)
        assert sum(map(len,case["pair_target_evolution"].values())) == 27
        assert all(len(row) == 3 and all(len(stage) == 5 for stage in row)
            for pairs in case["pair_target_evolution"].values() for row in pairs)
        assert all(fit["pairs"] == 9 and fit["outcome_slots"] == 18
            for fit in case["finite_fit"].values())
        assert case["finite_fit"]["finite300_line"]["coefficients"] == [
            [15.,-15.,15.],[30.,-30.,30.],[45.,-45.,45.]]
        assert len(case["by_line"]) == 3 and len(case["by_window"]) == 2
    stable, shock = (result["cases"][scenario]["overall"] for scenario in SCENARIOS)
    assert stable["models"]["zero_memory"] == [6,6,3,6,120.]
    assert stable["models"]["per_head_level"] == [6,6,3,0,90.]
    assert stable["models"]["shared_level"] == [6,6,5,4,60.]
    assert stable["shared_vs_controls"][0] == [6,-60.,2,0,60.,0.]
    assert stable["shared_vs_controls"][1] == [6,-30.,3,1,90.,60.]
    assert shock["models"]["zero_memory"] == [6,6,3,6,270.]
    assert shock["models"]["per_head_level"] == [6,6,6,3,0.]
    assert shock["models"]["shared_level"] == [6,6,3,0,300.]
    assert shock["shared_vs_controls"][0] == [6,30.,3,3,270.,300.]
    states = [dict(zip(result["columns"]["states"],row)) for row in result["states"]]
    sample = next(row for row in states if row["scenario"]=="stable" and row["slot"]=="7X/early")
    assert sample["true_wait_ride_cost_gain"] == [10.,-10.,10.]
    quality = dict(zip(result["columns"]["quality"],sample["quality"]))
    assert quality["zero_next_event"] == ["low","high"]
    assert quality["next_event_gain"] == [0.,0.,0.]
    payload = json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n"
    assert len(payload.encode()) <= 24000, len(payload.encode())


def test_heldout_truth_never_fits_readout(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    before = analysis.analyze_cases(root)
    for scenario in SCENARIOS:
        path = root/"collection"/scenario/"pair_truth.json"
        rows = json.loads(path.read_text())
        for row in rows:
            if row["role"] == "heldout":
                for key, value in (("wait",10**6),("ride",10**6),("cost",3*10**6)):
                    row["outcomes"]["low"]["truth"][key] += value
        write(path, rows)
    after = analysis.analyze_cases(root)
    for scenario in SCENARIOS:
        assert before["cases"][scenario]["finite_fit"] == after["cases"][scenario]["finite_fit"]
    model_column = before["columns"]["states"].index("models")
    assert before["states"][0][model_column] != after["states"][0][model_column]


def test_exact_zero_total_benefit_has_zero_regret_and_no_unique_direction(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    path = root/"collection"/"stable"/"pair_truth.json"
    rows = json.loads(path.read_text())
    row = next(row for row in rows if row["id"] == "heldout/705X/tail")
    row["outcomes"]["low"]["truth"] = dict(row["outcomes"]["high"]["truth"])
    write(path, rows)
    result = analysis.analyze_cases(root)
    states = [dict(zip(result["columns"]["states"],row)) for row in result["states"]]
    selected = next(row for row in states if row["scenario"]=="stable" and row["slot"]=="705X/tail")
    assert selected["true_wait_ride_cost_gain"] == [0.,0.,0.]
    direction = result["columns"]["state_model"].index("direction_correct")
    regret = result["columns"]["state_model"].index("regret")
    assert all(model[direction] is None and model[regret] == 0. for model in selected["models"])
    assert all(model[0:2] == [6,5] for model in result["cases"]["stable"]["overall"]["models"].values())


@pytest.mark.parametrize("failure", ["parent", "future", "stage", "join"])
def test_failed_evidence_cannot_produce_scientific_result(tmp_path, monkeypatch, failure):
    root = fixture(tmp_path, monkeypatch)
    if failure == "parent":
        path = root/"shadow"/"stable"/"summary.json"
        row = json.loads(path.read_text())
        row["integrity"]["all_pass"] = False
        write(path,row)
    elif failure == "future":
        path = root/"collection"/"stable"/"dataset.json"
        row = json.loads(path.read_text())
        row["packets"][0]["receipt"]["processed_at"] -= 1
        write(path,row)
    else:
        path = root/"shadow"/"stable"/"zero_memory"/("pair_targets.jsonl" if failure=="stage" else "heldout.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "stage":
            rows[0]["phase"] = "packet_after"
        else:
            rows[0]["sequence"] += 1
        write(path,rows,jsonl=True)
    with pytest.raises((RuntimeError,ValueError)):
        analysis.analyze_cases(root)
