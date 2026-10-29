"""Resident compact attribution summary; no fit arrays or checkpoints are pulled."""

from itertools import product
import json
from math import log1p
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def direction(predicted, factual):
    if predicted == 0 or factual == 0:
        return "tie"
    return "agree" if (predicted > 0) == (factual > 0) else "disagree"


def metrics(prediction, truth, normalizer, actual_gains):
    mse_heads, physical_errors, bellman_errors = [], [], []
    for candidate in ("low", "high"):
        p, y = prediction["candidates"][candidate], truth[candidate]
        heads = [log1p(y["global_decoded"][i]) if i < 2 else y["global_decoded"][i] for i in range(4)]
        if p["analytic_within_batch"]:
            predicted = [log1p(p["global_decoded"][i]) if i < 2 else p["global_decoded"][i] for i in range(4)]
            error = [(predicted[i]-heads[i])/normalizer["std"][i] for i in range(4)]
        else:
            error = [p["standardized_global"][i]-(heads[i]-normalizer["mean"][i])/normalizer["std"][i] for i in range(4)]
        mse_heads.append([v*v for v in error])
        physical_errors.append([abs(a-b) for a,b in zip(p["global_decoded"], y["global_decoded"])])
        bellman_errors.append((p["bellman"]["target"]-y["bellman"]["target"])**2)
    gains = prediction["gains"]
    return dict(gains=gains, g4_mse=mean(v for row in mse_heads for v in row),
        g4_mse_heads=[mean(row[i] for row in mse_heads) for i in range(4)],
        global_mae=[mean(row[i] for row in physical_errors) for i in range(4)],
        bellman_target_mse=mean(bellman_errors),
        false_immediate_gain_error=abs(gains["next_event_cost"]-actual_gains["next_event_cost"]),
        bellman_gain_error=abs(gains["bellman_target"]-actual_gains["bellman_target"]),
        direction_vs_300s=direction(gains["bellman_target"], actual_gains["cost_300_seconds"]),
        cost_direction_vs_300s=direction(gains["next_event_cost"], actual_gains["cost_300_seconds"]),
        hindsight_choice_gain=actual_gains["cost_300_seconds"] if gains["bellman_target"] > 0 else 0.)


def score(record):
    prediction = record["prediction"]
    actual = record["comparison"]["factual"]["gains"]
    arms = {name: metrics(p["branches"]["online"], record["truth"], record["global_normalizer"], actual)
            for name, p in prediction["arms"].items()}
    response = record["response"]
    coverage = response["selected_holding_coverage"]
    affected = {line: coverage["by_line"][line] for line in ("7X", "102X", "705X")}
    return dict(cell=record["cell"], selected=record["selected"],
        zero_duration=record["truth"]["low"]["zero_duration"], factual_gains=actual,
        arms=arms, frozen_base=metrics(prediction["arms"]["base_analytic"]["branches"]["frozen_common"],
                                     record["truth"], record["global_normalizer"], actual),
        full_group_contributions=prediction["full_group_contributions"]["online"],
        fits=dict(labels=response["labels"], retained_labels=response["retained_labels"],
                  refits=response["refits"], designs=response["designs"], fit_errors=response["fit_errors"],
                  coefficient_group_l2=response["coefficient_group_l2"],
                  holding_coverage=dict(overall=coverage["overall"], by_line=affected)),
        integrity=record["integrity"], budget=record["budget"])


def aggregate(pairs):
    result = dict(n=len(pairs))
    if not pairs:
        return result
    result["arms"] = {}
    for name in pairs[0]["arms"]:
        rows = [p["arms"][name] for p in pairs]
        result["arms"][name] = dict(
            mean_gains={key: mean(r["gains"][key] for r in rows) for key in ("next_event_cost", "bellman_target")},
            false_immediate_gain_mae=mean(r["false_immediate_gain_error"] for r in rows),
            bellman_gain_mae=mean(r["bellman_gain_error"] for r in rows),
            g4_mse=mean(r["g4_mse"] for r in rows),
            bellman_target_mse=mean(r["bellman_target_mse"] for r in rows),
            direction_vs_300s={v: sum(r["direction_vs_300s"] == v for r in rows) for v in ("agree", "disagree", "tie")},
            hindsight_choice_gain=mean(r["hindsight_choice_gain"] for r in rows))
    return result


def analyze(root=HERE / "runs"):
    root = Path(root)
    protocol = json.loads((HERE / "protocol.json").read_text())
    pairs, missing, failures, errors, budgets = [], [], [], [], []
    for seed, scenario, line in product(protocol["training_seeds"], protocol["scenarios"], protocol["target_lines"]):
        cell = dict(seed=seed, scenario=scenario, target_line=line)
        directory = root / scenario / line / f"seed{seed}"
        if (directory / "failure.json").is_file():
            failures.append(dict(cell=cell, failure=json.loads((directory / "failure.json").read_text())))
        if not (directory / "summary.json").is_file():
            missing.append(cell)
            continue
        record = json.loads((directory / "summary.json").read_text())
        budgets.append(record["budget"])
        if record.get("status") != "complete" or record.get("integrity", {}).get("all_pass") is not True or record.get("cell") != cell:
            errors.append(dict(cell=cell, status=record.get("status"), integrity=record.get("integrity")))
            continue
        pairs.append(score(record))
    groups = dict(overall=aggregate(pairs), positive_dt=aggregate([p for p in pairs if not p["zero_duration"]]))
    for scenario in protocol["scenarios"]:
        selected = [p for p in pairs if p["cell"]["scenario"] == scenario]
        groups[scenario] = aggregate(selected)
        groups[scenario + "/positive_dt"] = aggregate([p for p in selected if not p["zero_duration"]])
        for line in protocol["target_lines"]:
            groups[scenario + "/" + line] = aggregate([p for p in selected if p["cell"]["target_line"] == line])
    return dict(status="complete" if len(pairs) == 12 and not (missing or failures or errors) else "incomplete",
        integrity=dict(expected_states=12, complete_states=len(pairs), missing=missing, failures=failures, errors=errors),
        priority="high", pairs=pairs, groups=groups,
        budget={key: sum(b.get(key, 0) for b in budgets) for key in (
            "prefix_sim_seconds", "new_suffix_sim_seconds", "reused_v7_branches", "selected_prefix_labels",
            "response_fit_labels", "truth_labels_enter_training", "response_ridge_refits", "ridge_solutions")},
        interpretation=dict(scope="Same original development facts and reused truth; this is attribution, not independent confirmation.",
            nested="40/58/66 columns are independently refitted; drop arms zero full-model coefficients without refitting.",
            support="Coverage and numerical rank describe selected facts, not proof of causal identification.",
            storage="Coefficients stay in resident per-state summaries; this small file includes only necessary statistics.",
            next_step="Analyze stable/shock, positive-duration and grouped effects before locking a candidate for new-flow confirmation."))
