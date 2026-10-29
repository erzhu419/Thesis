"""Small resident development summary; absent states and ties stay explicit."""

from itertools import product
import json
from math import log1p
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
WORLDS = ("online", "frozen_common")


def direction(predicted, factual):
    if predicted == 0 or factual == 0:
        return "tie"
    return "agree" if (predicted > 0) == (factual > 0) else "disagree"


def score(record):
    truth = record["comparison"]["factual"]
    prediction = record["prediction"]
    # Reused real branch outputs remain separate from model-dependent Bellman values.
    low = record["truth"]["low"]
    high = record["truth"]["high"]
    norm = record["global_normalizer"]
    versions = {}
    for version, prediction_rows in (("legacy", prediction["legacy"]), ("response", prediction)):
        versions[version] = {}
        for world in WORLDS:
            p = prediction_rows["branches"][world]
            head_errors, standardized_squared, bellman_squared = [], [], []
            for candidate, actual in (("low", low), ("high", high)):
                predicted = p["candidates"][candidate]
                head_errors.append([a-b for a, b in zip(predicted["global_decoded"], actual["global_decoded"])])
                true_heads = [log1p(actual["global_decoded"][i]) if i < 2 else actual["global_decoded"][i]
                              for i in range(4)]
                # Exact same-batch predictions are scored by their deterministic output.
                if predicted.get("analytic_within_batch"):
                    decoded_heads = [log1p(predicted["global_decoded"][i]) if i < 2 else predicted["global_decoded"][i]
                                     for i in range(4)]
                    errors = [(decoded_heads[i]-true_heads[i])/norm["std"][i] for i in range(4)]
                else:
                    errors = [predicted["standardized_global"][i]-(true_heads[i]-norm["mean"][i])/norm["std"][i]
                              for i in range(4)]
                standardized_squared.extend(value*value for value in errors)
                bellman_squared.append((predicted["bellman"]["target"]-actual["bellman"]["target"])**2)
            versions[version][world] = dict(
                gains=p["gains"],
                directions={key: direction(p["gains"][key], truth["gains"][key])
                            for key in ("next_event_cost", "bellman_target")},
                direction_vs_300s=direction(p["gains"]["bellman_target"], truth["gains"]["cost_300_seconds"]),
                global_mae=[mean(abs(error[i]) for error in head_errors) for i in range(4)],
                g4_mse=mean(standardized_squared), bellman_target_mse=mean(bellman_squared),
                mean_trust=mean(p["candidates"][c]["trust"]["weight"] for c in ("low", "high")),
                candidates={c: {k: value for k, value in p["candidates"][c].items()
                                if k in ("holding_seconds", "global_decoded", "bellman", "response_correction_standardized", "analytic_within_batch")}
                            for c in ("low", "high")})
    return dict(cell=record["cell"], selected=record["selected"], factual=truth,
                zero_duration=low["zero_duration"], response=record["response"], versions=versions,
                integrity=record["integrity"], budget=record["budget"])


def aggregate(pairs):
    output = dict(n=len(pairs))
    if not pairs:
        return output
    output["actual_mean_300s_gain"] = mean(row["factual"]["gains"]["cost_300_seconds"] for row in pairs)
    for version in ("legacy", "response"):
        output[version] = {}
        for world in WORLDS:
            rows = [row["versions"][version][world] for row in pairs]
            output[version][world] = dict(
                mean_gains={key: mean(row["gains"][key] for row in rows) for key in rows[0]["gains"]},
                mean_abs_next_cost_gain_error=mean(abs(row["gains"]["next_event_cost"]-pair["factual"]["gains"]["next_event_cost"])
                                                   for row, pair in zip(rows, pairs)),
                g4_mse=mean(row["g4_mse"] for row in rows),
                bellman_target_mse=mean(row["bellman_target_mse"] for row in rows),
                direction_vs_300s={value: sum(row["direction_vs_300s"] == value for row in rows)
                                   for value in ("agree", "disagree", "tie")},
                next_event_direction={value: sum(row["directions"]["next_event_cost"] == value for row in rows)
                                      for value in ("agree", "disagree", "tie")},
                high_better_predictions=sum(row["gains"]["bellman_target"] > 0 for row in rows),
                mean_trust=mean(row["mean_trust"] for row in rows))
    return output


def analyze(root=HERE / "runs"):
    root = Path(root)
    p = json.loads((HERE / "protocol.json").read_text())
    missing, failures, errors, pairs, budgets = [], [], [], [], []
    for seed, scenario, line in product(p["training_seeds"], p["scenarios"], p["target_lines"]):
        cell = dict(seed=seed, scenario=scenario, target_line=line)
        directory = root / scenario / line / f"seed{seed}"
        if (directory / "failure.json").is_file():
            failures.append(dict(cell=cell, failure=json.loads((directory / "failure.json").read_text())))
        if not (directory / "summary.json").is_file():
            missing.append(cell)
            continue
        row = json.loads((directory / "summary.json").read_text())
        budgets.append(row["budget"])
        if row.get("status") != "complete" or row.get("integrity", {}).get("all_pass") is not True or row.get("cell") != cell:
            errors.append(dict(cell=cell, status=row.get("status"), integrity=row.get("integrity")))
            continue
        pairs.append(score(row))
    groups = dict(overall=aggregate(pairs), positive_dt=aggregate([row for row in pairs if not row["zero_duration"]]))
    for scenario in p["scenarios"]:
        selected = [row for row in pairs if row["cell"]["scenario"] == scenario]
        groups[scenario] = aggregate(selected)
        groups[scenario + "/positive_dt"] = aggregate([row for row in selected if not row["zero_duration"]])
        for line in p["target_lines"]:
            groups[scenario + "/" + line] = aggregate([row for row in selected if row["cell"]["target_line"] == line])
    return dict(status="complete" if len(pairs) == 12 and not (missing or failures or errors) else "incomplete",
        integrity=dict(expected_states=12, complete_states=len(pairs), missing=missing, failures=failures, errors=errors),
        priority="high", budget={key: sum(row.get(key, 0) for row in budgets) for key in (
            "prefix_sim_seconds", "new_suffix_sim_seconds", "reused_v7_branches", "selected_prefix_labels", "response_fit_labels", "truth_labels_enter_training", "response_ridge_refits")},
        pairs=pairs, groups=groups,
        interpretation=dict(scope="Original development states only; reused v7 truth never trains the candidate.",
            next_step="Review action differences before new-flow native confirmation, eight-step shadow updates or control.",
            missing="Missing/failed states remain explicit; no zero imputation.",
            analytic="Same-batch exact decoding changes diagnostic boundary only; those rows never entered original model TD."))
