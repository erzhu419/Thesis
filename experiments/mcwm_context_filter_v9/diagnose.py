"""Replay source-block filters with matched resets and cross-seed models."""

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from filters import WindowFilter, route

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("filter_original_context", HERE.parent / "mcwm_context_v7/context.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(seed, scene, arm):
    row = json.loads((HERE / "inputs" / scene / f"{arm}_seed{seed}.json").read_text())
    assert (row["seed"], row["scenario"], row["arm"]) == (seed, scene, arm)
    return row


def fit(provider, variant):
    indices = P["variants"][variant]
    labeled = {}
    for scene in P["contexts"]:
        rows = [b["features"] for arm in P["fit_arms"] for b in read(provider, scene, arm)["blocks"]]
        labeled[scene] = np.asarray(rows)[:, indices]
    return context.fit_recognizer(labeled, P["contexts"], P["variance_floor"])


def recover_resets(blocks, trace, model):
    # Match the two possible original updates; this recovers resets without labels.
    assert len(blocks) == len(trace)
    original = context.ContextFilter(model, np.zeros(len(model["classes"])), P["original_filter"])
    resets, maximum_residual = [], 0.
    for index, (block, recorded) in enumerate(zip(blocks, trace)):
        assert block["end_source_row"] == recorded["end_source_row"]
        previous = original.belief.copy()
        original.observe(block["features"])
        continued = original.belief.copy()
        original.reset()
        original.observe(block["features"])
        restarted = original.belief.copy()
        target = np.asarray(recorded["belief"])
        continued_error = float(np.max(np.abs(continued - target)))
        reset_error = float(np.max(np.abs(restarted - target)))
        assert min(continued_error, reset_error) <= 1e-8, "Original trace does not replay"
        if index and continued_error <= 1e-8 and reset_error <= 1e-8:
            assert np.allclose(previous, 1 / len(previous), atol=1e-10), "Reset is not identifiable"
        reset = index == 0 or continued_error > 1e-8
        resets.append(reset)
        maximum_residual = max(maximum_residual, min(continued_error, reset_error))
        original.belief = target
    return resets, maximum_residual


def metrics(beliefs, coefficients, accepted, true_class, true_coefficient, global_coefficient):
    beliefs, coefficients, accepted = np.asarray(beliefs), np.asarray(coefficients), np.asarray(accepted)
    correct = beliefs.argmax(1) == true_class
    return dict(blocks=len(beliefs), accuracy=float(correct.mean()),
        wrong_high_score_fraction=float((~correct & (beliefs.max(1) > P["high_score_threshold"])).mean()),
        mean_max_score=float(beliefs.max(1).mean()), accepted_fraction=float(accepted.mean()),
        accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
        accepted_blocks=int(accepted.sum()), accepted_correct=int((accepted & correct).sum()),
        mean_coefficient=float(coefficients.mean()),
        mapping_absolute_error=float(np.abs(coefficients - true_coefficient).mean()),
        global_mapping_absolute_error=abs(global_coefficient - true_coefficient))


def diagnose(consumer, variant):
    provider = next(seed for seed in P["seeds"] if seed != consumer)
    model = fit(provider, variant)
    outcomes, reset_records = {}, []
    for true_class, scene in enumerate(P["contexts"]):
        row = read(consumer, scene, "recognized_context")
        selection = row["selection"]
        assert (selection["provider_seed"], selection["consumer_seed"]) == (provider, consumer)
        assert model["classes"] == selection["recognizer"]["classes"]
        if variant == "full":
            for key in ("mean", "scale", "centers", "variance"):
                np.testing.assert_allclose(model[key], selection["recognizer"][key], atol=1e-10, rtol=0.)
        resets, residual = recover_resets(row["blocks"], row["trace"], selection["recognizer"])
        reset_records.append(dict(scenario=scene, reset_block_indices=[i for i, r in enumerate(resets) if r],
            original_replay_max_residual=residual))
        coefficients = np.asarray([selection["coefficients"][c] for c in model["classes"]])
        global_coefficient = row["global_coefficient"]
        true_coefficient = coefficients[true_class]
        for name, window in [("sticky", None)] + [(f"mean_w{w}", w) for w in P["windows"]]:
            learner = context.ContextFilter(model, coefficients, P["original_filter"]) if window is None else WindowFilter(
                model, coefficients, window, global_coefficient)
            beliefs = []
            for block, reset in zip(row["blocks"], resets):
                if reset:
                    learner.reset()
                learner.observe(np.asarray(block["features"])[P["variants"][variant]])
                beliefs.append(learner.belief.copy())
            if name == "sticky" and variant == "full":
                np.testing.assert_allclose(beliefs, [t["belief"] for t in row["trace"]], atol=1e-8, rtol=0.)
            for suffix, margin in [("", 0.), ("_fallback", P["fallback_score_margin"])]:
                decisions = [route(b, coefficients, global_coefficient, margin) for b in beliefs]
                result = metrics(beliefs, [d[0] for d in decisions], [d[1] for d in decisions],
                    true_class, true_coefficient, global_coefficient)
                outcomes.setdefault(name + suffix, []).append(dict(scenario=scene, **result))
    return dict(consumer_seed=consumer, provider_seed=provider, variant=variant, model=model,
        reset_source="matched_original_belief_updates", resets=reset_records,
        outcomes=[dict(method=name, contexts=rows, balanced_accuracy=float(np.mean([r["accuracy"] for r in rows])),
            wrong_high_score_fraction=float(np.mean([r["wrong_high_score_fraction"] for r in rows])),
            accepted_fraction=float(np.mean([r["accepted_fraction"] for r in rows])),
            mapping_absolute_error=float(np.mean([r["mapping_absolute_error"] for r in rows])))
            for name, rows in outcomes.items()], sumo_outcomes_used=False, policy_training=False)


def collect():
    paths = [HERE / "results" / f"{v}_seed{s}.json" for s in P["seeds"] for v in P["variants"]]
    results = [json.loads(path.read_text()) for path in paths if path.exists()]
    for row in results:
        assert row["provider_seed"] != row["consumer_seed"]
        assert not row["sumo_outcomes_used"] and not row["policy_training"]
    aggregate = []
    for variant in P["variants"]:
        folds = [r for r in results if r["variant"] == variant]
        if len(folds) != len(P["seeds"]):
            continue
        for index, method in enumerate(folds[0]["outcomes"]):
            rows = [f["outcomes"][index] for f in folds]
            assert all(r["method"] == method["method"] for r in rows)
            aggregate.append(dict(variant=variant, method=method["method"], **{
                key:float(np.mean([r[key] for r in rows])) for key in (
                    "balanced_accuracy", "wrong_high_score_fraction", "accepted_fraction", "mapping_absolute_error")}))
    row = dict(complete=len(results)==len(paths), ready=len(results), expected=len(paths),
        aggregate=aggregate, results=results, scope=P["scope"], coefficient_scope=P["coefficient_scope"])
    write(HERE / "summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=P["seeds"])
    parser.add_argument("--variant", choices=P["variants"])
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        result = collect()
        print(json.dumps(dict(complete=result["complete"], ready=result["ready"])))
    else:
        assert args.seed is not None and args.variant is not None
        write(HERE / "results" / f"{args.variant}_seed{args.seed}.json", diagnose(args.seed, args.variant))
        print("DONE matched context filter diagnosis", flush=True)
