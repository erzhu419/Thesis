"""Cross-seed source-block diagnosis; no policy fitting or welfare selection."""

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("diagnosis_source_context", HERE.parent / "mcwm_context_v7/context.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def write(path, r):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(r, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def read_blocks(seed, scene, arm):
    row = json.loads((HERE / "inputs" / scene / f"{arm}_seed{seed}.json").read_text())
    assert (row["seed"], row["scenario"], row["arm"]) == (seed, scene, arm)
    return np.asarray([b["features"] for b in row["blocks"]])


def score(model, rows):
    x = (rows-np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = -.5*((x[:,None,:]-np.asarray(model["centers"])[None,:,:])**2/np.asarray(model["variance"])).mean(2)
    weights = np.exp(logits-logits.max(1, keepdims=True))
    return weights/weights.sum(1, keepdims=True)


def diagnose(consumer, variant):
    provider = next(s for s in P["seeds"] if s != consumer)
    features = P["variants"][variant]
    labeled = {s:np.concatenate([read_blocks(provider,s,a)[:,features] for a in P["fit_arms"]]) for s in P["contexts"]}
    model = context.fit_recognizer(labeled, P["contexts"], P["variance_floor"])
    outcomes = []
    for arm in P["test_arms"]:
        confusion = np.zeros((3,3), dtype=int)
        per_context = []
        for index, scene in enumerate(P["contexts"]):
            rows = read_blocks(consumer,scene,arm)[:,features]
            scores = score(model, rows)
            prediction = scores.argmax(1)
            counts = np.bincount(prediction, minlength=3)
            confusion[index] = counts
            wrong = prediction != index
            per_context.append(dict(scenario=scene, blocks=len(rows), accuracy=float((~wrong).mean()),
                confident_wrong_fraction=float((wrong & (scores.max(1)>.9)).mean()),
                mean_true_class_score=float(scores[:,index].mean()), predicted_counts=counts.tolist()))
        outcomes.append(dict(arm=arm, confusion=confusion.tolist(),
            balanced_accuracy=float(np.mean([r["accuracy"] for r in per_context])), contexts=per_context))
    return dict(consumer_seed=consumer, provider_seed=provider, variant=variant, features=features,
        training_arms=P["fit_arms"], model=model, outcomes=outcomes, sumo_outcomes_used=False, scope=P["scope"])


def collect():
    paths = [HERE / "results" / f"{v}_seed{s}.json" for s in P["seeds"] for v in P["variants"]]
    rows = [json.loads(p.read_text()) for p in paths if p.exists()]
    for r in rows:
        assert r["provider_seed"] != r["consumer_seed"] and not r["sumo_outcomes_used"]
        assert r["features"] == P["variants"][r["variant"]]
    r = dict(complete=len(rows)==6, ready=len(rows), expected=6, results=rows, scope=P["scope"])
    write(HERE / "summary.json", r)
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=P["seeds"])
    parser.add_argument("--variant", choices=P["variants"])
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        r = collect()
        print(json.dumps(dict(complete=r["complete"], ready=r["ready"])))
    else:
        assert args.seed is not None and args.variant is not None
        r = diagnose(args.seed, args.variant)
        write(HERE / "results" / f"{args.variant}_seed{args.seed}.json", r)
        print("DONE offline context feature diagnosis", flush=True)
