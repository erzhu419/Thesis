"""Cross-fit coefficient mappings without opening final test outcomes."""

from statistics import mean
import json

from context_common import HERE, P, FIXED, CONTEXTS, bundle_path, write
from context import fit_recognizer


def make_bundle(provider, consumer):
    losses, labeled = {}, {scene: [] for scene in CONTEXTS}
    for scene in CONTEXTS:
        for arm in FIXED:
            folder = HERE / "runs" / scene / f"{arm}_seed{provider}" / "results"
            paths = [folder / "ready.json", folder / "context_blocks.json"] + [folder / f"eval_{c:02d}.json" for c in range(10)]
            if not all(path.exists() for path in paths):
                return None
            ready = json.loads(paths[0].read_text())
            assert (ready["arm"], ready["seed"], ready["scenario"]) == (arm, provider, scene)
            blocks = json.loads(paths[1].read_text())["blocks"]
            labeled[scene].extend(b["features"] for b in blocks)
            rows = [json.loads(path.read_text()) for path in paths[2:]]
            for c, row in enumerate(rows):
                block = P["development_blocks"][c % 5]
                assert (row["case"], row["split"], row["arm"], row["scenario"], row["training_seed"]) == (c, "development", arm, scene, provider)
                assert all(row[k] == v for k, v in block.items())
                assert row["horizon_seconds"] == 24000 and row["demand_scale"] == P["evaluation_demand_scales"][c // 5]
            losses[scene, arm] = mean(r["generalized_seconds_per_departed_passenger"] for r in rows)
    best = lambda scene: min(FIXED, key=lambda arm: (losses[scene, arm], FIXED[arm]))
    global_arm = min(FIXED, key=lambda arm: (mean(losses[s, arm] for s in CONTEXTS), FIXED[arm]))
    coefficients = {s: FIXED[best(s)] for s in CONTEXTS}
    return dict(provider_seed=provider, consumer_seed=consumer,
        coefficients=coefficients, selected_arms={s: best(s) for s in CONTEXTS},
        mapping_has_context_effect=len(set(coefficients.values())) > 1,
        best_global_fixed_arm=global_arm,
        development_costs={s: {a: losses[s, a] for a in FIXED} for s in CONTEXTS},
        recognizer=fit_recognizer(labeled, CONTEXTS, P["context_variance_floor"]),
        selection_scope=P["selection"], test_outcomes_used=False)


def prepare():
    created = []
    for consumer in P["training_seeds"]:
        if bundle_path(consumer).exists():
            continue
        provider = next(s for s in P["training_seeds"] if s != consumer)
        bundle = make_bundle(provider, consumer)
        if bundle is not None:
            write(bundle_path(consumer), bundle)
            created.append(consumer)
    return created


if __name__ == "__main__":
    print(json.dumps(dict(created=prepare())))
