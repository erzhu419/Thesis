"""Compute matched cohort effects on server storage."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
METRICS = ("mean_wait_seconds", "mean_onboard_seconds", "mean_generalized_seconds", "completion_rate")
COHORTS = ("all", "baseline", "extra", "baseline_before", "baseline_during", "baseline_after")


def collect():
    rows, missing = {}, []
    for method, seed, day, scene in product(
        P["methods"], P["training_seeds"], P["evaluation_seeds"], P["scenes"]
    ):
        path = HERE / "evaluate" / method / f"seed{seed}" / f"day{day}_{scene}.json"
        if not path.exists():
            missing.append(str(path.relative_to(HERE)))
            continue
        row = json.loads(path.read_text())
        assert (row["method"], row["training_seed"], row["evaluation_seed"], row["scene"]) == (
            method, seed, day, scene
        )
        assert row["phase"] == "evaluate" and row["horizon_seconds"] == P["horizon_seconds"]
        rows[method, seed, day, scene] = row
    aggregates, effects, comparisons = [], [], []
    for method, scene, cohort in product(P["methods"], P["scenes"], COHORTS):
        cases = [r for k, r in rows.items() if (k[0], k[3]) == (method, scene)]
        if len(cases) != 6 or not all(r["groups"][cohort]["passengers"] for r in cases):
            continue
        aggregates.append(dict(method=method, scene=scene, cohort=cohort, cases=6,
            passengers_mean=mean(r["groups"][cohort]["passengers"] for r in cases),
            **{m: mean(r["groups"][cohort][m] for r in cases) for m in METRICS}))
    for method, seed, day in product(P["methods"], P["training_seeds"], P["evaluation_seeds"]):
        normal, shock = rows.get((method, seed, day, "normal")), rows.get((method, seed, day, "burst50"))
        if normal is None or shock is None:
            continue
        for cohort in ("baseline", "baseline_before", "baseline_during", "baseline_after"):
            n, s = normal["groups"][cohort], shock["groups"][cohort]
            assert n["passengers"] == s["passengers"], "Baseline arrivals changed across scenes"
            effects.append(dict(method=method, training_seed=seed, evaluation_seed=day, cohort=cohort,
                **{m + "_delta": s[m] - n[m] for m in METRICS}))
    for seed, day, scene in product(P["training_seeds"], P["evaluation_seeds"], P["scenes"]):
        re, cs = rows.get(("resac", seed, day, scene)), rows.get(("csbapr", seed, day, scene))
        if re is None or cs is None:
            continue
        for cohort in COHORTS:
            a, b = re["groups"][cohort], cs["groups"][cohort]
            assert a["passengers"] == b["passengers"], "Arrivals changed across algorithms"
            if not a["passengers"]:
                continue
            comparisons.append(dict(training_seed=seed, evaluation_seed=day, scene=scene, cohort=cohort,
                **{m + "_delta": b[m] - a[m] for m in METRICS}))
    impact = []
    for method, cohort in product(P["methods"], ("baseline", "baseline_before", "baseline_during", "baseline_after")):
        pairs = [r for r in effects if (r["method"], r["cohort"]) == (method, cohort)]
        if len(pairs) != 6:
            continue
        normal = next(r for r in aggregates if (r["method"], r["scene"], r["cohort"]) == (method, "normal", cohort))
        row = dict(method=method, cohort=cohort, pairs=6,
                   **{m + "_delta": mean(r[m + "_delta"] for r in pairs) for m in METRICS})
        row["waiting_increase_percent"] = 100 * row["mean_wait_seconds_delta"] / normal["mean_wait_seconds"]
        row["generalized_increase_percent"] = 100 * row["mean_generalized_seconds_delta"] / normal["mean_generalized_seconds"]
        row["completion_change_pp"] = 100 * row["completion_rate_delta"]
        impact.append(row)
    report = dict(experiment=P["name"], analysis_location="server", evaluations=len(rows), expected=24,
                  complete=not missing, missing=missing, aggregates=aggregates, impact=impact,
                  paired_impacts=effects, paired_method_differences=comparisons,
                  maximum_wall_seconds=max((r["wall_seconds"] for r in rows.values()), default=0),
                  maximum_rss_mb=max((r["peak_rss_mb"] for r in rows.values()), default=0),
                  interpretation=P["contract"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("evaluations", "expected", "complete", "maximum_wall_seconds", "maximum_rss_mb")}))


if __name__ == "__main__":
    collect()
