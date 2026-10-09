"""Pool compact response summaries on the server, never export states or actors."""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def pool(values):
    n = sum(v["n"] for v in values)
    avg = sum(v["sum"] for v in values) / n
    return dict(n=n, mean=avg,
        std=max(0., sum(v["sum_squares"] for v in values) / n - avg * avg) ** .5,
        minimum=min(v["minimum"] for v in values), maximum=max(v["maximum"] for v in values))


def actions(rows):
    n = sum(r["holding"]["n"] for r in rows)
    return dict(holding=pool([r["holding"] for r in rows]), logit=pool([r["logit"] for r in rows]),
        above55_fraction=sum(r["above55"] for r in rows) / n,
        above59_fraction=sum(r["above59"] for r in rows) / n,
        below1_fraction=sum(r["below1"] for r in rows) / n)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/*.json")]
    expected = set(product(P["drivers"], P["scenes"], P["days"]))
    assert {(r["driver"], r["scene"], r["day"]) for r in rows} == expected and len(rows) == len(expected)
    assert all(r["learner_updates"] == 0 and r["rollout"]["timestamp_cost_identity_passed"] for r in rows)
    assert all(r["previous_rollout_matched"] for r in rows if r["scene"] != "static3" and r["driver"] != "fixed0")
    output = dict(complete=True, evaluations=len(rows), analysis_location="server", training_runs=0,
        state_exports=0, maximum_wall_seconds=max(r["wall_seconds"] for r in rows),
        maximum_rss_mb=max(r["peak_rss_mb"] for r in rows), groups={})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["driver"], row["scene"]].append(row)
    for (driver, scene), cases in grouped.items():
        n = sum(r["features"]["n"] for r in cases)
        group = dict(cost_seconds=mean(r["rollout"]["cost"] for r in cases),
            closed_loop_holding_seconds=mean(r["rollout"]["mean_holding_seconds"] for r in cases),
            feature_mean=[sum(r["features"]["sum"][i] for r in cases) / n for i in range(92)],
            feature_min=[min(r["features"]["minimum"][i] for r in cases) for i in range(92)],
            feature_max=[max(r["features"]["maximum"][i] for r in cases) for i in range(92)],
            per_day=[dict(day=r["day"], cost=r["rollout"]["cost"], holding=r["rollout"]["mean_holding_seconds"])
                     for r in sorted(cases, key=lambda r: r["day"])], responses={})
        for actor in P["actors"]:
            values = [r["responses"][actor] for r in cases]
            summary = actions(values)
            bins = defaultdict(list)
            for value in values:
                for bucket in value["time_bins"]:
                    bins[bucket["start"]].append(bucket)
            summary["time_bins"] = [dict(start=start, **actions(bucket)) for start, bucket in sorted(bins.items())]
            summary["feature_logit_sensitivity"] = [mean(v["feature_logit_sensitivity"][i] for v in values) for i in range(92)]
            summary["feature_absolute_logit_sensitivity"] = [mean(v["feature_absolute_logit_sensitivity"][i] for v in values) for i in range(92)]
            summary["sampled_original"] = actions([v["sampled_original"] for v in values])
            summary["group_probes"] = {}
            for name in values[0]["group_probes"]:
                summary["group_probes"][name] = {}
                for factor in ["0.0", "0.3"]:
                    probes = [v["group_probes"][name][factor] for v in values]
                    summary["group_probes"][name][factor] = dict(**actions(probes),
                        mean_logit_change=mean(v["mean_logit_change"] for v in probes),
                        mean_holding_change=mean(v["mean_holding_change"] for v in probes))
            group["responses"][actor] = summary
        output["groups"][f"{driver}/{scene}"] = group
    (HERE / "analysis_summary.json").write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in ["complete", "evaluations", "training_runs", "maximum_wall_seconds", "maximum_rss_mb"]}))


if __name__ == "__main__":
    collect()
