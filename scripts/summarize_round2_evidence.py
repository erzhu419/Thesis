"""Summarize saved evaluation completeness and inference costs; no reruns."""
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
H2O = ROOT.parent / "sumo-rl/H2Oplus"
FORMAL = H2O / "reproducibility/h2oplus_v20_formal_confirmation_results"


def read(path):
    return json.loads(path.read_text())


def main():
    audit = read(FORMAL / "evaluation_audit_v1.json")
    ends, batch_max, complete_trips, limited = [], [], [], 0
    inference = defaultdict(list)
    for record in audit["result_files"]:
        row = read(H2O / record["path"])
        episode = row["episode"]
        ends.append(episode["observed_end_time_seconds"])
        limited += bool(episode["stopped_by_event_limit"] or episode["stopped_by_time_limit"])
        complete_trips.append(episode["completed_trip_contexts_observed"])
        batch_max.append(row["operations"]["simultaneous_batch_size"]["max"])
        method = row["method"]
        inference[method].append(row["inference_seconds_per_decision"]["mean"])
    logs = list((FORMAL / "evaluation_v4/single_jtl110gpu").glob("*.log"))
    logged = sum(len(re.findall(r"^\[\d+/135\] completed ", p.read_text(), re.M)) for p in logs)
    failure_lines = [line for p in logs for line in p.read_text().splitlines()
                     if re.search(r"failed|Traceback|ERROR|retry", line)]
    formal = dict(planned=sum(audit["coverage"]["algorithms"].values()),
                  retained=len(ends), end_seconds_range=[min(ends), max(ends)],
                  event_or_time_limited=limited, maximum_batch=max(batch_max),
                  completed_trip_range=[min(complete_trips), max(complete_trips)],
                  final_batch_log_count=len(logs), logged_completions=logged,
                  logged_failure_lines=failure_lines,
                  earlier_attempt_count="Not reconstructed; final batch is reported separately",
                  inference_ms={k: 1000 * mean(v) for k, v in inference.items()})
    costs = defaultdict(list)
    for path in (ROOT / "experiments/enhancement_v1/runs/mc").glob("*/results/train_summary.json"):
        row = read(path)
        value = row["phases"]["inference_actor_1000_batch1"]["wall_seconds"]
        costs[row["arm"]].append(value)
    # Seconds for 1,000 single-state calls numerically equal milliseconds per call.
    actor_costs = {k: dict(seeds=len(v), mean_ms=mean(v), range_ms=[min(v), max(v)])
                   for k, v in costs.items()}
    output = dict(formal=formal, same_budget_actor_inference=actor_costs)
    dest = ROOT / "workpapers/manuscript_audit/第二轮评审现存记录汇总.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
