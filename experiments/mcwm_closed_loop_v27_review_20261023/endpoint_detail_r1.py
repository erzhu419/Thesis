"""Read resident v27 endpoints only; no training, checkpoint or trace reads."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path


METRICS = ("proposal_seconds", "holding_seconds", "reward_scaled",
           "bootstrap_contribution", "bellman_target", "q_advantage_mean", "q_advantage_min")
POINTS = ("before", "fresh", "no_fresh")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def point(row, arm):
    value = row[arm]
    advantages = {f"q_advantage_{kind}": value[f"q_on_post_actor_{kind}"]
                  - value[f"q_on_before_actor_{kind}"] for kind in ("mean", "min")}
    return [advantages[key] if key in advantages else value.get(key) for key in METRICS]


def f_relation(row):
    """Describe observed endpoint transmission, without guessing reserve/floor masks."""
    dp = row["fresh"]["proposal_seconds"]-row["no_fresh"]["proposal_seconds"]
    dh = row["fresh"]["holding_seconds"]-row["no_fresh"]["holding_seconds"]
    if abs(dp) <= 1e-6:
        return "unchanged_proposal" if abs(dh) <= 1e-6 else "holding_only_change"
    if abs(dh) <= 1e-6:
        return "flat_observed"
    if abs(dh-.125*dp) <= 2e-5:
        return "eighth_observed"
    return "other_transmission"


def rounded(value):
    if isinstance(value, float):
        return float(format(value, ".8g"))
    if isinstance(value, list):
        return [rounded(part) for part in value]
    if isinstance(value, dict):
        return {key: rounded(part) for key, part in value.items()}
    return value


def analyze(root):
    groups, state_rows, runs, missing = defaultdict(list), [], [], []
    counts, main_checks, shadow_checks = Counter(), defaultdict(Counter), defaultdict(Counter)
    windows = []
    for seed in (6201, 6202):
        for scenario in ("stable", "local_speed_drop"):
            directory = root/"runs"/f"seed{seed}"/scenario
            summary = json.loads((directory/"summary.json").read_text())
            if summary["status"] != "complete" or not summary["integrity"]["all_pass"]:
                raise ValueError(f"Native integrity is incomplete: {seed}/{scenario}")
            wall = summary["execution_wall"]
            windows.append([f"{seed}/{scenario}", wall["native_started_wall"], wall["native_finished_wall"]])
            runs.append(dict(seed=seed, scenario=scenario, post_budget_counts=summary["post_budget_counts"],
                main_training_stopped_at=summary["main_training_stopped_at"],
                native_wall_seconds=wall["native_finished_wall"]-wall["native_started_wall"],
                execution_wall=wall, native_check_count=len(summary["integrity"]["checks"])))
            with (directory/"shadows.jsonl").open() as handle:
                records = [json.loads(text) for text in handle if text.strip()]
            for record in records:
                meta = record["metadata"]
                prefix = [seed, scenario, meta["deadline"]]
                counts["anchors"] += 1
                for name, value in record["fresh_main_replay"]["checks"].items():
                    main_checks[name]["passed" if value is True else "failed"] += 1
                for name, value in record["integrity"].items():
                    shadow_checks[name]["passed" if value is True else "failed"] += 1
                for line in meta.get("missing_probes", []):
                    missing.append([*prefix, line])
                for row in record["probe_rows"]:
                    role, line = row["role"], row.get("line_id")
                    relation = f_relation(row)
                    groups[(*prefix, role, line)].append((row, relation))
                    counts["probe_occurrences"] += 1
                    counts["state_only" if not row["has_transition"] else "transition"] += 1
                    if row["has_transition"]:
                        dt = row.get("dt_seconds")
                        counts["missing_dt" if not finite(dt) else "zero_dt" if dt == 0
                               else "negative_dt" if dt < 0 else "positive_dt"] += 1
                        counts["terminal"] += bool(row.get("terminal"))
                    else:
                        clock = row.get("canonical_clock", row["issue_time"])
                        values = [point(row, arm) for arm in POINTS]
                        state_rows.append([*prefix, role, line, row.get("sequence"), row.get("started_at"), row["issue_time"],
                            clock, [value[0] for value in values], [value[1] for value in values],
                            [value[5:] for value in values], relation])
    grouped = []
    for key, entries in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        values = [[point(row, arm) for arm in POINTS] for row, _ in entries]
        ns, before, fresh_delta, paired_delta = [], [], [], []
        for index in range(len(METRICS)):
            valid = [[branch[index] for branch in value] for value in values
                     if all(finite(branch[index]) for branch in value)]
            n = len(valid)
            ns.append(n)
            before.append(sum(value[0] for value in valid)/n if n else None)
            fresh_delta.append(sum(value[1]-value[0] for value in valid)/n if n else None)
            paired_delta.append(sum(value[1]-value[2] for value in valid)/n if n else None)
        grouped.append([*key, len(entries), ns, before, fresh_delta, paired_delta,
                        dict(Counter(relation for _, relation in entries))])
    events = sorted([(start, 1) for _, start, _ in windows]+[(stop, -1) for _, _, stop in windows])
    running = peak = 0
    for _, change in events:
        running += change
        peak = max(peak, running)
    overlap = [[a[0], b[0], max(0., min(a[2], b[2])-max(a[1], b[1]))]
               for a, b in combinations(windows, 2)]
    return rounded(dict(schema=1, runs=runs, counts=dict(counts), missing_state_probes=missing,
        fresh_main_replay_checks=dict(main_checks), shadow_integrity_checks=dict(shadow_checks),
        metric_columns=list(METRICS), point_columns=list(POINTS),
        group_columns=["seed", "scenario", "deadline", "role", "line", "n", "finite_n",
            "before", "fresh_minus_before", "fresh_minus_no_fresh", "F_relation_counts"],
        grouped=grouped,
        state_columns=["seed", "scenario", "deadline", "role", "line", "sequence", "started_at", "issue_time",
            "canonical_clock", "proposal_seconds_by_point", "holding_seconds_by_point",
            "Q_advantage_mean_min_by_point", "F_relation"], states=state_rows,
        wall=dict(first_native_start=min(row[1] for row in windows),
            last_native_finish=max(row[2] for row in windows), max_simultaneous_native_days=peak,
            native_wall_sum_seconds=sum(row[2]-row[1] for row in windows),
            overlap_columns=["day_a", "day_b", "seconds"], pair_overlaps=overlap),
        interpretation="Q advantage uses each arm's own critic: Q(post actor)-Q(before actor). "
            "No-fresh-minus-before is derived as fresh-minus-before minus fresh-minus-no-fresh. "
            "Endpoint F transmission is observed, not a reserve/floor mask. State-only target nulls "
            "are intentional. started_at is the reference event clock; state-only issue_time "
            "is decoded from canonical state. These fresh-prefix shadows have no realized control returns."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    encoded = json.dumps(analyze(args.root), separators=(",", ":"), allow_nan=False)+"\n"
    size = len(encoded.encode("utf-8"))
    if size > 50000:
        raise ValueError(f"Endpoint detail exceeds small-pull budget: {size} bytes")
    with args.out.open("x") as handle:
        handle.write(encoded)
    print(json.dumps(dict(status="complete", output=str(args.out), bytes=size)))


if __name__ == "__main__":
    main()
