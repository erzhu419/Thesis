"""Pair frozen cases on the server; keep individual passenger records remote."""

from review_common import HERE, P, write, distribution
import argparse
from pathlib import Path
import json
import time


def jsonlines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def guard_pairs():
    rows = []
    for method in range(4):
        for block in range(9):
            off = HERE / "runs/guard" / f"case{method*18+block:03d}"
            on = HERE / "runs/guard" / f"case{method*18+9+block:03d}"
            if not all((p / "results/ready.json").exists() for p in (off, on)):
                continue
            a, b = ({r["person"]: r for r in jsonlines(p / "passengers.jsonl")} for p in (off, on))
            if a.keys() != b.keys():
                raise RuntimeError("Guard pair has different exogenous passenger populations")
            triggers = jsonlines(on / "triggers.jsonl")
            cohorts = {name: set().union(*(set(t[name]) for t in triggers)) for name in
                       ("waiting_at_decision", "onboard_at_decision", "boarded_after_decision")}
            cohorts["all_departed"] = set(a)
            groups = {}
            for name, members in cohorts.items():
                members = members & a.keys()
                wait = [b[x]["wait"]-a[x]["wait"] for x in members]
                ride = [b[x]["ride"]-a[x]["ride"] for x in members]
                completed = [int(b[x]["completion"] is not None)-int(a[x]["completion"] is not None) for x in members]
                groups[name] = dict(passengers=len(members), wait_difference=distribution(wait),
                    ride_difference=distribution(ride), completion_difference=sum(completed),
                    generalized_difference=distribution([2*w+r for w,r in zip(wait,ride)]),
                    completed_only_with_guard=sum(a[x]["completion"] is None and b[x]["completion"] is not None for x in members),
                    completed_only_without_guard=sum(a[x]["completion"] is not None and b[x]["completion"] is None for x in members))
            rows.append(dict(method=P["guard"]["methods"][method], demand=P["guard"]["demands"][block//3],
                             block=block%3, groups=groups,
                             interpretation="on-minus-off; all_departed is the paired policy-day effect; exposure-selected cohorts are descriptive, not single-trigger causal effects"))
    write(HERE / "runs/guard/paired_summary.json", dict(completed_pairs=len(rows), expected_pairs=36, rows=rows))
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic()+48*3600
    while True:
        count = guard_pairs()
        if count == 36 or not args.wait:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Guard pairs incomplete")
        time.sleep(120)


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
