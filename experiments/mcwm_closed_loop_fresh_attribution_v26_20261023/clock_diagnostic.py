"""Read-only correction of the issue-ledger versus source-event clock audit."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


LINES = ("7X", "102X", "705X")
ARMS = ("fresh", "no_fresh")
HORIZON = 24000.


def _phase(normalized_clock):
    # These are the actual ExecutionAction float32 comparisons, rather than
    # conversion of the bookkeeping issue timestamp to wall-clock thresholds.
    clock = np.float32(normalized_clock)
    return "pre_floor" if clock < np.float32(.15) else "reserve_floor" if clock < np.float32(.75) else "tail"


def _new_group():
    return dict(issues=0, canonical_source_clock_mismatches=0,
                event_source_clock_mismatches=0, old_issue_clock_mismatches=0,
                phase_misclassified=0, offsets=Counter(), phase_transitions=Counter())


def _cell(path, examples):
    groups = {line: _new_group() for line in ("all", *LINES)}
    with path.open() as handle:
        for text in handle:
            if not text.strip():
                continue
            row = json.loads(text)
            issue, source = float(row["started_at"]), float(row["raw_s"][0])
            actual = np.float32(row["s"][0])
            expected = np.float32(np.float32(source)/HORIZON)
            canonical_bad = actual != expected
            event_bad = float(row["event"][2]) != source
            old_clock_bad = actual != np.float32(np.float32(issue)/HORIZON)
            old_phase = "pre_floor" if issue < .15*HORIZON else "reserve_floor" if issue < .75*HORIZON else "tail"
            actual_phase = _phase(actual)
            phase_bad = old_phase != actual_phase
            offset = f"{issue-source:g}"
            for line in ("all", str(row["line_id"])):
                if line not in groups:
                    continue
                group = groups[line]
                group["issues"] += 1
                group["canonical_source_clock_mismatches"] += int(canonical_bad)
                group["event_source_clock_mismatches"] += int(event_bad)
                group["old_issue_clock_mismatches"] += int(old_clock_bad)
                group["phase_misclassified"] += int(phase_bad)
                group["offsets"][offset] += 1
                if phase_bad:
                    group["phase_transitions"][old_phase+" -> "+actual_phase] += 1
            if phase_bad and len(examples) < 10:
                examples.append(dict(sequence=row["sequence"], line=row["line_id"],
                    issue_seconds=issue, source_event_seconds=source,
                    old_phase=old_phase, actual_f_phase=actual_phase))
    return {line: {key: dict(value) if isinstance(value, Counter) else value
                  for key, value in group.items()} for line, group in groups.items()}


def summarize(root, scenarios):
    root = Path(root)
    cells, examples, missing = [], [], []
    for scenario in scenarios:
        for arm in ARMS:
            path = root/"runs"/scenario/"seed6202"/arm/"prefix_trace.jsonl"
            if not path.exists():
                missing.append(dict(scenario=scenario, arm=arm))
                continue
            sample_start = len(examples)
            groups = _cell(path, examples)
            for example in examples[sample_start:]:
                example.update(scenario=scenario, arm=arm)
            cells.append(dict(scenario=scenario, arm=arm, groups=groups))
    totals = [cell["groups"]["all"] for cell in cells]
    result = dict(status="complete" if not missing else "incomplete", role="passive_clock_audit_correction",
        canonical_and_event_clock_exact=bool(totals) and all(
            row["canonical_source_clock_mismatches"] == row["event_source_clock_mismatches"] == 0 for row in totals),
        cells=cells, missing=missing, boundary_examples=examples,
        semantics=dict(source_clock="canonical s[0] must equal float32(float32(raw_s[0])/24000); event[2] must equal raw_s[0]",
            offset="started_at minus raw source-event time, in seconds; started_at is the issue-ledger clock",
            phase="Old audit groups by started_at; actual unchanged F compares canonical float32 s[0] with .15 and .75",
            scope="Counters only; no actor/environment clock changes, training, model selection, or fresh-cost acceptance gate",
            coverage_windows="History causality and diagnostic coverage windows continue to use started_at, unchanged"))
    if len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) > 10000:
        raise ValueError("Clock correction exceeds 10KB summary budget")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("stable", "pilot"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scenarios = ["stable"] if args.stage == "stable" else ["stable", "local_speed_drop"]
    result = summarize(args.root, scenarios)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation preserves the earlier audit and prior results.
    with args.out.open("x") as output:
        json.dump(result, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=result["status"], clock_exact=result["canonical_and_event_clock_exact"],
                         cells=len(result["cells"]), out=str(args.out))))


if __name__ == "__main__":
    main()
