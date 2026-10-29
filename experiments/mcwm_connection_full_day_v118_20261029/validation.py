"""Resident parsed-JSON equality for the unchanged full-day continuation."""
from contextlib import ExitStack
from itertools import chain, zip_longest
import json
from pathlib import Path


SEGMENTED = ("actions", "commands", "ticks", "exchanges", "waiting_changes", "visit_events")
CUMULATIVE = ("passengers", "visits")
OBSERVER_FIELDS = ("status", "checks", "errors", "ledger_exact", "start", "end", "ticks",
    "issues", "commands", "passengers", "visits", "rejected_commands", "passenger_totals",
    "periods", "ledger", "maximum_count_gap", "maximum_ledger_gap", "traffic_or_other_reasons")


def compare_segments(actual_paths, reference_paths):
    """Compare both segmented streams once, irrespective of their fork boundaries."""
    counts, unequal, first = [0, 0], 0, None
    absent = object()
    try:
        with ExitStack() as stack:
            sides = [chain.from_iterable([stack.enter_context(Path(path).open()) for path in paths])
                     for paths in (actual_paths, reference_paths)]
            for index, pair in enumerate(zip_longest(*sides, fillvalue=absent), 1):
                values = []
                for side, line in enumerate(pair):
                    if line is absent:
                        values.append(absent)
                    else:
                        counts[side] += 1
                        values.append(json.loads(line))
                if values[0] != values[1]:
                    unequal += 1
                    if first is None:
                        first = index
    except (OSError, ValueError) as error:
        return dict(all_pass=False, observed_rows=counts, error=str(error))
    return dict(all_pass=unequal == 0, observed_rows=counts,
                unequal_rows=unequal, first_difference=first)


def compare_nominal(parent_dir, child_dir, reference_root):
    parent, child, old = map(Path, (parent_dir, child_dir, reference_root))
    old_child = old / "branches/low"
    files = {name + ".jsonl": compare_segments(
        [parent / (name + ".jsonl"), child / (name + ".jsonl")],
        [old / (name + ".jsonl"), old_child / (name + ".jsonl")]) for name in SEGMENTED}
    files.update({name + ".jsonl": compare_segments(
        [child / (name + ".jsonl")], [old_child / (name + ".jsonl")]) for name in CUMULATIVE})
    observer, ledger = {}, None
    try:
        actual = json.loads((child / "observer_summary.json").read_text())
        reference = json.loads((old_child / "observer_summary.json").read_text())
        equal = {key: key in actual and key in reference and actual[key] == reference[key]
                 for key in OBSERVER_FIELDS}
        complete = all(row.get("status") == "complete" and row.get("ledger_exact") is True
            and bool(row.get("checks")) and all(value is True for value in row["checks"].values())
            and row.get("end") == 24000 and row.get("ledger", {}).get("time_seconds") == 24000
            for row in (actual, reference))
        row_counts = {name: files[name + ".jsonl"]["observed_rows"][0] == actual.get(field)
                      for name, field in (("actions", "issues"), ("commands", "commands"),
                                          ("ticks", "ticks"), ("passengers", "passengers"), ("visits", "visits"))}
        observer = dict(all_pass=complete and all(equal.values()) and all(row_counts.values()),
                        fields_equal=equal, complete_horizon=complete, row_counts=row_counts)
        ledger = actual.get("ledger")
    except (OSError, ValueError) as error:
        observer = dict(all_pass=False, error=str(error))
    passed = all(row["all_pass"] for row in files.values()) and observer["all_pass"]
    return dict(all_pass=passed, files=files, observer=observer, final_ledger=ledger,
        reference_root=str(old), compared_branch="low",
        semantics="Full parent+low JSON records equal across different fork boundaries; cumulative passenger/visit and observer accounting exact. No learner, optimizer, update-log or policy-equivalence claim.")
